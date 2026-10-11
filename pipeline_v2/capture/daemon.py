"""Watch desktop audio, save the buffer when something loud happens.

This is the loop the whole capture design rests on, and the one part of it that
answers a question rather than implementing a known solution: **does the audio
trigger fire at the moments worth clipping?**

The threshold (+9.5 dB over a rolling baseline) comes from clip-level envelope
analysis, and the correlation justifying it (+0.237 with a clip being kept) was
measured on clips *another* tool had already decided to save. Whether the same
signal works as a live trigger is unproven. Hence `--dry-run`, which logs every
firing decision without recording anything, so the threshold can be tuned against
a real session cheaply.

Deliberately video-only for now. A silent clip is not shippable, but audio muxing
is engineering with a known fallback, whereas trigger quality is a question -- and
building the muxing on top of a threshold that might change is work against a
moving target.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline_v2 import config  # noqa: E402
from pipeline_v2.capture.recorder import RecorderConfig, SegmentRecorder  # noqa: E402
from pipeline_v2.capture.trigger import (  # noqa: E402
    LevelTrigger,
    SustainedConfig,
    SustainedTrigger,
    TriggerConfig,
)

# Meter cadence. 20Hz is well under the audio block rate and gives the rolling
# median enough samples to be stable without burning CPU on sorting.
POLL_HZ = 20

# Function keys are contiguous, so that half is arithmetic. The punctuation keys
# are the OEM range, whose codes are only stable on a US layout -- fine for a
# tuning tool, worth revisiting if this ever ships to users.
VK_MARK_KEYS = {f"F{n}": 0x6F + n for n in range(1, 13)} | {
    "[": 0xDB,
    "]": 0xDD,
    "\\": 0xDC,
    ";": 0xBA,
    "'": 0xDE,
    ",": 0xBC,
    ".": 0xBE,
    "/": 0xBF,
    "-": 0xBD,
    "=": 0xBB,
    "`": 0xC0,
}


def make_mark_poller(key: str):
    """Return a predicate that reports whether `key` was tapped since last call.

    `GetAsyncKeyState` rather than console input or a keyboard hook: the game
    holds focus for the whole session, so `msvcrt` never sees a keypress, and a
    low-level hook needs a message pump and a thread for no benefit here. This
    is a poll of global key state, costs nothing, and needs no admin rights.

    Bit 0 means "pressed since the previous call", which is what makes a 20Hz
    poll safe -- a quick tap landing between two polls is still reported rather
    than missed.
    """
    import ctypes

    vk = VK_MARK_KEYS.get(key.upper())
    if vk is None:
        supported = " ".join(sorted(VK_MARK_KEYS))
        raise SystemExit(f"unsupported mark key {key!r}; use one of: {supported}")
    user32 = ctypes.windll.user32
    user32.GetAsyncKeyState(vk)  # discard state predating the session

    def tapped() -> bool:
        return bool(user32.GetAsyncKeyState(vk) & 0x0001)

    return tapped


# Two labels because they answer different questions. `clip` is the real target
# -- what you would actually want saved -- but it is rare and subjective. `kill`
# is objective and plentiful, which is what makes recall measurable at all. If
# audio tracks kills but not clips, audio is a gate and the VLM does the judging.
MARK_LABELS = {"]": "clip", "[": "kill"}


def preflight(pollers: dict, timeout: float = 30.0) -> bool:
    """Require every mark key to be pressed once before the session starts.

    Exists because the obvious alternative -- pressing a key early to check it
    -- lands the press inside the trigger's warmup, where firing is impossible.
    Session-1 lost a third of its ground truth that way. Confirming before the
    clock starts keeps the check out of the data entirely.
    """
    pending = dict(pollers)
    print("\nkey check -- press each once:")
    for key, label in ((k, MARK_LABELS.get(k, "mark")) for k in pending):
        print(f"   {key}  = {label}")
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for key in list(pending):
            if pending[key]():
                print(f"   {key} ok")
                del pending[key]
        time.sleep(1.0 / POLL_HZ)
    if pending:
        missing = " ".join(sorted(pending))
        print(f"\nno response from: {missing}")
        print(
            "Keys are not reaching the poll. Try borderless windowed rather "
            "than exclusive fullscreen, or pick different keys."
        )
        return False
    print("all keys ok\n")
    return True


@dataclass
class Session:
    fired: int = 0
    saved: int = 0
    marked: int = 0
    started: float = 0.0

    def summary(self) -> str:
        minutes = max((time.monotonic() - self.started) / 60.0, 1e-6)
        return (
            f"{minutes:.1f} min | {self.fired} triggers "
            f"({self.fired / minutes * 60:.1f}/hour) | {self.marked} marked "
            f"| {self.saved} clips saved"
        )


def _db(block) -> float:
    """Peak of an audio block in dBFS. Peak, not RMS: a gunshot is a transient,
    and RMS over a 50ms window flattens exactly the thing being detected."""
    import numpy as np

    peak = float(np.abs(block).max()) if len(block) else 0.0
    return 20.0 * np.log10(peak) if peak > 1e-7 else -120.0


def pick_speaker(name_fragment: str | None):
    """Resolve which output to listen to.

    The default output is not necessarily where game audio goes -- this machine
    defaults to a Steam virtual device. Listening to the wrong endpoint produces
    silence and a trigger that never fires, which looks identical to a broken
    threshold.
    """
    import soundcard as sc

    if not name_fragment:
        return sc.default_speaker()
    needle = name_fragment.lower()
    for speaker in sc.all_speakers():
        if needle in str(speaker.name).lower():
            return speaker
    names = [str(s.name) for s in sc.all_speakers()]
    joined = ("\n  ").join(names)
    raise SystemExit(f"no output matching {name_fragment!r}. Available:\n  {joined}")


def run(
    out_dir: Path,
    dry_run: bool,
    threshold_db: float,
    seconds: float,
    device: str | None = None,
    monitor: bool = False,
    log_path: Path | None = None,
    mark_keys: str = "][",
    detector: str = "sustained",
) -> int:
    import csv

    import soundcard as sc

    speaker = pick_speaker(device)
    loopback = sc.get_microphone(str(speaker.name), include_loopback=True)
    print(f"listening to: {speaker.name}")

    # Both run on every sample regardless of which one is armed: the log needs
    # both columns so one session can settle the peak-vs-sustained question
    # offline, rather than costing a session per detector.
    peak = LevelTrigger(TriggerConfig(threshold_db=threshold_db))
    sustained = SustainedTrigger(SustainedConfig(threshold_db=threshold_db))
    trigger = sustained if detector == "sustained" else peak
    print(f"armed detector: {detector} at +{threshold_db} dB")

    recorder = None
    status = 0
    pending_at = None
    saving = None
    with ExitStack() as resources:
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            recorder = SegmentRecorder(config.DATA_DIR / "ring", RecorderConfig())
            resources.callback(recorder.stop)
            recorder.start()
            print(f"recording: {recorder.buffer_seconds}s buffer -> {out_dir}")
            time.sleep(2)
            if not recorder.running:
                print("ffmpeg failed:", recorder.error_output()[:300])
                return 1
        else:
            print("DRY RUN - logging trigger decisions only, nothing recorded")

        # Ground truth, marked by hand while playing. Without it a session records
        # only what the trigger *did*, never whether it should have -- which makes
        # the recording unusable for measuring recall, the number that matters.
        pollers = {key: make_mark_poller(key) for key in mark_keys}
        if pollers and not preflight(pollers):
            return 1

        # Every sample, not just the firings. A full level history lets any
        # candidate threshold be replayed offline against one recorded session,
        # instead of costing a fresh session per value tried.
        log_file = log_writer = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            resources.callback(log_file.close)
            log_writer = csv.writer(log_file)
            log_writer.writerow(
                [
                    "elapsed_s",
                    "wall_clock",
                    "level_db",
                    "baseline_db",
                    "delta_db",
                    "sustained_db",
                    "event",
                ]
            )
            print(f"logging samples -> {log_path}")

        session = Session(started=time.monotonic())
        last_meter = 0.0
        block = int(48000 / POLL_HZ)
        deadline = time.monotonic() + seconds if seconds > 0 else float("inf")

        executor = resources.enter_context(ThreadPoolExecutor(max_workers=1))
        try:
            with loopback.recorder(
                samplerate=48000, channels=2, blocksize=block
            ) as mic:
                while time.monotonic() < deadline:
                    level = _db(mic.record(numframes=block))
                    now = time.monotonic()

                    # Reproduce the trigger's own comparison so the threshold is
                    # tunable from the command line without editing the module.
                    # Computed once per sample because both the meter and the log
                    # need it; the trigger derives its own copy internally.
                    median = -120.0
                    if monitor or log_writer is not None:
                        levels = sorted(v for _, v in trigger._samples)
                        if levels:
                            median = levels[len(levels) // 2]

                    # Polled before the trigger so a mark and a firing in the same
                    # sample both land, and the log shows they coincided.
                    hits = [
                        MARK_LABELS.get(k, k) for k, poll in pollers.items() if poll()
                    ]
                    for label in hits:
                        stamp = datetime.now().strftime("%H:%M:%S")
                        if monitor:
                            print()
                        print(
                            f"  [{stamp}] MARK {label:<5} {level:6.1f} dB  "
                            f"(+{level - median:.1f})"
                        )

                    if monitor:
                        # A live meter: current level, the rolling baseline it is
                        # judged against, and how far above it sits. This is the tool
                        # for choosing a threshold against real gameplay rather than
                        # inheriting one from offline clip analysis.
                        bar = "#" * max(0, min(40, int((level + 60) / 1.5)))
                        # Throttled to ~2Hz: a 20Hz meter is unreadable and
                        # wastes a sort per frame. Overwrites in place on a
                        # terminal; falls back to plain lines when piped, so a
                        # captured log stays legible.
                        if now - last_meter >= 0.5:
                            last_meter = now
                            body = (
                                f"{level:7.1f} dB  base {median:7.1f}  "
                                f"+{level - median:5.1f}  {bar:<40}"
                            )
                            if sys.stdout.isatty():
                                print("\r  " + body, end="", flush=True)
                            else:
                                print("  " + body, flush=True)

                    # Both advance on every sample; only the armed one decides.
                    fired_peak = peak.observe(level, now=now)
                    fired_sustained = sustained.observe(level, now=now)
                    fired = fired_sustained if trigger is sustained else fired_peak
                    if fired:
                        if monitor:
                            print()
                        session.fired += 1
                        stamp = datetime.now().strftime("%H:%M:%S")
                        if recorder is None:
                            print(f"  [{stamp}] TRIGGER  {level:6.1f} dB  (dry run)")
                        else:
                            # Allow post-roll and wait for its segment to close.
                            if pending_at is None and saving is None:
                                pending_at = now + recorder.config.segment_seconds + 3.0
                                print(f"  [{stamp}] TRIGGER queued (3s post-roll)")

                    if saving is not None and saving.done():
                        try:
                            saved = saving.result()
                            session.saved += int(saved is not None)
                            print(f"  saved: {saved or 'SAVE FAILED'}")
                            if saved is None:
                                status = 1
                        except Exception as exc:
                            print(f"  SAVE FAILED: {exc}")
                            status = 1
                        saving = None
                    if pending_at is not None and now >= pending_at and saving is None:
                        name = datetime.now().strftime("clip-%Y%m%d-%H%M%S-%f.mp4")
                        saving = executor.submit(recorder.save, out_dir / name)
                        pending_at = None

                    session.marked += len(hits)
                    if log_writer is not None:
                        # Several events can land on one sample, so the column
                        # records them jointly rather than as exclusive states.
                        parts = [f"MARK:{label}" for label in hits]
                        if fired:
                            parts.append("TRIGGER")
                        log_writer.writerow(
                            [
                                f"{now - session.started:.2f}",
                                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                f"{level:.1f}",
                                f"{median:.1f}",
                                f"{level - median:.1f}",
                                f"{sustained.delta():.1f}",
                                "+".join(parts),
                            ]
                        )

                    if recorder is not None and not recorder.running:
                        print("ffmpeg died:", recorder.error_output()[:300])
                        status = 1
                        break
        except KeyboardInterrupt:
            print("\nstopping")
    # ExitStack has joined an outstanding save before stopping the recorder.
    if saving is not None:
        try:
            saved = saving.result()
            session.saved += int(saved is not None)
            if saved is None:
                status = 1
        except Exception as exc:
            print(f"  SAVE FAILED: {exc}")
            status = 1
    print("\n" + session.summary())
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="audio-triggered clip capture")
    parser.add_argument("--out", default=str(Path.home() / "Videos" / "Salience"))
    parser.add_argument(
        "--dry-run", action="store_true", help="log trigger decisions without recording"
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=9.5,
        help="dB above rolling median that counts as an event",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="stop after N seconds (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--device", default=None, help="substring of the output device to listen to"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="print a live level meter for threshold tuning",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="CSV path for the full level history (enables offline threshold sweeps)",
    )
    parser.add_argument(
        "--mark-keys",
        default="][",
        help="keys pressed in-game to mark ground truth; "
        "] = clip-worthy, [ = kill (default: ][)",
    )
    parser.add_argument(
        "--detector",
        choices=("peak", "sustained"),
        default="sustained",
        help="which detector is armed. Both are always logged; "
        "this only changes the live TRIGGER lines.",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        import soundcard as sc

        for s in sc.all_speakers():
            print(f"  {s.name}")
        return 0
    return run(
        Path(args.out),
        args.dry_run,
        args.threshold_db,
        args.seconds,
        args.device,
        args.monitor,
        Path(args.log) if args.log else None,
        args.mark_keys,
        args.detector,
    )


if __name__ == "__main__":
    raise SystemExit(main())
