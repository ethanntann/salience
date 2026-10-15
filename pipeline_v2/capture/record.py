"""Record a play session, then turn it into highlights. The product's entry point.

    python -m pipeline_v2.capture.record          (or double-click Start Recording.cmd)

Nothing here touches the game process: video is Windows Graphics Capture of the
game window, audio is WASAPI loopback plus the microphone, input is
`GetAsyncKeyState`/`GetLastInputInfo` polling. Anti-cheat sees an ordinary
screen recorder.

While the game runs, nothing but capture happens. Detectors only *journal*
moments (free). When recording stops, the whole session is cut into 8s
windows (stream copy, on the encoder's 2s keyframes), the mic is transcribed
locally, every window is described, and the lot is ranked by the player's
taste (`highlights.run`). Then the highlights folder opens.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

DEFAULT_OUT = Path.home() / "Videos" / "Salience" / "Recordings"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record gameplay, then cut and rank highlights by your taste."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"recording directory (default: {DEFAULT_OUT})")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="8M")
    parser.add_argument("--monitor", dest="window", action="store_false",
                        help="capture monitor 1 instead of the game window")
    # Kept so the command from earlier sessions still works; the window is the default.
    parser.add_argument("--window", dest="window", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--wait", type=float, default=300.0,
                        help="seconds to wait for the game window (default 300)")
    parser.add_argument("--no-audio", dest="audio", action="store_false",
                        help="record video only (audio is on by default)")
    parser.add_argument("--mic", default="HyperX",
                        help="substring of the microphone to record, or '' for none")
    parser.add_argument("--game", default="fortnite",
                        help="which game adapter to add to the universal detectors")
    parser.add_argument("--no-highlights", dest="highlights", action="store_false",
                        help="skip cutting and ranking highlights when recording stops")
    parser.add_argument("--top", type=int, default=10,
                        help="how many highlights to keep (default 10)")
    parser.add_argument("--no-triggers", dest="triggers", action="store_false",
                        help="record without journalling detector moments")
    parser.add_argument("--no-speech", dest="speech", action="store_false",
                        help="do not transcribe the mic when ranking highlights")
    parser.add_argument("--no-open", dest="open", action="store_false",
                        help="do not open the highlights folder when done")
    return parser


def _summary(stats) -> str:
    """The line printed when recording stops.

    Written frames above captured ones are the pacer holding 30fps through
    static scenes, not loss; superseded ones are captures overwritten before
    encoding. (This once read fields `WgcStats` does not have, and the crash in
    the `finally` skipped making highlights.)
    """
    return (f"\n  recorded {stats.written} frames from {stats.delivered} captured "
            f"({stats.superseded_rate:.1%} superseded)")


def _save_stats(session: Path, stats, error: str | None) -> None:
    """What the capture produced, kept beside the footage: the console that
    printed it is usually closed before anyone reads it."""
    from dataclasses import asdict

    from pipeline_v2.jsonio import write_json

    record = {k: v for k, v in asdict(stats).items() if k != "first_frame_at"}
    record.update(superseded_rate=stats.superseded_rate, error=error)
    if stats.first_frame_at is not None:
        # Wall-clock t=0 of the video, which `gameplay` lines Fortnite's log up
        # against. `first_frame_at` is monotonic and means nothing once the
        # process exits, so it is converted while both clocks are at hand.
        from datetime import datetime, timezone

        started = time.time() - (time.monotonic() - stats.first_frame_at)
        record["first_frame_utc"] = datetime.fromtimestamp(started, tz=timezone.utc).isoformat()
    try:
        write_json(session / "capture_stats.json", record)
    except OSError:
        pass  # never lose the highlights step over a stats file


def _wait_for_game(wait: float):
    from pipeline_v2.capture.wgc import game_window

    deadline = time.monotonic() + wait
    hwnd = game_window()
    if hwnd is None:
        print("waiting for the Fortnite window... (Ctrl+C to cancel)")
    while hwnd is None and time.monotonic() < deadline:
        time.sleep(2.0)
        hwnd = game_window()
    return hwnd


def main(argv: list[str] | None = None) -> int:
    from pipeline_v2.capture import clips
    from pipeline_v2.capture.input_activity import ActivitySampler
    from pipeline_v2.capture.marks import MarkRecorder
    from pipeline_v2.capture.triggers import TriggerBus, detectors_for
    from pipeline_v2.capture.wgc import WgcRecorder

    args = _parser().parse_args(argv)

    hwnd = None
    if args.window:
        try:
            hwnd = _wait_for_game(args.wait)
        except KeyboardInterrupt:
            return 1
        if hwnd is None:
            raise SystemExit("Fortnite window not found -- start the game first, "
                             "or pass --monitor to record the whole screen")

    session = args.out / time.strftime("%Y%m%d-%H%M%S")
    rec = WgcRecorder(session, fps=args.fps, bitrate=args.bitrate,
                      monitor_index=None if hwnd else 1, window_hwnd=hwnd,
                      audio=args.audio, mic_hint=args.mic or None)
    rec.start()
    marker = MarkRecorder(journal=session / "marks.tsv").start()
    activity = ActivitySampler(journal=session / "input.tsv").start()
    print(f"recording -> {session}")
    print(f"  target: {'game window' if hwnd else 'monitor 1'} @ {args.fps}fps {args.bitrate}")
    print(f"  audio : {'game + mic' if args.audio else 'OFF'}")
    print("  F9    = mark this moment as worth keeping")
    print("  F10   = mark it as nothing happening")
    print("  Ctrl+C to stop")

    # Detector moments are journalled as a hint and for measurement; with full
    # coverage they no longer decide what gets described.
    bus = TriggerBus(detectors_for(args.game)) if args.triggers else None
    journal = session / clips.JOURNAL_NAME

    stop = False

    def handle(*_):
        nonlocal stop
        stop = True

    # SIGTERM as well as SIGINT: the paired benchmark driver stops this with
    # `terminate()`, which otherwise skips the cleanup below and loses both the
    # marks and the audio statistics for that round.
    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    seen_samples = 0
    try:
        while not stop:
            time.sleep(0.5)
            if bus is None:
                continue
            fresh = activity.samples[seen_samples:]
            seen_samples += len(fresh)
            mic = rec.mic_level()
            # Triggers are placed on the *video's* timeline, whose t=0 is the
            # first frame the encoder received -- not when the sampler started.
            origin = rec.stats.first_frame_at
            for sample in fresh:
                # Stamped with the sample's own time, not whenever this loop
                # woke: a 0.5s sleep hands over several windows at once, and
                # stamping them all "now" collapses the rolling baseline.
                fired = bus.observe(
                    press_rate=sample.rate, mic_db=mic,
                    now=activity.started + sample.at,
                )
                if fired is None or origin is None:
                    continue  # no frame encoded yet, so nothing to cut from
                clips.append(journal, clips.ClipRequest.from_trigger(fired, origin))
    finally:
        s = rec.stop()
        activity.stop()
        marks = marker.stop()
        marker.save(session / "marks.json")
        keeps = sum(1 for m in marks if m.kind == "keep")
        if marks:
            print(f"  marks : {keeps} keep, {len(marks) - keeps} dull -> marks.json")
        print(_summary(s))
        _save_stats(session, s, rec.error)
        if rec.error:
            print(f"  error: {rec.error}")

    if not args.highlights or rec.error:
        print(f"  make highlights later with:\n"
              f"  python -m pipeline_v2.capture.highlights {session}")
        return 1 if rec.error else 0

    # Runs after capture has fully stopped, so cutting (GPU) and describing
    # (cloud) never compete with the game. Interrupting loses nothing: cutting
    # resumes, and every paid description is cached.
    from pipeline_v2.capture import highlights

    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\nmaking highlights (Ctrl+C to stop; resume later with")
    print(f"  python -m pipeline_v2.capture.highlights {session})")
    try:
        highlights.report(highlights.run(session, top=args.top, speech=args.speech))
    except KeyboardInterrupt:
        print("  stopped; nothing lost")
        return 0
    out = session / "highlights"
    print(f"  -> {out}")
    if args.open and out.is_dir() and hasattr(os, "startfile"):
        os.startfile(out)  # noqa: S606 -- the player's own folder, in Explorer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
