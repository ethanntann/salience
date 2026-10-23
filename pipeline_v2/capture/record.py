"""Record a play session, then turn it into highlights. The product's entry point.

    python -m pipeline_v2.capture.record          (or double-click Start Recording.cmd)

Nothing here touches the game process: video is Windows Graphics Capture of the
game window, audio is WASAPI loopback plus the microphone, input is
`GetAsyncKeyState`/`GetLastInputInfo` polling. Anti-cheat sees an ordinary
screen recorder.

While the game runs, each finished 30s segment is cut into 8s windows (stream
copy, on the encoder's 2s keyframes) and the ones inside a match are described
(`live`), so describing is done by the time the player stops. Detectors only
*journal* moments (free). When recording stops, the mic is transcribed locally,
anything live did not reach is described, and the lot is ranked by the
player's taste (`highlights.run`). Then the highlights folder opens.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

DEFAULT_OUT = Path.home() / "Videos" / "Salience" / "Recordings"

# (fps, video bitrate). "standard" is what sessions 1-5 recorded at: fine for
# the describer, blocky in fights next to ShadowPlay's 1080p60. NVENC is its
# own chip on the GPU, so a higher preset costs disk rather than game frames.
QUALITY = {
    "standard": (30, "8M"),
    "high": (60, "25M"),
    "max": (60, "40M"),
}
DEFAULT_QUALITY = "high"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record gameplay, then cut and rank highlights by your taste."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"recording directory (default: {DEFAULT_OUT})")
    parser.add_argument("--fps", type=int, default=QUALITY[DEFAULT_QUALITY][0])
    parser.add_argument("--bitrate", default=QUALITY[DEFAULT_QUALITY][1])
    parser.add_argument("--height", type=int, default=None,
                        help="record at this many lines, e.g. 1080 or 720 (default: the "
                             "game window's own size; never scaled up)")
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


def _save_stats(session: Path, stats, error: str | None, audio=None,
                input_started: float | None = None) -> None:
    """What the capture produced, kept beside the footage: the console that
    printed it is usually closed before anyone reads it."""
    from dataclasses import asdict

    from pipeline_v2.jsonio import write_json

    record = {k: v for k, v in asdict(stats).items() if k != "first_frame_at"}
    record.update(superseded_rate=stats.superseded_rate, error=error)
    if audio is not None:
        record["audio"] = audio.summary()
    if stats.first_frame_at is not None:
        # Wall-clock t=0 of the video, which `gameplay` lines Fortnite's log up
        # against. `first_frame_at` is monotonic and means nothing once the
        # process exits, so it is converted while both clocks are at hand.
        from datetime import datetime, timezone

        started = time.time() - (time.monotonic() - stats.first_frame_at)
        record["first_frame_utc"] = datetime.fromtimestamp(started, tz=timezone.utc).isoformat()
        if input_started is not None:
            # The input and mouse logs count from when they started, the video
            # from its first frame, 0.5-0.8s later (measured): a flick is a
            # tenth of a second, so the gap is kept to line them up.
            record["input_offset_sec"] = round(stats.first_frame_at - input_started, 3)
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


class Session:
    """One recording: capture, input, detectors and live describing.

    `start`, then `tick` about every 0.5s, then `stop`. The terminal command
    drives it from a Ctrl+C loop and the tray app from a thread, so neither
    owns the other's signal handling.
    """

    def __init__(self, args: argparse.Namespace, hwnd=None, root: Path | None = None):
        self.args = args
        self.hwnd = hwnd
        self.path = (root or args.out) / time.strftime("%Y%m%d-%H%M%S")
        self.live = None
        self.warning: str | None = None
        self._seen_samples = 0
        self._origin_saved = False

    def start(self) -> "Session":
        from pipeline_v2.capture import clips
        from pipeline_v2.capture.input_activity import ActivitySampler
        from pipeline_v2.capture.mouse import JOURNAL_NAME as MOUSE_JOURNAL, SHOTS_NAME, MouseMotion
        from pipeline_v2.capture.triggers import TriggerBus, detectors_for
        from pipeline_v2.capture import gfx
        from pipeline_v2.capture.wgc import WgcRecorder

        args, session = self.args, self.path
        # Inside FFmpeg where it can (frames stay on the GPU: 3% of a core
        # against 63-112% through Python); the Python path for an FFmpeg
        # older than 8, which has no `gfxcapture`.
        gpu = gfx.available()
        recorder = gfx.GfxRecorder if gpu else WgcRecorder
        # Scaling is the GPU capture's; the Python fallback records the window's size.
        sized = {"height": args.height} if gpu and args.height else {}
        self.rec = recorder(session, fps=args.fps, bitrate=args.bitrate,
                            monitor_index=None if self.hwnd else 1, window_hwnd=self.hwnd,
                            audio=args.audio, mic_hint=args.mic or None, **sized)
        self.rec.start()
        self.activity = ActivitySampler(journal=session / "input.tsv").start()
        # On the input log's clock, so the two line up window for window.
        self.mouse = MouseMotion(journal=session / MOUSE_JOURNAL, shots=session / SHOTS_NAME,
                                 started=self.activity.started).start()
        print(f"recording -> {session}")
        print(f"  target: {'game window' if self.hwnd else 'monitor 1'} @ {args.fps}fps {args.bitrate}"
              + (f" {args.height}p" if sized else ""))
        print(f"  audio : {'game + mic' if args.audio else 'OFF'}")
        # Detector moments are journalled as a hint and for measurement; with
        # full coverage they no longer decide what gets described.
        self.bus = TriggerBus(detectors_for(args.game)) if args.triggers else None
        self.journal = session / clips.JOURNAL_NAME
        if args.highlights:
            from pipeline_v2.capture.live import LiveDescriber

            self.live = LiveDescriber(session).start()
            print("  live  : describing each finished 30s segment during play")
        return self

    @property
    def error(self) -> str | None:
        return self.rec.error

    def _check_audio(self) -> None:
        """Warn once if the game track is pure silence two minutes in.

        Session 2 recorded 43 minutes of digital silence on the game track
        (Fortnite was playing through another output than Windows' default),
        and nobody knew until the analysis.
        """
        stats = getattr(self.rec, "audio_stats", None)
        game = next((t for t in getattr(stats, "tracks", []) if t.name == "game"), None)
        if game is None or game.seconds < 120 or self.warning is not None:
            return
        if game.silent_fraction > 0.99:
            self.warning = ("Game audio is silent. Fortnite may be playing through a "
                            "different output than Windows' default sound device.")
            print(f"  WARNING: {self.warning}")

    def tick(self) -> None:
        from pipeline_v2.capture import clips

        rec = self.rec
        self._check_audio()
        if not self._origin_saved and rec.stats.first_frame_at is not None:
            # Written now rather than at stop: `live` lines the game log up
            # with the video while it is still recording.
            _save_stats(self.path, rec.stats, None, input_started=self.activity.started)
            self._origin_saved = True
        if self.bus is None:
            return
        fresh = self.activity.samples[self._seen_samples:]
        self._seen_samples += len(fresh)
        mic = rec.mic_level()
        # Triggers are placed on the *video's* timeline, whose t=0 is the
        # first frame the encoder received -- not when the sampler started.
        origin = rec.stats.first_frame_at
        for sample in fresh:
            # Stamped with the sample's own time, not whenever this loop
            # woke: a 0.5s sleep hands over several windows at once, and
            # stamping them all "now" collapses the rolling baseline.
            fired = self.bus.observe(
                press_rate=sample.rate, mic_db=mic,
                now=self.activity.started + sample.at,
            )
            if fired is None or origin is None:
                continue  # no frame encoded yet, so nothing to cut from
            clips.append(self.journal, clips.ClipRequest.from_trigger(fired, origin))

    def stop(self):
        """Stop everything and save what was recorded; the capture stats."""
        # It used to stop in silence while live finished its queue, and a second
        # Ctrl+C then landed in (and cancelled) the highlights step.
        print("\nstopping -- a few seconds, no need to press Ctrl+C again")
        s = self.rec.stop()
        if self.live is not None:
            print(f"  live  : {self.live.stop()} window(s) described during play"
                  + (f" (last error: {self.live.error})" if self.live.error else ""))
        self.activity.stop()
        self.mouse.stop()
        if self.mouse.error:
            print(f"  mouse : {self.mouse.error}")
        print(_summary(s))
        _save_stats(self.path, s, self.rec.error, getattr(self.rec, "audio_stats", None),
                    input_started=self.activity.started)
        if self.rec.error:
            print(f"  error: {self.rec.error}")
        return s


def main(argv: list[str] | None = None) -> int:
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

    recording = Session(args, hwnd).start()
    session = recording.path
    print("  Ctrl+C to stop")
    stop = False

    def handle(*_):
        nonlocal stop
        stop = True

    # SIGTERM as well as SIGINT: the paired benchmark driver stops this with
    # `terminate()`, which otherwise skips the cleanup below and loses the
    # audio statistics for that round.
    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    try:
        while not stop:
            time.sleep(0.5)
            recording.tick()
    finally:
        recording.stop()

    if not args.highlights or recording.error:
        print(f"  make highlights later with:\n"
              f"  python -m pipeline_v2.capture.highlights {session}")
        return 1 if recording.error else 0

    # What live did not reach (the last segment or two) is described here. Interrupting loses nothing: cutting resumes, and every
    # paid description is cached.
    from pipeline_v2.capture import highlights

    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\nmaking highlights, usually under a minute (Ctrl+C here postpones them)")
    try:
        highlights.report(highlights.run(session, top=args.top))
    except KeyboardInterrupt:
        print("  postponed; nothing lost. Finish them with:\n"
              f"  python -m pipeline_v2.capture.highlights {session}")
        return 0
    out = session / "highlights"
    print(f"  -> {out}")
    if args.open and out.is_dir() and hasattr(os, "startfile"):
        os.startfile(out)  # noqa: S606 -- the player's own folder, in Explorer
    print(f"  reels : {highlights.reels(session)} made, 9:16, in {out / highlights.REEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
