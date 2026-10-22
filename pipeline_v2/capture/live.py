"""Describe the session while it is being played, so highlights are ready on stop.

Describing after the session was the wait: one 40-minute session is about 270
windows, and even in parallel that is minutes of frame extraction and cloud
calls after the player has quit. But the recorder closes a 30s segment every
30s, and Fortnite writes its log as the match happens, so by the time a
segment is on disk everything needed to describe its windows exists. This
thread does that, a segment at a time, and `highlights.run` at the end finds
nearly every window already described.

It only ever does work the final run would do anyway, in exactly the same
way, so it can never change the result:

- A window is cut only once all of it is on disk. A window cut short at the
  live edge would differ from the one the final run cuts, and be cut and paid
  for twice.
- A window is described only if the log already places it inside a match.
  Anything the log has not decided yet (no match logged, log missing) is left
  for the final run, which applies the same rule with the whole log.
- Descriptions go to the same cache and store; embeddings and ranking are left
  to the final run, which needs the whole session anyway.

Mid-match work is a trickle: one window and one frame at a time, decoded on
the GPU's video decoder (which the game does not use), below normal priority.
Measured on a 165fps game (2026-10-18): as a burst of 16 CPU decoders every
30s it dipped the game to 100-120; as the trickle it held ~160, the same as
capture alone. Between matches it catches up at full width. Cuts are stream
copies; the cloud calls cost the machine nothing.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import gameplay as gameplay_mod

# How often to look for newly finished footage: the segment length, since
# nothing new can be cut in between.
INTERVAL = 30.0
# Segments transcribed per look while the player is out of a match: about 10s
# of CPU, which is also the longest stopping the recorder can wait on it.
LISTEN_BATCH = 5
# Windows in flight mid-match: one, so the decoding is a
# trickle over the 30s rather than a burst. A segment holds about four windows.
WORKERS = 1
# Between matches, when nothing the player sees depends on the machine.
IDLE_WORKERS = 16


def _low_priority_run(*args, **kwargs):
    if hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
        kwargs.setdefault("creationflags", subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    return subprocess.run(*args, **kwargs)


def ready_windows(session: Path) -> tuple[list[clips_mod.ClipRequest], dict]:
    """Coverage windows that are fully recorded and inside a logged match."""
    segments = clips_mod.read_segments(session)
    if not segments:
        return [], {"spans": None, "why": "no footage yet"}
    frontier = segments[-1].end
    played = gameplay_mod.spans(session, frontier)
    if played["spans"] is None:
        return [], played
    half = clips_mod.COVERAGE_STRIDE / 2
    requests = clips_mod.coverage_requests(segments)
    return [
        r for r in requests
        if r.at + half <= frontier + 1e-6
        and gameplay_mod.overlaps(played["spans"], r.at - half, r.at + half)
    ], played


def step(session: Path, *, seen: set | None = None, cfg=None, workers: int = WORKERS,
         harvest_fn=None, describe_fn=None, store_fn=None, connect=None,
         stopping: threading.Event | None = None, listen=None) -> int:
    """Cut and describe whatever has become ready; how many were described.

    `seen` holds windows already dealt with, so a long session does not re-hash
    every window it has every 30s.
    """
    from pipeline_v2 import cache, cli, config, store

    seen = set() if seen is None else seen
    ready = [r for r in ready_windows(session)[0] if r.at not in seen]
    if not ready:
        return 0
    wanted = {r.at for r in ready}
    clips_mod.write_coverage(session)
    half = clips_mod.COVERAGE_STRIDE / 2
    clip_dir = session / "windows"
    cut = (harvest_fn or clips_mod.harvest)(
        session, clip_dir, journal_name=clips_mod.COVERAGE_JOURNAL,
        analysis_span=clips_mod.COVERAGE_STRIDE, lead=half, lag=half,
        skip=lambda r: r.at not in wanted, runner=_low_priority_run,
    )
    paths = [clip_dir / record["video_file"] for record in cut]
    if harvest_fn is None:
        segments = clips_mod.read_segments(session)
        gameplay_mod.annotate(session, clip_dir, segments[-1].end)
    conn = (connect or store.connect)()
    try:
        described = {row[0] for row in conn.execute("select clip_sha from descriptions")}
        fresh = [p for p in paths if p.is_file() and cache.clip_sha(p) not in described]
        # Tried once, whatever the outcome: a window that failed to cut or
        # describe here is the final run's to retry, not every later step's.
        seen.update(wanted)
        if not fresh:
            return 0
        if listen is not None:
            # Before describing: the prompt is told what was said (`notes`).
            listen(session, fresh)
        cfg = cfg or config.resolve(config.load_config())
        done = 0
        # One at a time, each stored before the next is asked about: a window's
        # prompt includes the one before it (`notes`). Catching up between
        # matches goes wide instead and lets the final run fill any gaps.
        batches = [[clip] for clip in fresh] if workers == 1 else [fresh]
        for batch in batches:
            for clip, outcome in (describe_fn or cli.describe_many)(cfg, batch, len(batch), workers):
                if outcome is None or isinstance(outcome, Exception):
                    continue  # the final run retries it, from cache if it was paid
                (store_fn or cli.store_described)(conn, cfg, clip, *outcome)
                done += 1
            if stopping is not None and stopping.is_set():
                break  # the rest is the final run's; stopping must not wait on it
        return done
    finally:
        conn.close()


def between_matches(session: Path) -> bool:
    """Whether the newest footage is outside a match the log knows about.

    False when the log places no match at all: that is also what a missing log
    looks like, and Whisper must not run mid-fight on a guess.
    """
    segments = clips_mod.read_segments(session)
    if not segments:
        return False
    frontier = segments[-1].end
    played = gameplay_mod.spans(session, frontier)
    return played["spans"] is not None and not gameplay_mod.overlaps(
        played["spans"], frontier - INTERVAL, frontier)


# Whisper's threads while a match may be on: one core's worth of work in
# bursts of ~0.4s a segment, never all of them.
LIVE_WHISPER_THREADS = 2


def _transcriber(holder: dict):
    if "model" not in holder:
        from pipeline_v2.audio import Transcriber

        holder["model"] = Transcriber(cpu_threads=LIVE_WHISPER_THREADS)
    return holder["model"]


def listen_windows(session: Path, windows: list[Path], holder: dict) -> bool:
    """Transcribe what is new and attach it to `windows`; False without Whisper."""
    from pipeline_v2.capture import voice

    try:
        voice.attach(session, windows, transcriber=_transcriber(holder))
    except ImportError:
        return False
    return True


def listen_step(session: Path, holder: dict, transcribe=None) -> bool:
    """Transcribe a batch of the mic between matches. False once Whisper is
    unavailable, so the caller stops asking."""
    from pipeline_v2.capture import voice

    try:
        _transcriber(holder)
    except ImportError:
        return False
    (transcribe or voice.transcribe_session)(
        session, transcriber=holder["model"], limit=LISTEN_BATCH)
    return True


class LiveDescriber:
    """`step` every `interval` seconds on a background thread until `stop`.

    With `speech`, the mic is also transcribed whenever the player is between
    matches (lobby, island, spectating), so little of it is left at the end.
    """

    def __init__(self, session: Path, interval: float = INTERVAL, step_fn=step,
                 speech: bool = True, idle_fn=between_matches, listen_fn=listen_step):
        self.session = session
        self.interval = interval
        self.step_fn = step_fn
        self.speech = speech
        self.idle_fn = idle_fn
        self.listen_fn = listen_fn
        self._whisper: dict = {}
        self.seen: set[float] = set()
        self.described = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="live-describe", daemon=True)

    def _listen(self, session: Path, windows: list[Path]) -> None:
        if self.speech:
            self.speech = listen_windows(session, windows, self._whisper)

    def start(self) -> "LiveDescriber":
        from salience_media import ffmpeg

        ffmpeg.LOW_PRIORITY = True
        self._thread.start()
        return self

    def _loop(self) -> None:
        try:
            # What ranking needs, loaded now rather than in the wait after the
            # player quits: sklearn alone is ~2s to import.
            import pipeline_v2.ranking  # noqa: F401
            import sklearn.linear_model  # noqa: F401
        except ImportError:
            pass
        # Every `interval` from the start of the last look, not after it: a
        # look takes about 7s a window mid-match, and waiting the whole 30s
        # after it covered about half the footage as it was recorded (session 5,
        # 2026-10-21: 36 windows behind at the end, 38s of the wait after it).
        due = time.monotonic() + self.interval
        while not self._stop.wait(max(0.0, due - time.monotonic())):
            due = time.monotonic() + self.interval
            try:
                idle = self.idle_fn(self.session)
                # Out of a match the machine is free, so catch up at full width.
                extra = {"workers": IDLE_WORKERS} if idle else {}
                if self.speech:
                    extra["listen"] = self._listen
                self.described += self.step_fn(
                    self.session, seen=self.seen, stopping=self._stop, **extra)
                if self.speech and idle and not self._stop.is_set():
                    self.speech = self.listen_fn(self.session, self._whisper)
            except Exception as exc:  # noqa: BLE001 -- never take the recording down
                # The final run describes and transcribes whatever this missed.
                self.error = f"{type(exc).__name__}: {exc}"

    def stop(self, timeout: float = 120.0) -> int:
        """Stop after the step in progress, then restore normal priority."""
        from salience_media import ffmpeg

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
        ffmpeg.LOW_PRIORITY = False
        return self.described
