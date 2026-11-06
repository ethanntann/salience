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

The trickle still cost frames on a CPU-bound game: one match each
(2026-10-31, i5-14500HX), 161fps and 76 at the 1% low with only capture and
the input logs, 150/54 and 148/51 with this thread cutting and scanning,
nothing described. Its ffmpegs had been beside Fortnite on the performance
cores; they now run on the efficiency cores, in efficiency mode, on two
threads (`salience_media.ffmpeg.run`).

A window's prompt used to include what was said in it, so every window
waited for its stretch of the mic to be transcribed, and with speech off
mid-match nothing was described during play. It no longer does
(`notes.window_notes`), and a window is described as soon as it is cut:
mid-match where the efficiency cores take its frames, otherwise between
matches. Speech and laughter run on their own
thread (`LiveDescriber._listen_loop`), a segment at a time, at the lowest
priority: between matches always, mid-match only with schedule D
(`MATCH_SPEECH_KEY`). Session 7 (2026-10-22) transcribed mid-match before
each describe, at normal priority: about 2,000 CPU-seconds over the last
match, the player's fps dropping, and a look still running 12 minutes after
the stop. Session 2026-10-27 ran D in the describer's own loop on one thread:
no frames lost, but ten segments a look took ten minutes, nothing was
described during play, and highlights took 28 minutes.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import gameplay as gameplay_mod

# How often to look for newly finished footage: the segment length, since
# nothing new can be cut in between.
INTERVAL = 30.0
# Segments transcribed (and listened to for laughter) a call: one, so stopping
# the recorder waits on at most a segment's work. Ten mid-match took ten
# minutes a call and outlived the stop by nine (2026-10-27).
LISTEN_BATCH = 1
# Seconds the speech thread rests once it has caught up, or may not run.
LISTEN_WAIT = 5.0
# Windows in flight mid-match: one, so the decoding is a
# trickle over the 30s rather than a burst. A segment holds about four windows.
WORKERS = 1
# Between matches, when nothing the player sees depends on the machine.
IDLE_WORKERS = 16


def _low_priority_run(*args, **kwargs):
    if hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
        kwargs.setdefault("creationflags", subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    from salience_media import ffmpeg

    return ffmpeg.run(*args, **kwargs)  # off the game's cores while it runs


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


def _inside(spans: list[tuple[float, float]], start: float, end: float) -> bool:
    """Whether [start, end) is covered by `spans` (back-to-back segments join)."""
    at = start
    for first, last in sorted(spans):
        if first <= at + 1e-6 < last:
            at = last
    return at >= end - 1e-6


BANK_WORKERS = 2  # one segment on the GPU's decoder and one on the CPU
PLAN_WORKERS = 8


def ffmpeg_low_priority() -> bool:
    from salience_media import ffmpeg

    return ffmpeg.LOW_PRIORITY


def bank_frames(session: Path, windows: list[Path]) -> int:
    """Every frame `windows` will be described from, taken a segment at a time
    (`frames.bank`), so describing finds them there; how many were."""
    from collections import defaultdict

    from pipeline_v2 import cli, frames

    segments = clips_mod.read_segments(session)
    if not segments:
        return 0
    grid = clips_mod.keyframe_interval(segments[0].path)
    if not grid:
        return 0  # re-encoded windows start where asked; nothing to line up with
    from concurrent.futures import ThreadPoolExecutor

    def planned(clip):
        # Each a probe, a hash and the banners: 0.33s a window one after
        # another, half a minute of a 96-window session (2026-11-05).
        try:
            return clip, cli.frame_plan(clip), float(clips_mod._read_sidecar(clip)["segment_start_sec"])
        except (KeyError, TypeError, ValueError, OSError):
            return clip, None, None

    with ThreadPoolExecutor(1 if ffmpeg_low_priority() else PLAN_WORKERS) as pool:
        plans = list(pool.map(planned, windows))
    wanted: dict[Path, list] = defaultdict(list)
    for clip, plan, asked in plans:
        if plan is None:
            continue
        # A window is a stream copy: it begins on the first keyframe at or
        # after where it was asked to, a keyframe every `grid` seconds from
        # its segment's start (272.0 began at 272.021, 2026-11-03).
        home = next((s for s in segments if s.start <= asked < s.end), None)
        if home is None:
            continue
        start = home.start + math.ceil(round((asked - home.start) / grid, 6)) * grid
        folder, window, kills = plan
        for tag, times in (("window", window), ("kill", kills)):
            for index, at in enumerate(times):
                when = start + at
                segment = next((s for s in segments if s.start <= when < s.end), None)
                if segment is not None:
                    wanted[segment.path].append((folder, tag, index, at, when - segment.start))
    from concurrent.futures import ThreadPoolExecutor

    def take(item) -> int:
        index, (segment, frames_wanted) = item
        # Every other segment on the CPU: the GPU's decoder is one engine, so
        # four segments at once on it were no faster than one (49s for 24
        # windows), and after the game the CPU is as fast (2.3s a segment
        # against 2.2). During play, the GPU alone, one at a time.
        gpu = workers == 1 or index % 2 == 0
        try:
            return frames.bank(segment, frames_wanted, gpu=gpu)
        except (OSError, ValueError):
            return 0  # each window takes its own, as before

    workers = 1 if ffmpeg_low_priority() else BANK_WORKERS
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(take, enumerate(wanted.items())))


def prepare(paths: list[Path]) -> None:
    """The cheap work on a window as it is cut, mid-match: its kill banners
    (on the GPU's decoder) and its mic level, kept in its sidecar.

    About a fifth of a core while playing (2026-10-25), and the wait after
    the session no longer holds them: cutting alone was 2.5 minutes of it.
    Not cheap for the game, though: the banner scan's ffmpeg is 2.2
    CPU-seconds a window (the GPU decoder's start-up), cutting 1.0, the mic
    0.3 (2026-10-31), so all of it runs off the game's cores (`ffmpeg.run`).
    """
    from pipeline_v2 import banner
    from pipeline_v2.capture import highlights as hl, levels, voice as voice_mod
    from pipeline_v2.jsonio import write_json

    for path in paths:
        record = clips_mod._read_sidecar(path)
        if not record or not path.is_file():
            continue
        length = float(record.get("segment_end_sec", 0) or 0) - float(record.get("segment_start_sec", 0) or 0)
        banner.cached_found(path, length or clips_mod.COVERAGE_STRIDE, runner=_low_priority_run)
        record = clips_mod._read_sidecar(path)
        if hl.MIC_LEVEL_KEY not in record:
            record[hl.MIC_LEVEL_KEY] = levels.loudness(path, voice_mod.MIC_TRACK)
            try:
                write_json(path.with_suffix(".capture.json"), record)
            except OSError:
                pass


def step(session: Path, *, seen: set | None = None, cfg=None, workers: int = WORKERS,
         harvest_fn=None, describe_fn=None, store_fn=None, connect=None,
         stopping: threading.Event | None = None, listen=None, heard=None,
         prepared: set | None = None, prepare_fn=prepare, describe_now: bool = True) -> int:
    """Cut and describe whatever has become ready; how many were described.

    `seen` holds windows already dealt with, so a long session does not re-hash
    every window it has every 30s. With `heard` (the transcribed stretches), a
    window is ready only once all of it is transcribed; `listen` then attaches
    what was said, without transcribing. With `prepared`, every finished
    window is cut and prepared (`prepare`) as it is ready, heard or not.
    Without `describe_now`, windows are only cut and prepared.
    """
    from pipeline_v2 import cache, cli, config, store

    seen = set() if seen is None else seen
    half = clips_mod.COVERAGE_STRIDE / 2
    finished = ready_windows(session)[0]
    ready = [r for r in finished if r.at not in seen] if describe_now else []
    if ready and heard is not None:
        spans = heard(session)
        ready = [r for r in ready if _inside(spans, r.at - half, r.at + half)]
    early = {r.at for r in finished if prepared is not None and r.at not in prepared}
    wanted = {r.at for r in ready}
    if not wanted and not early:
        return 0
    clips_mod.write_coverage(session)
    clip_dir = session / "windows"
    cut = (harvest_fn or clips_mod.harvest)(
        session, clip_dir, journal_name=clips_mod.COVERAGE_JOURNAL,
        analysis_span=clips_mod.COVERAGE_STRIDE, lead=half, lag=half,
        skip=lambda r: r.at not in wanted | early, runner=_low_priority_run,
    )
    by_at = {}
    for record in cut:
        path = clip_dir / record["video_file"]
        side = clips_mod._read_sidecar(path)
        start = side.get("segment_start_sec")
        by_at[round(float(start) + half, 3) if start is not None else None] = path
    if early:
        prepare_fn([p for at, p in by_at.items() if at in {round(a, 3) for a in early}])
        prepared.update(early)
    if not wanted:
        return 0
    if early:  # describe only the ready ones; the rest were cut to be ready
        paths = [p for at, p in by_at.items() if at in {round(a, 3) for a in wanted}]
    else:
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
        if harvest_fn is None:
            bank_frames(session, fresh)
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
    looks like, and transcription must not run mid-fight on a guess.
    """
    segments = clips_mod.read_segments(session)
    if not segments:
        return False
    frontier = segments[-1].end
    played = gameplay_mod.spans(session, frontier)
    return played["spans"] is not None and not gameplay_mod.overlaps(
        played["spans"], frontier - INTERVAL, frontier)


# The audio models' threads (onnxruntime's pool), never all of them: four of
# the eight efficiency cores (2026-10-31: 14s a 30s segment, speech and
# laughter; two in efficiency mode were 48s, behind the recording).
LIVE_SPEECH_THREADS = 4

# Speech and laughter during matches too (schedule D, 2026-10-25): what makes
# the wait after a session a minute instead of ten, and what last cost the
# player frames (Parakeet mid-match, normal priority, spinning: session 7).
# "alternate" turns it on for every other match, so one session compares the
# two (`experiment.jsonl`, `presentmon`). Session 2026-10-27, one thread at the
# lowest priority: 134 fps with it, 150 without (2 minutes of it); and at half
# the speed of speech, so it never caught up. Now its own thread, every model
# thread at the lowest, on the efficiency cores and out of efficiency mode
# (`load_lowest`), as the ffmpegs that had cost 10fps (2026-10-31).
# Before the first match is in the log, a match may be on: D's rule applies.
MATCH_SPEECH_KEY = "SALIENCE_MATCH_SPEECH"
MATCH_SPEECH_MODES = ("off", "on", "alternate")
EXPERIMENT_LOG = "experiment.jsonl"


def match_speech_mode() -> str:
    mode = os.environ.get(MATCH_SPEECH_KEY, "off").strip().lower()
    return mode if mode in MATCH_SPEECH_MODES else "off"


def match_number(session: Path) -> int | None:
    """Which match the newest footage is in, counting from 0, or None in none."""
    segments = clips_mod.read_segments(session)
    if not segments:
        return None
    frontier = segments[-1].end
    played = gameplay_mod.spans(session, frontier) or {}
    for index, (start, end) in enumerate(sorted(played.get("spans") or [])):
        if start <= frontier <= end + INTERVAL:
            return index
    return None


def speech_allowed(mode: str, idle: bool, match: int | None) -> bool:
    """Whether speech and laughter may run now: between matches always; in a
    match (or before the log knows of one, `match` None) by schedule D."""
    if idle:
        return True
    return mode == "on" or (mode == "alternate" and (match is None or match % 2 == 0))


THREAD_PRIORITY_LOWEST = -2


def _full_speed(kernel, handle) -> None:
    """A thread out of efficiency mode (EcoQoS), which the tray is in while a
    game runs: on the efficiency cores it halved the audio models, 29s a
    30s segment against 14 (2026-10-31, four threads)."""
    import ctypes

    class Throttling(ctypes.Structure):  # THREAD_POWER_THROTTLING_STATE
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    kernel.SetThreadInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    state = Throttling(1, 0x1, 0)  # THREAD_POWER_THROTTLING_EXECUTION_SPEED, off
    kernel.SetThreadInformation(handle, 3, ctypes.byref(state), ctypes.sizeof(state))  # ThreadPowerThrottling


DESCRIBE_KEY = "SALIENCE_DESCRIBE_DURING_PLAY"


def work_during_play() -> bool:
    """Whether anything of the highlights runs while the game is open.

    No, unless describing or speech during play is asked for: cutting each
    window, scanning it for banners and measuring its mic as each segment
    closed still cost 7fps in bursts (154 against 161, 2026-11-05), and the
    player would rather wait a little longer after the game. Everything is
    done once it closes: speech on the GPU, frames a segment at a time.
    """
    return describe_during_play() or match_speech_mode() != "off"


def describe_during_play() -> bool:
    return os.environ.get(DESCRIBE_KEY, "off").strip().lower() == "on"


def speech_threads() -> int:
    """The audio models' threads: `LIVE_SPEECH_THREADS`, never more than the
    efficiency cores there are (a laptop may have two), and two without any."""
    mask = _efficiency_cores()
    return min(LIVE_SPEECH_THREADS, bin(mask).count("1")) if mask else 2


def _efficiency_cores() -> int | None:
    from salience_media import ffmpeg

    try:
        return ffmpeg._efficiency_cores()
    except (OSError, AttributeError, ValueError):
        return None


@contextlib.contextmanager
def lowest_thread_priority():
    """This thread at the lowest priority, on the efficiency cores where the
    CPU has them, while inside; then back."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentThread.restype = ctypes.c_void_p
    kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel.GetThreadPriority.argtypes = [ctypes.c_void_p]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    thread = kernel.GetCurrentThread()
    before = kernel.GetThreadPriority(thread)
    kernel.SetThreadPriority(thread, THREAD_PRIORITY_LOWEST)
    mask = _efficiency_cores()
    was = kernel.SetThreadAffinityMask(thread, mask) if mask else 0
    if was:
        _full_speed(kernel, thread)
    try:
        yield
    finally:
        kernel.SetThreadPriority(thread, before)
        if was:
            kernel.SetThreadAffinityMask(thread, was)


def _thread_ids() -> set[int]:
    """This process's threads (Windows), or none elsewhere."""
    if sys.platform != "win32":
        return set()
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.Thread32First.argtypes = kernel.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return set()
    found, me = set(), os.getpid()
    entry = Entry(dwSize=ctypes.sizeof(Entry))
    try:
        more = kernel.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == me:
                found.add(entry.th32ThreadID)
            more = kernel.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return found


def _set_lowest(thread_ids: set[int]) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    kernel = ctypes.WinDLL("kernel32")
    kernel.OpenThread.restype = ctypes.c_void_p
    kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mask = _efficiency_cores()
    for tid in thread_ids:
        handle = kernel.OpenThread(0x0060, False, tid)  # THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION
        if handle:
            kernel.SetThreadPriority(handle, THREAD_PRIORITY_LOWEST)
            if mask and kernel.SetThreadAffinityMask(handle, mask):
                _full_speed(kernel, handle)  # only where it is kept off the game's cores
            kernel.CloseHandle(handle)


def load_lowest(load):
    """`load()`, with every thread it starts at the lowest priority and on
    the efficiency cores (`_efficiency_cores`).

    A model's pool threads start at normal priority whatever the thread that
    made them, so lowering the calling thread alone left the rest of a
    two-thread model at the game's level. And the lowest priority still put
    them on the performance cores beside the game, as it did the ffmpegs
    (2026-10-31, `salience_media.ffmpeg._beside_game`).
    """
    before = _thread_ids()
    made = load()
    _set_lowest(_thread_ids() - before)
    return made


def _transcriber(holder: dict):
    if "model" not in holder:
        from pipeline_v2.audio import Transcriber

        holder["model"] = load_lowest(lambda: Transcriber(cpu_threads=speech_threads()))
    return holder["model"]


def _cached(session: Path, name: str, read) -> int:
    return len(read(session / name))


def laugh_step(session: Path, holder: dict, detect=None) -> int | None:
    """Listen for laughter in the next `LISTEN_BATCH` segments: how many it
    did, or None once the model is unavailable, so the caller stops asking."""
    from pipeline_v2.capture import reactions

    try:
        if "laughs" not in holder:
            holder["laughs"] = load_lowest(lambda: reactions.Detector(cpu_threads=speech_threads()))
    except ImportError:
        return None
    before = _cached(session, reactions.CACHE_NAME, reactions._read_cache)
    (detect or reactions.listen)(session, detector=holder["laughs"], limit=LISTEN_BATCH, runner=_low_priority_run)
    return _cached(session, reactions.CACHE_NAME, reactions._read_cache) - before


def listen_step(session: Path, holder: dict, transcribe=None) -> int | None:
    """Transcribe the next `LISTEN_BATCH` segments of the mic: how many it did,
    or None once the speech model is unavailable, so the caller stops asking."""
    from pipeline_v2.capture import voice

    try:
        _transcriber(holder)
    except ImportError:
        return None
    before = _cached(session, voice.CACHE_NAME, voice._read_cache)
    (transcribe or voice.transcribe_session)(
        session, transcriber=holder["model"], limit=LISTEN_BATCH, runner=_low_priority_run)
    return _cached(session, voice.CACHE_NAME, voice._read_cache) - before


class LiveDescriber:
    """`step` every `interval` seconds on a background thread until `stop`.

    With `speech`, a second thread transcribes the mic and listens for
    laughter a segment at a time whenever it may (`speech_allowed`), and a
    window is described once its stretch is heard. Without the speech model
    windows are described as they come.
    """

    def __init__(self, session: Path, interval: float = INTERVAL, step_fn=step,
                 speech: bool = True, idle_fn=between_matches, listen_fn=listen_step,
                 laugh_fn=laugh_step):
        self.session = session
        self.interval = interval
        self.step_fn = step_fn
        self.speech = speech
        self.idle_fn = idle_fn
        self.listen_fn = listen_fn
        self.laugh_fn = laugh_fn
        self.laughter = speech
        self.mode = match_speech_mode()
        self.match_fn = match_number
        self.listen_wait = min(LISTEN_WAIT, interval)
        self._speech_model: dict = {}
        self.seen: set[float] = set()
        self.prepared: set[float] = set()  # cut and prepared mid-match (`prepare`)
        self.described = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="live-describe", daemon=True)
        self._listener = threading.Thread(target=self._listen_loop, name="live-listen", daemon=True)

    def start(self) -> "LiveDescriber":
        from salience_media import ffmpeg

        ffmpeg.LOW_PRIORITY = True
        self._thread.start()
        # Speech and laughter wait for the game to close unless schedule D asks
        # for them during it: transcribing the lobby cost the 1% lows
        # (2026-11-05), and nothing waits on it any more (describing does not;
        # after the game, the GPU does a session in seconds).
        if self.speech and self.mode != "off":
            self._listener.start()
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
                extra = {"workers": IDLE_WORKERS} if idle else {}
                # Described without waiting on speech: what was said is no
                # longer in the prompt (`notes.window_notes`). Between matches,
                # and mid-match only with DESCRIBE_KEY: even on the efficiency
                # cores, its frames (the GPU's decoder and scaler, started for
                # every window) took the game from 160fps to 149 in both
                # sessions that tried it, GPU time a frame 2.7ms to 3.6
                # (2026-11-03).
                extra["describe_now"] = idle or (describe_during_play() and _efficiency_cores() is not None)
                self.described += self.step_fn(
                    self.session, seen=self.seen, stopping=self._stop, prepared=self.prepared, **extra)
            except Exception as exc:  # noqa: BLE001 -- never take the recording down
                # The final run describes and transcribes whatever this missed.
                self.error = f"{type(exc).__name__}: {exc}"

    def _listen_loop(self) -> None:
        """Speech, then laughter, a segment at a time while allowed; a rest
        once caught up. Every thread of it at the lowest priority."""
        with lowest_thread_priority():
            while not self._stop.is_set():
                done = 0
                try:
                    idle = self.idle_fn(self.session)
                    match = None if idle else self.match_fn(self.session)
                    # Mid-match only where the efficiency cores take it: below
                    # normal on the game's cores it cost 16fps (2026-10-27).
                    allowed = speech_allowed(self.mode, idle, match) and (idle or _efficiency_cores() is not None)
                    started, began = time.monotonic(), time.time()
                    if allowed and self.speech:
                        got = self.listen_fn(self.session, self._speech_model)
                        if got is None:
                            self.speech = False
                        done += got or 0
                    if allowed and self.laughter and not self._stop.is_set():
                        got = self.laugh_fn(self.session, self._speech_model)
                        if got is None:
                            self.laughter = False
                        done += got or 0
                    if self.mode != "off":
                        self._log({"start": began, "wall": time.time(), "idle": idle, "match": match,
                                   "speech_live": allowed and not idle, "segments": done,
                                   "speech_s": round(time.monotonic() - started, 2)})
                except Exception as exc:  # noqa: BLE001 -- the final run hears what this missed
                    self.error = f"{type(exc).__name__}: {exc}"
                    done = 0
                if not (self.speech or self.laughter):
                    return
                if not done:
                    self._stop.wait(self.listen_wait)

    def _log(self, entry: dict) -> None:
        try:
            with (self.session / EXPERIMENT_LOG).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def stop(self, timeout: float = 120.0) -> int:
        """Stop after the steps in progress, then restore normal priority. The
        final run transcribes too, so it must not start beside a call still
        going: at most a segment's work."""
        from salience_media import ffmpeg

        self._stop.set()
        for thread in (self._thread, self._listener):
            if thread.is_alive():
                thread.join(timeout)
        ffmpeg.LOW_PRIORITY = False
        return self.described
