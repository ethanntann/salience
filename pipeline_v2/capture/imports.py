"""Clips and montages the player already likes, as a start on their taste.

A new player has no ratings, so their first highlights are guesses. A clip
they kept, or a montage they made, says what they like before they have rated
anything. Simulated on the ratings so far (2026-11-06), held out a session at
a time: 10 of a player's own ratings ranked a kept kill above a skipped one
78% of the time, and 85% with 20 liked clips beside them, at half weight. But
20 random windows passed off as liked fell to 57%: a liked clip that is not
one is worse than none, which is why `ranking` checks imports against the
player's own ratings once there are enough of them.

A montage is edited: cuts between moments, often from different matches. It
is split at its cuts first, then into windows of the length the recorder's
are, so the model sees one moment at a time, described by the same prompt as
the player's own windows (a `coverage` sidecar), and lands in the same corpus.

Only the picture counts. A montage's music is not the player's voice, so
imported windows never get a transcript and add nothing to speech.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

IMPORT_DIR = Path.home() / "Videos" / "Salience" / "Imports"
# The recorder's window length, so an imported moment is described like one.
WINDOW_SECONDS = 8.0
# A shot shorter than this is a transition or a flash, not a moment.
MIN_SHOT_SECONDS = 3.0
# How different a frame must be from the one before to be an edit's cut.
SCENE_CHANGE = 0.35
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


@dataclass(frozen=True)
class Window:
    start: float
    end: float


def duration(path: Path, runner=subprocess.run) -> float | None:
    out = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float((out.stdout or "").strip())
    except ValueError:
        return None


def cuts(path: Path, runner=subprocess.run) -> list[float]:
    """Where the edit cuts, in seconds: frames unlike the one before."""
    out = runner(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-an",
                  "-vf", f"scale=320:-2,select='gt(scene,{SCENE_CHANGE})',showinfo",
                  "-f", "null", "-"], capture_output=True, text=True)
    return sorted(float(t) for t in re.findall(r" pts_time:([\d.]+)", out.stderr or ""))


def windows(length: float, cut_at: list[float]) -> list[Window]:
    """Windows inside each shot, none across a cut.

    A shot is split into equal windows of at most `WINDOW_SECONDS`, so a 12 s
    shot is two of 6 s rather than one of 8 and a 4 s scrap.
    """
    edges = [0.0] + [t for t in cut_at if 0.0 < t < length] + [length]
    found = []
    for a, b in zip(edges, edges[1:]):
        if b - a < MIN_SHOT_SECONDS:
            continue
        n = max(1, round((b - a) / WINDOW_SECONDS))
        step = (b - a) / n
        found += [Window(round(a + i * step, 3), round(a + (i + 1) * step, 3)) for i in range(n)]
    return found


def folder_for(source: Path) -> Path:
    """One folder per imported file, named so the same file twice is one."""
    digest = hashlib.sha256(f"{source.resolve()}|{source.stat().st_size}".encode()).hexdigest()[:10]
    stem = re.sub(r"[^\w-]+", "_", source.stem)[:40] or "clip"
    return IMPORT_DIR / f"{stem}-{digest}"


def sidecar(source: Path, video: Path, window: Window) -> dict:
    length = window.end - window.start
    return {
        "schema": "salience.capture/1", "session_id": "Imports", "run_id": video.parent.name,
        "sampling": "coverage", "video_file": video.name, "audio": False,
        "run_started_utc": None,
        "segment_start_sec": window.start, "segment_end_sec": window.end,
        "duration_sec": round(length, 3),
        "analysis_start_sec": 0.0, "analysis_end_sec": round(length, 3),
        "event_sec": round(length / 2, 3),
        "trigger_kind": "import", "trigger_source": "import", "trigger_confidence": 0.0,
        "trigger_detail": str(source), "encoder": "h264_nvenc",
    }


def cut(source: Path, runner=subprocess.run, *, encoder: str = "h264_nvenc") -> list[Path]:
    """Cut `source` into windows on disk; the window files, existing ones kept."""
    length = duration(source, runner)
    if not length:
        return []
    folder = folder_for(source)
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for index, window in enumerate(windows(length, cuts(source, runner))):
        video = folder / f"{index:03d}-coverage.mkv"
        if not video.is_file():
            for codec in (encoder, "libx264"):
                result = runner(["ffmpeg", "-hide_banner", "-nostdin", "-y",
                                 "-ss", f"{window.start:.3f}", "-i", str(source),
                                 "-t", f"{window.end - window.start:.3f}", "-an",
                                 "-c:v", codec, "-b:v", "8M", str(video)],
                                capture_output=True, text=True)
                if getattr(result, "returncode", 1) == 0 and video.is_file():
                    break
                video.unlink(missing_ok=True)
            else:
                continue
        side = video.with_suffix(".capture.json")
        if not side.is_file():
            side.write_text(json.dumps(sidecar(source, video, window), indent=2), encoding="utf-8")
        made.append(video)
    return made


def sources(paths: list[Path]) -> list[Path]:
    """The video files among `paths`, a folder standing for the videos in it."""
    found = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            found.append(path)
    return found


def remember(conn, videos: list[Path], source: dict[Path, Path], now: float | None = None) -> int:
    """Mark described windows as imported likes; how many were described."""
    from pipeline_v2 import cache

    described = {r[0] for r in conn.execute("select clip_sha from clips")}
    marked = 0
    with conn:
        for video in videos:
            sha = cache.clip_sha(video)
            if sha not in described:
                continue
            conn.execute("insert or ignore into imported (clip_sha, source, added_at) values (?, ?, ?)",
                         (sha, str(source[video]), now or time.time()))
            marked += 1
    return marked


def run(paths: list[Path], *, max_calls: int, runner=subprocess.run, describe=None, connect=None) -> dict:
    """Cut, describe and mark everything in `paths`. Describing is paid."""
    from pipeline_v2 import store
    from pipeline_v2.capture import highlights

    describe = describe or highlights.describe
    connect = connect or store.connect
    source: dict[Path, Path] = {}
    for path in sources(paths):
        for video in cut(path, runner):
            source[video] = path
    for folder in sorted({v.parent for v in source}):
        describe(folder, max_calls)
    conn = connect()
    try:
        marked = remember(conn, list(source), source)
    finally:
        conn.close()
    return {"files": len({*source.values()}), "windows": len(source), "liked": marked}


def liked_count(connect=None) -> int:
    from pipeline_v2 import store

    conn = (connect or store.connect)()
    try:
        return conn.execute("select count(*) from imported").fetchone()[0]
    finally:
        conn.close()


class Job:
    """One import at a time, for the window: cut (free), then describe (paid)
    only once the player has seen how many calls it takes and said yes.

    Both wait while the game runs: cutting is video encoding, and the
    window's work during a match froze it (2026-11-06).
    """

    def __init__(self, *, playing=lambda: False, runner=subprocess.run, describe=None,
                 connect=None, wait_seconds: float = 5.0) -> None:
        import threading

        self._lock = threading.Lock()
        self._playing, self._runner, self._describe = playing, runner, describe
        self._connect, self._wait = connect, wait_seconds
        self.stage, self.files, self.videos, self.liked, self.error = "idle", [], {}, 0, ""

    def view(self) -> dict:
        with self._lock:
            return {"stage": self.stage, "files": len(self.files), "windows": len(self.videos),
                    "liked": self.liked, "error": self.error}

    def _set(self, **changes) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(self, name, value)

    def _until_game_closes(self, stage: str) -> None:
        while self._playing():
            self._set(stage=f"{stage}-waiting")
            time.sleep(self._wait)
        self._set(stage=stage)

    def _background(self, work) -> None:
        import threading

        def guarded() -> None:
            try:
                work()
            except Exception as error:  # noqa: BLE001 - shown to the player, not raised in a thread
                self._set(stage="error", error=str(error) or type(error).__name__)

        threading.Thread(target=guarded, name="import", daemon=True).start()

    def busy(self) -> bool:
        return self.view()["stage"].split("-")[0] in ("cutting", "describing")

    def cut(self, paths: list[Path], *, wait: bool = False) -> bool:
        files = sources(paths)
        if self.busy() or not files:
            return False
        self._set(stage="cutting", files=files, videos={}, liked=0, error="")

        def work() -> None:
            self._until_game_closes("cutting")
            videos = {}
            for path in files:
                for video in cut(path, self._runner):
                    videos[video] = path
            self._set(videos=videos, stage="ready" if videos else "empty")

        work() if wait else self._background(work)
        return True

    def describe(self, *, max_calls: int, wait: bool = False) -> bool:
        if self.view()["stage"] != "ready":
            return False
        self._set(stage="describing")

        def work() -> None:
            from pipeline_v2 import store
            from pipeline_v2.capture import highlights

            self._until_game_closes("describing")
            videos = dict(self.videos)
            for folder in sorted({v.parent for v in videos}):
                (self._describe or highlights.describe)(folder, max_calls)
            conn = (self._connect or store.connect)()
            try:
                marked = remember(conn, list(videos), videos)
            finally:
                conn.close()
            self._set(liked=marked, stage="done")

        work() if wait else self._background(work)
        return True

    def discard(self) -> None:
        if not self.busy():
            self._set(stage="idle", files=[], videos={}, liked=0, error="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="clips, montages, or folders of them")
    parser.add_argument("--max-calls", type=int, default=200,
                        help="most model calls to pay for (one per window)")
    parser.add_argument("--count", action="store_true",
                        help="only cut and count the windows; describe nothing")
    args = parser.parse_args(argv)
    if args.count:
        made = [v for p in sources(args.paths) for v in cut(p)]
        print(f"{len(made)} windows from {len(sources(args.paths))} file(s), "
              f"one model call each to describe")
        return 0
    done = run(args.paths, max_calls=args.max_calls)
    print(f"{done['liked']} of {done['windows']} windows from {done['files']} file(s) now count as liked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
