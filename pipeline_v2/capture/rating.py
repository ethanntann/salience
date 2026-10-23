"""What the rating page needs from a recorded session, plugged into `review`.

A described window's copy is deleted after its session (`clips.release_window`),
so watching it re-cuts it from the raw footage, about a second, into a few
reused temporary files. And a keep there becomes a highlight of its session
(`highlights.add_kept`), so a moment the ranking missed is one key away.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod

RECUT_DIR = Path(tempfile.gettempdir()) / "salience-rating"
RECUT_KEEP = 8
_lock = threading.Lock()


def playable(path: Path, recut=clips_mod.recut_window) -> Path | None:
    """`path`, or a fresh cut of its window when its copy is gone; None if neither."""
    if path.is_file():
        return path
    if not clips_mod.released(path):
        return None
    copy = RECUT_DIR / f"{path.parent.parent.name}-{path.name}"
    with _lock:
        if copy.is_file():
            return copy
        RECUT_DIR.mkdir(parents=True, exist_ok=True)
        for stale in sorted(RECUT_DIR.glob("*.mkv"), key=lambda p: p.stat().st_mtime)[:-RECUT_KEEP]:
            try:
                stale.unlink()
            except OSError:
                pass  # still being played
        partial = copy.with_name("cutting-" + copy.name)
        if recut(path, partial) is None:
            return None
        partial.replace(copy)  # never half a file to a reader
        return copy


def rated(sha: str, label: int) -> None:
    """A keep becomes a highlight of its session; re-rated a skip, it goes again."""
    from pipeline_v2.capture import highlights

    try:
        (highlights.add_kept if label == 1 else highlights.drop_kept)(sha)
    except Exception as exc:  # noqa: BLE001 -- the rating is stored either way
        print(f"highlight for {sha[:12]} failed: {type(exc).__name__}: {exc}", flush=True)
