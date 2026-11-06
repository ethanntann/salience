"""Deleting raw footage once it has done its job.

Nothing was ever deleted: 86 GB of recordings by 2026-11-03, about 11 GB an
hour of play, which fills a drive. After a session's highlights are made, its
raw segments are needed only to rate the windows that were not kept (the
rating page cuts them again) and to make a rated window a highlight later.
Highlights and reels are files of their own, and trimming one keeps its
original beside it; neither reads the raw footage.

So a session's raw segments go `KEEP_DAYS` after it was recorded, or sooner,
oldest first, while the drive has less than `MIN_FREE_GB` free. Never the
session being recorded, never one whose highlights are not made yet, and
only the raw segments: highlights, reels and what was learned stay.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

KEEP_KEY = "SALIENCE_KEEP_FOOTAGE_DAYS"
KEEP_DAYS = 7.0
MIN_FREE_GB = 20.0
RECORD = "cleanup.json"
SESSION = re.compile(r"^\d{8}-\d{6}$")


def keep_days() -> float | None:
    """Days raw footage is kept: `KEEP_KEY`, "forever" (or "never") for always."""
    value = os.environ.get(KEEP_KEY, "").strip().lower()
    if value in ("forever", "never", "off"):
        return None
    try:
        return max(0.0, float(value)) if value else KEEP_DAYS
    except ValueError:
        return KEEP_DAYS


def recorded_minutes(session: Path) -> float | None:
    """How long the session was, from its segment listing, footage or not."""
    try:
        rows = list(csv.reader((session / "segments.csv").read_text(encoding="utf-8").splitlines()))
        return round(max(float(row[2]) for row in rows if len(row) == 3) / 60.0, 1)
    except (OSError, ValueError):
        return None


def footage_deleted(session: Path) -> bool:
    return (session / RECORD).is_file()


def _raw(session: Path) -> list[Path]:
    return sorted(session.glob("part-*.mkv"))


def _started(session: Path) -> float | None:
    try:
        return datetime.strptime(session.name, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


def _delete(session: Path, why: str) -> int:
    freed = 0
    for part in _raw(session):
        try:
            size = part.stat().st_size
            part.unlink()
            freed += size
        except OSError:
            continue  # open in a player: next time
    try:
        (session / RECORD).write_text(json.dumps(
            {"deleted": datetime.now().isoformat(timespec="seconds"), "why": why, "bytes": freed}),
            encoding="utf-8")
    except OSError:
        pass
    return freed


_SETTING = object()


def clean(root: Path, *, active: set[str] = frozenset(), now: float | None = None,
          days=_SETTING, min_free_gb: float = MIN_FREE_GB,
          free_bytes=lambda path: shutil.disk_usage(path).free) -> list[tuple[str, str, int]]:
    """Delete raw footage due to go; (session, why, bytes freed) for each.

    `days` defaults to the setting (`keep_days`); None keeps footage while
    the drive has room, and a drive short of room still frees the oldest first.
    """
    now = time.time() if now is None else now
    days = keep_days() if days is _SETTING else days
    try:
        sessions = sorted(p for p in Path(root).iterdir() if p.is_dir() and SESSION.match(p.name))
    except OSError:
        return []
    candidates = [s for s in sessions if s.name not in active
                  and (s / "highlights" / "highlights.json").is_file() and _raw(s)]
    done: list[tuple[str, str, int]] = []
    if days is not None:
        for session in list(candidates):
            started = _started(session)
            if started is not None and now - started > days * 86400:
                done.append((session.name, f"older than {days:g} days", _delete(session, "age")))
                candidates.remove(session)
    try:
        while candidates and free_bytes(root) < min_free_gb * 1e9:
            session = candidates.pop(0)
            done.append((session.name, f"under {min_free_gb:g} GB free", _delete(session, "space")))
    except OSError:
        pass
    return done
