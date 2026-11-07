"""Deleting raw footage once it has done its job.

Nothing was ever deleted: 86 GB of recordings by 2026-11-03, about 11 GB an
hour of play, which fills a drive. After a session's highlights are made, its
raw segments are needed only to rate the windows that were not kept (the
rating page cuts them again) and to make a rated window a highlight later.
Highlights and reels are files of their own, and trimming one keeps its
original beside it; neither reads the raw footage.

So a session's raw segments go once every window that could be a highlight
is rated (nothing is left for them to do), or `KEEP_DAYS` after it was
recorded, or sooner, oldest first, while the drive has less than
`MIN_FREE_GB` free. Never the session being recorded, never one whose
highlights are not made yet, and only the raw segments: highlights, reels and
what was learned stay.

While recording, the lobby goes too (`drop_lobby`): footage outside every
match is never cut into a window, and was a fifth to a third of a session.
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


# Days after a session's last rating before its footage goes: a skip changed
# to a keep needs the footage to become a highlight.
RATED_SETTLE_DAYS = 3.0


def fully_rated(session: Path, now: float | None = None, settle_days: float = RATED_SETTLE_DAYS) -> bool:
    """Whether every window of `session` that could be a highlight is rated,
    the newest rating at least `settle_days` ago.

    False with nothing described: describing that failed, or waits for the
    month's allowance, would otherwise read as "nothing left to rate".
    """
    windows = session / "windows"
    if not windows.is_dir():
        return False
    from pipeline_v2 import store
    from pipeline_v2.capture import highlights

    conn = store.connect()
    try:
        _, inside = highlights._session_clips(conn, windows)
        if not inside:
            return False
        offered = highlights.rateable(session, inside)
        rated = dict(conn.execute("select clip_sha, decided_at from preferences").fetchall())
    finally:
        conn.close()
    if not offered <= rated.keys():
        return False
    newest = max((rated[sha] for sha in offered), default=0.0)
    return (time.time() if now is None else now) - newest >= settle_days * 86400


_SETTING = object()


def clean(root: Path, *, active: set[str] = frozenset(), now: float | None = None,
          days=_SETTING, min_free_gb: float = MIN_FREE_GB,
          free_bytes=lambda path: shutil.disk_usage(path).free,
          rated=fully_rated) -> list[tuple[str, str, int]]:
    """Delete raw footage due to go; (session, why, bytes freed) for each.

    `days` defaults to the setting (`keep_days`); None keeps footage while
    the drive has room, and a drive short of room still frees the oldest first.
    `rated(session)` says a session has nothing left to rate.
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
        for session in list(candidates):
            try:
                finished = rated(session)
            except Exception:  # noqa: BLE001 -- unknown: the footage stays
                finished = False
            if finished:
                done.append((session.name, "every moment rated", _delete(session, "rated")))
                candidates.remove(session)
    try:
        while candidates and free_bytes(root) < min_free_gb * 1e9:
            session = candidates.pop(0)
            done.append((session.name, f"under {min_free_gb:g} GB free", _delete(session, "space")))
    except OSError:
        pass
    return done


# Kept either side of a match: a highlight's lead-in, the death and the
# Victory Royale banner all reach a little past the log's start and end.
LOBBY_MARGIN = 60.0
LOBBY_RECORD = "lobby.json"


def lobby_segments(segments: list, spans: list[tuple[float, float]] | None,
                   margin: float = LOBBY_MARGIN) -> list:
    """The segments that are lobby, loading screens or spectating.

    Wholly outside every match by `margin`, and before the newest match
    began: what comes after it may yet be a match the log has not finished
    saying (the pre-game island reads as no match until the bus). None with
    no match in the log, as a missing log looks the same.
    """
    if not spans:
        return []
    newest = max(a for a, _ in spans)
    return [s for s in segments if s.end <= newest - margin
            and all(s.end <= a - margin or s.start >= b + margin for a, b in spans)]


def drop_lobby(session: Path, log_dir: Path | None = None) -> int:
    """Delete `session`'s lobby footage so far (`lobby_segments`); bytes freed.

    Kept while footage is kept forever (`keep_days`), which is for keeping it.
    """
    if keep_days() is None:
        return 0
    from pipeline_v2.capture import clips, gameplay

    segments = clips.read_segments(session)
    if len(segments) < 2:
        return 0
    played = gameplay.spans(session, segments[-1].end, log_dir)
    freed, names = 0, []
    for segment in lobby_segments(segments, played["spans"]):
        try:
            size = segment.path.stat().st_size
            segment.path.unlink()
        except OSError:
            continue
        freed += size
        names.append(segment.path.name)
    if names:
        record = session / LOBBY_RECORD
        try:
            known = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {"segments": [], "bytes": 0}
        known = {"segments": known.get("segments", []) + names, "bytes": known.get("bytes", 0) + freed}
        try:
            record.write_text(json.dumps(known), encoding="utf-8")
        except OSError:
            pass
    return freed
