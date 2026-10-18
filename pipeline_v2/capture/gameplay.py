"""When the recording shows the player's own gameplay, from Fortnite's log.

A coverage session is not all gameplay. Session 2 (2026-10-14) was 320 windows,
and 35 of them were the login, lobby and item shop, loading screens, and
spectating whoever killed the player. None of the 35 was a window the player
kept. Describing them is paid for and, when spectating, is where descriptions
went wrong (14 of 18 spectator windows misdescribed in the September pilot).

The game writes match start, the local player's death and match end to its log
on every launch, with no setting to turn it off, so this is a signal the
product can rely on. When the log is missing or covers no match in the
session, `spans` returns None and every window is described, as before: the
log can only remove windows, never add a gap where there was none.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline_v2.capture import game_log

# Kept after a match ends: the Victory Royale banner is drawn just after the
# log's match end (session 2: 5s after), and it is the best moment of a win.
WIN_TAIL = 16.0
# Kept after the player's death: the death itself, before the camera moves to
# the killer.
DEATH_TAIL = 4.0
STATS_NAME = "capture_stats.json"


def video_origin(session: Path) -> tuple[datetime | None, str]:
    """UTC time of the video's first frame, and where that came from.

    Recorded exactly by the recorder from 2026-10-14 on; older sessions fall
    back to the folder name, the local time the session began, which precedes
    the first frame by the encoder's startup (about a second).
    """
    try:
        stats = json.loads((session / STATS_NAME).read_text(encoding="utf-8"))
        if stats.get("first_frame_utc"):
            return datetime.fromisoformat(stats["first_frame_utc"]), "recorded"
    except (OSError, ValueError):
        pass
    try:
        local = time.strptime(session.name, "%Y%m%d-%H%M%S")
    except ValueError:
        return None, "unknown"
    return datetime.fromtimestamp(time.mktime(local), tz=timezone.utc), "folder name"


def _logs(log_dir: Path, since: datetime) -> list[Path]:
    """Fortnite logs (current and rotated backups) written to after `since`."""
    if not log_dir.is_dir():
        return []
    return sorted(
        p for p in log_dir.glob("FortniteGame*.log")
        if datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) >= since
    )


def play_spans(
    events: list[game_log.GameEvent], origin: datetime, duration: float
) -> list[tuple[float, float]]:
    """Seconds on the video timeline during which the player was playing.

    A match runs from its start to the player's death, the match end, or the
    next match start, whichever comes first, minus the pre-game island. A match still running when the
    recording stopped runs to the end of the footage.
    """
    def at(event) -> float:
        return (event.at - origin).total_seconds()

    spans: list[tuple[float, float]] = []
    start: float | None = None
    for event in sorted(events, key=lambda e: e.at):
        if event.kind == "match_start":
            if start is not None:
                spans.append((start, at(event)))
            start = at(event)
        elif event.kind == "death" and start is not None:
            spans.append((start, at(event) + DEATH_TAIL))
            start = None
        elif event.kind == "match_end" and start is not None:
            spans.append((start, at(event) + WIN_TAIL))
            start = None
    if start is not None:
        spans.append((start, duration))
    spans = _without(spans, _warmups(events, at))
    return [(max(0.0, a), min(duration, b)) for a, b in spans if b > 0 and a < duration and b - a > 0.5]


def _warmups(events, at) -> list[tuple[float, float]]:
    """The pre-game island: Warmup until the Battle Bus. The match is already
    `InProgress` there, but kills do not count and the player skipped every
    window of it."""
    found, began = [], None
    for event in sorted(events, key=lambda e: e.at):
        if event.kind != "phase":
            continue
        if event.detail == "Warmup":
            began = at(event)
        elif began is not None:
            found.append((began, at(event)))
            began = None
    if began is not None:
        # Still on the island when the log ends: while the session is live
        # (`live`), or a recording stopped before the bus.
        found.append((began, float("inf")))
    return found


def _without(spans, holes):
    """`spans` with every interval in `holes` cut out."""
    for hole_start, hole_end in holes:
        cut = []
        for a, b in spans:
            if hole_end <= a or b <= hole_start:
                cut.append((a, b))
                continue
            if a < hole_start:
                cut.append((a, hole_start))
            if hole_end < b:
                cut.append((hole_end, b))
        spans = cut
    return spans


def spans(session: Path, duration: float, log_dir: Path | None = None) -> dict:
    """The session's gameplay spans and how they were found.

    `spans` is None when they could not be (no log, no match in the session);
    the caller then treats the whole recording as gameplay.
    """
    origin, source = video_origin(session)
    if origin is None:
        return {"spans": None, "why": "session start unknown"}
    log_dir = log_dir or game_log.DEFAULT_LOG.parent
    end = origin + timedelta(seconds=duration)
    events = [
        e for path in _logs(log_dir, origin) for e in game_log.read_events(path)
        if origin - timedelta(minutes=30) <= e.at <= end
    ]
    found = play_spans(events, origin, duration)
    if not any(e.kind == "match_start" for e in events) or not found:
        return {"spans": None, "why": "no match in the Fortnite log for this session"}
    return {
        "spans": found,
        "matches": sum(e.kind == "match_start" for e in events if e.at >= origin),
        "origin": origin.isoformat(),
        "origin_source": source,
    }


def overlaps(spans_: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(start < b and a < end for a, b in spans_)
