"""What the v3.9 prompt is told about a window beyond its frames.

Two lines, each something the frames cannot carry:

- what the player said on their own mic during it (`voice`, stored per clip);
- what happened in the 8 seconds before it, as already described.

Tested against two other arms on sessions 2 and 3 (trained on the sessions
before each): v3.8 put 17 of 33 kept moments in the top 10, these notes 19,
HUD banner facts 18, both together 19. The HUD added nothing on top and costs
minutes of CPU a session, so only these two lines are sent.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

_INDEX = re.compile(r"^(\d+)-")


def _previous_window(clip: Path) -> Path | None:
    """The coverage window just before `clip`: same folder, index one lower."""
    match = _INDEX.match(clip.name)
    if not match:
        return None
    index = int(match.group(1))
    if index == 0:
        return None
    return clip.with_name(f"{index - 1:0{len(match.group(1))}d}{clip.name[match.end() - 1:]}")


def _adjacent(earlier: Path, later: Path) -> bool:
    """Whether `earlier` ends where `later` starts (no skipped lobby between)."""
    try:
        a = json.loads(earlier.with_suffix(".capture.json").read_text(encoding="utf-8"))
        b = json.loads(later.with_suffix(".capture.json").read_text(encoding="utf-8"))
        return abs(float(a["segment_end_sec"]) - float(b["segment_start_sec"])) < 1.0
    except (OSError, ValueError, KeyError, TypeError):
        return False


def window_notes(conn: sqlite3.Connection, clip: Path, sha: str, variant: str) -> tuple[str, ...]:
    """The notes for one window. The previous window's summary is taken from
    the same variant when it is already described, so in a live session the
    windows of one segment are described in order."""
    notes = []
    row = conn.execute("select text from transcripts where clip_sha = ? and text != ''",
                       (sha,)).fetchone()
    if row:
        notes.append(f'Said on the player\'s own microphone during these frames: "{row[0]}"')
    before = _previous_window(clip)
    if before is not None and before.is_file() and _adjacent(before, clip):
        summary = conn.execute(
            "select d.summary from descriptions d join clips c using (clip_sha)"
            " where c.path = ? and d.prompt_ver = ?", (str(before), variant)).fetchone()
        if summary and summary[0]:
            notes.append(f"In the 8 seconds just before these frames: {summary[0]}")
    return tuple(notes)
