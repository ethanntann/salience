"""What the v3.9 prompt is told about a window beyond its frames.

Each something the frames cannot carry:

- what happened in the 8 seconds before it, as already described;
- (until 2026-11-02) what the player said on their own mic during it.

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
    # What was said is not among them (2026-11-02). Quoted in the summary, it
    # made describing wait on speech, and taste was better without it: held
    # out a session at a time, 0.946 with the words and 0.954 without, kills
    # 0.649 and 0.664, windows the player spoke in 0.902 and 0.943. Speech
    # reaches taste on its own (`ranking.SPEECH_VARIANT`).
    before = _previous_window(clip)
    # By its sidecar: a described window's copy may be gone (`clips.release_window`).
    if before is not None and _adjacent(before, clip):
        summary = conn.execute(
            "select d.summary from descriptions d join clips c using (clip_sha)"
            " where c.path = ? and d.prompt_ver = ?", (str(before), variant)).fetchone()
        if summary and summary[0]:
            notes.append(f"In the 8 seconds just before these frames: {summary[0]}")
    earlier = recent_banners(clip)
    if earlier:
        notes.append(earlier)
    here = window_banners(clip)
    if here:
        notes.append(here)
    return tuple(notes)


def window_banners(clip: Path) -> str | None:
    """The player's own knock and elimination banners in `clip` itself, or
    that there are none: the kill feed's lines are anyone's, the banner only
    the player's. Kill-feed lines were taken for the player's kills when the
    banner said otherwise (2026-11-04)."""
    from pipeline_v2 import banner

    try:
        record = json.loads(clip.with_suffix(".capture.json").read_text(encoding="utf-8"))
        start, end = float(record["segment_start_sec"]), float(record["segment_end_sec"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    session = clip.parent.parent
    if not (session / "segments.csv").is_file():
        return None
    try:
        hits = banner.session_found(session, start, end)
    except Exception:  # noqa: BLE001 -- a note, never a failure
        return None
    if not hits:
        return ("The player's own KNOCKED or ELIMINATION banner does not appear in these frames: "
                "no knock or elimination here is theirs.")
    words = {"kill": "ELIMINATION", "knock": "KNOCKED"}
    return ("The player's own banners in these frames: "
            + ", ".join(f"{words.get(kind, kind)} at {max(0.0, at):.1f}s" for at, kind in hits) + ".")


# How far back a window is told of the player's kill and knock banners.
RECENT_SEC = 30.0


def recent_banners(clip: Path) -> str | None:
    """The player's own knock and elimination banners in the `RECENT_SEC`
    before `clip`, from the session's banner scan (`banner.session_found`).

    An ELIMINATION! banner for an opponent knocked a while before, bleeding
    out or finished by a teammate, was described as a new kill: with only the
    8 seconds before, the knock was out of sight (2026-11-04).
    """
    from pipeline_v2 import banner

    try:
        start = float(json.loads(clip.with_suffix(".capture.json").read_text(encoding="utf-8"))["segment_start_sec"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    session = clip.parent.parent
    if not (session / "segments.csv").is_file():
        return None
    since = max(0.0, start - RECENT_SEC)
    try:
        hits = banner.session_found(session, since, start)
    except Exception:  # noqa: BLE001 -- a note, never a failure
        return None
    words = {"kill": "ELIMINATION", "knock": "KNOCKED"}
    said = [f"{words.get(kind, kind)} {start - (since + at):.0f}s before"
            for at, kind in hits if since + at < start]
    if not said:
        return None
    return (f"The player's own banners in the {RECENT_SEC:.0f} seconds before these frames: "
            + ", ".join(said) + ". An ELIMINATION banner in these frames for an opponent knocked "
            "earlier can be them bleeding out or a teammate finishing them, not a new shot.")


# "...; the player says 'what was that'", the way v3.10-v3.12 summaries ended
# while what was said was a note (`prompt.REACTION_RULE`).
_SAID = re.compile(
    r"""[,;]?\s*(?:and |then |while |as )?(?:the (?:recording )?player )?"""
    r"""(?:says|said|saying|yells|shouts|asks|exclaims|remarks)\b.*$""", re.IGNORECASE)


def without_said(summary: str) -> str:
    """A summary without the player's words, as summaries are written now: what
    taste reads, so the ones written with the words read like the rest.
    Stripped from all 2,838 stored, taste held out went 0.946 to 0.954."""
    text = (summary or "").rstrip()
    kept = _SAID.sub("", text).rstrip(" ,;")
    if kept == text.rstrip(" ,;") or not kept:
        return text
    return kept if kept.endswith((".", "!", "?")) else kept + "."
