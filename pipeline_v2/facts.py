"""What we know about a clip before the model looks at it.

This file used to be the largest in the pipeline: banner localization, marker
grouping, weapon attribution timed against the shot, kill-feed parsing, a vitals
reader. Ablation removed all of it.

Measured on 189 clips, 175 labelled, four fold settings:

* injecting those facts into the prompt left summary ranking unchanged and made
  the model's *structured* output 4-10 points worse -- it copied OCR errors it
  would not have made on its own;
* replacing OCR localization with ``0.75 x duration`` cost nothing (73.2% vs
  71.3% mean pairwise accuracy, naive slightly ahead);
* the two together were 98% of local runtime.

So the answer to "where is the event" is now arithmetic, and the answer to
"what happened" is the model's alone. What remains here is the one fact that is
free and exact: the event type the recording tool wrote into the filename.

The deleted code is in git history if a future measurement disagrees.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

# Auto-clippers name the file after the engine event that triggered the save --
# free, exact, and unavailable from pixels at any price.
_FILENAME_EVENTS = (
    ("double elimination", "double_elimination"),
    ("multi elimination", "multi_elimination"),
    ("elimination", "elimination"),
    ("knock", "knock"),
    ("down", "knock"),
    ("victory", "victory"),
)

# Where the triggering event sits in an auto-clipped file. Measured across 119
# clips whose event was located by OCR: median offset from this fraction was
# +0.20s, 77% within +/-1.0s, 94% within +/-2.0s. The frame window spans 2.5s
# before to 1.5s after, so that residual error still contains the event.
EVENT_FRACTION = 0.75


@dataclass
class ClipFacts:
    source_event: str | None = None
    event_sec: float | None = None
    event_source: str = "fraction"
    analysis_start_sec: float | None = None
    analysis_end_sec: float | None = None
    capture_session: str | None = None
    # From the game's own log (`gameplay.annotate`), when it covers the window.
    player_name: str | None = None
    match_phase: str | None = None
    # More about this window, one line each, for the v3.9 prompts.
    notes: tuple[str, ...] = ()


def event_from_filename(path: Path) -> str | None:
    lowered = path.stem.lower()
    for needle, label in _FILENAME_EVENTS:
        if needle in lowered:
            return label
    return None


def gather(clip: Path, duration_sec: float | None) -> ClipFacts:
    """Everything known before the model runs. No decoding, no OCR, no model."""
    sidecar = clip.with_suffix(".capture.json")
    if (clip.parent / "capture-run.json").exists() and not sidecar.exists():
        raise ValueError(f"Capture chunk is not finalized: {clip}")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        start, end = (
            metadata.get("analysis_start_sec"),
            metadata.get("analysis_end_sec"),
        )
        if (
            metadata.get("schema") != "salience.capture/1"
            or metadata.get("video_file") != clip.name
            or not isinstance(start, (float, int))
            or not isinstance(end, (float, int))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not duration_sec
            or not 0 <= start < end <= duration_sec + 0.1
        ):
            raise ValueError(f"Invalid capture timing metadata: {sidecar}")
        end = min(end, duration_sec)
        return ClipFacts(
            event_sec=(start + end) / 2,
            event_source="capture_window",
            analysis_start_sec=start,
            analysis_end_sec=end,
            capture_session=metadata.get("session_id"),
            player_name=metadata.get("player_name"),
            match_phase=metadata.get("match_phase"),
        )
    return ClipFacts(
        source_event=event_from_filename(clip),
        event_sec=(duration_sec or 30.0) * EVENT_FRACTION,
        event_source="fraction",
    )
