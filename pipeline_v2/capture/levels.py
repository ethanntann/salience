"""Turn a recorded audio track into the level series a detector replays against.

Tuning a live detector by playing sessions is the expensive way: every threshold
costs a session, and the one audio result this project has was tuned on its own
test data. `sweep.py` already solves that for game audio by replaying a recorded
level log -- this extends it to any track of any recording the recorder has
already made.

That matters because the recorder now writes game and microphone as **separate
tracks**. A single mixed track cannot answer whether a reaction or an explosion
fired the detector; two can, and every recording made from here on is a tuning
dataset without extra instrumentation.

Levels are dBFS over short windows, which is what `LevelTrigger` and
`VoiceReactionDetector` both expect.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 50ms: fine enough to catch the onset of a shout, coarse enough that a
# 20-minute session is a few thousand samples rather than millions.
WINDOW_SECONDS = 0.05
SILENCE_DB = -120.0


@dataclass(frozen=True)
class Level:
    at: float  # seconds from the start of the file
    db: float


def track_count(clip: Path) -> int:
    """How many audio tracks a recording carries."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return 0
    return len(json.loads(out.stdout or "{}").get("streams", []))


def to_db(amplitude: float) -> float:
    """Linear amplitude to dBFS, with a floor instead of -inf for silence."""
    if amplitude <= 0:
        return SILENCE_DB
    return max(SILENCE_DB, 20.0 * math.log10(amplitude))


def extract_levels(
    clip: Path, track: int = 0, window_seconds: float = WINDOW_SECONDS
) -> list[Level]:
    """RMS level per window for one audio track of a recording.

    Decoded to raw mono f32 and reduced here rather than parsed out of
    ffmpeg's `astats` text: the numbers are then the same ones a live pump would
    compute from the same samples, so a replay and a live run cannot disagree
    about what the level was.
    """
    import numpy as np

    rate = 48000
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip),
         "-map", f"0:a:{track}", "-ac", "1", "-ar", str(rate),
         "-f", "f32le", "-"],
        capture_output=True, timeout=600,
    )
    if out.returncode != 0 or not out.stdout:
        return []
    samples = np.frombuffer(out.stdout, dtype=np.float32)
    per_window = max(1, int(rate * window_seconds))
    usable = len(samples) - (len(samples) % per_window)
    if usable < per_window:
        return []
    blocks = samples[:usable].reshape(-1, per_window).astype(np.float64)
    rms = np.sqrt((blocks**2).mean(axis=1))
    return [
        Level(at=index * window_seconds, db=to_db(float(value)))
        for index, value in enumerate(rms)
    ]


def summarise(levels: list[Level]) -> dict:
    """Enough to tell a usable track from a silent or clipped one."""
    if not levels:
        return {"windows": 0}
    values = sorted(level.db for level in levels)
    audible = [v for v in values if v > SILENCE_DB + 1]
    return {
        "windows": len(values),
        "seconds": round(len(values) * WINDOW_SECONDS, 1),
        "median_db": round(values[len(values) // 2], 1),
        "p95_db": round(values[int(len(values) * 0.95)], 1),
        "peak_db": round(values[-1], 1),
        "audible_fraction": round(len(audible) / len(values), 3),
    }
