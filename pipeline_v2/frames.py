"""Extract sampled full frames around the estimated event."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Imported for its side effect: config loads the root .env at import time, so
# V2_FRAME_WIDTH set there is visible when FRAME_WIDTH is evaluated below.
from pipeline_v2 import config  # noqa: F401

from salience_media.ffmpeg import extract_frame, parallel_map
from salience_media.probe import safe_probe_media

# Full frame, no crops.
#
# 960px was chosen for upload size and assumed "large enough for HUD text to
# survive". That assumption was wrong: at 960px the personal hit feed is a few
# pixels tall, and a clip whose HUD plainly read "SHIELD CRACKED X2 / ASSIST /
# ELIMINATION" was described as a single-opponent kill because the model could
# not read it. The game states the facts on screen; the pipeline was shrinking
# them out of legibility.
#
# 1440 is the measured setting. At 960 the model read an ADS-to-third-person view
# change as a weapon change and invented "switches to a pistol"; at 1440 it
# correctly reports no weapon change, and it also fixed a misread weapon name
# ("Strike" -> "Striker" Pump Shotgun). In that historical comparison, 1920 gave
# identical answers to 1440 for 75% more tokens; this is not a universal limit.
#
# Resolution did not fix entity tracking in that comparison. The newer session
# also shows kill-weapon errors at 1440 with both 7 and 19 frames. See the
# attribution evaluation; frame density, legibility and reasoning are distinct.
#
# V2_FRAME_WIDTH overrides for experiments. Input tokens scale roughly with
# pixel count: ~4.2k at 960, ~8.6k at 1440, ~15k at 1920.
FRAME_WIDTH = int(os.environ.get("V2_FRAME_WIDTH", "1440"))
FRAME_FILTER = f"scale=w={FRAME_WIDTH}:h=-2:force_original_aspect_ratio=decrease"


@dataclass(frozen=True)
class Frame:
    path: Path
    timestamp_sec: float


def probe_duration(clip: Path) -> float | None:
    return safe_probe_media(clip).duration_sec


def _extract(
    clip: Path, timestamps: list[float], out_dir: Path, tag: str
) -> list[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (ts, out_dir / f"{tag}-{index:03d}.jpg") for index, ts in enumerate(timestamps)
    ]
    results = parallel_map(
        specs, lambda spec: extract_frame(clip, spec[0], spec[1], FRAME_FILTER)
    )
    return [
        Frame(path=path, timestamp_sec=ts)
        for (ts, path), ok in zip(specs, results, strict=True)
        if ok
    ]


# Named so the review pages can bound playback to exactly what the model saw,
# instead of each guessing its own span.
WINDOW_PRE_SEC = 2.5
WINDOW_POST_SEC = 1.5


def window_frames(
    clip: Path,
    duration_sec: float | None,
    event_sec: float,
    *,
    pre_sec: float = WINDOW_PRE_SEC,
    post_sec: float = WINDOW_POST_SEC,
    samples: int = 7,
    out_dir: Path,
) -> list[Frame]:
    """Frames tightly packed around the located event -- what the VLM sees.

    Biased *backwards*: the located time comes from the kill banner, which
    appears after the shot that caused it, so the interesting action is behind
    that mark rather than ahead of it.
    """
    duration = max(duration_sec or 0.0, 1.0)
    step = (pre_sec + post_sec) / max(samples - 1, 1)
    timestamps = sorted(
        {
            round(max(0.1, min(duration - 0.1, event_sec - pre_sec + step * i)), 3)
            for i in range(samples)
        }
    )
    return _extract(clip, timestamps, out_dir, "window")
