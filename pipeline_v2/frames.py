"""Extract sampled full frames around the estimated event."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Imported for its side effect: config loads the root .env at import time, so
# V2_FRAME_WIDTH set there is visible when FRAME_WIDTH is evaluated below.
from pipeline_v2 import config  # noqa: F401

from salience_media.ffmpeg import extract_frame, parallel_map, priority_flags, run as run_ffmpeg
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
# Frames per window. Most of the input tokens; 4 measured a third cheaper and
# missed kills (agreement with the player 97.3% -> 94.5%), so 7.
# V2_FRAME_COUNT overrides for experiments, stored as its own variant.
# 14 on the fight windows (2026-10-22, v3.11, sessions 1-3 and 5, each ranked
# by the other three): keeps in the top 10 18 of 27 either way, a kept kill
# above a skipped one 62% against 75% at 7, headshots seen on 15 kept kills
# against 19, and 70% more tokens. It also moved 19 kills to no_event, mostly
# windows opening just after a kill (two of them kept), so 7 stays.
DEFAULT_FRAME_COUNT = 7
FRAME_COUNT = int(os.environ.get("V2_FRAME_COUNT", "") or DEFAULT_FRAME_COUNT)
FRAME_FILTER = f"scale=w={FRAME_WIDTH}:h=-2:force_original_aspect_ratio=decrease"


@dataclass(frozen=True)
class Frame:
    path: Path
    timestamp_sec: float


def probe_duration(clip: Path) -> float | None:
    return safe_probe_media(clip).duration_sec


BANK = "bank.json"


def _banked(timestamps: list[float], out_dir: Path, tag: str) -> list[Frame] | None:
    """The frames already taken for these times from their segment (`bank`)."""
    import json

    try:
        kept = json.loads((out_dir / BANK).read_text(encoding="utf-8")).get(tag) or {}
    except (OSError, ValueError):
        return None
    found = []
    for ts in timestamps:
        name = kept.get(f"{ts:.3f}")
        if not name or not (out_dir / name).is_file():
            return None
        found.append(Frame(path=out_dir / name, timestamp_sec=ts))
    return found


def bank(segment: Path, wanted: list[tuple[Path, str, int, float, float]], gpu: bool = True) -> int:
    """Take many windows' frames from one decode of `segment`; how many were.

    `wanted` is (window's frame folder, tag, index, time in the window, time
    in the segment). Each window then finds its frames already there
    (`_banked`). A window's frames, a decode each and another for its kills,
    started the GPU's decoder eight times a segment: 17.8 CPU-seconds for a
    segment's four windows, against 6.9 for one pass over the segment
    (2026-11-03), and mid-match the game lost 11fps to it.
    """
    import json

    if not wanted:
        return 0
    out = segment.parent / ".frames"
    out.mkdir(exist_ok=True)
    tag = f"bank-{segment.stem}-{abs(hash(tuple(w[4] for w in wanted))) % 10**8}"
    taken = _extract_one_pass(segment, sorted({round(w[4], 3) for w in wanted}), out, tag, gpu=gpu)
    if taken is None:
        return 0
    by_time = {round(f.timestamp_sec, 3): f.path for f in taken}
    done = 0
    for folder, name, index, window_ts, segment_ts in wanted:
        source = by_time.get(round(segment_ts, 3))
        if source is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{name}-{index:03d}.jpg"
        import shutil

        shutil.copyfile(source, target)
        record_path = folder / BANK
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        record.setdefault(name, {})[f"{window_ts:.3f}"] = target.name
        record_path.write_text(json.dumps(record), encoding="utf-8")
        done += 1
    for path in taken:
        path.path.unlink(missing_ok=True)
    return done


def _extract(
    clip: Path, timestamps: list[float], out_dir: Path, tag: str
) -> list[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    banked = _banked(timestamps, out_dir, tag)
    if banked is not None:
        return banked
    one_pass = _extract_one_pass(clip, timestamps, out_dir, tag)
    if one_pass is not None:
        return one_pass
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


def _frame_rate(clip: Path) -> float | None:
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", str(clip)],
                         capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        top, _, bottom = out.stdout.strip().partition("/")
        return float(top) / float(bottom or 1)
    except (ValueError, ZeroDivisionError):
        return None


def _extract_one_pass(clip: Path, timestamps: list[float], out_dir: Path, tag: str,
                      gpu: bool = True) -> list[Frame] | None:
    """Every wanted frame from one decode, picked and scaled on the GPU.

    Seven seeks were seven ffmpeg runs, each downloading, scaling and
    encoding on the CPU: 4.3 CPU-seconds a window. One pass that keeps only
    the wanted frames (by number) and scales them on the GPU is 0.6
    (2026-10-25). None when there is no CUDA decoder, or anything is off:
    the seeks are the fallback.
    """
    import os
    import subprocess

    if len(timestamps) < 2 or (gpu and os.getenv("SALIENCE_FFMPEG_HWACCEL", "cuda") != "cuda"):
        return None
    rate = _frame_rate(clip)
    if not rate:
        return None
    numbers = [round(ts * rate) for ts in timestamps]
    if len(set(numbers)) != len(numbers):
        return None
    pattern = out_dir / f"{tag}-pass-%03d.jpg"
    picked = "+".join(rf"eq(n\,{n})" for n in sorted(numbers))
    # On the CPU too, where the GPU's decoder is already busy (`bank`, after a
    # session): a 30s segment 2.3s there against 2.2 on the GPU (2026-11-05).
    decode = (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(clip), "-an",
               "-vf", f"select='{picked}',scale_cuda={FRAME_WIDTH}:-2,hwdownload,format=nv12"] if gpu
              else ["-i", str(clip), "-an", "-vf", f"select='{picked}',scale={FRAME_WIDTH}:-2"])
    done = run_ffmpeg(
        ["ffmpeg", "-v", "error", "-y", *decode,
         "-fps_mode", "passthrough", "-q:v", "3", str(pattern)],
        capture_output=True, creationflags=priority_flags())
    written = sorted(out_dir.glob(f"{tag}-pass-*.jpg"))
    if done.returncode != 0 or len(written) != len(numbers):
        for path in written:
            path.unlink(missing_ok=True)
        return None
    order = sorted(range(len(numbers)), key=lambda i: numbers[i])
    frames = [None] * len(numbers)
    for path, i in zip(written, order):
        final = out_dir / f"{tag}-{i:03d}.jpg"
        path.replace(final)
        frames[i] = Frame(path=final, timestamp_sec=timestamps[i])
    return frames


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
    samples: int = FRAME_COUNT,
    out_dir: Path,
) -> list[Frame]:
    """Frames tightly packed around the located event -- what the VLM sees.

    Biased *backwards*: the located time comes from the kill banner, which
    appears after the shot that caused it, so the interesting action is behind
    that mark rather than ahead of it.
    """
    timestamps = window_times(duration_sec, event_sec, pre_sec=pre_sec, post_sec=post_sec, samples=samples)
    return _extract(clip, timestamps, out_dir, "window")


def window_times(duration_sec: float | None, event_sec: float, *, pre_sec: float = WINDOW_PRE_SEC,
                 post_sec: float = WINDOW_POST_SEC, samples: int = FRAME_COUNT) -> list[float]:
    """The times `window_frames` takes, without taking them."""
    duration = max(duration_sec or 0.0, 1.0)
    step = (pre_sec + post_sec) / max(samples - 1, 1)
    return sorted(
        {
            round(max(0.1, min(duration - 0.1, event_sec - pre_sec + step * i)), 3)
            for i in range(samples)
        }
    )
