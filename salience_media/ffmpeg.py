"""Single-frame FFmpeg extraction and a bounded parallel runner.

Extracted from the v1 keyframe module, which mixed these general helpers with
OCR crops and composite layouts that only the retired student model used.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
from typing import TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")

# Set by work that runs while a game is being played (the live describer), so
# frame extraction yields the CPU to the game. Only processes started through
# this module see it; the recorder's own encoder is never lowered.
LOW_PRIORITY = False


def _hwaccel() -> list[str]:
    """Decode on the GPU's video decoder (NVDEC), which a game does not use,
    instead of the CPU, which it does. Measured on 28 frames: 11.8s of CPU
    down to 3.3s, and the JPEGs byte-identical (H.264 decoding is exact), so
    cached descriptions still match. SALIENCE_FFMPEG_HWACCEL=none turns it off."""
    accel = os.getenv("SALIENCE_FFMPEG_HWACCEL", "cuda")
    return [] if accel in ("", "none") else ["-hwaccel", accel]


def priority_flags() -> int:
    """`creationflags` for a helper ffmpeg: below normal while a game runs."""
    if LOW_PRIORITY and os.name == "nt":
        return subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return 0


def ffmpeg_worker_count() -> int:
    """Return a bounded extraction pool size with a sequential override."""
    if LOW_PRIORITY:
        # While a game runs: one decoder at a time. Seven at once was a burst
        # that took a 165fps game to 100-120 every 30s (2026-10-18).
        return 1
    configured = os.getenv("SALIENCE_FFMPEG_WORKERS")
    if configured:
        try:
            return max(1, min(8, int(configured)))
        except ValueError:
            pass
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, min(4, cpu_count))


def parallel_map(items: list[_T], worker: Callable[[_T], _U]) -> list[_U]:
    """Run independent FFmpeg jobs concurrently while preserving input order."""
    workers = ffmpeg_worker_count()
    if workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(workers, len(items)),
        thread_name_prefix="salience-ffmpeg",
    ) as executor:
        return list(executor.map(worker, items))


def extract_frame(
    path: Path, timestamp: float, output: Path, filter_graph: str
) -> bool:
    for accel in (_hwaccel(), []):
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                *accel,
                "-ss",
                str(timestamp),
                "-i",
                str(path),
                "-vf",
                filter_graph,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(output),
            ],
            capture_output=True,
            text=True,
            creationflags=priority_flags(),
        )
        if completed.returncode == 0 and output.exists():
            return True
        if not accel:
            break  # no GPU decoder (or no NVIDIA GPU): the CPU attempt was the last
    return False
