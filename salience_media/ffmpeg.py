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


def ffmpeg_worker_count() -> int:
    """Return a bounded extraction pool size with a sequential override."""
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
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
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
    )
    return completed.returncode == 0 and output.exists()
