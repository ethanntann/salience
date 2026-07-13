from dataclasses import dataclass
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TypeVar


_T = TypeVar("_T")
_U = TypeVar("_U")
_BATCH_DECODE_MAX_DURATION_SEC = 60.0
_OCR_FILTER = (
    "crop=w=iw*0.48:h=ih*0.26:x=iw*0.32:y=ih*0.70,"
    "scale=1440:-2:flags=lanczos,unsharp=5:5:0.8:3:3:0.4"
)


@dataclass(frozen=True)
class Keyframe:
    path: Path
    timestamp_sec: float
    view: str = "full gameplay frame"
    event_index: int | None = None
    event_center_sec: float | None = None
    ocr_path: Path | None = None


def _ffmpeg_worker_count() -> int:
    """Return a bounded extraction pool size with a sequential override."""
    configured = os.getenv("SALIENCE_FFMPEG_WORKERS")
    if configured:
        try:
            return max(1, min(8, int(configured)))
        except ValueError:
            pass
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, min(4, cpu_count))


def _parallel_map(items: list[_T], worker: Callable[[_T], _U]) -> list[_U]:
    """Run independent FFmpeg jobs concurrently while preserving input order."""
    workers = _ffmpeg_worker_count()
    if workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(workers, len(items)),
        thread_name_prefix="salience-ffmpeg",
    ) as executor:
        return list(executor.map(worker, items))


@lru_cache(maxsize=64)
def _probe_constant_fps(path: str) -> float | None:
    """Return a nominal CFR rate, or None when frame-index mapping is unsafe."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
        )
        rates = [
            float(Fraction(value.strip()))
            for value in completed.stdout.strip().split(",")
            if value.strip() and value.strip() != "0/0"
        ]
    except (OSError, ValueError, ZeroDivisionError):
        return None
    if completed.returncode != 0 or not rates or rates[0] <= 0:
        return None
    nominal = rates[0]
    average = rates[1] if len(rates) > 1 else nominal
    if abs(nominal - average) > max(0.5, nominal * 0.01):
        return None
    return nominal


def _extract_many(
    path: Path,
    timestamps: list[float],
    outputs: list[Path],
    filter_graph: str,
    fps: float | None,
) -> bool:
    """Decode selected CFR frames in one pass, preserving current filters."""
    if not timestamps or fps is None or len(timestamps) != len(outputs):
        return False
    indices = [
        max(0, math.ceil(timestamp * fps - 1e-9)) for timestamp in timestamps
    ]
    unique_indices = sorted(set(indices))
    select_expression = "+".join(f"eq(n\\,{index})" for index in unique_indices)
    output_dir = Path(tempfile.mkdtemp(prefix="salience-batch-"))
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vf",
                f"select='{select_expression}',{filter_graph}",
                "-vsync",
                "0",
                "-an",
                "-q:v",
                "3",
                str(output_dir / "frame-%06d.jpg"),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return False
        extracted = sorted(output_dir.glob("frame-*.jpg"))
        if len(extracted) != len(unique_indices):
            return False
        by_index = dict(zip(unique_indices, extracted, strict=True))
        for index, output in zip(indices, outputs, strict=True):
            shutil.copyfile(by_index[index], output)
        return True
    except (OSError, shutil.Error):
        return False
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def keyframe_timestamps(
    duration_sec: float | None, *, count: int | None = None
) -> list[float]:
    """Return uniformly spaced timestamps covering the complete clip."""
    duration = max(duration_sec or 0, 1.0)
    if count is None:
        count = max(20, min(60, int(duration / 2)))
    if duration <= 2 or count <= 1:
        return [0.0]
    interval = duration / (count + 1)
    return [
        round(max(0.1, min(duration - 0.1, interval * (index + 1))), 3)
        for index in range(count)
    ]


def event_window_timestamps(
    duration_sec: float | None,
    event_timestamps: list[float],
    *,
    pre_sec: float = 0.75,
    post_sec: float = 1.5,
    samples: int = 7,
) -> list[tuple[int, float]]:
    """Build independent event windows so evidence never crosses candidates.

    Windows are biased forward: finish UI often appears after the damaging shot,
    especially when the player swaps weapons immediately after impact.
    """
    duration = max(duration_sec or 0, 1.0)
    pre_sec = max(0.0, float(pre_sec))
    post_sec = max(0.0, float(post_sec))
    if samples <= 1:
        offsets = [0.0]
    else:
        span = pre_sec + post_sec
        step = span / (samples - 1) if span > 0 else 0.0
        offsets = [-pre_sec + (step * index) for index in range(samples)]

    windows: list[tuple[int, float]] = []
    for event_index, center in enumerate(event_timestamps):
        timestamps = {
            round(max(0.1, min(duration - 0.1, float(center) + offset)), 3)
            for offset in offsets
        }
        windows.extend((event_index, timestamp) for timestamp in sorted(timestamps))
    return windows


def _extract_frame(
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


def extract_timeline_keyframes(
    path: Path,
    duration_sec: float | None,
    *,
    count: int | None = None,
) -> list[Keyframe]:
    if not path.exists():
        return []
    output_dir = Path(tempfile.mkdtemp(prefix="salience-timeline-"))
    frames: list[Keyframe] = []
    specs = [
        (index, timestamp, output_dir / f"timeline-{index}.jpg")
        for index, timestamp in enumerate(
            keyframe_timestamps(duration_sec, count=count)
        )
    ]

    def extract(spec: tuple[int, float, Path]) -> bool:
        _, timestamp, output = spec
        return _extract_frame(
            path,
            timestamp,
            output,
            "scale=w=960:h=-2:force_original_aspect_ratio=decrease",
        )

    timestamps = [timestamp for _, timestamp, _ in specs]
    outputs = [output for _, _, output in specs]
    fps = (
        _probe_constant_fps(str(path))
        if duration_sec is not None
        and duration_sec <= _BATCH_DECODE_MAX_DURATION_SEC
        else None
    )
    batched = _extract_many(
        path,
        timestamps,
        outputs,
        "scale=w=960:h=-2:force_original_aspect_ratio=decrease",
        fps,
    )
    extracted_results = (
        [True] * len(specs) if batched else _parallel_map(specs, extract)
    )

    for (index, timestamp, output), extracted in zip(
        specs, extracted_results, strict=True
    ):
        if extracted:
            frames.append(
                Keyframe(
                    path=output,
                    timestamp_sec=timestamp,
                    view="coarse full gameplay frame",
                )
            )
    if not frames:
        shutil.rmtree(output_dir, ignore_errors=True)
    return frames


def extract_event_keyframes(
    path: Path,
    duration_sec: float | None,
    event_timestamps: list[float],
) -> list[Keyframe]:
    if not path.exists() or not event_timestamps:
        return []
    output_dir = Path(tempfile.mkdtemp(prefix="salience-events-"))
    frames: list[Keyframe] = []
    filter_graph = (
        "split=3[full][action][hud];"
        "[full]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:432:(ow-iw)/2:(oh-ih)[fullout];"
        "[action]crop=w=iw*0.72:h=ih*0.72:x=iw*0.14:y=ih*0.12,"
        "scale=640:432:force_original_aspect_ratio=decrease,"
        "pad=640:432:(ow-iw)/2:(oh-ih)[actionout];"
        "[hud]crop=w=iw*0.56:h=ih*0.40:x=iw*0.42:y=ih*0.58,"
        "scale=400:432:force_original_aspect_ratio=decrease,"
        "pad=400:432:(ow-iw)/2:(oh-ih)[hudout];"
        "[fullout][actionout][hudout]hstack=inputs=3"
    )
    specs: list[tuple[int, int, float, Path, Path]] = []
    for frame_index, (event_index, timestamp) in enumerate(
        event_window_timestamps(duration_sec, event_timestamps)
    ):
        specs.append(
            (
                frame_index,
                event_index,
                timestamp,
                output_dir / f"event-{event_index}-{frame_index}.jpg",
                output_dir / f"event-ocr-{event_index}-{frame_index}.jpg",
            )
        )

    def extract_full(spec: tuple[int, int, float, Path, Path]) -> bool:
        _, _, timestamp, output, _ = spec
        return _extract_frame(path, timestamp, output, filter_graph)

    def extract_ocr(spec: tuple[int, int, float, Path, Path]) -> bool:
        _, _, timestamp, _, ocr_output = spec
        return _extract_frame(path, timestamp, ocr_output, _OCR_FILTER)

    timestamps = [timestamp for _, _, timestamp, _, _ in specs]
    full_outputs = [output for _, _, _, output, _ in specs]
    ocr_outputs = [ocr_output for _, _, _, _, ocr_output in specs]
    fps = (
        _probe_constant_fps(str(path))
        if duration_sec is not None
        and duration_sec <= _BATCH_DECODE_MAX_DURATION_SEC
        else None
    )
    full_batched = _extract_many(
        path, timestamps, full_outputs, filter_graph, fps
    )
    ocr_batched = _extract_many(path, timestamps, ocr_outputs, _OCR_FILTER, fps)
    full_results = (
        [True] * len(specs)
        if full_batched
        else _parallel_map(specs, extract_full)
    )
    ocr_results = (
        [True] * len(specs)
        if ocr_batched
        else _parallel_map(specs, extract_ocr)
    )

    for (
        frame_index,
        event_index,
        timestamp,
        output,
        ocr_output,
    ), full_ok, has_ocr_crop in zip(
        specs, full_results, ocr_results, strict=True
    ):
        if full_ok:
            frames.append(
                Keyframe(
                    path=output,
                    timestamp_sec=timestamp,
                    view=(
                        "event composite (left full gameplay, center action crop without most "
                        "peripheral UI, right enlarged weapon HUD)"
                    ),
                    event_index=event_index,
                    event_center_sec=float(event_timestamps[event_index]),
                    ocr_path=ocr_output if has_ocr_crop else None,
                )
            )
    if not frames:
        shutil.rmtree(output_dir, ignore_errors=True)
    return frames


def cleanup_keyframes(frames: list[Keyframe]) -> None:
    for directory in {frame.path.parent for frame in frames}:
        shutil.rmtree(directory, ignore_errors=True)
