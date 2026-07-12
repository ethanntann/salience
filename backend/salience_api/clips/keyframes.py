from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass(frozen=True)
class Keyframe:
    path: Path
    timestamp_sec: float
    view: str = "full gameplay frame"
    event_index: int | None = None
    event_center_sec: float | None = None
    ocr_path: Path | None = None


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
    for index, timestamp in enumerate(keyframe_timestamps(duration_sec, count=count)):
        output = output_dir / f"timeline-{index}.jpg"
        if _extract_frame(
            path,
            timestamp,
            output,
            "scale=w=960:h=-2:force_original_aspect_ratio=decrease",
        ):
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
    for frame_index, (event_index, timestamp) in enumerate(
        event_window_timestamps(duration_sec, event_timestamps)
    ):
        output = output_dir / f"event-{event_index}-{frame_index}.jpg"
        ocr_output = output_dir / f"event-ocr-{event_index}-{frame_index}.jpg"
        if _extract_frame(path, timestamp, output, filter_graph):
            has_ocr_crop = _extract_frame(
                path,
                timestamp,
                ocr_output,
                (
                    "crop=w=iw*0.48:h=ih*0.26:x=iw*0.32:y=ih*0.70,"
                    "scale=1440:-2:flags=lanczos,unsharp=5:5:0.8:3:3:0.4"
                ),
            )
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
