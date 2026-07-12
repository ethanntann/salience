from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import subprocess


@dataclass(frozen=True)
class MediaMetadata:
    duration_sec: float | None
    width: int | None
    height: int | None
    fps: float | None
    size_bytes: int | None


def parse_ffprobe(payload: dict) -> MediaMetadata:
    video_stream = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        {},
    )
    format_info = payload.get("format", {})

    fps = None
    fps_raw = video_stream.get("r_frame_rate")
    if fps_raw and fps_raw != "0/0":
        fps = float(Fraction(fps_raw))

    duration_raw = format_info.get("duration")
    size_raw = format_info.get("size")

    return MediaMetadata(
        duration_sec=float(duration_raw) if duration_raw else None,
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=fps,
        size_bytes=int(size_raw) if size_raw else None,
    )


def probe_media(path: Path) -> MediaMetadata:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ffprobe(json.loads(completed.stdout))


def fallback_metadata(path: Path) -> MediaMetadata:
    stat = path.stat()
    return MediaMetadata(
        duration_sec=None,
        width=None,
        height=None,
        fps=None,
        size_bytes=stat.st_size,
    )


def safe_probe_media(path: Path) -> MediaMetadata:
    try:
        return probe_media(path)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return fallback_metadata(path)
