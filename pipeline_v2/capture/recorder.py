"""Experimental video-only DDA/NVENC replay buffer.
Each run owns an isolated segment ring. Snapshots exclude the active writer.
Capture hardware impact, audio muxing, recovery, and event timing remain open;
see SCOPE.md. No zero-overhead or arbitrary frame-cut guarantees are implied."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass
class RecorderConfig:
    segment_seconds: int = 5
    # Ring depth. buffer length = segment_seconds * segments. 6 x 5s = 30s.
    segments: int = 6
    framerate: int = 60
    bitrate: str = "12M"
    # Output height follows the capture; ddagrab grabs the whole monitor.
    encoder: str = "h264_nvenc"
    monitor: int = 0

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0 or self.segments < 2 or self.framerate <= 0:
            raise ValueError("need positive duration/FPS and at least two segments")


class SegmentRecorder:
    """Continuous capture into a wrapping segment ring."""

    def __init__(self, ring_dir: Path, config: RecorderConfig | None = None) -> None:
        self.config = config or RecorderConfig()
        self.ring_dir = ring_dir
        self._process: subprocess.Popen | None = None
        self._session = None
        self._stderr = None
        self._error = ""

    @property
    def active_dir(self) -> Path:
        return Path(self._session.name) if self._session else self.ring_dir

    @property
    def buffer_seconds(self) -> int:
        return self.config.segment_seconds * (self.config.segments - 1)

    def _command(self) -> list[str]:
        cfg = self.config
        pattern = str(self.active_dir / "seg%03d.mp4")
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Desktop Duplication straight into GPU memory.
            "-filter_complex",
            f"ddagrab=output_idx={cfg.monitor}:framerate={cfg.framerate}[v]",
            "-map",
            "[v]",
            "-c:v",
            cfg.encoder,
            "-b:v",
            cfg.bitrate,
            # A keyframe at least every segment, so every segment is independently
            # decodable and concatenation lands exactly on a frame boundary.
            "-g",
            str(cfg.framerate * cfg.segment_seconds),
            "-force_key_frames",
            f"expr:gte(t,n_forced*{cfg.segment_seconds})",
            "-f",
            "segment",
            "-segment_time",
            str(cfg.segment_seconds),
            # The ring itself: ffmpeg overwrites seg000..segNNN cyclically.
            "-segment_wrap",
            str(cfg.segments),
            "-segment_format",
            "mp4",
            "-reset_timestamps",
            "1",
            pattern,
        ]

    def start(self) -> "SegmentRecorder":
        if self.running:
            raise RuntimeError("recorder is already running")
        self.stop()
        self.ring_dir.mkdir(parents=True, exist_ok=True)
        self._session = tempfile.TemporaryDirectory(
            prefix="session-", dir=self.ring_dir
        )
        # A file cannot fill a pipe and deadlock a long-running recorder.
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.stop()
            raise
        return self

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        if self._process is not None:
            if self.running:
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        if self._stderr is not None:
            self._error = self.error_output()
            self._stderr.close()
            self._stderr = None
        if self._session is not None:
            self._session.cleanup()
            self._session = None

    def error_output(self) -> str:
        if self._stderr is None:
            return self._error
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace")

    def _ordered_segments(self) -> list[Path]:
        """Segments oldest to newest.

        Ordered by mtime, not by filename: the ring wraps, so seg000 may be the
        newest file on disk rather than the oldest.
        """
        files = [p for p in self.active_dir.glob("seg*.mp4") if p.stat().st_size > 0]
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def save(self, out_path: Path, drop_newest: bool = True) -> Path | None:
        """Concatenate the ring into one clip.

        `drop_newest` excludes the segment ffmpeg is still writing -- including a
        half-written file produces a clip that ends mid-frame.
        """
        segments = self._ordered_segments()
        if drop_newest:
            segments = segments[:-1]
        if not segments:
            return None

        # Copy before concatenating: the ring keeps turning, and ffmpeg would
        # otherwise read a file that gets overwritten underneath it.
        if out_path.exists():
            raise FileExistsError(out_path)
        with tempfile.TemporaryDirectory(prefix="save-", dir=self.ring_dir) as name:
            staging = Path(name)
            copied = []
            for index, segment in enumerate(segments):
                before = segment.stat()
                target = staging / f"{index:03d}.mp4"
                shutil.copy2(segment, target)
                after = segment.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    self._error = "ring changed during save; snapshot discarded"
                    return None
                copied.append(target)
            (staging / "list.txt").write_text(
                "".join(f"file '{p.name}'\n" for p in copied), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    "list.txt",
                    "-c",
                    "copy",
                    "saved.mp4",
                ],
                cwd=staging,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            saved = staging / "saved.mp4"
            if completed.returncode != 0 or not saved.exists():
                self._error = completed.stderr.decode("utf-8", errors="replace")
                return None
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("xb") as dest, saved.open("rb") as source:
                try:
                    shutil.copyfileobj(source, dest)
                except BaseException:
                    dest.close()
                    out_path.unlink()
                    raise
        return out_path
