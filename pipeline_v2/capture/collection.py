"""Hands-free evaluation collection: preserve foreground gameplay, not just peaks.

No model calls, audio devices, keyboard hooks, or game injection. FFmpeg supplies
media timestamps for complete chunks; metadata selects a reproducible four-second
window for later v2 analysis without pretending to know a game-event timestamp.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
import uuid

from pipeline_v2.capture.windows import Display
from pipeline_v2.jsonio import utc_now, write_json


@dataclass(frozen=True)
class CollectionConfig:
    output_dir: Path
    # Overridable without editing code, so an encoder or frame rate can be A/B
    # tested against the game's frame times rather than argued about.
    fps: int = field(
        default_factory=lambda: int(os.environ.get("SALIENCE_CAPTURE_FPS", "60"))
    )
    bitrate: str = field(
        default_factory=lambda: os.environ.get("SALIENCE_CAPTURE_BITRATE", "12M")
    )
    # Forces one encoder and skips the preference order AND the cached choice in
    # capture-settings.json. That cache pinned h264_qsv on a hybrid laptop whose
    # Quick Sync cannot even open an MFX session, while the game runs on the
    # dGPU -- capture was competing with Fortnite on the wrong GPU.
    encoder: str | None = field(
        default_factory=lambda: os.environ.get("SALIENCE_CAPTURE_ENCODER") or None
    )
    # "1536x960" to downscale before encoding. DDA duplicates the whole desktop
    # every frame, so this cuts the encode but NOT the duplication itself. This
    # build cannot scale on the GPU -- scale_d3d11 fails to allocate and CUDA
    # interop is unimplemented -- so it routes through system memory, moving
    # work off the GPU and onto the CPU. For a GPU-bound game that is the useful
    # direction, but it is a trade, not a saving: measure it, do not assume it.
    scale: str | None = field(
        default_factory=lambda: os.environ.get("SALIENCE_CAPTURE_SCALE") or None
    )
    segment_seconds: int = 30
    max_seconds: float = 7200
    max_bytes: int = 15 * 1024**3
    min_free_bytes: int = 5 * 1024**3
    exit_grace_seconds: float = 30

    def __post_init__(self):
        if self.fps < 1 or self.segment_seconds < 1 or self.max_seconds <= 0:
            raise ValueError(
                "positive frame rate, segment duration, and time limit required"
            )
        if self.max_bytes <= 0 or self.min_free_bytes < 0:
            raise ValueError("invalid storage limits")


def video_command(
    display: Display, encoder: str, config: CollectionConfig
) -> list[str]:
    source = f"ddagrab=output_idx={display.output}:framerate={config.fps}:draw_mouse=0"
    if encoder == "h264_qsv":
        source += ",hwmap=derive_device=qsv,format=qsv,vpp_qsv=format=nv12"
    elif encoder == "h264_nvenc":
        if config.scale:
            width, _, height = config.scale.lower().partition("x")
            if not (width.isdigit() and height.isdigit()):
                raise ValueError("scale must look like 1536x960")
            source += (
                f",hwdownload,format=bgra,scale={width}:{height}"
                ",format=nv12,hwupload_cuda"
            )
    else:
        raise ValueError("only tested hardware encoding paths are allowed")
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-init_hw_device",
        f"d3d11va=desktop:{display.adapter}",
        "-filter_hw_device",
        "desktop",
        "-filter_complex",
        source + "[v]",
        "-map",
        "[v]",
        "-c:v",
        encoder,
        "-preset",
        "p1" if encoder == "h264_nvenc" else "veryfast",
        "-b:v",
        config.bitrate,
        # One keyframe per segment, not one per second. `-g fps` forced an IDR
        # every 30 frames, five times more than the original recorder, and IDRs
        # are the most expensive frames to encode. `-force_key_frames` on the
        # output already guarantees a keyframe at each segment boundary, which
        # is all that independently-decodable chunks require, so the old value
        # was pure extra work.
        "-g",
        str(config.fps * config.segment_seconds),
        # No -maxrate/-bufsize. Pinning maxrate to the target bitrate makes the
        # rate controller effectively CBR, which costs encoder time and can
        # stall; the original recorder omitted both and total storage is already
        # bounded by max_bytes.
        "-bf",
        "0",
    ]


class CaptureRun:
    """One uninterrupted foreground stretch, finalized with FFmpeg's q command."""

    def __init__(
        self,
        directory: Path,
        display: Display,
        encoder: str,
        config: CollectionConfig,
        command: list[str] | None = None,
    ):
        self.directory = directory
        self.display = display
        self.encoder = encoder
        self.config = config
        self.started_utc = utc_now()
        self.started = time.monotonic()
        self.stopped = None
        self.directory.mkdir(parents=True)
        write_json(
            directory / "capture-run.json",
            {
                "schema": "salience.capture-run/1",
                "started_utc": self.started_utc,
                "display": asdict(display),
                "encoder": encoder,
            },
        )
        self.log = (directory / "ffmpeg.log").open("wb")
        self.seen: set[str] = set()
        self.fault = None
        self._last_bytes = 0
        self._last_growth = self.started
        args = command or video_command(display, encoder, config)
        args = args + [
            "-force_key_frames",
            f"expr:gte(t,n_forced*{config.segment_seconds})",
            "-f",
            "segment",
            "-segment_time",
            str(config.segment_seconds),
            "-segment_list",
            str(directory / "segments.csv"),
            "-segment_list_type",
            "csv",
            "-reset_timestamps",
            "1",
            str(directory / "part-%05d.mkv"),
        ]
        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.log.close()
            raise

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def bytes_written(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*.mkv"))

    def stalled(self, now: float) -> bool:
        size = self.bytes_written()
        if size != self._last_bytes:
            self._last_bytes = size
            self._last_growth = now
        return now - self._last_growth > 15

    def stop(self) -> None:
        if self.stopped is not None:
            return
        if self.running:
            try:
                self.process.stdin.write(b"q\n")
                self.process.stdin.flush()
            except (OSError, ValueError):
                pass
        try:
            self.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
            self.fault = "forced_stop: last chunk may be incomplete"
        self.stopped = time.monotonic()
        self.process.stdin.close()
        self.log.close()

    def publish_closed(self, session_id: str) -> list[dict]:
        listing = self.directory / "segments.csv"
        if not listing.exists():
            return []
        published = []
        for row in csv.reader(listing.read_text(encoding="utf-8").splitlines()):
            if len(row) != 3 or row[0] in self.seen:
                continue
            path = (self.directory / Path(row[0]).name).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                start, end = float(row[1]), float(row[2])
            except ValueError:
                continue  # writer may not have finished this CSV row yet
            duration = end - start
            if duration <= 0:
                continue
            # Frame timestamps are local to each reset segment, not wall-clock guesses.
            span = min(4.0, duration)
            seed = hashlib.sha256(
                f"{session_id}/{self.directory.name}/{path.name}".encode()
            ).digest()
            offset = random.Random(int.from_bytes(seed, "big")).uniform(
                0, max(0, duration - span)
            )
            metadata = {
                "schema": "salience.capture/1",
                "session_id": session_id,
                "run_id": self.directory.name,
                "sampling": "uniform",
                "video_file": path.name,
                "audio": False,
                "run_started_utc": self.started_utc,
                "segment_start_sec": start,
                "segment_end_sec": end,
                "duration_sec": duration,
                "analysis_start_sec": offset,
                "analysis_end_sec": offset + span,
                "event_sec": None,
                "encoder": self.encoder,
                "display": asdict(self.display),
            }
            write_json(path.with_suffix(".capture.json"), metadata)
            self.seen.add(row[0])
            published.append(metadata)
        return published


class Collector:
    """Foreground gate and recovery controller. tick() never starts paid inference."""

    def __init__(self, config: CollectionConfig, *, run_factory=CaptureRun):
        self.config = config
        self.run_factory = run_factory
        self.session_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        )
        self.directory = config.output_dir.resolve() / self.session_id
        self.directory.mkdir(parents=True)
        self.run = None
        self.run_number = 0
        self.clips = 0
        self.finished_bytes = 0
        self.finished_seconds = 0.0
        self.ever_recorded = False
        self.absent_since = None
        self.retry_at = 0.0
        self.failures = 0
        self.encoder_for: dict[tuple[int, int], str] = {}
        self.next_encoder: dict[tuple[int, int], int] = {}
        self.done = False
        self.status = "Waiting for Fortnite"
        self.detail = (
            "Open Fortnite and play. Recording starts when the game is foreground."
        )
        self.started_utc = utc_now()
        self.log_path = self.directory / "events.jsonl"
        self._record("armed")
        self._save_manifest()

    def _record(self, kind: str, **fields) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utc_now(), "kind": kind, **fields}) + "\n")

    def _publish(self):
        if self.run:
            for metadata in self.run.publish_closed(self.session_id):
                self.clips += 1
                self._record(
                    "clip_saved",
                    run_id=metadata["run_id"],
                    file=metadata["video_file"],
                    duration=metadata["duration_sec"],
                )

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "detail": self.detail,
            "clips": self.clips,
            "output": str(self.directory),
            "done": self.done,
            "bytes": self.finished_bytes
            + (self.run.bytes_written() if self.run else 0),
            "recorded_seconds": self.finished_seconds
            + (time.monotonic() - self.run.started if self.run else 0),
            "audio": False,
            "updated_utc": utc_now(),
        }

    def _save_manifest(self):
        write_json(
            self.directory / "session.json",
            {
                "schema": "salience.session/1",
                "started_utc": self.started_utc,
                **self.snapshot(),
                "policy": "all foreground Fortnite footage; uniform analysis windows",
                "limits": {
                    "seconds": self.config.max_seconds,
                    "bytes": self.config.max_bytes,
                },
            },
        )

    def _end_run(self, reason: str):
        if self.run:
            self.run.stop()
            self._publish()
            self.finished_bytes += self.run.bytes_written()
            self.finished_seconds += max(0, self.run.stopped - self.run.started)
            self._record(
                "capture_stopped",
                reason=reason,
                run_id=self.run.directory.name,
                fault=self.run.fault,
            )
            self.run = None

    def stop(self, reason="Stopped — footage saved"):
        self._end_run(reason)
        self.done = True
        self.status = reason
        self.detail = "Your recordings are saved locally. No clips were uploaded."
        self._record("session_stopped", reason=reason)
        self._save_manifest()

    def tick(self, display: Display | None, game_is_running: bool):
        if self.done:
            return
        now = time.monotonic()
        state = self.snapshot()
        if (
            state["recorded_seconds"] >= self.config.max_seconds
            or state["bytes"] >= self.config.max_bytes
        ):
            self.stop("Session limit reached — footage saved")
            return
        if shutil.disk_usage(self.directory).free < self.config.min_free_bytes:
            self.stop("Low disk space — recording stopped")
            return
        if self.run and (
            display != self.run.display or not self.run.running or self.run.stalled(now)
        ):
            failed = display == self.run.display
            old = self.run
            self._end_run("capture_error" if failed else "game_not_foreground")
            if failed:
                self.failures += 1
                key = (old.display.adapter, old.display.output)
                self.next_encoder[key] = 0 if old.encoder == "h264_qsv" else 1
                self.retry_at = now + min(15, self.failures * 2)
                self._record("retry", encoder=old.encoder, failures=self.failures)
                if self.failures >= 6:
                    self.stop("Capture failed — check the session log")
                    return
        if display is None:
            self.status = (
                "Paused — Fortnite is not foreground"
                if game_is_running
                else "Waiting for Fortnite"
            )
            self.detail = "Return to the game to record. Switching away pauses capture."
            if self.ever_recorded and not game_is_running:
                self.absent_since = (
                    self.absent_since if self.absent_since is not None else now
                )
                if now - self.absent_since >= self.config.exit_grace_seconds:
                    self.stop("Game closed — footage saved")
                    return
            else:
                self.absent_since = None
        elif self.run is None and now >= self.retry_at:
            self.absent_since = None
            key = (display.adapter, display.output)
            # Only encoders the adapter driving this display can actually run.
            # Probing Quick Sync on an NVIDIA-driven panel wastes a whole run:
            # the MFX session fails and the output file stays empty.
            encoders = [
                name
                for name in ("h264_nvenc", "h264_qsv")
                if display.supports(name)
            ] or ["h264_nvenc"]
            index = min(self.next_encoder.get(key, 0), len(encoders) - 1)
            encoder = self.encoder_for.get(key, encoders[index])
            # On a failed cached encoder, retry an alternative instead of pinning it forever.
            if self.failures or not display.supports(encoder):
                encoder = encoders[index]
            # An explicit choice outranks both the preference order and the cache.
            if self.config.encoder:
                encoder = self.config.encoder
            self.run_number += 1
            self.run = self.run_factory(
                self.directory / f"run-{self.run_number:03d}",
                display,
                encoder,
                self.config,
            )
            self.status = "Starting capture"
            self.detail = f"Checking {encoder} on {display.name}…"
            self._record(
                "capture_started",
                display=asdict(display),
                encoder=encoder,
                run_id=self.run.directory.name,
            )
        elif self.run:
            self._publish()
            if now - self.run.started >= 2 and self.run.bytes_written() > 0:
                self.ever_recorded = True
                self.failures = 0
                self.encoder_for[(display.adapter, display.output)] = self.run.encoder
                self.status = "Recording Fortnite"
                self.detail = f"{self.config.fps} FPS · video only · saves every {self.config.segment_seconds}s · {self.run.encoder}"
        else:
            self.status = "Recovering capture"
            self.detail = "Retrying with a hardware encoder. Saved chunks are retained."
        self._save_manifest()

    def watch(self, stop_event: threading.Event, emit):
        from pipeline_v2.capture.windows import foreground_game, game_running

        try:
            while not self.done and not stop_event.is_set():
                self.tick(foreground_game(), game_running())
                emit(self.snapshot())
                stop_event.wait(0.2)
        except Exception as exc:
            self._record("error", message=str(exc))
            self.stop(f"Capture error: {exc}")
        finally:
            if not self.done:
                self.stop()
            emit(self.snapshot())
