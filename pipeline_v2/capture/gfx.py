"""Windows Graphics Capture inside FFmpeg: frames never leave the GPU.

`wgc` captured the same way but handed every frame to Python: a 9MB copy out
of the GPU 51 times a second at 1920x1200, then 30 of them down a pipe for
FFmpeg to upload again. That cost 63% of a core in the lobby and 112% over
session 4 (2026-10-19), whose player saw frame drops in fights. FFmpeg 8 added
`gfxcapture`, the same capture API as a filter whose frames stay D3D11
textures all the way into NVENC: measured at 3% of one core for the same
1920x1200 at 30fps.

What the Python recorder did by hand, FFmpeg now does:

- the `fps` filter holds a constant rate, repeating a frame when the window
  has not changed (29.93fps against the wall clock on a still screen);
- the capture ends by itself when the window closes, which is how the game
  quitting is seen (`source_closed`).

What stays here is the clock. Every timeline the session uses (the game log,
key presses, the audio pumps) needs to know when frame 0 was captured: FFmpeg
logs it from a side branch of the capture graph, and its arrival is stamped.
The frame count comes from `-progress`, ten times a second.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from pipeline_v2.capture.wgc import WgcStats

# The GPU that encodes. On a laptop the desktop belongs to the integrated GPU,
# and a capture device created there cannot reach NVENC ("no encode device").
NVIDIA = "0x10de"
# What FFmpeg logs for frame 0 (`showinfo` on the capture's first frame).
FIRST_FRAME = "Parsed_showinfo"


def available(runner=subprocess.run) -> bool:
    """Whether this FFmpeg has `gfxcapture` (8.0 and later)."""
    try:
        out = runner(["ffmpeg", "-hide_banner", "-h", "filter=gfxcapture"],
                     capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return "gfxcapture AVOptions" in (out.stdout or "")


def source(window_hwnd: int | None, monitor_index: int | None, capture_fps: int) -> str:
    """The `gfxcapture` filter for one window, or a monitor (1-based)."""
    target = (f"hwnd={window_hwnd}" if window_hwnd
              else f"monitor_idx={max(0, (monitor_index or 1) - 1)}")
    # scale_aspect: a resolution change mid-session keeps the whole picture
    # rather than cropping it to the first size.
    return (f"gfxcapture={target}:max_framerate={capture_fps}:capture_cursor=0"
            ":resize_mode=scale_aspect")


def command(capture: str, fps: int, bitrate: str, segment_seconds: int,
            pattern: Path, segment_list: Path, audio_pipes: list[str]) -> list[str]:
    from pipeline_v2.capture import audio_pump

    return audio_pump.mux_command(
        None, fps, bitrate, "h264_nvenc", segment_seconds, pattern, segment_list,
        audio_pipes, gpu_source=capture, gpu_vendor=NVIDIA,
    )


class Progress:
    """Frame count from FFmpeg's `-progress` lines, frame-0 time from its log."""

    def __init__(self, fps: int, clock=time.monotonic) -> None:
        self.fps = fps
        self.clock = clock
        self.frames = 0
        self.origin: float | None = None

    def feed(self, line: str) -> None:
        if line.startswith("frame="):
            try:
                self.frames = int(line.split("=", 1)[1])
            except ValueError:
                pass

    def logged(self, line: str) -> None:
        """A line of FFmpeg's log. Placing frame 0 by the muxed frame count
        instead put it ~180ms late, so the audio came out that much early."""
        if self.origin is None and FIRST_FRAME in line and " n:" in line:
            self.origin = self.clock()

    def timeline(self) -> float | None:
        """Seconds of video so far, for the audio pumps to hold to; None until
        frame 0 is placed (`TrackPump` then lets audio flow).

        From the clock, not the frame count: FFmpeg counts a frame once it is
        muxed, and the muxer waits on the audio it is interleaving, so a count
        held back by audio would hold audio back in turn (it swung 13s of
        padding and trimming in 40s). The `fps` filter makes the video exactly
        one frame per 1/fps of capture time, so elapsed time is its length.
        """
        if self.origin is None:
            return None
        return max(0.0, self.clock() - self.origin)


class GfxRecorder:
    """Same surface as `WgcRecorder`; FFmpeg does the capture."""

    def __init__(self, directory: Path, *, fps: int = 30, bitrate: str = "8M",
                 encoder: str = "h264_nvenc", segment_seconds: int = 30,
                 monitor_index: int | None = None, window_hwnd: int | None = None,
                 audio: bool = False, mic_hint: str | None = None,
                 capture_fps: int | None = None) -> None:
        self.directory = directory
        self.fps = fps
        self.capture_fps = capture_fps or fps * 2
        self.bitrate = bitrate
        self.segment_seconds = segment_seconds
        self.monitor_index = monitor_index
        self.window_hwnd = window_hwnd
        self.audio = audio
        self.mic_hint = mic_hint
        self.stats = WgcStats()
        self.audio_stats = None
        self.error: str | None = None
        self.progress = Progress(fps)
        self._pumps: list = []
        self._sinks: list = []
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self._stopping = False

    def _audio(self) -> list[str]:
        from pipeline_v2.capture import audio_pump

        stats = audio_pump.AudioStats()
        pipes = []
        for name, device in (("game", audio_pump.loopback_device()),
                             ("mic", audio_pump.microphone_device(self.mic_hint))):
            if device is None:
                stats.tracks.append(audio_pump.TrackStats(name=name, error="no such device"))
                continue
            sink = audio_pump.NamedPipeSink(audio_pump.pipe_name(name)).open()
            self._sinks.append(sink)
            pump = audio_pump.TrackPump(name, device, sink, timeline=self.progress.timeline)
            self._pumps.append(pump)
            stats.tracks.append(pump.stats)
            pipes.append(sink.name)
        self.audio_stats = stats
        return pipes

    def start(self) -> "GfxRecorder":
        self.directory.mkdir(parents=True, exist_ok=True)
        pipes = self._audio() if self.audio else []
        self._log = (self.directory / "ffmpeg.log").open("wb")
        self._process = subprocess.Popen(
            command(source(self.window_hwnd, self.monitor_index, self.capture_fps),
                    self.fps, self.bitrate, self.segment_seconds,
                    self.directory / "part-%05d.mkv", self.directory / "segments.csv", pipes),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._reader = threading.Thread(target=self._read, name="gfx-progress", daemon=True)
        self._reader.start()
        self._logger = threading.Thread(target=self._read_log, name="gfx-log", daemon=True)
        self._logger.start()
        # Audio flows freely until the first frames are counted, then is held
        # to the video; the startup excess is cut within about a second.
        for pump in self._pumps:
            pump.start()
        return self

    def _read_log(self) -> None:
        for raw in self._process.stderr:
            line = raw.decode("utf-8", "replace")
            if FIRST_FRAME in line:
                if self.progress.origin is not None:
                    continue  # a line per frame; only frame 0's is kept
                self.progress.logged(line)
                self.stats.first_frame_at = self.progress.origin
            self._log.write(raw)

    def _read(self) -> None:
        for raw in self._process.stdout:
            self.progress.feed(raw.decode("utf-8", "replace").strip())
            self.stats.written = self.stats.delivered = self.progress.frames
        code = self._process.wait()
        if not self._stopping:
            if code == 0:
                self._closed.set()  # the window closed: the game quit
            else:
                self.error = f"ffmpeg exited with code {code} (see ffmpeg.log)"
                self._closed.set()

    @property
    def source_closed(self) -> bool:
        return self._closed.is_set()

    def mic_level(self) -> float | None:
        for pump in self._pumps:
            if pump.stats.name == "mic":
                return pump.stats.latest_db
        return None

    def stop(self) -> WgcStats:
        self._stopping = True
        for pump in self._pumps:
            pump.signal_stop()
        if self._process is not None:
            try:
                self._process.stdin.write(b"q")  # FFmpeg's own clean stop
                self._process.stdin.flush()
            except OSError:
                pass
        # The audio pipes closed before waiting: FFmpeg finishes the last
        # segment only once every input has ended.
        for pump in self._pumps:
            pump.stop()
        self._pumps = []
        for sink in self._sinks:
            sink.close()
        self._sinks = []
        if self._process is not None:
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        for reader in (self._reader, getattr(self, "_logger", None)):
            if reader is not None:
                reader.join(timeout=5)
        if self._process is not None:
            self._log.close()
            self._process = None
        return self.stats
