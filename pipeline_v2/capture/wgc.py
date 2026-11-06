"""Windows Graphics Capture backend: frames in Python, encoding in FFmpeg.

Desktop Duplication failed the performance gate on 2026-09-16 by eleven times --
55.2% of the game's 1% lows against a 5% budget -- with every encode setting
already tuned. The damage was stalls rather than throughput: GPU utilisation
barely moved while the 99th-percentile frame tripled. DDA copies the entire
desktop every frame regardless of what is done with the result, so no flag
removes that cost.

WGC differs in the two ways that matter here:

* it captures **one window**, not the whole desktop, and
* ``minimum_update_interval`` throttles at the **source**, so it composites a
  bounded number of frames rather than every one the game draws.

The throttle is set to twice the output rate, not equal to it. WGC releases
frames on display refreshes, so at an interval equal to the output period
delivery quantises to *below* 30fps (24.9-28.8/s measured on a 180Hz panel)
and the pacer repeats a frame every ~6: 6.9-10.5% of frames during motion, the
stutter seen in session 2's recordings (10.4%). At twice the rate, delivery is
~44/s, the pacer always has a fresh frame, and repeats fall to 0.9-1.1% for
about a tenth of a core more (bench on a 120fps source, 2026-10-14).

It also needs no OBS install, which rules the obs-websocket route out as an
end-user dependency, and it never injects into the game.

The trade is that frames arrive in a Python callback instead of staying inside
FFmpeg. That costs a copy per delivered frame, and it is the reason this is
worth measuring rather than assuming: run `capture.bench` against it and compare
with the recorded DDA baseline.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture.windows import game_window  # noqa: F401

# A second WGC session in one process segfaults once audio is involved -- the
# native capture thread and `soundcard`'s COM state do not both survive a
# restart. Measured: video-only then audio-enabled crashes the interpreter with
# no traceback, which is the worst failure a shipped recorder can have.
#
# The fix is architectural: one recording session per process, which is what
# `.local-data/tools/wgc_record.py` already does and what a launcher must do
# (spawn a subprocess per session rather than hosting captures in-process).
# This flag turns the crash into an error that says so.
_SESSION_STARTED = False


class WgcSessionReused(RuntimeError):
    """A second capture session was started in a process that already had one."""


def encoder_command(
    width: int,
    height: int,
    fps: int,
    bitrate: str,
    encoder: str,
    segment_seconds: int,
    pattern: Path,
    segment_list: Path,
    audio_pipes: list[str] | None = None,
) -> list[str]:
    """FFmpeg reading raw BGRA on stdin, optionally muxing audio tracks.

    One builder for both cases. Video-only and with-audio were separate
    functions whose encode settings had to be kept in step by hand, which is
    exactly how `-g` drifted out of sync with the original recorder before.
    """
    from pipeline_v2.capture import audio_pump

    return audio_pump.mux_command(
        f"{width}x{height}", fps, bitrate, encoder, segment_seconds,
        pattern, segment_list, audio_pipes or [],
    )


@dataclass
class WgcStats:
    """What the capture produced.

    `written` exceeding `delivered` is normal and expected: the pacer repeats
    the last frame to hold a constant rate when the scene is static.
    `superseded` is the opposite case -- WGC delivered faster than the pacer
    consumed, so a frame was overwritten before it was ever encoded. That is
    by design since the source captures at twice the output rate: about a
    third of deliveries are superseded, and it is what keeps repeats rare.
    """

    delivered: int = 0
    written: int = 0
    superseded: int = 0
    # `time.monotonic()` when frame 0 reached the encoder, which is t=0 of every
    # segment timeline the clip cutter reads. Triggers are stamped on the same
    # clock, so subtracting this places them on the video's own timeline. Using
    # any other start -- when the recorder was constructed, when the input
    # sampler began -- shifts every clip by however long WGC and ffmpeg took to
    # come up, which was never measured.
    first_frame_at: float | None = None

    @property
    def superseded_rate(self) -> float:
        return self.superseded / self.delivered if self.delivered else 0.0


class WgcRecorder:
    """Pumps WGC frames into an FFmpeg segment ring.

    One rule governs the pump: **never block the capture callback**. WGC
    delivers on its own thread, and stalling it to wait for the encoder would
    push back on the compositor -- reintroducing exactly the stall behaviour
    that disqualified DDA. A frame that cannot be written immediately is
    superseded and counted instead.
    """

    def __init__(
        self,
        directory: Path,
        *,
        fps: int = 30,
        bitrate: str = "8M",
        encoder: str = "h264_nvenc",
        segment_seconds: int = 30,
        monitor_index: int | None = None,
        window_hwnd: int | None = None,
        audio: bool = False,
        mic_hint: str | None = None,
        capture_fps: int | None = None,
    ) -> None:
        self.directory = directory
        self.fps = fps
        self.capture_fps = capture_fps or fps * 2
        self.bitrate = bitrate
        self.encoder = encoder
        self.segment_seconds = segment_seconds
        self.monitor_index = monitor_index
        self.window_hwnd = window_hwnd
        self.audio = audio
        self.mic_hint = mic_hint
        self.stats = WgcStats()
        self.audio_stats = None
        self._pumps: list = []
        self._sinks: list = []
        self._audio_devices: list = []
        self.error: str | None = None
        self._process: subprocess.Popen | None = None
        self._control = None
        self._latest: bytes | None = None
        self._repeat: bytes | None = None
        self._pacer: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = threading.Event()

    def _start_encoder(self, width: int, height: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._log = (self.directory / "ffmpeg.log").open("wb")
        if self.audio:
            command = self._audio_command(width, height)
        else:
            command = encoder_command(
                width, height, self.fps, self.bitrate, self.encoder,
                self.segment_seconds,
                self.directory / "part-%05d.mkv",
                self.directory / "segments.csv",
            )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # Audio and the video pacer both begin once the encoder exists, so the
        # streams start together and stay the same length.
        if getattr(self, "_pending_pumps", False):
            for pump in self._pumps:
                pump.start()
            self._pending_pumps = False
        self._pacer = threading.Thread(target=self._pace, name="wgc-pacer", daemon=True)
        self._pacer.start()

    def _audio_command(self, width: int, height: int) -> list[str]:
        """Open the audio pipes BEFORE FFmpeg starts, then start the pumps.

        A named pipe must exist before FFmpeg tries to open it, and the pump
        cannot write until FFmpeg connects -- so the order is create, launch,
        then connect from the pump thread.
        """
        from pipeline_v2.capture import audio_pump

        # Devices were resolved on the main thread in `start()`. Enumerating
        # them here instead fails with "Error 0x100000001": this runs on WGC's
        # native callback thread, whose COM apartment soundcard cannot use.
        devices = self._audio_devices
        stats = audio_pump.AudioStats()
        pipes = []
        choosers = {"game": audio_pump.loopback_device,
                    "mic": lambda: audio_pump.microphone_device(self.mic_hint)}
        # Both tracks always exist, so the mic is always the second: one with
        # no device yet records silence until one is plugged in.
        for name, device in devices:
            sink = audio_pump.NamedPipeSink(audio_pump.pipe_name(name)).open()
            self._sinks.append(sink)
            # Held to the video's own timeline: frames written / fps.
            pump = audio_pump.TrackPump(
                name, device, sink, timeline=lambda: self.stats.written / self.fps,
                choose=choosers[name])
            self._pumps.append(pump)
            stats.tracks.append(pump.stats)
            pipes.append(sink.name)
        self.audio_stats = stats
        command = audio_pump.mux_command(
            f"{width}x{height}", self.fps, self.bitrate, self.encoder,
            self.segment_seconds,
            self.directory / "part-%05d.mkv",
            self.directory / "segments.csv",
            pipes,
        )
        # Pumps start only after FFmpeg is launched, and FFmpeg is launched
        # immediately after this returns. Starting them here would let audio
        # accumulate while the encoder is still opening, which is what produced
        # a 657ms duration drift: the audio track ran longer than the video
        # because it began before the first frame existed.
        self._pending_pumps = True
        return command

    def _pace(self) -> None:
        """Write exactly `fps` frames per wall-clock second, duplicating as needed.

        WGC delivers a frame only when the captured surface changes, so a quiet
        scene yields fewer frames than the declared rate. Feeding those straight
        to a `-framerate N` input makes the video shorter than real time while
        audio runs at real time -- measured as 877ms of drift over 12 seconds.
        Pacing against the clock and repeating the last frame keeps the video
        truly constant-rate, which is what `ddagrab` did internally (its
        "More than 1000 frames duplicated" notice is this same mechanism).

        It also means the capture callback never touches the encoder: it only
        swaps in the newest frame, so a slow write can never stall WGC.
        """
        period = 1.0 / self.fps
        next_at = time.perf_counter()
        while not self._stop.is_set():
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -period * 4:
                # Far behind (a long stall); resync rather than burst-writing.
                next_at = time.perf_counter()
            latest, self._latest = self._latest, None
            if latest is None:
                # Nothing new since the last tick: repeat, to hold the rate.
                latest = self._repeat
            if latest is None or self._process is None:
                continue
            self._repeat = latest
            try:
                if self.stats.first_frame_at is None:
                    self.stats.first_frame_at = time.monotonic()
                self._process.stdin.write(latest)
                self.stats.written += 1
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.error = f"encoder closed the pipe: {exc}"
                return

    def _on_frame(self, frame, control) -> None:
        if self._stop.is_set():
            control.stop()
            return
        self.stats.delivered += 1
        if self._process is None:
            # An exception here vanishes into the native callback and the
            # capture simply stops with no explanation, so it is caught and
            # recorded instead.
            try:
                self._start_encoder(frame.width, frame.height)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self.error = f"could not start encoder: {type(exc).__name__}: {exc}"
                control.stop()
                return
        # Hand the frame to the pacer and return immediately. The callback does
        # no I/O at all, so the encoder can never back-pressure the compositor.
        if self._latest is not None:
            self.stats.superseded += 1
        self._latest = frame.frame_buffer.tobytes()

    def start(self) -> "WgcRecorder":
        import windows_capture as wc

        global _SESSION_STARTED
        if _SESSION_STARTED:
            raise WgcSessionReused(
                "a WGC capture already ran in this process; starting a second "
                "one crashes the interpreter. Run each recording session in "
                "its own process (see .local-data/tools/wgc_record.py)."
            )
        _SESSION_STARTED = True

        if self.audio:
            from pipeline_v2.capture import audio_pump

            self._audio_devices = [
                ("game", audio_pump.loopback_device()),
                ("mic", audio_pump.microphone_device(self.mic_hint)),
            ]

        capture = wc.WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            # Throttle at the source. This is the property DDA lacked: fewer
            # frames are composited, not just fewer encoded.
            minimum_update_interval=max(1, round(1000 / self.capture_fps)),
            monitor_index=self.monitor_index,
            window_hwnd=self.window_hwnd,
        )
        # `event` dispatches on the handler's __name__, so these must be named
        # exactly `on_frame_arrived` and `on_closed` -- a bound method or a
        # lambda is rejected at registration.
        def on_frame_arrived(frame, control):
            self._on_frame(frame, control)

        def on_closed():
            self._closed.set()

        capture.event(on_frame_arrived)
        capture.event(on_closed)
        self._control = capture.start_free_threaded()
        return self

    @property
    def source_closed(self) -> bool:
        """Did WGC end the capture itself?

        For a window-targeted recording this is how the game exiting is
        observed: the session closes rather than reporting an error.
        """
        return self._closed.is_set()

    def mic_level(self) -> float | None:
        """Loudness of the microphone's most recent block, or None if not recording.

        The voice detector needs the level *during* the session. Extracting it
        from the finished file is the offline path `levels.py` serves for
        threshold sweeps; a trigger cannot wait that long.
        """
        for pump in self._pumps:
            if pump.stats.name == "mic":
                return pump.stats.latest_db
        return None

    def stop(self) -> WgcStats:
        # Signal every producer before joining any of them. Joining the pacer
        # first let the audio pumps keep recording for the duration of that
        # join, which left the audio track 191ms longer than the video.
        self._stop.set()
        for pump in self._pumps:
            pump.signal_stop()
        if self._pacer is not None:
            self._pacer.join(timeout=5)
            self._pacer = None
        for pump in self._pumps:
            pump.stop()
        self._pumps = []
        if self._control is not None:
            try:
                self._control.stop()
                # `stop` only *requests* the native thread finish. Without this
                # wait it is still running when the next session starts, and a
                # second capture in the same process segfaults -- which is
                # exactly what "start recording, stop, start again" does.
                self._control.wait()
            except Exception:  # noqa: BLE001 - stopping must never raise
                pass
            self._control = None
        # A sink whose pump never ran still holds an open pipe handle; the pump
        # closes its own on stop, so only orphans remain here.
        for sink in self._sinks:
            sink.close()
        self._sinks = []
        if self._process is not None:
            # No lock needed: the pacer is the only writer and was joined above.
            try:
                self._process.stdin.close()
            except OSError:
                pass
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
            self._log.close()
        return self.stats
