"""Game and microphone audio, on separate tracks, alongside WGC video.

All 194 chunks of the September 7 session carry video and no audio at all, so
the A/V drift gate has never been evaluable and the saved highlights are silent.
SCOPE's first audio priority is blunt about it: usable recorded audio is
required before recorder release.

FFmpeg cannot supply it on this machine. `-f dshow` lists microphones and no
loopback device, and this build has no `wasapi` input, so game audio has to come
from WASAPI loopback via `soundcard` and be piped in.

Two tracks, not a mix. SCOPE asks for it -- "separate tracks preserve future
options" -- and the reason is concrete: the ranking question of whether voice
or game effects predict a keep is unanswerable once they are summed, which is
exactly the state the mixed-stereo NVIDIA clips left the corpus in. Mix for
playback; keep them separable for measurement.

The pump never blocks its reader. A device that stalls drops a block and counts
it rather than backing up into the capture threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline_v2.capture.levels import SILENCE_DB, to_db

SAMPLE_RATE = 48000
CHANNELS = 2
# ~21ms per block: small enough that a dropped one is inaudible, large enough
# not to wake the thread thousands of times a second.
BLOCK_FRAMES = 1024
# How far a track may drift from the video before it is corrected. Under the
# ~45ms at which a late sound starts to look wrong, over one block of jitter.
SYNC_TOLERANCE = int(0.04 * SAMPLE_RATE)
# How often a track asks Windows which device it should be recording. A device
# opened once at the start stays the one recorded: session 4 (2026-10-19) began
# before the player plugged in their mic, and its whole mic track came from the
# device that was the default before it.
FOLLOW_SECONDS = 2.0


@dataclass
class TrackStats:
    name: str
    blocks: int = 0
    dropped: int = 0
    silent_blocks: int = 0
    # WASAPI lost samples: a real gap, after which this track no longer lines
    # up with the video. Counted so a recording can be judged rather than
    # assumed intact.
    discontinuities: int = 0
    # Corrections that kept this track on the video's timeline: silence added
    # where the device delivered nothing (loopback sends no samples while the
    # game is silent), samples dropped where it ran ahead.
    padded_samples: int = 0
    trimmed_samples: int = 0
    # Times the track moved to another device (plugged in, unplugged, or the
    # Windows default changed).
    switches: int = 0
    error: str | None = None
    # Loudness of the most recent block, so a detector can read the level while
    # the session is still running. The block maximum is already computed to
    # count silence, so this costs one logarithm per block rather than a second
    # pass over the audio -- which matters because the alternative is extracting
    # levels from the finished file, and a trigger has to fire during play.
    latest_db: float = SILENCE_DB

    @property
    def seconds(self) -> float:
        return self.blocks * BLOCK_FRAMES / SAMPLE_RATE

    @property
    def silent_fraction(self) -> float:
        return self.silent_blocks / self.blocks if self.blocks else 0.0


@dataclass
class AudioStats:
    tracks: list[TrackStats] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            t.name: {
                "seconds": round(t.seconds, 2),
                "dropped": t.dropped,
                "silent_fraction": round(t.silent_fraction, 3),
                "discontinuities": t.discontinuities,
                "padded_seconds": round(t.padded_samples / SAMPLE_RATE, 3),
                "trimmed_seconds": round(t.trimmed_samples / SAMPLE_RATE, 3),
                "switches": t.switches,
                "error": t.error,
            }
            for t in self.tracks
        }


# Must divide both the segment length and the coverage stride, so segment
# boundaries and coverage window starts all land on a keyframe.
KEYFRAME_SECONDS = 2

def loopback_device(speaker_name: str | None = None):
    """The WASAPI loopback matching an output device, or None.

    Loopback follows the *speaker*, so a user switching from headphones to
    desktop speakers mid-session changes which device carries game audio;
    `TrackPump` asks again every `FOLLOW_SECONDS`.
    """
    import soundcard as sc

    try:
        name = speaker_name or sc.default_speaker().name
        return sc.get_microphone(name, include_loopback=True)
    except (RuntimeError, IndexError, AssertionError):
        return None


def microphone_device(name_hint: str | None = None):
    """A real (non-loopback) microphone, preferring one matching the hint."""
    import soundcard as sc

    mics = [m for m in sc.all_microphones(include_loopback=False)]
    if not mics:
        return None
    if name_hint:
        for mic in mics:
            if name_hint.lower() in mic.name.lower():
                return mic
    try:
        return sc.default_microphone()
    except (RuntimeError, IndexError, AssertionError):
        return mics[0]


def pipe_name(label: str) -> str:
    """A unique Windows named pipe path.

    FFmpeg cannot inherit a `pipe:N` file descriptor across the Windows process
    boundary -- it reports "Bad file descriptor" -- so each extra audio input is
    served as a named pipe that FFmpeg opens like a file.
    """
    import uuid

    return "\\\\.\\pipe\\salience-" + f"{label}-{uuid.uuid4().hex[:8]}"


class NamedPipeSink:
    """Serves one named pipe, blocking until FFmpeg connects."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._handle = None

    def open(self) -> "NamedPipeSink":
        import win32pipe

        self._handle = win32pipe.CreateNamedPipe(
            self.name,
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1, 1 << 20, 1 << 20, 0, None,
        )
        return self

    def connect(self) -> None:
        import win32pipe

        win32pipe.ConnectNamedPipe(self._handle, None)

    def write(self, data: bytes) -> None:
        import win32file

        win32file.WriteFile(self._handle, data)

    def close(self) -> None:
        if self._handle is not None:
            import win32file

            try:
                win32file.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001 - closing must never raise
                pass
            self._handle = None


def correction(written: int, block: int, target: int,
               tolerance: int = SYNC_TOLERANCE) -> tuple[int, int]:
    """(silence to insert before the block, samples to cut from its start) so
    the track ends near `target`, the video's length in samples.

    FFmpeg stamps a raw PCM input by its sample count alone, so a track is
    only in sync while it holds exactly as many samples as the video has
    seconds. It did not: loopback delivers nothing while the game is silent,
    and the mic runs on its own clock. Measured with a flash-and-beep test on
    2026-10-18: audio slid ~2.5ms per second against the video, which over a
    51-minute session is seconds.
    """
    ahead = written + block - target
    if ahead < -tolerance:
        return -ahead, 0
    if ahead > tolerance:
        return 0, min(block, ahead)
    return 0, 0


class _SilentRecorder:
    def __init__(self, channels: int) -> None:
        self.channels = channels

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def record(self, numframes: int):
        import numpy as np

        time.sleep(numframes / SAMPLE_RATE)
        return np.zeros((numframes, self.channels), np.float32)


class Silence:
    """Stands in while there is no device: zeros at the pace of real audio.

    The track has to keep flowing, not pause: FFmpeg interleaves its inputs,
    so a pipe that went quiet would hold the video back with it.
    """

    id = None
    name = "no device"

    def recorder(self, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=None):
        return _SilentRecorder(channels)


SILENCE = Silence()


def _same(a, b) -> bool:
    return getattr(a, "id", a) == getattr(b, "id", b)


class TrackPump:
    """Reads one device and writes raw float32 PCM to a pipe.

    With `timeline` (seconds of video written so far), the track is held to
    the video's length: see `correction`.

    With `choose` (the device this track should be recording now), the device
    is chosen again every `FOLLOW_SECONDS` and whenever it fails, so a mic
    plugged in mid-session is recorded from then on and one unplugged leaves
    silence rather than ending the track. The gap a switch leaves is padded by
    `correction` like any other.
    """

    def __init__(self, name: str, device, sink, timeline=None, choose=None,
                 clock=time.monotonic) -> None:
        self.stats = TrackStats(name=name)
        self._device = device
        self._sink = sink
        self._timeline = timeline
        self._choose = choose
        self._clock = clock
        self._written = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _chosen(self):
        try:
            return self._choose() or SILENCE
        except Exception:  # noqa: BLE001 - asking must never end the track
            return SILENCE

    def _push(self, block, np) -> bool:
        """Correct and write one block; False once the sink is gone."""
        self.stats.blocks += 1
        # Silence is diagnostic, not an error: a muted game or an
        # unplugged mic produces a valid but useless track, and the
        # report should be able to say which.
        loudest = float(np.abs(block).max())
        self.stats.latest_db = to_db(loudest)
        if loudest < 1e-5:
            self.stats.silent_blocks += 1
        # None: no video yet to hold to. FFmpeg that captures
        # its own video (`gfx`) counts no frames until audio
        # flows, so holding audio back then would wait forever;
        # the startup excess is cut once frames are counted.
        target = self._timeline() if self._timeline is not None else None
        if target is not None:
            pad, cut = correction(self._written, len(block), int(target * SAMPLE_RATE))
            if pad:
                block = np.concatenate(
                    [np.zeros((pad, block.shape[1]), np.float32), block])
                self.stats.padded_samples += pad
            if cut:
                block = block[cut:]
                self.stats.trimmed_samples += cut
            if not len(block):
                return True
        self._written += len(block)
        try:
            self._sink.write(np.ascontiguousarray(block, dtype=np.float32).tobytes())
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.stats.error = f"sink closed: {exc}"
            return False
        return True

    def _loop(self) -> None:
        import warnings

        import numpy as np

        try:
            # soundcard raises SoundcardRuntimeWarning("data discontinuity in
            # recording") when WASAPI drops samples. That is a real gap in the
            # track -- audio that will not line up with the video after it --
            # so it is counted rather than printed and forgotten.
            warnings.simplefilter("always")
            # Blocks until FFmpeg opens its side; nothing can be written before.
            connect = getattr(self._sink, "connect", None)
            if connect is not None:
                connect()
        except Exception as exc:  # noqa: BLE001 - a device fault must not kill capture
            self.stats.error = f"{type(exc).__name__}: {exc}"
            return
        device = self._device or SILENCE
        while not self._stop.is_set():
            switch = None
            try:
                with device.recorder(
                    samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_FRAMES
                ) as recorder:
                    check = self._clock() + FOLLOW_SECONDS
                    while not self._stop.is_set():
                        with warnings.catch_warnings(record=True) as caught:
                            block = recorder.record(numframes=BLOCK_FRAMES)
                        if any("discontinuity" in str(w.message) for w in caught):
                            self.stats.discontinuities += 1
                        if block is not None and len(block) and not self._push(block, np):
                            return
                        if self._choose is not None and self._clock() >= check:
                            check = self._clock() + FOLLOW_SECONDS
                            found = self._chosen()
                            if not _same(found, device):
                                switch = found
                                break
            except Exception as exc:  # noqa: BLE001 - a device fault must not kill capture
                if self._choose is None:
                    self.stats.error = f"{type(exc).__name__}: {exc}"
                    return
                # Unplugged mid-read: whatever is there now, or silence.
                self._stop.wait(0.2)
                switch = self._chosen()
                if _same(switch, device):
                    switch = SILENCE  # still listed but failing; ask again later
            if switch is not None:
                device = switch
                self.stats.switches += 1

    def start(self) -> "TrackPump":
        self._thread = threading.Thread(
            target=self._loop, name=f"audio-{self.stats.name}", daemon=True
        )
        self._thread.start()
        return self

    def signal_stop(self) -> None:
        """Ask the thread to finish without waiting for it.

        Separate from `stop` so every producer can be signalled before any is
        joined; otherwise the ones signalled later keep recording while an
        earlier join blocks, and the tracks end up different lengths.
        """
        self._stop.set()

    def stop(self) -> TrackStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        close = getattr(self._sink, "close", None)
        if close is not None:
            close()
        return self.stats


def mux_command(
    video_size: str | None,
    fps: int,
    bitrate: str,
    encoder: str,
    segment_seconds: int,
    pattern: Path,
    segment_list: Path,
    audio_pipes: list[str],
    *,
    gpu_source: str | None = None,
    gpu_vendor: str | None = None,
) -> list[str]:
    """FFmpeg taking BGRA video on stdin and one raw PCM input per audio track.

    With `gpu_source` (a `gfxcapture` filter, see `gfx`), the video is
    captured inside FFmpeg instead, on the GPU of `gpu_vendor`, and its frame
    count is reported on stdout (`-progress`).

    Each track is mapped to its own output stream rather than mixed, and
    `-shortest` is deliberately absent: a track that starts late or drops out
    must not truncate the video.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning" if gpu_source is None else "info",
        "-y",
    ]
    if gpu_source is None:
        command += [
            "-f", "rawvideo",
            "-pixel_format", "bgra",
            "-video_size", video_size,
            "-framerate", str(fps),
            "-i", "pipe:0",
        ]
    else:
        command += [
            "-progress", "pipe:1", "-stats_period", "0.1", "-nostats",
            "-init_hw_device", f"d3d11va=gpu:,vendor_id={gpu_vendor}",
            "-filter_hw_device", "gpu",
            # `showinfo` logs each frame as it is captured, before encoding
            # or muxing: when frame 0's line arrives is when the video starts
            # (`gfx.Progress.logged`). A side branch ending in `nullsink`
            # would log only frame 0, but crashes FFmpeg 9's scheduler.
            "-filter_complex", f"{gpu_source},showinfo,fps={fps}[video]",
        ]
    first_audio = 1 if gpu_source is None else 0
    for pipe in audio_pipes:
        if gpu_source is not None:
            # Opened before the first frame exists; read no more than needed.
            command += ["-probesize", "32", "-analyzeduration", "0"]
        command += [
            "-f", "f32le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", pipe,
        ]
    command += ["-map", "0:v:0" if gpu_source is None else "[video]"]
    for index in range(len(audio_pipes)):
        command += ["-map", f"{index + first_audio}:a:0"]
    command += [
        "-c:v", encoder,
        "-preset", "p1" if encoder == "h264_nvenc" else "veryfast",
    ]
    if gpu_source is None:
        # GPU frames go to NVENC as they are; it writes yuv420p itself.
        command += ["-pix_fmt", "yuv420p"]
    command += [
        "-b:v", bitrate,
        # A keyframe every KEYFRAME_SECONDS, so highlight windows can be
        # stream-copied out after the session instead of re-encoded: 0.2s a
        # window instead of 2.4s, which for full coverage of half an hour is
        # under a minute instead of ten. Two seconds is OBS's default, so it
        # is the interval every streamer already plays under. (It was one per
        # 30s segment; `-g fps`, one a second, was measured as too costly.)
        "-g", str(fps * KEYFRAME_SECONDS),
        "-bf", "0",
    ]
    if audio_pipes:
        command += ["-c:a", "aac", "-b:a", "160k"]
    command += [
        "-force_key_frames", f"expr:gte(t,n_forced*{KEYFRAME_SECONDS})",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-segment_list", str(segment_list),
        "-segment_list_type", "csv",
        "-reset_timestamps", "1",
        str(pattern),
    ]
    return command
