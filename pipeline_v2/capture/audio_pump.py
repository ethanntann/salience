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
from dataclasses import dataclass, field
from pathlib import Path

from pipeline_v2.capture.levels import SILENCE_DB, to_db

SAMPLE_RATE = 48000
CHANNELS = 2
# ~21ms per block: small enough that a dropped one is inaudible, large enough
# not to wake the thread thousands of times a second.
BLOCK_FRAMES = 1024


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
    desktop speakers mid-session changes which device carries game audio.
    Resolving it by the current default at start time is correct for a session;
    it is not a substitute for handling a device change mid-run, which the
    recorder does not yet do.
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


class TrackPump:
    """Reads one device and writes raw float32 PCM to a pipe."""

    def __init__(self, name: str, device, sink) -> None:
        self.stats = TrackStats(name=name)
        self._device = device
        self._sink = sink
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
            with self._device.recorder(
                samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_FRAMES
            ) as recorder:
                while not self._stop.is_set():
                    with warnings.catch_warnings(record=True) as caught:
                        block = recorder.record(numframes=BLOCK_FRAMES)
                    if any("discontinuity" in str(w.message) for w in caught):
                        self.stats.discontinuities += 1
                    if block is None or not len(block):
                        continue
                    self.stats.blocks += 1
                    # Silence is diagnostic, not an error: a muted game or an
                    # unplugged mic produces a valid but useless track, and the
                    # report should be able to say which.
                    loudest = float(np.abs(block).max())
                    self.stats.latest_db = to_db(loudest)
                    if loudest < 1e-5:
                        self.stats.silent_blocks += 1
                    try:
                        self._sink.write(
                            np.ascontiguousarray(block, dtype=np.float32).tobytes()
                        )
                    except (BrokenPipeError, OSError, ValueError) as exc:
                        self.stats.error = f"sink closed: {exc}"
                        return
        except Exception as exc:  # noqa: BLE001 - a device fault must not kill capture
            self.stats.error = f"{type(exc).__name__}: {exc}"

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
    video_size: str,
    fps: int,
    bitrate: str,
    encoder: str,
    segment_seconds: int,
    pattern: Path,
    segment_list: Path,
    audio_pipes: list[str],
) -> list[str]:
    """FFmpeg taking BGRA video on stdin and one raw PCM input per audio track.

    Each track is mapped to its own output stream rather than mixed, and
    `-shortest` is deliberately absent: a track that starts late or drops out
    must not truncate the video.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgra",
        "-video_size", video_size,
        "-framerate", str(fps),
        "-i", "pipe:0",
    ]
    for pipe in audio_pipes:
        command += [
            "-f", "f32le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", pipe,
        ]
    command += ["-map", "0:v:0"]
    for index in range(len(audio_pipes)):
        command += ["-map", f"{index + 1}:a:0"]
    command += [
        "-c:v", encoder,
        "-preset", "p1" if encoder == "h264_nvenc" else "veryfast",
        "-pix_fmt", "yuv420p",
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
