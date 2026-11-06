"""Single-frame FFmpeg extraction and a bounded parallel runner.

Extracted from the v1 keyframe module, which mixed these general helpers with
OCR crops and composite layouts that only the retired student model used.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import functools
import os
from pathlib import Path
import subprocess
from typing import TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")

# Set by work that runs while a game is being played (the live describer), so
# frame extraction yields the CPU to the game. Only processes started through
# this module see it; the recorder's own encoder is never lowered.
LOW_PRIORITY = False


def _hwaccel() -> list[str]:
    """Decode on the GPU's video decoder (NVDEC), which a game does not use,
    instead of the CPU, which it does. Measured on 28 frames: 11.8s of CPU
    down to 3.3s, and the JPEGs byte-identical (H.264 decoding is exact), so
    cached descriptions still match. SALIENCE_FFMPEG_HWACCEL=none turns it off."""
    accel = os.getenv("SALIENCE_FFMPEG_HWACCEL", "cuda")
    return [] if accel in ("", "none") else ["-hwaccel", accel]


def priority_flags() -> int:
    """`creationflags` for a helper ffmpeg: below normal while a game runs."""
    if LOW_PRIORITY and os.name == "nt":
        return subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return 0


# Threads a helper ffmpeg may use while a game runs.
GAME_THREADS = 2


def efficient_mask(cpus: list[tuple[int, int, int]]) -> int | None:
    """Affinity mask of the efficiency cores among (group, processor, efficiency
    class) entries, or None on a CPU whose cores are all alike. The lowest
    class is the efficient one; only group 0 (the first 64) can be masked."""
    classes = {efficiency for _, _, efficiency in cpus}
    if len(classes) < 2:
        return None
    low = min(classes)
    mask = sum(1 << index for group, index, efficiency in cpus if group == 0 and efficiency == low and index < 64)
    return mask or None


# A test: behave as on a CPU with no efficiency cores (`simulate_no_efficiency_cores`).
NO_ECORES_KEY = "SALIENCE_TEST_NO_EFFICIENCY_CORES"


def simulating_no_efficiency_cores() -> bool:
    return os.environ.get(NO_ECORES_KEY, "").strip().lower() in ("1", "on", "yes")


def simulate_no_efficiency_cores() -> bool:
    """This process (and what it starts) kept to the performance cores, and
    no efficiency cores reported: how Salience runs on a CPU without them,
    beside the game on the same cores. Whether it was set."""
    if os.name != "nt" or not simulating_no_efficiency_cores():
        return False
    real = _detected_efficiency_cores()
    if not real:
        return False
    import ctypes

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    every = (1 << (os.cpu_count() or 1)) - 1
    return bool(kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), every & ~real))


def _efficiency_cores() -> int | None:
    """`efficient_mask` of this machine, or None while simulating a CPU without."""
    return None if simulating_no_efficiency_cores() else _detected_efficiency_cores()


@functools.cache
def _detected_efficiency_cores() -> int | None:
    """`efficient_mask` of this machine (GetSystemCpuSetInformation)."""
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not size.value or not kernel.GetSystemCpuSetInformation(buffer, size, ctypes.byref(size), None, 0):
        return None
    raw, cpus, at = buffer.raw, [], 0
    while at + 19 <= size.value:
        length = int.from_bytes(raw[at:at + 4], "little")
        if length <= 0:
            break
        if int.from_bytes(raw[at + 4:at + 8], "little") == 0:  # CpuSetInformation
            # Id(4) Group(2) LogicalProcessorIndex CoreIndex LastLevelCacheIndex NumaNodeIndex EfficiencyClass
            cpus.append((int.from_bytes(raw[at + 12:at + 14], "little"), raw[at + 14], raw[at + 18]))
        at += length
    return efficient_mask(cpus)


def thread_beside_game() -> bool:
    """This thread below normal priority, and on the efficiency cores where
    the CPU has them; whether it was moved there. For the work that runs all
    through a match (recording the audio, the key and mouse logs): out of
    the game's way, as its ffmpegs are. Without efficiency cores the
    priority is what keeps it behind the game; it can wait: audio is read
    every 100ms from a half-second buffer, and the logs only count."""
    if os.name != "nt":
        return False
    import ctypes

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentThread.restype = ctypes.c_void_p
    kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel.SetThreadPriority(kernel.GetCurrentThread(), -1)  # THREAD_PRIORITY_BELOW_NORMAL
    mask = _efficiency_cores()
    if not mask:
        return False
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return bool(kernel.SetThreadAffinityMask(kernel.GetCurrentThread(), mask))


def _beside_game(process: subprocess.Popen) -> None:
    """Keep a started helper off the cores a game plays on: the efficiency
    cores where the CPU has them, and efficiency mode (EcoQoS) besides.

    Below normal priority was not enough. Fortnite is held back by its threads
    on the performance cores, and an ffmpeg's threads were put beside them
    there: the kill-banner scan and cutting, about 3.5 CPU-seconds a window,
    in bursts every 30s, cost 10fps and a third of the 1% lows with nothing
    described (i5-14500HX, 2026-10-31: 161 alone, 150 and 148 with them).
    """
    import ctypes

    class Throttling(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    handle = int(process._handle)  # type: ignore[attr-defined]
    mask = _efficiency_cores()
    if mask:
        kernel.SetProcessAffinityMask(handle, mask)
    state = Throttling(1, 0x1, 0x1)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, on
    kernel.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))  # ProcessPowerThrottling


def never_throttled(process: subprocess.Popen) -> None:
    """The recorder, out of efficiency mode for good, at normal priority.

    Windows puts a windowless process in efficiency mode (EcoQoS) on its own,
    by heuristics that change from one session to the next: the tray was (5x
    slower speech, 2026-10-27), and a recorder there kept a quarter of the
    game's frames: 15 new a second of 60 written, for a whole session, the
    game at 161fps (2026-11-02, session 20261102-213553; the one before it,
    every frame). Opted out, Windows may not choose for it.
    """
    if os.name != "nt":
        return
    import ctypes

    class Throttling(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    handle = int(process._handle)  # type: ignore[attr-defined]
    kernel.SetPriorityClass(handle, 0x20)  # NORMAL: never the tray's below normal, inherited
    # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, both controlled, both off.
    state = Throttling(1, 0x1 | 0x4, 0)
    kernel.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))  # ProcessPowerThrottling


def run(command: list[str], *, confine=None, **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run` for a helper ffmpeg or ffprobe. While a game runs
    (LOW_PRIORITY): below normal, `GAME_THREADS` threads, and kept beside the
    game (`_beside_game`). Otherwise exactly `subprocess.run`."""
    if not LOW_PRIORITY or os.name != "nt":
        return subprocess.run(command, **kwargs)
    command = list(command)
    if Path(command[0]).stem.lower() == "ffmpeg":
        command[1:1] = ["-threads", str(GAME_THREADS), "-filter_threads", str(GAME_THREADS)]
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    timeout, data, check = kwargs.pop("timeout", None), kwargs.pop("input", None), kwargs.pop("check", False)
    if data is not None:
        kwargs["stdin"] = subprocess.PIPE
    with subprocess.Popen(command, **kwargs) as process:
        try:
            (confine or _beside_game)(process)
        except (OSError, AttributeError, ValueError):
            pass  # a helper that could not be moved still runs below normal
        try:
            out, err = process.communicate(data, timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
    done = subprocess.CompletedProcess(process.args, process.returncode, out, err)
    if check:
        done.check_returncode()
    return done


def ffmpeg_worker_count() -> int:
    """Return a bounded extraction pool size with a sequential override."""
    if LOW_PRIORITY:
        # While a game runs: one decoder at a time. Seven at once was a burst
        # that took a 165fps game to 100-120 every 30s (2026-10-18).
        return 1
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
    for accel in (_hwaccel(), []):
        completed = run(
            [
                "ffmpeg",
                "-y",
                *accel,
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
            creationflags=priority_flags(),
        )
        if completed.returncode == 0 and output.exists():
            return True
        if not accel:
            break  # no GPU decoder (or no NVIDIA GPU): the CPU attempt was the last
    return False
