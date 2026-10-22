"""How fast the player's aim moved, from the mouse's own movement counts.

Key presses measured as a ranking signal (kept kills came with more of them),
but they are blind to aim: a flick, a tracking spray or a 180 is the mouse
moving, and in game the cursor is pinned to the screen centre, so reading its
position says nothing. Raw Input is the mouse's own movement counts, before
any cursor exists.

Raw Input, not a hook. A low-level mouse hook (`WH_MOUSE_LL`) makes Windows
wait on this process for every movement before the game gets it, which is how
a busy Python process stutters the whole desktop's mouse. With Raw Input,
Windows posts a copy of each movement to our own hidden window and moves on;
nothing waits on us, nothing touches the game, and the game's own raw input is
untouched (each process registers for itself). `RIDEV_INPUTSINK` is what lets
a background window receive it.

Drained in batches (`GetRawInputBuffer`) every 10ms rather than one message at
a time: a 1000Hz mouse sends a thousand messages a second, and a Python call
per message is real CPU mid-match. Per 0.5s window it journals the distance
moved and the fastest 50ms in it, in mouse counts (DPI-dependent, so only
comparable within a session, which is how ranking uses every signal).
"""

from __future__ import annotations

import ctypes
import math
import struct
import threading
from collections import deque
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

JOURNAL_NAME = "mouse.tsv"
# One line per shot (left button down): when, how far the aim moved in the
# 150ms before it, and in the 100ms after. A flick is a big move ending in a
# shot, then the aim holding still; a box fight's spinning moves as far but
# not into a shot. Fastest-50ms alone could not tell them apart: the kept
# kill of session 5 was a flick and ranked 10th of 16 kills by it.
SHOTS_NAME = "shots.tsv"
SHOT_BEFORE = 0.15
SHOT_AFTER = 0.10

_RIDEV_INPUTSINK = 0x00000100
_RID_INPUT = 0x10000003
_RIM_TYPEMOUSE = 0
_MOUSE_MOVE_ABSOLUTE = 0x01  # tablets, remote desktop: positions, not motion
# usButtonFlags, the low half of the button union: RI_MOUSE_LEFT_BUTTON_DOWN.
_LEFT_BUTTON_DOWN = 0x0001
_WM_INPUT = 0x00FF
_PM_REMOVE = 0x0001
_HWND_MESSAGE = wintypes.HWND(-3)

# RAWINPUTHEADER on 64-bit Windows: two DWORDs, a HANDLE and a WPARAM.
_HEADER = 8 + 2 * ctypes.sizeof(ctypes.c_void_p)
# RAWMOUSE: usFlags, the button union, ulRawButtons, lLastX, lLastY, extra.
_MOUSE = struct.Struct("<H2xIIiiI")


class _RawInputDevice(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]


def batch(buffer: bytes, count: int) -> tuple[list[tuple[int, int]], int]:
    """(dx, dy) of each relative mouse record in a `GetRawInputBuffer` batch,
    and how many times the left button went down in it."""
    found, presses, at = [], 0, 0
    for _ in range(count):
        kind, size = struct.unpack_from("<II", buffer, at)
        if kind == _RIM_TYPEMOUSE:
            flags, buttons, _, dx, dy, _ = _MOUSE.unpack_from(buffer, at + _HEADER)
            if not flags & _MOUSE_MOVE_ABSOLUTE:
                found.append((dx, dy))
            presses += bool(buttons & _LEFT_BUTTON_DOWN)
        at = (at + size + 7) & ~7  # NEXTRAWINPUTBLOCK: records are 8-aligned
    return found, presses


def motions(buffer: bytes, count: int) -> list[tuple[int, int]]:
    return batch(buffer, count)[0]


@dataclass(frozen=True)
class Shot:
    at: float  # seconds from the start of the recording
    before: float  # mouse counts moved in the SHOT_BEFORE leading into it
    after: float  # and in the SHOT_AFTER following it


@dataclass(frozen=True)
class MouseSample:
    at: float  # seconds from the start of the recording, end of the window
    distance: float  # mouse counts moved in the window
    peak: float  # counts moved in its fastest 50ms: a flick


class MouseMotion:
    """Journals mouse movement per 0.5s window, from Raw Input."""

    WINDOW_SECONDS = 0.5
    BIN_SECONDS = 0.05
    DRAIN_SECONDS = 0.01

    def __init__(self, journal: Path | None = None, started: float | None = None,
                 shots: Path | None = None):
        self.journal = journal
        self.shots_journal = shots
        self.shots: list[Shot] = []
        self._recent: deque[tuple[float, float]] = deque()  # (moment, moved) per drain
        self._pending: list[list[float]] = []  # [at, before, after] still collecting
        self.started = started if started is not None else time.monotonic()
        self.samples: list[MouseSample] = []
        self.error: str | None = None
        self._window_end = self.WINDOW_SECONDS
        self._distance = 0.0
        self._bins = 1  # bins ended so far, counted rather than summed in floats
        self._bin = 0.0
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, moment: float, moved: float) -> list[MouseSample]:
        """Movement drained at `moment` (seconds since start). Returns the
        windows it closed."""
        closed = []
        per_window = round(self.WINDOW_SECONDS / self.BIN_SECONDS)
        while moment >= self._bins * self.BIN_SECONDS - 1e-9:
            self._peak = max(self._peak, self._bin)
            self._bin = 0.0
            if self._bins % per_window == 0:
                closed.append(self._close())
            self._bins += 1
        self._bin += moved
        self._distance += moved
        return closed

    def track(self, moment: float, moved: float, presses: int = 0) -> list[Shot]:
        """One drain: movement and left presses at `moment`. Returns the shots
        whose after-window it closed."""
        self.add(moment, moved)
        finished = []
        for pending in list(self._pending):
            if moment <= pending[0] + SHOT_AFTER + 1e-9:
                pending[2] += moved
            else:
                self._pending.remove(pending)
                finished.append(self._shot(*pending))
        self._recent.append((moment, moved))
        while self._recent and self._recent[0][0] <= moment - SHOT_BEFORE - 1e-9:
            self._recent.popleft()
        if presses:
            # This drain's own movement led into the press, so it counts before.
            self._pending.append([moment, sum(m for _, m in self._recent), 0.0])
        return finished

    def _shot(self, at: float, before: float, after: float) -> Shot:
        shot = Shot(round(at, 3), round(before, 1), round(after, 1))
        self.shots.append(shot)
        if self.shots_journal is not None:
            try:
                with self.shots_journal.open("a", encoding="utf-8") as handle:
                    handle.write(f"{shot.at}\t{shot.before}\t{shot.after}\n")
            except OSError:
                pass
        return shot

    @staticmethod
    def read_shots(path: Path) -> list[Shot]:
        found = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return found
        for line in lines:
            try:
                found.append(Shot(*(float(v) for v in line.split("\t"))))
            except (TypeError, ValueError):
                continue
        return found

    def _close(self) -> MouseSample:
        sample = MouseSample(round(self._window_end, 2), round(self._distance, 1),
                             round(self._peak, 1))
        self.samples.append(sample)
        if self.journal is not None:
            try:
                with self.journal.open("a", encoding="utf-8") as handle:
                    handle.write(f"{sample.at}\t{sample.distance}\t{sample.peak}\n")
            except OSError:
                pass  # telemetry must never interrupt a recording
        self._distance = self._peak = 0.0
        self._window_end = round(self._window_end + self.WINDOW_SECONDS, 6)
        return sample

    @staticmethod
    def read_journal(path: Path) -> list[MouseSample]:
        found = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return found
        for line in lines:
            try:
                at, distance, peak = (float(v) for v in line.split("\t"))
            except ValueError:
                continue
            found.append(MouseSample(at, distance, peak))
        return found

    def _loop(self) -> None:
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.CreateWindowExW.restype = wintypes.HWND
        user.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        user.GetRawInputBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.UINT),
                                           wintypes.UINT]
        user.GetRawInputBuffer.restype = wintypes.UINT
        user.PeekMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                      wintypes.UINT, wintypes.UINT]
        user.DispatchMessageW.argtypes = [ctypes.c_void_p]
        # A message-only window of a stock class: it is never shown, and its
        # window procedure (DefWindowProc) frees each raw input it is handed.
        hwnd = user.CreateWindowExW(0, "STATIC", "salience-mouse", 0, 0, 0, 0, 0,
                                    _HWND_MESSAGE, None, None, None)
        device = _RawInputDevice(0x01, 0x02, _RIDEV_INPUTSINK, hwnd)  # generic mouse
        if not hwnd or not user.RegisterRawInputDevices(
                ctypes.byref(device), 1, ctypes.sizeof(device)):
            self.error = f"raw input unavailable (error {ctypes.get_last_error()})"
            return
        buffer = ctypes.create_string_buffer(64 * 1024)
        message = ctypes.create_string_buffer(64)  # MSG
        try:
            while not self._stop.wait(self.DRAIN_SECONDS):
                moved, presses = 0.0, 0
                while True:
                    size = wintypes.UINT(len(buffer))
                    count = user.GetRawInputBuffer(buffer, ctypes.byref(size), _HEADER)
                    if count in (0, 0xFFFFFFFF):
                        break
                    moves, pressed = batch(buffer.raw, count)
                    moved += sum(math.hypot(dx, dy) for dx, dy in moves)
                    presses += pressed
                # What the batch read leaves queued as messages, cleared so the
                # queue never fills.
                while user.PeekMessageW(message, hwnd, _WM_INPUT, _WM_INPUT, _PM_REMOVE):
                    user.DispatchMessageW(message)
                self.track(time.monotonic() - self.started, moved, presses)
        finally:
            device.dwFlags, device.hwndTarget = 0x00000001, None  # RIDEV_REMOVE
            user.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device))
            user.DestroyWindow(hwnd)

    def start(self) -> "MouseMotion":
        self._thread = threading.Thread(target=self._loop, name="mouse-motion", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[MouseSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        for pending in self._pending:  # the last shot's after-window, cut short
            self._shot(*pending)
        self._pending = []
        return self.samples
