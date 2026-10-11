"""Read foreground process and DXGI display identity without injection/hooks.

Every function here is Windows-only, but ``Display`` is a plain record that the
collector and the frozen studies pass around. ``ctypes.wintypes`` raises on
import off Windows, so it is bound lazily: importing this module must stay safe
everywhere, and only *calling* a capture function requires Windows.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
import sys
import uuid

if sys.platform == "win32":
    from ctypes import wintypes as wt
else:  # pragma: no cover - import guard exercised only off Windows
    wt = None


@dataclass(frozen=True)
class Display:
    adapter: int
    output: int
    name: str
    left: int
    top: int
    right: int
    bottom: int
    # PCI vendor of the adapter driving this output. Quick Sync only exists on
    # Intel, and on a laptop whose panel is wired to the dGPU there is no Intel
    # device to derive one from -- `hwmap=derive_device=qsv` fails with
    # "Error creating a MFX session: -9" and the run produces nothing. Knowing
    # the vendor lets the collector skip an encoder that cannot work here
    # instead of rediscovering it every session.
    vendor_id: int = 0

    @property
    def vendor(self) -> str:
        return {0x10DE: "nvidia", 0x8086: "intel", 0x1002: "amd"}.get(
            self.vendor_id, "unknown"
        )

    def supports(self, encoder: str) -> bool:
        """Can this adapter plausibly run the encoder at all?"""
        if self.vendor == "unknown":
            return True  # not enumerated; let the probe decide
        if encoder == "h264_qsv":
            return self.vendor == "intel"
        if encoder == "h264_nvenc":
            return self.vendor == "nvidia"
        return True


GAME_EXE = "fortniteclient-win64-shipping.exe"


def _require_windows() -> None:
    if wt is None:
        raise RuntimeError("Desktop capture requires Windows")


def game_running() -> bool:
    """Toolhelp process enumeration; does not attach to the game."""
    _require_windows()

    class ProcessEntry(ct.Structure):
        _fields_ = [
            ("size", wt.DWORD),
            ("usage", wt.DWORD),
            ("pid", wt.DWORD),
            ("heap", ct.c_size_t),
            ("module", wt.DWORD),
            ("threads", wt.DWORD),
            ("parent", wt.DWORD),
            ("priority", wt.LONG),
            ("flags", wt.DWORD),
            ("exe", wt.WCHAR * 260),
        ]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel.Process32FirstW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ct.c_void_p(-1).value:
        # Unknown must not falsely conclude that the game exited.
        return True
    try:
        entry = ProcessEntry(size=ct.sizeof(ProcessEntry))
        more = kernel.Process32FirstW(snapshot, ct.byref(entry))
        while more:
            if entry.exe.lower() == GAME_EXE:
                return True
            more = kernel.Process32NextW(snapshot, ct.byref(entry))
        return False
    finally:
        kernel.CloseHandle(snapshot)


def displays() -> list[Display]:
    """Enumerate the same adapter/output indices FFmpeg ddagrab uses."""
    _require_windows()

    class GUID(ct.Structure):
        _fields_ = [
            ("a", ct.c_ulong),
            ("b", ct.c_ushort),
            ("c", ct.c_ushort),
            ("d", ct.c_ubyte * 8),
        ]

    class AdapterDesc(ct.Structure):
        _fields_ = [
            ("description", wt.WCHAR * 128),
            ("vendor_id", ct.c_uint),
            ("device_id", ct.c_uint),
            ("subsys_id", ct.c_uint),
            ("revision", ct.c_uint),
            ("dedicated_video_memory", ct.c_size_t),
            ("dedicated_system_memory", ct.c_size_t),
            ("shared_system_memory", ct.c_size_t),
            ("adapter_luid", ct.c_longlong),
        ]

    class OutputDesc(ct.Structure):
        _fields_ = [
            ("name", wt.WCHAR * 32),
            ("rect", wt.RECT),
            ("attached", wt.BOOL),
            ("rotation", ct.c_uint),
            ("monitor", wt.HMONITOR),
        ]

    def call(obj, index, *types):
        address = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents[index]
        return ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *types)(address)

    iid = GUID.from_buffer_copy(
        uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le
    )
    factory = ct.c_void_p()
    create = ct.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ct.POINTER(GUID), ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_long
    if create(ct.byref(iid), ct.byref(factory)) < 0:
        raise OSError("DXGI display enumeration failed")
    found = []
    try:
        for adapter_index in range(16):
            adapter = ct.c_void_p()
            if (
                call(factory, 12, ct.c_uint, ct.POINTER(ct.c_void_p))(
                    factory, adapter_index, ct.byref(adapter)
                )
                < 0
            ):
                break
            adapter_desc = AdapterDesc()
            vendor_id = (
                adapter_desc.vendor_id
                if call(adapter, 8, ct.POINTER(AdapterDesc))(
                    adapter, ct.byref(adapter_desc)
                )
                >= 0
                else 0
            )
            try:
                for output_index in range(16):
                    output = ct.c_void_p()
                    if (
                        call(adapter, 7, ct.c_uint, ct.POINTER(ct.c_void_p))(
                            adapter, output_index, ct.byref(output)
                        )
                        < 0
                    ):
                        break
                    try:
                        desc = OutputDesc()
                        if (
                            call(output, 7, ct.POINTER(OutputDesc))(
                                output, ct.byref(desc)
                            )
                            >= 0
                            and desc.attached
                        ):
                            r = desc.rect
                            found.append(
                                Display(
                                    adapter_index,
                                    output_index,
                                    desc.name,
                                    r.left,
                                    r.top,
                                    r.right,
                                    r.bottom,
                                    vendor_id,
                                )
                            )
                    finally:
                        call(output, 2)(output)
            finally:
                call(adapter, 2)(adapter)
    finally:
        call(factory, 2)(factory)
    return found


def window_is_game(hwnd: int) -> bool:
    """Is this window owned by the game's executable?

    Matched on the executable, never the title, so a browser tab or chat
    message naming the game is never mistaken for it. Shared by the foreground
    check and the WGC window lookup, which each had their own copy of this
    ctypes dance.
    """
    _require_windows()
    user = ct.WinDLL("user32", use_last_error=True)
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    user.GetWindowThreadProcessId.argtypes = [wt.HWND, ct.POINTER(wt.DWORD)]
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [
        wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)
    ]
    kernel.CloseHandle.argtypes = [wt.HANDLE]

    pid = wt.DWORD()
    user.GetWindowThreadProcessId(hwnd, ct.byref(pid))
    process = kernel.OpenProcess(0x1000, False, pid)
    if not process:
        return False
    try:
        size = wt.DWORD(32768)
        name = ct.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(process, 0, name, ct.byref(size)):
            return False
        return Path(name.value).name.lower() == GAME_EXE
    finally:
        kernel.CloseHandle(process)


def game_window() -> int | None:
    """HWND of the game's first visible top-level window, or None."""
    _require_windows()
    user = ct.WinDLL("user32", use_last_error=True)
    found: list[int] = []

    def visit(hwnd, _):
        if user.IsWindowVisible(hwnd) and window_is_game(hwnd):
            found.append(hwnd)
            return False
        return True

    prototype = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    user.EnumWindows.argtypes = [prototype, wt.LPARAM]
    user.EnumWindows(prototype(visit), 0)
    return found[0] if found else None


def foreground_game() -> Display | None:
    """Only recognize the actual Fortnite executable, never a title match."""
    _require_windows()
    user = ct.WinDLL("user32", use_last_error=True)
    user.GetForegroundWindow.restype = wt.HWND
    hwnd = user.GetForegroundWindow()
    if not hwnd or not window_is_game(hwnd):
        return None

    class MonitorInfo(ct.Structure):
        _fields_ = [
            ("size", wt.DWORD),
            ("monitor", wt.RECT),
            ("work", wt.RECT),
            ("flags", wt.DWORD),
            ("device", wt.WCHAR * 32),
        ]

    user.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
    user.MonitorFromWindow.restype = wt.HMONITOR
    user.GetMonitorInfoW.argtypes = [wt.HMONITOR, ct.POINTER(MonitorInfo)]
    info = MonitorInfo(size=ct.sizeof(MonitorInfo))
    if not user.GetMonitorInfoW(user.MonitorFromWindow(hwnd, 2), ct.byref(info)):
        return None
    return next((d for d in displays() if d.name == info.device), None)
