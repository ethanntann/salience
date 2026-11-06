"""Which output device the game is playing through, from Windows' audio sessions.

Game audio is the loopback of an output device, and it followed Windows'
default. A player who picks their headphones in Fortnite's own settings
(2026-10-24) left the default on the speakers, and the track was silent:
"game audio not detected". Windows keeps an audio session per process per
device; the device where the game's session is playing is the one to record.

Read-only Core Audio (`IAudioSessionManager2`), through ctypes: nothing is
loaded into the game, nothing is hooked.
"""

from __future__ import annotations

import ctypes as ct
import ctypes.wintypes as wt
import sys
import uuid
from pathlib import Path

from pipeline_v2.capture.windows import GAME_EXE

ACTIVE = 1  # AudioSessionStateActive
E_RENDER, DEVICE_STATE_ACTIVE, CLSCTX_ALL = 0, 1, 23


def _guid(text: str):
    class GUID(ct.Structure):
        _fields_ = [("bytes", ct.c_ubyte * 16)]
    g = GUID()
    ct.memmove(g.bytes, uuid.UUID(text).bytes_le, 16)
    return g


CLSID_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E"
IID_ENUMERATOR = "A95664D2-9614-4F35-A746-DE8DB63617E6"
IID_SESSION_MANAGER2 = "77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"
IID_SESSION_CONTROL2 = "BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D"


def _call(obj: ct.c_void_p, index: int, *args, types=()):
    """Method `index` of a COM object's vtable, returning its HRESULT."""
    vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p)))[0]
    fn = ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *types)(vtable[index])
    return fn(obj, *args)


def _release(obj: ct.c_void_p) -> None:
    if obj:
        vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p)))[0]
        ct.WINFUNCTYPE(ct.c_ulong, ct.c_void_p)(vtable[2])(obj)


def _exe(pid: int) -> str:
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)]
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    process = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return ""
    try:
        size = wt.DWORD(32768)
        name = ct.create_unicode_buffer(size.value)
        ok = kernel.QueryFullProcessImageNameW(process, 0, name, ct.byref(size))
        return Path(name.value).name.lower() if ok else ""
    finally:
        kernel.CloseHandle(process)


def sessions() -> list[tuple[str, int, int]]:
    """(device id, process id, state) for every audio session on every output device."""
    if sys.platform != "win32":
        return []
    ole = ct.OleDLL("ole32")
    ole.CoInitializeEx.argtypes = [ct.c_void_p, wt.DWORD]
    try:
        ole.CoInitializeEx(None, 0)  # multithreaded; already initialised is fine
    except OSError:
        pass
    found: list[tuple[str, int, int]] = []
    enumerator, devices = ct.c_void_p(), ct.c_void_p()
    clsid, iid = _guid(CLSID_ENUMERATOR), _guid(IID_ENUMERATOR)
    try:
        ole.CoCreateInstance(ct.byref(clsid), None, CLSCTX_ALL, ct.byref(iid), ct.byref(enumerator))
    except OSError:
        return []
    try:
        if _call(enumerator, 3, E_RENDER, DEVICE_STATE_ACTIVE, ct.byref(devices),
                 types=(ct.c_int, wt.DWORD, ct.c_void_p)) != 0:
            return []
        count = wt.UINT()
        _call(devices, 3, ct.byref(count), types=(ct.c_void_p,))
        for i in range(count.value):
            device = ct.c_void_p()
            if _call(devices, 4, i, ct.byref(device), types=(wt.UINT, ct.c_void_p)) != 0:
                continue
            try:
                found += _device_sessions(device, ole)
            finally:
                _release(device)
    finally:
        _release(devices)
        _release(enumerator)
    return found


def _device_sessions(device: ct.c_void_p, ole) -> list[tuple[str, int, int]]:
    out = []
    raw_id = ct.c_wchar_p()
    if _call(device, 5, ct.byref(raw_id), types=(ct.c_void_p,)) != 0:  # GetId
        return out
    device_id = raw_id.value or ""
    ole.CoTaskMemFree.argtypes = [ct.c_void_p]
    ole.CoTaskMemFree(raw_id)
    manager, listing = ct.c_void_p(), ct.c_void_p()
    iid = _guid(IID_SESSION_MANAGER2)
    if _call(device, 3, ct.byref(iid), CLSCTX_ALL, None, ct.byref(manager),  # Activate
             types=(ct.c_void_p, wt.DWORD, ct.c_void_p, ct.c_void_p)) != 0:
        return out
    try:
        if _call(manager, 5, ct.byref(listing), types=(ct.c_void_p,)) != 0:  # GetSessionEnumerator
            return out
        count = ct.c_int()
        _call(listing, 3, ct.byref(count), types=(ct.c_void_p,))
        control2_iid = _guid(IID_SESSION_CONTROL2)
        for i in range(count.value):
            control, control2 = ct.c_void_p(), ct.c_void_p()
            if _call(listing, 4, i, ct.byref(control), types=(ct.c_int, ct.c_void_p)) != 0:
                continue
            try:
                if _call(control, 0, ct.byref(control2_iid), ct.byref(control2),  # QueryInterface
                         types=(ct.c_void_p, ct.c_void_p)) != 0:
                    continue
                state, pid = ct.c_int(), wt.DWORD()
                _call(control2, 3, ct.byref(state), types=(ct.c_void_p,))  # GetState
                _call(control2, 14, ct.byref(pid), types=(ct.c_void_p,))  # GetProcessId
                out.append((device_id, pid.value, state.value))
            finally:
                _release(control2)
                _release(control)
    finally:
        _release(listing)
        _release(manager)
    return out


def game_device(exe: str = GAME_EXE, listing=sessions, name_of=_exe) -> str | None:
    """The id of the output device the game is playing through, or None.

    Playing beats merely open (a game opens sessions on devices it never uses);
    None when the game has no session at all, so the caller keeps the default.
    """
    try:
        mine = [(device, state) for device, pid, state in listing() if pid and name_of(pid) == exe]
    except OSError:
        return None
    playing = [device for device, state in mine if state == ACTIVE]
    if playing:
        return playing[0]
    return mine[0][0] if len(mine) == 1 else None
