"""Everything the PC plays except one program: the game left out of the
friends' audio.

The PC's audio track is everything the PC plays, so transcribing it to hear
friends also heard Fortnite: its voice lines and Jam Tracks came back as
"what friends said" (2026-11-05). Windows can capture a process's audio, or
everything but a process's (process loopback, Windows 10 2004 and later):
excluding the game's process tree leaves Discord, party chat or any voice
app, whichever the player uses, with nothing to configure.

Plain ctypes against the Windows audio API (ActivateAudioInterfaceAsync,
IAudioClient, IAudioCaptureClient): soundcard, which records the other
tracks, has no process loopback. Nothing here touches the game's process:
the audio engine is asked for a mix that leaves it out.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes

SAMPLE_RATE = 48000
CHANNELS = 2

_S_OK = 0
_VT_BLOB = 65
_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
INCLUDE, EXCLUDE = 0, 1  # PROCESS_LOOPBACK_MODE: the target's tree, or all but it
_SHARED = 0
_LOOPBACK = 0x00020000
_AUTOCONVERT = 0x80000000 | 0x08000000  # AUTOCONVERTPCM | SRC_DEFAULT_QUALITY
_SILENT = 0x2  # AUDCLNT_BUFFERFLAGS_SILENT
_VIRTUAL_DEVICE = "VAD\\Process_Loopback"


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "_GUID":
        h = text.strip("{}").replace("-", "")
        return cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16),
                   (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:32])))

    def __eq__(self, other) -> bool:
        return bytes(self) == bytes(other)


IID_IUnknown = _GUID.parse("00000000-0000-0000-C000-000000000046")
IID_IAgileObject = _GUID.parse("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")
IID_ICompletionHandler = _GUID.parse("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_IAudioClient = _GUID.parse("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2")
IID_IAudioCaptureClient = _GUID.parse("C8ADBD64-E71E-48a0-A4DE-185C395CD317")


class _LoopbackParams(ctypes.Structure):
    _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", ctypes.c_int)]


class _ActivationParams(ctypes.Structure):
    _fields_ = [("ActivationType", ctypes.c_int), ("ProcessLoopbackParams", _LoopbackParams)]


class _Blob(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", ctypes.c_void_p)]


class _PropVariant(ctypes.Structure):
    _fields_ = [("vt", wintypes.USHORT), ("r1", wintypes.WORD), ("r2", wintypes.WORD), ("r3", wintypes.WORD),
                ("blob", _Blob), ("pad", ctypes.c_void_p)]


class _WaveFormat(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


def _method(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    """Method `index` of a COM object's vtable, bound to it."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    function = prototype(vtable[index])
    return lambda *args: function(obj, *args)


def _release(obj) -> None:
    if obj:
        _method(obj, 2, wintypes.ULONG)()


class _Handler:
    """IActivateAudioInterfaceCompletionHandler, agile, as a COM object of
    our own: Windows calls ActivateCompleted on a thread of its own."""

    def __init__(self) -> None:
        self.done = threading.Event()
        HRESULT = ctypes.c_long
        self._qi = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.POINTER(_GUID),
                                      ctypes.POINTER(ctypes.c_void_p))(self._query)
        self._addref = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(lambda this: 1)
        self._release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(lambda this: 1)
        self._completed = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)(self._activated)
        self._vtable = (ctypes.c_void_p * 4)(*(ctypes.cast(f, ctypes.c_void_p) for f in
                                               (self._qi, self._addref, self._release, self._completed)))
        self._object = ctypes.pointer(ctypes.cast(self._vtable, ctypes.c_void_p))
        self.pointer = ctypes.cast(self._object, ctypes.c_void_p)

    def _query(self, this, iid, out):
        if iid.contents in (IID_IUnknown, IID_IAgileObject, IID_ICompletionHandler):
            out[0] = this
            return _S_OK
        out[0] = None
        return ctypes.c_long(0x80004002).value  # E_NOINTERFACE

    def _activated(self, this, operation):
        self.done.set()
        return _S_OK


def supported() -> bool:
    """Whether this Windows has process loopback (10, version 2004 and later)."""
    if os.name != "nt":
        return False
    try:
        ctypes.WinDLL("mmdevapi").ActivateAudioInterfaceAsync  # noqa: B018
    except (OSError, AttributeError):
        return False
    build = getattr(__import__("sys").getwindowsversion(), "build", 0)
    return build >= 19041


class Recorder:
    """Float32 stereo at 48kHz of everything but (or only) `pid`'s process tree,
    read like soundcard's recorders: `record(numframes)`."""

    def __init__(self, pid: int, mode: int = EXCLUDE, buffer_seconds: float = 0.5) -> None:
        import numpy as np

        self._np = np
        ole = ctypes.WinDLL("ole32")
        ole.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED, this thread's
        params = _ActivationParams(_ACTIVATION_TYPE_PROCESS_LOOPBACK, _LoopbackParams(pid, mode))
        variant = _PropVariant(vt=_VT_BLOB)
        variant.blob = _Blob(ctypes.sizeof(params), ctypes.cast(ctypes.pointer(params), ctypes.c_void_p))
        handler = _Handler()
        operation = ctypes.c_void_p()
        mmdev = ctypes.WinDLL("mmdevapi")
        mmdev.ActivateAudioInterfaceAsync.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(_GUID),
                                                      ctypes.POINTER(_PropVariant), ctypes.c_void_p,
                                                      ctypes.POINTER(ctypes.c_void_p)]
        hr = mmdev.ActivateAudioInterfaceAsync(_VIRTUAL_DEVICE, ctypes.byref(IID_IAudioClient),
                                               ctypes.byref(variant), handler.pointer, ctypes.byref(operation))
        if hr != _S_OK or not handler.done.wait(5):
            raise OSError(f"process loopback not activated ({hr & 0xFFFFFFFF:#x})")
        result, client = ctypes.c_long(), ctypes.c_void_p()
        _method(operation, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(result), ctypes.byref(client))
        _release(operation)
        if result.value != _S_OK or not client:
            raise OSError(f"process loopback refused ({result.value & 0xFFFFFFFF:#x})")
        self._client = client
        self._handler = handler  # alive while Windows may still hold it
        fmt = _WaveFormat(3, CHANNELS, SAMPLE_RATE, SAMPLE_RATE * CHANNELS * 4, CHANNELS * 4, 32, 0)
        initialize = _method(client, 3, ctypes.c_long, ctypes.c_int, wintypes.DWORD, ctypes.c_longlong,
                             ctypes.c_longlong, ctypes.POINTER(_WaveFormat), ctypes.c_void_p)
        hr = initialize(_SHARED, _LOOPBACK | _AUTOCONVERT, int(buffer_seconds * 1e7), 0, ctypes.byref(fmt), None)
        if hr != _S_OK:
            _release(client)
            raise OSError(f"process loopback not initialised ({hr & 0xFFFFFFFF:#x})")
        capture = ctypes.c_void_p()
        hr = _method(client, 14, ctypes.c_long, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))(
            ctypes.byref(IID_IAudioCaptureClient), ctypes.byref(capture))
        if hr != _S_OK:
            _release(client)
            raise OSError(f"no capture client ({hr & 0xFFFFFFFF:#x})")
        self._capture = capture
        self._get = _method(capture, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT),
                            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, ctypes.c_void_p)
        self._put = _method(capture, 4, ctypes.c_long, wintypes.UINT)
        self._next = _method(capture, 5, ctypes.c_long, ctypes.POINTER(wintypes.UINT))
        self._pending = np.zeros((0, CHANNELS), np.float32)
        self._last = time.monotonic()
        _method(client, 10, ctypes.c_long)()  # Start

    def _drain(self) -> None:
        np = self._np
        got = []
        size = wintypes.UINT()
        while self._next(ctypes.byref(size)) == _S_OK and size.value:
            data, frames, flags = ctypes.c_void_p(), wintypes.UINT(), wintypes.DWORD()
            if self._get(ctypes.byref(data), ctypes.byref(frames), ctypes.byref(flags), None, None) != _S_OK:
                break
            n = frames.value
            if flags.value & _SILENT or not data.value:
                got.append(np.zeros((n, CHANNELS), np.float32))
            else:
                raw = ctypes.string_at(data.value, n * CHANNELS * 4)
                got.append(np.frombuffer(raw, np.float32).reshape(n, CHANNELS).copy())
            self._put(n)
        if got:
            self._pending = np.concatenate([self._pending, *got])
            self._last = time.monotonic()

    def record(self, numframes: int):
        """`numframes` of audio, waiting for them; silence where nothing
        played (a process loopback delivers nothing then, not zeros)."""
        np = self._np
        deadline = time.monotonic() + 2 * numframes / SAMPLE_RATE
        while len(self._pending) < numframes:
            self._drain()
            if len(self._pending) >= numframes:
                break
            idle = time.monotonic() - self._last
            if idle >= numframes / SAMPLE_RATE:
                # Nothing played for a block: as long a silence as went by.
                self._pending = np.concatenate([self._pending, np.zeros((int(idle * SAMPLE_RATE), CHANNELS), np.float32)])
                self._last = time.monotonic()
                break
            if time.monotonic() > deadline:
                break
            time.sleep(0.005)
        block, self._pending = self._pending[:numframes], self._pending[numframes:]
        return block

    def close(self) -> None:
        if self._client:
            _method(self._client, 11, ctypes.c_long)()  # Stop
            _release(self._capture)
            _release(self._client)
            self._client = self._capture = None

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def window_pid(hwnd: int) -> int | None:
    """The process that owns a window: the game's, from the window recorded."""
    if os.name != "nt" or not hwnd:
        return None
    user = ctypes.WinDLL("user32")
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    pid = wintypes.DWORD()
    user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value or None


class Device:
    """A stand-in for a soundcard device: everything the PC plays but `pid`'s
    process tree, for `audio_pump.TrackPump`."""

    def __init__(self, pid: int, mode: int = EXCLUDE) -> None:
        self.pid, self.mode = pid, mode
        self.name = f"everything but process {pid}" if mode == EXCLUDE else f"process {pid}"
        self.id = f"process-loopback:{mode}:{pid}"  # one device to `audio_pump._same`

    def recorder(self, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=None):
        if samplerate != SAMPLE_RATE or channels != CHANNELS:
            raise ValueError("process loopback records 48kHz stereo")
        return Recorder(self.pid, self.mode)
