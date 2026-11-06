"""Which device runs the audio models after a session: the CPU or the GPU, by
measurement, never by assumption.

Speech and laughter were the whole wait once describing stopped waiting on
them. On an RTX 4060 laptop (DirectML, 2026-11-03), 5 minutes of mic: speech
17.9s on the CPU and 4.0s on the GPU with the same int8 model, laughter 3.2s
and 1.1s a track. But a fast CPU beside a weak GPU is as common as the
reverse, so each model is timed once on each device, on the player's own
audio, and the faster kept for that hardware (`CHOICES`, beside the app).

The GPU is the game's. It is used only while `free()` says nothing is
playing; a match started meanwhile sends the rest to the CPU.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

GPU = ["DmlExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]
CHOICES = "devices.json"
# The GPU is kept only when clearly faster: a tie is the CPU, which never
# touches what the game draws with.
MARGIN = 0.8


def gpu_available() -> bool:
    try:
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


def _hardware() -> str:
    """A fingerprint of this machine's CPU and display adapters: a new GPU is
    timed again. Only the fingerprint is kept: knowing the hardware changed
    needs no record of what it is."""
    import hashlib

    return hashlib.sha256(_hardware_names().encode("utf-8")).hexdigest()[:16]


def _hardware_names() -> str:
    names = [platform.processor()]
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Device(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                        ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                        ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

        index, seen = 0, set()
        while True:
            device = Device(cb=ctypes.sizeof(Device))
            if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                break
            seen.add(device.DeviceString)
            index += 1
        names += sorted(seen)
    return " | ".join(names)


class Choices:
    """What was measured, per hardware and model, kept in `path`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, kind: str) -> dict | None:
        return self._read().get(_hardware(), {}).get(kind)

    def put(self, kind: str, record: dict) -> None:
        everything = self._read()
        everything.setdefault(_hardware(), {})[kind] = record
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(everything, indent=1), encoding="utf-8")
        except OSError:
            pass


class Fastest:
    """A model on whichever device measured faster, behind `call`.

    `make(providers)` builds the model; `call(model, *args)` runs it. The
    first call on a machine without a choice times `probe(model)`, a short
    fixed piece of work, on each device (the GPU's second, once its graph is
    built) and keeps the faster. Timed on the call itself, the first 10
    minutes of a session ran three times: 45s of speech for 15 minutes.
    """

    def __init__(self, kind: str, make, call, choices: Choices | None, free=lambda: True,
                 probe=None) -> None:
        self.kind, self.make, self.call, self.choices, self.free = kind, make, call, choices, free
        self.probe = probe
        self._models: dict[str, object] = {}
        self.device: str | None = None

    def _model(self, device: str):
        if device not in self._models:
            self._models[device] = self.make(GPU if device == "gpu" else CPU)
        return self._models[device]

    def _timed(self, device: str, args) -> tuple[float, object]:
        model = self._model(device)
        started = time.perf_counter()
        out = self.probe(model) if self.probe else self.call(model, *args)
        return time.perf_counter() - started, (None if self.probe else out)

    def _choose(self, args):
        """(device, the result of this call), timing both if there is no choice yet."""
        known = self.choices.get(self.kind) if self.choices else None
        if known is not None:
            return known["device"], None
        cpu_s, out = self._timed("cpu", args)
        try:
            self._timed("gpu", args)  # builds the GPU's graph; not the time
            gpu_s, _ = self._timed("gpu", args)
        except Exception:  # noqa: BLE001 -- a GPU that will not run this is no choice
            gpu_s = None
        device = "gpu" if gpu_s is not None and gpu_s < cpu_s * MARGIN else "cpu"
        if self.choices:
            self.choices.put(self.kind, {"device": device, "cpu_s": round(cpu_s, 3),
                                         "gpu_s": None if gpu_s is None else round(gpu_s, 3)})
        if device == "cpu":
            self._models.pop("gpu", None)  # its memory back
        return device, out

    def __call__(self, *args):
        if self.device is None:
            if gpu_available() and self.free():
                self.device, out = self._choose(args)
                if out is not None:
                    return out
            else:
                self.device = "cpu"
        if self.device == "gpu" and not self.free():
            self.device = "cpu"  # a match began: the GPU is the game's now
            self._models.pop("gpu", None)
        try:
            return self.call(self._model(self.device), *args)
        except Exception:  # noqa: BLE001 -- the GPU failing is the CPU's work
            if self.device != "gpu":
                raise
            self.device = "cpu"
            self._models.pop("gpu", None)
            return self.call(self._model("cpu"), *args)
