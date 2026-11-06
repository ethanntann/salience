"""The audio models on whichever device measured faster, never the game's GPU."""

from __future__ import annotations

import time

from pipeline_v2 import devices


def _model(speeds: dict, made: list, fail_on_gpu: bool = False):
    def make(providers):
        device = "gpu" if providers == devices.GPU else "cpu"
        made.append(device)
        return device

    def call(device, x):
        if device == "gpu" and fail_on_gpu:
            raise RuntimeError("no DirectML here")
        time.sleep(speeds[device])
        return f"{device}:{x}"

    return make, call


def test_the_faster_device_is_measured_once_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "gpu_available", lambda: True)
    choices = devices.Choices(tmp_path / devices.CHOICES)
    made: list = []
    make, call = _model({"cpu": 0.05, "gpu": 0.005}, made)

    first = devices.Fastest("speech", make, call, choices)
    assert first("a") == "cpu:a"  # measured on the way: this call's answer is the CPU's
    assert first("b") == "gpu:b"
    assert choices.get("speech")["device"] == "gpu"

    second = devices.Fastest("speech", make, call, choices)
    assert second("c") == "gpu:c" and made[-1] == "gpu"  # not timed again


def test_a_slower_gpu_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "gpu_available", lambda: True)
    choices = devices.Choices(tmp_path / devices.CHOICES)
    make, call = _model({"cpu": 0.005, "gpu": 0.05}, [])
    run = devices.Fastest("speech", make, call, choices)
    run("a")
    assert run("b") == "cpu:b" and choices.get("speech")["device"] == "cpu"


def test_a_match_begun_meanwhile_sends_the_rest_to_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "gpu_available", lambda: True)
    choices = devices.Choices(tmp_path / devices.CHOICES)
    choices.put("speech", {"device": "gpu"})
    playing = {"now": False}
    make, call = _model({"cpu": 0.0, "gpu": 0.0}, [])
    run = devices.Fastest("speech", make, call, choices, free=lambda: not playing["now"])
    assert run("a") == "gpu:a"
    playing["now"] = True
    assert run("b") == "cpu:b"


def test_a_gpu_that_fails_is_the_cpus_work(tmp_path, monkeypatch):
    monkeypatch.setattr(devices, "gpu_available", lambda: True)
    choices = devices.Choices(tmp_path / devices.CHOICES)
    choices.put("speech", {"device": "gpu"})
    make, call = _model({"cpu": 0.0, "gpu": 0.0}, [], fail_on_gpu=True)
    assert devices.Fastest("speech", make, call, choices)("a") == "cpu:a"


def test_without_directml_or_while_playing_it_is_never_timed(tmp_path, monkeypatch):
    made: list = []
    make, call = _model({"cpu": 0.0, "gpu": 0.0}, made)
    monkeypatch.setattr(devices, "gpu_available", lambda: False)
    assert devices.Fastest("speech", make, call, devices.Choices(tmp_path / "a.json"))("x") == "cpu:x"
    monkeypatch.setattr(devices, "gpu_available", lambda: True)
    assert devices.Fastest("speech", make, call, devices.Choices(tmp_path / "b.json"), free=lambda: False)("y") == "cpu:y"
    assert made == ["cpu", "cpu"]
