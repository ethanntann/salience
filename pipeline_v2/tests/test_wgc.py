"""The WGC backend.

Desktop Duplication failed the gate by 11x on 2026-09-16, and the damage was
stalls rather than throughput. These cover the properties chosen to avoid
repeating that: source-level throttling, a non-blocking frame pump, and output
that stays byte-compatible with the DDA segment ring.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pipeline_v2.capture import wgc


def test_the_encoder_writes_the_same_segment_shape_as_the_dda_backend():
    """Both backends must produce interchangeable chunks."""
    command = wgc.encoder_command(
        1920, 1200, 30, "8M", "h264_nvenc", 30,
        Path("part-%05d.mkv"), Path("segments.csv"),
    )
    joined = " ".join(command)

    assert "-f rawvideo" in joined and "-pixel_format bgra" in joined
    assert "-video_size 1920x1200" in joined
    assert "-i pipe:0" in joined
    assert "part-%05d.mkv" in joined
    # A keyframe every 2s (so highlight windows stream-copy), and no CBR
    # pinning -- the 2026-09-14 fix.
    assert command[command.index("-g") + 1] == "60"
    assert "expr:gte(t,n_forced*2)" in command
    assert "-maxrate" not in command and "-bufsize" not in command


def test_the_gop_follows_the_frame_rate():
    """Two seconds of frames at any rate; segments stay on the same grid."""
    command = wgc.encoder_command(
        1280, 720, 15, "4M", "h264_nvenc", 10,
        Path("p.mkv"), Path("s.csv"),
    )

    assert command[command.index("-g") + 1] == "30"
    assert command[command.index("-segment_time") + 1] == "10"


def test_capture_is_throttled_at_the_source(monkeypatch, tmp_path, isolated_data):
    """The property DDA lacked: fewer frames composited, not just encoded."""
    seen = {}

    class FakeCapture:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def event(self, handler):
            return handler

        def start_free_threaded(self):
            return object()

    monkeypatch.setitem(
        __import__("sys").modules, "windows_capture",
        type("m", (), {"WindowsCapture": FakeCapture}),
    )

    wgc.WgcRecorder(tmp_path, fps=30).start()

    # Twice the output rate: equal to it, delivery quantises below 30fps on
    # display refreshes and ~10% of frames during motion were repeats.
    assert seen["minimum_update_interval"] == 17
    assert seen["cursor_capture"] is False
    assert seen["draw_border"] is False


def test_the_pacer_writes_at_a_constant_rate_duplicating_when_idle(tmp_path):
    """WGC delivers only on change; a `-framerate N` input needs true CFR.

    Without this the video is shorter than real time while audio runs at real
    time -- measured as 877ms of drift over 12 seconds.
    """
    recorder = wgc.WgcRecorder(tmp_path, fps=50)
    written = []

    class Stdin:
        def write(self, data):
            written.append(data)

    recorder._process = type("p", (), {"stdin": Stdin()})()
    recorder._latest = b"frame"

    pacer = threading.Thread(target=recorder._pace, daemon=True)
    pacer.start()
    time.sleep(0.25)
    recorder._stop.set()
    pacer.join(timeout=2)

    # One frame was ever delivered; the pacer must have written it repeatedly.
    assert len(written) > 5
    assert set(written) == {b"frame"}
    assert recorder.stats.written == len(written)


def test_the_pacer_reports_a_broken_encoder_pipe(tmp_path):
    recorder = wgc.WgcRecorder(tmp_path, fps=50)

    class Stdin:
        def write(self, _):
            raise BrokenPipeError("encoder died")

    recorder._process = type("p", (), {"stdin": Stdin()})()
    recorder._latest = b"frame"

    pacer = threading.Thread(target=recorder._pace, daemon=True)
    pacer.start()
    pacer.join(timeout=2)

    assert recorder.error and "pipe" in recorder.error


def test_the_pacer_writes_nothing_before_the_first_frame(tmp_path):
    recorder = wgc.WgcRecorder(tmp_path, fps=50)

    class Stdin:
        def write(self, _):
            raise AssertionError("wrote before any frame arrived")

    recorder._process = type("p", (), {"stdin": Stdin()})()

    pacer = threading.Thread(target=recorder._pace, daemon=True)
    pacer.start()
    time.sleep(0.1)
    recorder._stop.set()
    pacer.join(timeout=2)

    assert recorder.stats.written == 0


def test_superseded_frames_are_reported():
    """A frame overwritten before the pacer saw it means the throttle is loose.

    `written` exceeding `delivered` is the normal case, not a fault: the pacer
    repeats the last frame to hold a constant rate when the scene is static.
    """
    stats = wgc.WgcStats(delivered=100, written=120, superseded=10)

    assert stats.superseded_rate == pytest.approx(0.1)
    assert wgc.WgcStats().superseded_rate == 0.0


def test_stop_is_safe_before_anything_started(tmp_path):
    stats = wgc.WgcRecorder(tmp_path).stop()

    assert stats.delivered == 0


def test_the_pump_state_is_thread_safe_by_construction(tmp_path):
    recorder = wgc.WgcRecorder(tmp_path)

    assert isinstance(recorder._stop, threading.Event)
    assert isinstance(recorder._closed, threading.Event)


def test_a_superseded_frame_is_counted(tmp_path):
    """Two frames between pacer ticks: the first is never encoded."""
    recorder = wgc.WgcRecorder(tmp_path)
    recorder._process = object()
    control = type("c", (), {"stop": lambda s: None})()

    class Frame:
        width, height = 8, 8
        frame_buffer = type("b", (), {"tobytes": lambda s: b"a"})()

    recorder._on_frame(Frame(), control)
    recorder._on_frame(Frame(), control)

    assert recorder.stats.delivered == 2
    assert recorder.stats.superseded == 1


def test_the_source_closing_is_observable(tmp_path):
    """A window-targeted capture learns the game exited this way."""
    recorder = wgc.WgcRecorder(tmp_path)

    assert recorder.source_closed is False
    recorder._closed.set()
    assert recorder.source_closed is True


def test_the_callback_does_no_io_so_it_cannot_stall_the_compositor(tmp_path):
    """The pacer writes; the callback only swaps in the newest frame."""
    recorder = wgc.WgcRecorder(tmp_path)
    recorder._process = object()          # a write would explode if attempted

    class Frame:
        width, height = 8, 8
        frame_buffer = type("b", (), {"tobytes": lambda s: b"pixels"})()

    recorder._on_frame(Frame(), type("c", (), {"stop": lambda s: None})())

    assert recorder.stats.delivered == 1
    assert recorder._latest == b"pixels"
    assert recorder.stats.written == 0    # writing is the pacer's job


def test_a_failure_to_start_the_encoder_is_reported_not_swallowed(tmp_path):
    """It used to vanish into the native callback and capture just stopped."""
    recorder = wgc.WgcRecorder(tmp_path, audio=True)
    recorder._audio_devices = []

    def boom(*_):
        raise RuntimeError("Error 0x100000001")

    recorder._start_encoder = boom

    class Frame:
        width, height = 8, 8
        frame_buffer = type("b", (), {"tobytes": lambda s: b""})()

    stopped = []
    recorder._on_frame(Frame(), type("c", (), {"stop": lambda s: stopped.append(1)})())

    assert recorder.error and "could not start encoder" in recorder.error
    assert stopped == [1]


def test_audio_devices_are_resolved_before_capture_starts(monkeypatch, tmp_path, isolated_data):
    """soundcard fails with 0x100000001 on WGC's native callback thread."""
    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def event(self, handler):
            return handler

        def start_free_threaded(self):
            return object()

    monkeypatch.setitem(
        __import__("sys").modules, "windows_capture",
        type("m", (), {"WindowsCapture": FakeCapture}),
    )
    monkeypatch.setattr(
        "pipeline_v2.capture.audio_pump.loopback_device", lambda *a, **k: "game-dev"
    )
    monkeypatch.setattr(
        "pipeline_v2.capture.audio_pump.microphone_device", lambda *a, **k: "mic-dev"
    )

    recorder = wgc.WgcRecorder(tmp_path, audio=True).start()

    assert [name for name, _ in recorder._audio_devices] == ["game", "mic"]
    assert [d for _, d in recorder._audio_devices] == ["game-dev", "mic-dev"]


def test_a_second_session_in_one_process_is_refused_not_crashed(tmp_path, monkeypatch):
    """A second WGC session segfaults once audio is involved.

    The crash has no traceback, which is the worst failure a shipped recorder
    can have, so it is converted into an error naming the fix: one session per
    process.
    """
    monkeypatch.setattr(wgc, "_SESSION_STARTED", True)

    with pytest.raises(wgc.WgcSessionReused, match="own process"):
        wgc.WgcRecorder(tmp_path).start()


def test_the_first_session_is_allowed(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def event(self, handler):
            return handler

        def start_free_threaded(self):
            return object()

    monkeypatch.setattr(wgc, "_SESSION_STARTED", False)
    monkeypatch.setitem(
        __import__("sys").modules, "windows_capture",
        type("m", (), {"WindowsCapture": FakeCapture}),
    )

    recorder = wgc.WgcRecorder(tmp_path).start()

    assert recorder._control is not None
