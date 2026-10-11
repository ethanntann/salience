from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import os
import shutil
import subprocess
import sys

import pytest

from pipeline_v2.capture import daemon, recorder


def test_only_active_segment_is_never_saved(tmp_path):
    (tmp_path / "seg000.mp4").write_bytes(b"unfinished")
    assert recorder.SegmentRecorder(tmp_path).save(tmp_path / "out.mp4") is None


def test_failed_start_cleans_session_and_handles(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(recorder.subprocess, "Popen", fail)
    rec = recorder.SegmentRecorder(tmp_path)
    with pytest.raises(FileNotFoundError):
        rec.start()
    assert rec._stderr is None
    assert rec._session is None
    assert list(tmp_path.iterdir()) == []


def test_new_session_ignores_previous_ring_and_cleans_up(tmp_path, monkeypatch):
    stale = tmp_path / "seg000.mp4"
    stale.write_bytes(b"old session")
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(recorder.subprocess, "Popen", lambda *a, **k: process)
    rec = recorder.SegmentRecorder(tmp_path).start()
    assert rec._ordered_segments() == []
    assert rec.buffer_seconds == 25
    with pytest.raises(RuntimeError):
        rec.start()
    rec.stop()
    process.terminate.assert_called_once()
    assert stale.exists()


def test_preflight_failure_stops_recorder(isolated_data, monkeypatch):
    rec = Mock()
    rec.running = True
    speaker = SimpleNamespace(name="fake")
    monkeypatch.setitem(
        sys.modules, "soundcard", SimpleNamespace(get_microphone=lambda *a, **k: None)
    )
    monkeypatch.setattr(daemon, "pick_speaker", lambda _: speaker)
    monkeypatch.setattr(daemon, "SegmentRecorder", lambda *a: rec)
    monkeypatch.setattr(daemon.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon, "make_mark_poller", lambda _: lambda: False)
    monkeypatch.setattr(daemon, "preflight", lambda _: False)
    assert daemon.run(isolated_data / "out", False, 9.5, 1) == 1
    rec.stop.assert_called_once()


def test_copy_race_discards_snapshot_and_cleans_staging(tmp_path, monkeypatch):
    for i in range(2):
        p = tmp_path / f"seg{i:03d}.mp4"
        p.write_bytes(b"segment")
        os.utime(p, (100 + i, 100 + i))
    original = shutil.copy2

    def racing_copy(source, target):
        original(source, target)
        source.write_bytes(b"overwritten while copying")

    monkeypatch.setattr(recorder.shutil, "copy2", racing_copy)
    rec = recorder.SegmentRecorder(tmp_path)
    assert rec.save(tmp_path / "out.mp4") is None
    assert not list(tmp_path.glob("save-*"))


def test_live_trigger_waits_for_postroll_and_saves_off_meter_thread(
    isolated_data, monkeypatch
):
    import threading

    clock = [0.0]
    observed = []
    rec = Mock()
    rec.running = True
    rec.config = SimpleNamespace(segment_seconds=5)
    rec.save.side_effect = lambda path: (
        observed.append((clock[0], threading.current_thread().name)) or path
    )
    mic = MagicMock()
    mic.__enter__.return_value = mic

    def sample(**kwargs):
        clock[0] += 1.0
        return None

    mic.record.side_effect = sample
    loopback = SimpleNamespace(recorder=lambda **kwargs: mic)
    monkeypatch.setitem(
        sys.modules,
        "soundcard",
        SimpleNamespace(get_microphone=lambda *a, **k: loopback),
    )
    monkeypatch.setattr(daemon, "pick_speaker", lambda _: SimpleNamespace(name="fake"))
    monkeypatch.setattr(daemon, "SegmentRecorder", lambda *a: rec)
    monkeypatch.setattr(daemon.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(daemon.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon, "_db", lambda _: -10.0)

    class Trigger:
        def __init__(self, config):
            self.fired = False

        def observe(self, value, now):
            fire = not self.fired
            self.fired = True
            return fire

    monkeypatch.setattr(daemon, "LevelTrigger", Trigger)
    monkeypatch.setattr(daemon, "SustainedTrigger", Trigger)
    assert daemon.run(isolated_data / "out", False, 9.5, 20, mark_keys="") == 0
    assert len(observed) == 1
    assert observed[0][0] >= 9.0  # trigger at 1s + 3s post-roll + 5s closure
    assert observed[0][1] != threading.current_thread().name
    assert mic.record.call_count == 20
    rec.stop.assert_called_once()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_closed_segment_snapshot_is_playable_and_does_not_overwrite(tmp_path):
    # Synthetic source only: no screen or audio device is accessed.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10",
            "-t",
            "1",
            "-c:v",
            "mpeg4",
            str(tmp_path / "seg000.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    shutil.copy2(tmp_path / "seg000.mp4", tmp_path / "seg001.mp4")
    os.utime(tmp_path / "seg000.mp4", (100, 100))
    os.utime(tmp_path / "seg001.mp4", (101, 101))
    rec = recorder.SegmentRecorder(tmp_path)
    output = tmp_path / "saved.mp4"
    assert rec.save(output) == output
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert float(probe.stdout.strip()) == pytest.approx(1.0, abs=0.1)
    with pytest.raises(FileExistsError):
        rec.save(output)
    assert not list(tmp_path.glob("save-*"))
