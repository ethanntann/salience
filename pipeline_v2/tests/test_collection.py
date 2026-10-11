import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from pipeline_v2 import cli, facts, prompt
from pipeline_v2.capture import collection
from pipeline_v2.capture.collection import CaptureRun, CollectionConfig, Collector
from pipeline_v2.capture.windows import Display

DISPLAY = Display(0, 0, "test-display", 0, 0, 1920, 1080)


def test_json_write_recovers_from_transient_windows_lock(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"clips": 1}')
    replace = Path.replace
    attempts = []

    def locked_once(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            raise PermissionError("file is open")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(collection.time, "sleep", lambda _: None)
    collection.write_json(path, {"clips": 2})
    assert json.loads(path.read_text()) == {"clips": 2}
    assert len(attempts) == 2
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_json_lock_preserves_previous_metadata(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"clips": 1}')

    def locked(source, target):
        raise PermissionError("file is open")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(collection.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        collection.write_json(path, {"clips": 2})
    assert json.loads(path.read_text()) == {"clips": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_status_write_failure_does_not_escape_into_gui(tmp_path, monkeypatch):
    from pipeline_v2.capture import launcher

    def locked(*args):
        raise PermissionError("file is open")

    monkeypatch.setattr(launcher, "write_json", locked)
    assert not launcher.publish_status(tmp_path / "status.json", {"clips": 2})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = [0.0]
    created = []
    monkeypatch.setattr(collection.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        collection.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )

    class Run:
        def __init__(self, directory, display, encoder, config):
            self.directory, self.display, self.encoder = directory, display, encoder
            directory.mkdir()
            self.started = clock[0]
            self.stopped = None
            self.running = True
            self.fault = None
            created.append(self)

        def bytes_written(self):
            return 100

        def stalled(self, now):
            return False

        def stop(self):
            self.running = False
            self.stopped = clock[0]

        def publish_closed(self, session_id):
            return []

    return (
        Collector(CollectionConfig(tmp_path, exit_grace_seconds=3), run_factory=Run),
        clock,
        created,
    )


def test_waits_for_foreground_game_and_resumes_into_fresh_run(harness):
    c, clock, runs = harness
    c.tick(None, False)
    c.tick(None, True)
    assert runs == []
    c.tick(DISPLAY, True)
    clock[0] = 3
    c.tick(DISPLAY, True)
    assert c.status == "Recording Fortnite"
    c.tick(None, True)
    assert not runs[0].running
    assert c.status.startswith("Paused")
    c.tick(DISPLAY, True)
    assert len(runs) == 2
    assert runs[0].directory != runs[1].directory
    c.stop()


def test_game_exit_finalizes_and_stops_automatically(harness):
    c, clock, _ = harness
    c.tick(DISPLAY, True)
    clock[0] = 3
    c.tick(DISPLAY, True)
    c.tick(None, False)
    clock[0] = 7
    c.tick(None, False)
    assert c.done
    assert c.status.startswith("Game closed")
    assert json.loads((c.directory / "session.json").read_text())["done"]


def test_failed_cached_encoder_retries_alternative(harness):
    c, clock, runs = harness
    c.encoder_for[(0, 0)] = "h264_qsv"
    c.tick(DISPLAY, True)
    assert runs[0].encoder == "h264_qsv"
    runs[0].running = False
    clock[0] = 1
    c.tick(DISPLAY, True)
    assert c.run is None
    clock[0] = 4
    c.tick(DISPLAY, True)
    assert runs[1].encoder == "h264_nvenc"
    c.stop()


def test_low_disk_space_stops_without_deleting_footage(harness, monkeypatch):
    c, _, runs = harness
    c.tick(DISPLAY, True)
    original = runs[0].directory / "saved.mkv"
    original.write_bytes(b"keep")
    monkeypatch.setattr(
        collection.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    c.tick(DISPLAY, True)
    assert c.done
    assert original.read_bytes() == b"keep"


def test_incomplete_capture_is_not_indexed(tmp_path):
    (tmp_path / "capture-run.json").write_text("{}")
    clip = tmp_path / "part-00000.mkv"
    clip.write_bytes(b"recording")
    assert cli._clips(tmp_path, None) == []
    with pytest.raises(ValueError, match="not finalized"):
        facts.gather(clip, 30)


def test_capture_timing_overrides_legacy_fraction_and_prompt(tmp_path):
    clip = tmp_path / "part-00000.mkv"
    clip.with_suffix(".capture.json").write_text(
        json.dumps(
            {
                "schema": "salience.capture/1",
                "video_file": clip.name,
                "session_id": "heldout-session",
                "analysis_start_sec": 3,
                "analysis_end_sec": 7,
            }
        )
    )
    got = facts.gather(clip, 30)
    assert got.event_sec == 5
    assert got.capture_session == "heldout-session"
    assert "uniformly sampled gameplay window" in prompt.build_prompt(got)
    assert "centred on one event" not in prompt.build_prompt(got)
    with pytest.raises(ValueError, match="Invalid capture"):
        facts.gather(clip, 2)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_closed_chunks_have_stable_media_relative_windows(tmp_path):
    cfg = CollectionConfig(tmp_path, fps=10, segment_seconds=1)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x90:rate=10",
        "-t",
        "2.5",
        "-c:v",
        "mpeg4",
        "-g",
        "10",
    ]
    run = CaptureRun(tmp_path / "run", DISPLAY, "synthetic", cfg, command=command)
    run.process.wait(timeout=20)
    run.stop()
    results = run.publish_closed("session")
    assert len(results) >= 2
    assert run.publish_closed("session") == []
    for meta in results:
        assert (
            0
            <= meta["analysis_start_sec"]
            < meta["analysis_end_sec"]
            <= meta["duration_sec"]
        )
        path = run.directory / meta["video_file"]
        media_duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(path),
                ]
            )
        )
        assert facts.gather(path, media_duration).event_source == "capture_window"
    saved = (
        (run.directory / results[0]["video_file"])
        .with_suffix(".capture.json")
        .read_text()
    )
    run.seen.clear()
    run.publish_closed("session")
    assert (run.directory / results[0]["video_file"]).with_suffix(
        ".capture.json"
    ).read_text() == saved


def test_capture_settings_come_from_the_environment(monkeypatch, tmp_path):
    """Encoder and rate must be A/B-testable without editing code."""
    monkeypatch.setenv("SALIENCE_CAPTURE_FPS", "30")
    monkeypatch.setenv("SALIENCE_CAPTURE_BITRATE", "8M")
    monkeypatch.setenv("SALIENCE_CAPTURE_ENCODER", "h264_nvenc")

    config = collection.CollectionConfig(tmp_path)

    assert (config.fps, config.bitrate, config.encoder) == (30, "8M", "h264_nvenc")


def test_capture_settings_default_when_unset(monkeypatch, tmp_path):
    for name in ("SALIENCE_CAPTURE_FPS", "SALIENCE_CAPTURE_BITRATE",
                 "SALIENCE_CAPTURE_ENCODER"):
        monkeypatch.delenv(name, raising=False)

    config = collection.CollectionConfig(tmp_path)

    assert (config.fps, config.bitrate, config.encoder) == (60, "12M", None)


def test_nvenc_captures_straight_from_the_d3d11_surface(tmp_path):
    """No hwdownload: ddagrab's surface feeds NVENC directly.

    The Quick Sync path needs `hwmap=derive_device=qsv`, and on a hybrid laptop
    whose iGPU cannot open an MFX session that fails outright while the game
    runs on the dGPU.
    """
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path, fps=30, bitrate="8M")

    nvenc = " ".join(collection.video_command(display, "h264_nvenc", config))
    qsv = " ".join(collection.video_command(display, "h264_qsv", config))

    assert "hwmap" not in nvenc and "qsv" not in nvenc
    assert "framerate=30" in nvenc and "-b:v 8M" in nvenc
    assert "hwmap=derive_device=qsv" in qsv


def test_an_explicit_encoder_overrides_a_cached_one(monkeypatch, tmp_path):
    """capture-settings.json pinned h264_qsv over the code's own preference."""
    monkeypatch.setenv("SALIENCE_CAPTURE_ENCODER", "h264_nvenc")
    config = collection.CollectionConfig(tmp_path)
    collector = collection.Collector(config)
    collector.encoder_for[(0, 0)] = "h264_qsv"

    chosen = config.encoder or collector.encoder_for.get((0, 0))

    assert chosen == "h264_nvenc"


def test_scaling_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SALIENCE_CAPTURE_SCALE", raising=False)
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path)

    command = " ".join(collection.video_command(display, "h264_nvenc", config))

    assert "hwdownload" not in command and "scale=" not in command


def test_scaling_routes_through_system_memory(tmp_path):
    """This build cannot scale on the GPU, so the trade must be explicit."""
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path, scale="1536x960")

    command = " ".join(collection.video_command(display, "h264_nvenc", config))

    assert "hwdownload,format=bgra,scale=1536:960,format=nv12,hwupload_cuda" in command


def test_a_malformed_scale_is_refused(tmp_path):
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path, scale="huge")

    with pytest.raises(ValueError, match="1536x960"):
        collection.video_command(display, "h264_nvenc", config)


def test_keyframes_are_one_per_segment_not_one_per_second(tmp_path):
    """`-g fps` forced 5x more IDRs than the original recorder needed."""
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path, fps=30, segment_seconds=30)

    command = collection.video_command(display, "h264_nvenc", config)
    gop = command[command.index("-g") + 1]

    assert gop == "900"                      # 30fps x 30s, not 30
    assert "-maxrate" not in command         # CBR pinning costs encoder time
    assert "-bufsize" not in command


def test_the_gop_follows_the_configured_rate_and_segment(tmp_path):
    display = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    config = collection.CollectionConfig(tmp_path, fps=15, segment_seconds=10)

    command = collection.video_command(display, "h264_nvenc", config)

    assert command[command.index("-g") + 1] == "150"


def _nvidia(): return Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200, 0x10DE)
def _intel():  return Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200, 0x8086)


def test_an_adapter_only_offers_encoders_it_can_run():
    assert _nvidia().supports("h264_nvenc") and not _nvidia().supports("h264_qsv")
    assert _intel().supports("h264_qsv") and not _intel().supports("h264_nvenc")


def test_an_unenumerated_adapter_still_lets_the_probe_decide():
    unknown = Display(0, 0, r"\.\DISPLAY1", 0, 0, 1920, 1200)
    assert unknown.vendor == "unknown"
    assert unknown.supports("h264_qsv") and unknown.supports("h264_nvenc")


def test_quick_sync_is_never_attempted_on_an_nvidia_panel(harness):
    """The Sep 14 session burned run-001 on an MFX session that cannot exist."""
    collector, clock, runs = harness
    collector.encoder_for[(0, 0)] = "h264_qsv"      # stale cache, as shipped

    collector.tick(_nvidia(), True)

    assert runs[0].encoder == "h264_nvenc"
