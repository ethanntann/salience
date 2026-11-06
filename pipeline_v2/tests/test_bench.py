"""The capture performance gate.

The recorder was paused on a subjective report and could not be cleared because
nothing measured it. These cover the parts that decide pass or fail, especially
the rule that a gate with no data reports `null` rather than passing.
"""

from __future__ import annotations

import json

import pytest

from pipeline_v2.capture import bench


def test_one_percent_low_catches_stutter_that_mean_fps_hides():
    """A few very slow frames barely move the mean and wreck the 1% low."""
    smooth = [8.0] * 1000
    stuttery = [8.0] * 990 + [80.0] * 10

    a, b = bench.frame_metrics(smooth), bench.frame_metrics(stuttery)

    assert a["mean_fps"] == 125.0
    assert b["mean_fps"] > 90            # mean looks nearly fine
    assert a["one_percent_low_fps"] == 125.0
    assert b["one_percent_low_fps"] == pytest.approx(12.5, abs=0.1)


def test_frame_metrics_reports_percentiles_and_the_worst_frame():
    metrics = bench.frame_metrics([10.0] * 99 + [200.0])

    assert metrics["frames"] == 100
    assert metrics["worst_frame_ms"] == 200.0
    assert metrics["p95_frame_ms"] == 10.0
    assert metrics["seconds"] == pytest.approx(1.19, abs=0.01)


def test_frame_metrics_survives_an_empty_or_junk_trace():
    assert bench.frame_metrics([])["frames"] == 0
    assert bench.frame_metrics([0.0, -1.0])["frames"] == 0


def test_present_csv_is_read_whatever_the_column_case(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(
        "Application,MsBetweenPresents\nFortnite,8.1\nFortnite,8.3\n", encoding="utf-8"
    )

    assert bench.read_present_csv(path) == [8.1, 8.3]

    lower = tmp_path / "q.csv"
    lower.write_text("app,msBetweenPresents\nx,9.0\n", encoding="utf-8")
    assert bench.read_present_csv(lower) == [9.0]


def test_an_unrecognised_trace_is_refused_not_silently_empty(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("Application,SomethingElse\nFortnite,8.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="present-interval column"):
        bench.read_present_csv(path)


def _run(low=None, drift=None):
    record = {"frames": {}, "av_drift": []}
    if low is not None:
        record["frames"] = {"frames": 1000, "one_percent_low_fps": low}
    if drift is not None:
        record["av_drift"] = [{"clip": "a.mkv", "duration_drift_ms": drift}]
    return record


def test_a_small_frame_cost_passes_the_gate():
    result = bench.compare(_run(low=100.0), _run(low=97.0, drift=10.0))

    assert result["gates"]["one_percent_low"]["pass"] is True
    assert result["gates"]["one_percent_low"]["loss"] == pytest.approx(0.03)
    assert result["verdict"] == "pass"


def test_a_large_frame_cost_fails_the_gate():
    result = bench.compare(_run(low=100.0), _run(low=80.0, drift=10.0))

    assert result["gates"]["one_percent_low"]["pass"] is False
    assert result["verdict"] == "fail"


def test_drift_beyond_the_limit_fails_even_when_frames_are_fine():
    result = bench.compare(_run(low=100.0), _run(low=100.0, drift=250.0))

    assert result["gates"]["av_drift"]["pass"] is False
    assert result["verdict"] == "fail"


def test_drift_is_judged_on_magnitude_in_either_direction():
    result = bench.compare(_run(low=100.0), _run(low=100.0, drift=-250.0))

    assert result["gates"]["av_drift"]["worst_ms"] == -250.0
    assert result["gates"]["av_drift"]["pass"] is False


def test_a_gate_with_no_data_is_inconclusive_never_a_pass():
    """The failure mode that let the recorder be called acceptable before."""
    result = bench.compare(_run(), _run())

    assert result["gates"]["one_percent_low"]["pass"] is None
    assert result["gates"]["av_drift"]["pass"] is None
    assert result["verdict"] == "inconclusive"
    assert result["gates_evaluated"] == 0
    assert "PresentMon" in result["gates"]["one_percent_low"]["why"]


def test_one_measured_gate_does_not_certify_the_unmeasured_one():
    result = bench.compare(_run(low=100.0), _run(low=99.0))

    assert result["gates"]["one_percent_low"]["pass"] is True
    assert result["gates"]["av_drift"]["pass"] is None
    assert result["gates_evaluated"] == 1
    assert result["gates_total"] == 2
    # One gate passing does not certify the run.
    assert result["verdict"] == "inconclusive"


def test_gpu_summary_handles_no_readings():
    assert bench.GpuSampler().summary()["samples"] == 0


def test_gpu_summary_averages_what_it_collected():
    sampler = bench.GpuSampler()
    sampler.samples = [
        {"gpu_util_pct": 50.0, "vram_used_mb": 1000.0, "temp_c": 60.0, "power_w": None},
        {"gpu_util_pct": 90.0, "vram_used_mb": 2000.0, "temp_c": 70.0, "power_w": None},
    ]

    summary = sampler.summary()

    assert summary["gpu_util_pct"] == {"mean": 70.0, "max": 90.0}
    assert summary["vram_used_mb"]["max"] == 2000.0
    assert summary["power_w"] is None


def test_av_drift_reports_a_missing_audio_track_rather_than_zero(tmp_path, monkeypatch):
    """Recorder audio is unfinished; silence must not read as perfect sync."""
    payload = json.dumps({"streams": [
        {"codec_type": "video", "duration": "30.0", "start_time": "0.0"}]})

    class Done:
        returncode = 0
        stdout = payload

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: Done())

    result = bench.av_drift_ms(tmp_path / "clip.mkv")

    assert "duration_drift_ms" not in result
    assert "no audio stream" in result["error"]


def test_av_drift_measures_both_start_offset_and_duration(tmp_path, monkeypatch):
    payload = json.dumps({"streams": [
        {"codec_type": "video", "duration": "30.000", "start_time": "0.000"},
        {"codec_type": "audio", "duration": "29.880", "start_time": "0.021"}]})

    class Done:
        returncode = 0
        stdout = payload

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: Done())

    result = bench.av_drift_ms(tmp_path / "clip.mkv")

    assert result["duration_drift_ms"] == pytest.approx(-120.0)
    assert result["start_offset_ms"] == pytest.approx(21.0)


def test_matroska_duration_tags_are_understood():
    """MKV keeps track length in a DURATION tag, not the stream duration field."""
    assert bench._seconds("31.000000") == 31.0
    assert bench._seconds("00:00:31.000000000") == pytest.approx(31.0)
    assert bench._seconds("01:02:03.5") == pytest.approx(3723.5)
    assert bench._seconds("") is None
    assert bench._seconds("not a time") is None


def test_a_video_only_recorder_chunk_is_read_not_rejected(tmp_path, monkeypatch):
    """Every Sep 7 chunk looked like "no video stream" until DURATION was read."""
    payload = json.dumps({"streams": [{
        "codec_type": "video", "start_time": "0.000000",
        "tags": {"DURATION": "00:00:31.000000000"}}]})

    class Done:
        returncode = 0
        stdout = payload

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: Done())

    result = bench.av_drift_ms(tmp_path / "part-00000.mkv")

    assert result["video_sec"] == 31.0
    assert "no audio stream" in result["error"]


def test_the_tracer_command_uses_presentmon_2_syntax():
    """2.x takes --double-dash; 1.x took -single, and 2.x silently ignores it."""
    from pathlib import Path

    command = bench.tracer_command("PresentMon.exe", Path("out.csv"), 120)

    assert command[0] == "PresentMon.exe"
    for flag in ("--process_name", "--output_file", "--timed",
                 "--terminate_after_timed", "--v1_metrics"):
        assert flag in command
    assert not any(a.startswith("-") and not a.startswith("--") for a in command[1:])
    assert command[command.index("--timed") + 1] == "120"


def test_v1_metrics_is_requested_so_the_column_name_is_known():
    """read_present_csv looks for MsBetweenPresents; v2's schema differs."""
    from pathlib import Path

    assert "--v1_metrics" in bench.tracer_command("p.exe", Path("o.csv"), 5)


def test_a_relative_results_path_anchors_to_the_repo(monkeypatch, tmp_path):
    """The elevated PowerShell for PresentMon starts in system32.

    `.local-data\bench` there became a second results directory, so the
    baseline and capture phases landed in different places and `report` could
    not find its pair.
    """
    from pathlib import Path

    from pipeline_v2.config import REPO_ROOT

    relative, absolute = Path(".local-data/bench"), tmp_path / "elsewhere"

    assert (REPO_ROOT / relative).is_absolute()
    # An explicit absolute path is still honoured untouched.
    assert (absolute if absolute.is_absolute() else REPO_ROOT / absolute) == absolute


def test_a_capture_faster_than_baseline_is_refused_not_passed():
    """Capture cannot speed a game up; such a pair measured nothing.

    Fortnite's framerate is dominated by match stage, so a baseline on the bus
    and a capture phase late-game differ far more than any capture overhead.
    """
    result = bench.compare(_run(low=26.6), _run(low=35.16, drift=10.0))

    gate = result["gates"]["one_percent_low"]
    assert gate["pass"] is None
    assert "FASTER than baseline" in gate["why"]
    assert result["verdict"] != "pass"


def test_a_marginal_improvement_still_passes():
    """Small variation either way is ordinary, not a broken pairing."""
    result = bench.compare(_run(low=100.0), _run(low=102.0, drift=10.0))

    assert result["gates"]["one_percent_low"]["pass"] is True
    assert result["verdict"] == "pass"


def test_paired_summary_uses_medians_not_a_single_pair():
    """One pair cannot resolve this: baselines ranged 26.6-37.3 fps unrecorded."""
    assert bench.median([10.0, 20.0, 30.0]) == 20.0
    assert bench.median([]) is None


def test_the_noise_floor_is_reported_alongside_the_effect():
    """A difference smaller than the baseline spread is not a measurement."""
    import inspect

    source = inspect.getsource(bench.paired_runs)

    assert "baseline_spread" in source
    assert "median" in source


def test_the_warmup_round_is_discarded_by_rule_not_by_inspection():
    """Choosing which round to drop after seeing it would pick the answer.

    The first 90 seconds hold the drop, shader compilation and asset
    streaming -- session stutter, not capture cost. Measured 2026-09-23: the
    warm-up baseline's 1% low was 35.2 fps against 88.9 and 95.8 after it,
    while mean fps barely moved (157 vs 163).
    """
    import inspect

    source = inspect.getsource(bench.paired_runs)

    assert "warmup" in source
    assert "usable[0], usable[1:]" in source
    # It only applies when there is still more than one round left to score.
    assert "len(usable) > 2" in source


def test_capture_settings_reach_the_recorder_and_are_recorded():
    """A hardcoded rate would silently invalidate a `--fps 15` comparison."""
    import inspect

    source = inspect.getsource(bench.paired_runs)

    assert '"--fps", str(fps)' in source
    assert "SALIENCE_CAPTURE_SCALE" in source
    # Stored in the summary, so a result cannot be read against wrong settings.
    assert '"fps": fps' in source
    assert '"scale": scale' in source


def _pairs(values):
    """Build a paired summary the way `paired_runs` does, to test its rules."""
    usable = [(b, c) for b, c in values]
    bm = bench.median([b for b, _ in usable])
    cm = bench.median([c for _, c in usable])
    loss = (bm - cm) / bm
    summary = {"pass": bool(loss <= bench.MAX_ONE_PERCENT_LOW_LOSS), "loss": loss}
    faster = sum(1 for b, c in usable if c > b)
    if loss < -bench.NOISE_TOLERANCE or faster:
        summary["pass"] = None
    return summary


def test_paired_never_passes_when_capture_measured_faster():
    """The driver reported PASS on runs where capture beat baseline 3/3 times."""
    # Real data, 2026-09-23 --fps 15 run.
    assert _pairs([(58.77, 90.55), (29.84, 77.28), (78.07, 85.47)])["pass"] is None
    # Real data, --scale 1280x800 run.
    assert _pairs([(66.47, 90.46), (27.35, 121.31), (28.47, 20.95)])["pass"] is None


def test_paired_still_passes_a_genuinely_cheap_capture():
    assert _pairs([(100.0, 98.0), (102.0, 99.0), (101.0, 98.5)])["pass"] is True


def test_one_faster_round_is_enough_to_refuse():
    """A single impossible pair means the rounds were not comparable."""
    assert _pairs([(100.0, 95.0), (100.0, 104.0), (100.0, 96.0)])["pass"] is None
