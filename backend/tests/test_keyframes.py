from pathlib import Path
import subprocess

from salience_api.clips import keyframes as keyframes_module
from salience_api.clips.keyframes import (
    _EVENT_COMPOSITE_FILTER,
    _OCR_FILTER,
    Keyframe,
    _extract_many_variants,
    _frame_indices,
    cleanup_keyframes,
    event_window_timestamps,
    keyframe_timestamps,
)
from salience_api.features.fireworks_teacher import max_event_candidates
from salience_api.student.schema import COMPOSITE_HUD_LEFT_RATIO


def test_keyframe_timestamps_cover_the_whole_clip_uniformly():
    timestamps = keyframe_timestamps(20.152267)

    assert len(timestamps) == 20
    assert timestamps[0] < 1.5
    assert timestamps[-1] > 18.5
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    assert max(gaps) - min(gaps) < 0.01


def test_event_windows_bias_forward_and_keep_event_identity():
    timestamps = event_window_timestamps(20.0, [3.0, 16.0])

    first = [timestamp for event, timestamp in timestamps if event == 0]
    second = [timestamp for event, timestamp in timestamps if event == 1]
    assert first[0] == 2.25
    assert first[-1] == 4.5
    assert second[0] == 15.25
    assert second[-1] == 17.5
    # More lookahead than lookbehind so finish UI after impact stays in-window.
    assert (first[-1] - 3.0) > (3.0 - first[0])
    assert (second[-1] - 16.0) > (16.0 - second[0])


def test_event_windows_preserve_all_locator_events():
    timestamps = event_window_timestamps(20.0, [2.0, 6.0, 10.0, 15.0, 18.0])

    assert {event for event, _ in timestamps} == {0, 1, 2, 3, 4}


def test_long_montage_uses_more_context_and_event_capacity():
    assert len(keyframe_timestamps(120.0)) == 60
    assert max_event_candidates(20.0) == 4
    assert max_event_candidates(120.0) == 12


def test_frame_indices_dedupe_timestamps_landing_on_one_frame():
    indices, unique = _frame_indices([1.0, 1.001, 2.0], 60.0)

    # 1.0 and 1.001 both round up into frame 60 at 60fps.
    assert indices == [60, 61, 120]
    assert unique == [60, 61, 120]

    indices, unique = _frame_indices([1.0, 1.0, 2.0], 60.0)
    assert indices == [60, 60, 120]
    assert unique == [60, 120]


def test_extract_many_variants_declines_inputs_it_cannot_fuse(tmp_path: Path):
    outputs = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    clip = tmp_path / "clip.mp4"
    single = [(_OCR_FILTER, outputs)]
    pair = [(_OCR_FILTER, outputs), (_EVENT_COMPOSITE_FILTER, outputs)]

    # A single variant is the legacy path's job, not the fused one.
    assert _extract_many_variants(clip, [1.0, 2.0], single, 60.0) is False
    # Variable-rate clips have no safe timestamp -> frame-index mapping.
    assert _extract_many_variants(clip, [1.0, 2.0], pair, None) is False
    assert _extract_many_variants(clip, [], pair, 60.0) is False
    # Output count must line up with the requested timestamps.
    assert _extract_many_variants(clip, [1.0], pair, 60.0) is False


def _fake_ffmpeg(frames_per_output: list[int]):
    """Stand in for ffmpeg, writing N frames into each mapped output directory."""
    calls: list[list[str]] = []

    def run(command, capture_output=True, text=True):
        calls.append(command)
        patterns = [arg for arg in command if arg.endswith("frame-%06d.jpg")]
        for pattern, count in zip(patterns, frames_per_output, strict=True):
            directory = Path(pattern).parent
            for index in range(count):
                (directory / f"frame-{index + 1:06d}.jpg").write_bytes(
                    f"{directory.name}-{index}".encode()
                )
        return subprocess.CompletedProcess(command, 0, "", "")

    return run, calls


def test_fused_decode_runs_one_ffmpeg_pass_for_both_variants(tmp_path, monkeypatch):
    composite = [tmp_path / f"c{index}.jpg" for index in range(3)]
    ocr = [tmp_path / f"o{index}.jpg" for index in range(3)]
    run, calls = _fake_ffmpeg([3, 3])
    monkeypatch.setattr(keyframes_module.subprocess, "run", run)

    fused = _extract_many_variants(
        tmp_path / "clip.mp4",
        [1.0, 2.0, 3.0],
        [(_EVENT_COMPOSITE_FILTER, composite), (_OCR_FILTER, ocr)],
        60.0,
    )

    assert fused is True
    assert len(calls) == 1, "fused decode must not spawn a pass per variant"
    command = calls[0]
    assert command.count("-filter_complex") == 1
    assert command.count("-map") == 2
    assert "split=2[s0][s1]" in ";".join(command)
    assert all(path.exists() for path in composite + ocr)
    # Each variant keeps its own frames rather than aliasing the other's.
    assert composite[0].read_bytes() != ocr[0].read_bytes()


def test_fused_decode_writes_nothing_when_a_variant_comes_up_short(
    tmp_path, monkeypatch
):
    composite = [tmp_path / f"c{index}.jpg" for index in range(3)]
    ocr = [tmp_path / f"o{index}.jpg" for index in range(3)]
    # Second variant loses a frame; the whole fused attempt must be discarded so
    # the caller's fallback starts from a clean slate.
    run, _ = _fake_ffmpeg([3, 2])
    monkeypatch.setattr(keyframes_module.subprocess, "run", run)

    fused = _extract_many_variants(
        tmp_path / "clip.mp4",
        [1.0, 2.0, 3.0],
        [(_EVENT_COMPOSITE_FILTER, composite), (_OCR_FILTER, ocr)],
        60.0,
    )

    assert fused is False
    assert not any(path.exists() for path in composite + ocr)


def test_event_composite_layout_matches_the_hud_crop_ratio():
    """The student slices the HUD panel back out by ratio, so widths are a contract."""
    assert "scale=640:360" in _EVENT_COMPOSITE_FILTER
    assert "scale=640:432" in _EVENT_COMPOSITE_FILTER
    assert "scale=400:432" in _EVENT_COMPOSITE_FILTER
    assert "hstack=inputs=3" in _EVENT_COMPOSITE_FILTER

    composite_width = 640 + 640 + 400
    assert composite_width == 1680
    assert round(COMPOSITE_HUD_LEFT_RATIO * composite_width) == 1280


def test_cleanup_keyframes_removes_temporary_parent(tmp_path: Path):
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    frame_path = output_dir / "frame.jpg"
    frame_path.write_bytes(b"frame")

    cleanup_keyframes([Keyframe(path=frame_path, timestamp_sec=1.0)])

    assert not output_dir.exists()
