from pathlib import Path

from salience_api.clips.keyframes import (
    Keyframe,
    _parallel_map,
    cleanup_keyframes,
    event_window_timestamps,
    keyframe_timestamps,
)
from salience_api.features.fireworks_teacher import max_event_candidates


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


def test_parallel_map_preserves_frame_order(monkeypatch):
    monkeypatch.setenv("SALIENCE_FFMPEG_WORKERS", "2")

    assert _parallel_map([3, 1, 2], lambda value: value * 2) == [6, 2, 4]


def test_cleanup_keyframes_removes_temporary_parent(tmp_path: Path):
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    frame_path = output_dir / "frame.jpg"
    frame_path.write_bytes(b"frame")

    cleanup_keyframes([Keyframe(path=frame_path, timestamp_sec=1.0)])

    assert not output_dir.exists()
