"""Laughter: decoded as the model was calibrated, placed on the session's clock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline_v2.capture import reactions
from pipeline_v2.capture.clips import Segment

POINT = {"threshold": 0.97, "min_duration_s": 0.16, "merge_gap_s": 1.0}


def test_a_laugh_is_frames_over_the_threshold_merged_across_short_gaps():
    grid = np.zeros(100)
    grid[10:14] = 0.99  # 0.64s
    grid[18:20] = 0.98  # 0.64s later: merged into the first
    grid[60] = 0.99  # one frame: shorter than a 0.2s minimum
    found = reactions.decode(grid, {**POINT, "min_duration_s": 0.2}, 0.16, 16.0)
    assert len(found) == 1
    start, end, score = found[0]
    assert (round(start, 2), round(end, 2), score) == (1.6, 3.2, 0.99)


def test_overlapping_windows_keep_the_highest_probability():
    probs = np.array([[0.1, 0.9], [0.5, np.nan]])
    grid = reactions.stitch(probs, np.array([0.0, 0.16]), 0.5, 0.16)
    assert grid[1] == 0.9 and grid[0] == 0.1


def test_laughs_land_on_the_session_clock_and_by_track(tmp_path):
    chunk = [Segment(tmp_path / "part-00000.mkv", 0.0, 30.0), Segment(tmp_path / "part-00001.mkv", 40.0, 70.0)]
    found = {s.path.name: {} for s in chunk}
    reactions._place(chunk, [(35.0, 36.0, 0.99)], "game", found)  # 5s into the second segment
    assert found["part-00001.mkv"]["game"] == [[45.0, 46.0, 0.99]]
    assert found["part-00000.mkv"]["game"] == []


def test_a_laugh_just_after_a_moment_is_that_moments():
    laughs = [reactions.Laugh(103.0, 104.5, 0.99, "game")]
    assert reactions.laughed_in(laughs, 92.0, 100.0)  # 3s after its end
    assert not reactions.laughed_in(laughs, 80.0, 88.0)


def test_a_cache_made_another_way_is_redone(tmp_path):
    (tmp_path / reactions.CACHE_NAME).write_text(json.dumps(
        {"_source": "old", "part-00000.mkv": {"mic": [[1.0, 2.0, 0.99]]}}), encoding="utf-8")
    assert reactions.laughs(tmp_path) == []


def test_several_laughs_come_back_in_time_order(tmp_path):
    # Two Laughs could not be compared, so a session with more than one had none (2026-10-24).
    reactions._write_cache(tmp_path / reactions.CACHE_NAME, {
        "part-00001.mkv": {"mic": [[40.0, 41.0, 0.9]], "game": [[35.0, 36.0, 0.8], [40.0, 41.5, 0.7]]},
        "part-00000.mkv": {"mic": [[3.0, 4.0, 0.95]]},
    })
    got = reactions.laughs(tmp_path)
    assert [(laugh.start, laugh.track) for laugh in got] == [(3.0, "mic"), (35.0, "game"), (40.0, "mic"), (40.0, "game")]  # then by end
