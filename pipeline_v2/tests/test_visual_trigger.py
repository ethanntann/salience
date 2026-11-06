"""The coarse-grid visual trigger, and the measurement that rejected it.

Run against 80 elimination clips with a 1.5s temporal control, the strongest
cell separated event from non-event at Cohen's d = 0.58, with none above 0.8 and
the top cells scattered across rows 6-22. A real HUD element would be a tight
cluster with a large effect. These tests keep the machinery honest so the
experiment can be repeated if the labelling improves.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline_v2.capture import visual_trigger as vt


def test_a_frame_reduces_to_one_value_per_cell():
    grid = vt.to_grid(np.zeros((240, 320, 3), dtype=np.uint8))

    assert len(grid.cells) == vt.GRID_ROWS * vt.GRID_COLUMNS
    assert grid.width == 320 and grid.height == 240


def test_a_bright_patch_lands_in_the_right_cell():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[10:30, 100:200] = 255  # top strip, centre-left

    grid = vt.to_grid(image)
    brightest = max(range(len(grid.cells)), key=lambda i: grid.cells[i])

    assert brightest // vt.GRID_COLUMNS <= 3           # near the top
    assert 8 <= brightest % vt.GRID_COLUMNS <= 22      # centre-ish


def test_greyscale_and_colour_frames_both_reduce():
    colour = vt.to_grid(np.full((240, 320, 3), 128, dtype=np.uint8))
    grey = vt.to_grid(np.full((240, 320), 128, dtype=np.uint8))

    assert colour.cells == pytest.approx(grey.cells)


def test_deltas_are_per_cell_differences():
    dark = vt.to_grid(np.zeros((48, 64, 3), dtype=np.uint8))
    bright = vt.to_grid(np.full((48, 64, 3), 100, dtype=np.uint8))

    assert vt.cell_deltas(bright, dark) == pytest.approx([100.0] * len(dark.cells))


def test_separation_finds_a_cell_that_consistently_changes():
    """A planted signal must be detectable, or the metric is not measuring."""
    size = vt.GRID_ROWS * vt.GRID_COLUMNS
    target = 100
    events, nulls = [], []
    for _ in range(30):
        row = [0.0] * size
        row[target] = 50.0                # consistent brightening
        events.append(row)
        nulls.append([0.0] * size)

    scores = vt.separation(events, nulls)

    assert vt.top_cells(scores, 1)[0][:2] == (
        target // vt.GRID_COLUMNS, target % vt.GRID_COLUMNS
    )


def test_a_large_but_inconsistent_change_does_not_score():
    """Scene changes move many cells a lot; only consistency is signal."""
    size = vt.GRID_ROWS * vt.GRID_COLUMNS
    rng = np.random.default_rng(0)
    events = [list(rng.normal(0, 60, size)) for _ in range(40)]
    nulls = [list(rng.normal(0, 60, size)) for _ in range(40)]

    scores = vt.separation(events, nulls)

    assert max(abs(s) for s in scores) < 1.0


def test_separation_survives_empty_input():
    assert vt.separation([], []) == [0.0] * (vt.GRID_ROWS * vt.GRID_COLUMNS)


def test_regions_are_described_in_screen_terms():
    assert vt.describe_region([(1, 2, 1.0)]).startswith("top")
    assert vt.describe_region([(23, 30, 1.0)]).startswith("bottom")
    assert vt.describe_region([]) == "nowhere"
