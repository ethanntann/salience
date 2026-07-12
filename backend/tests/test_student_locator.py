import pytest

from salience_api.student.locator import timestamps_from_frame_scores
from salience_api.student.schema import (
    WEAPON_EVIDENCE_FRAME_INDEX,
    hud_crop_left,
    select_window_indices,
)


def test_peak_decode_picks_separated_maxima():
    timestamps = [i * 0.5 for i in range(20)]
    scores = [0.1] * 20
    scores[4] = 0.9   # 2.0s
    scores[5] = 0.85  # near peak, suppressed
    scores[14] = 0.95 # 7.0s
    result = timestamps_from_frame_scores(scores, timestamps, max_events=4, min_separation_sec=0.75)
    assert result == [7.0, 2.0] or result == [2.0, 7.0]
    assert len(result) == 2


def test_peak_decode_returns_no_event_below_threshold():
    assert timestamps_from_frame_scores(
        [0.2, 0.49], [1.0, 2.0], max_events=2
    ) == []


def test_peak_decode_falls_back_to_best_frame_when_all_below_threshold():
    result = timestamps_from_frame_scores(
        [0.2, 0.35, 0.1],
        [1.0, 2.0, 3.0],
        max_events=4,
        always_return_best=True,
    )
    assert result == [2.0]


def test_peak_decode_fallback_still_respects_max_events_of_one():
    result = timestamps_from_frame_scores(
        [0.1, 0.2],
        [1.0, 2.0],
        max_events=1,
        always_return_best=True,
    )
    assert result == [2.0]


def test_peak_decode_fallback_is_opt_in():
    # Default behavior (always_return_best=False) is unchanged.
    assert timestamps_from_frame_scores(
        [0.2, 0.49], [1.0, 2.0], max_events=2
    ) == []


def test_select_window_indices_evenly_covers_timeline():
    assert select_window_indices(7) == list(range(7))
    assert select_window_indices(3) == [0, 1, 2]
    assert select_window_indices(20) == [0, 3, 6, 10, 13, 16, 19]


def test_weapon_evidence_uses_prefinish_frame():
    # Offsets [-0.75, -0.375, 0, ...]; index 1 is the last sample before finish.
    assert WEAPON_EVIDENCE_FRAME_INDEX == 1


def test_hud_crop_starts_at_composite_hud_panel():
    assert hud_crop_left(1680) == 1280


def test_locator_net_output_shape():
    torch = pytest.importorskip("torch")
    from salience_api.student.locator import build_locator_net

    net = build_locator_net()
    batch = torch.zeros(2, 3, 224, 224)
    logits = net(batch)
    assert logits.shape == (2, 1)
