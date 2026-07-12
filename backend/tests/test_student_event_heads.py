from salience_api.features.fireworks_teacher import resolve_event_summaries
from salience_api.student.event_heads import (
    EventHeadPrediction,
    collapse_same_kill_finishes,
    event_summary_from_heads,
)


def _finish(index: int, timestamp: float, *, confidence: float = 0.5, kind: str = "elimination"):
    return {
        "event_index": index,
        "event_timestamp": timestamp,
        "event_kind": kind,
        "weapon_confidence": confidence,
    }


def test_collapse_merges_banner_tail_chain_into_one_finish():
    # One kill at 10s whose lingering banner produces finishes at 11.5s and 13s;
    # chained gaps < 3s collapse to the single most confident summary.
    summaries = [
        _finish(0, 10.0, confidence=0.9),
        _finish(1, 11.5, confidence=0.6),
        _finish(2, 13.0, confidence=0.7),
    ]
    collapsed = collapse_same_kill_finishes(summaries)
    assert len(collapsed) == 1
    assert collapsed[0]["event_timestamp"] == 10.0


def test_collapse_keeps_distinct_kills_separated_by_quiet_gap():
    summaries = [
        _finish(0, 10.0, confidence=0.9),
        _finish(1, 11.0, confidence=0.5),
        _finish(2, 20.0, confidence=0.8),
    ]
    collapsed = collapse_same_kill_finishes(summaries)
    assert [s["event_timestamp"] for s in collapsed] == [10.0, 20.0]


def test_collapse_passes_non_finish_summaries_through():
    summaries = [
        _finish(0, 10.0),
        _finish(1, 11.0, kind="damage_only"),
        _finish(2, 12.0, kind="none"),
    ]
    collapsed = collapse_same_kill_finishes(summaries)
    kinds = sorted(s["event_kind"] for s in collapsed)
    assert kinds == ["damage_only", "elimination", "none"]


def test_event_summary_requires_visible_damage_for_damage_claim():
    pred = EventHeadPrediction(
        event_kind="elimination",
        target_state="active",
        weapon="shotgun",
        weapon_confidence=0.92,
        aim_state="hipfire",
        pov_shot_visible="yes",
        new_damage_visible="no",
        single_shot_damage_known="yes",
        single_shot_damage=140,
    )
    summary = event_summary_from_heads(pred, event_index=0, event_timestamp=10.0)
    assert summary["single_shot_damage"] is None
    assert summary["high_damage_one_shot"] is False
    # Raw value is preserved for offline cutoff analysis.
    assert summary["raw_single_shot_damage"] == 140

    pred.new_damage_visible = "yes"
    summary = event_summary_from_heads(pred, event_index=0, event_timestamp=10.0)
    assert summary["single_shot_damage"] == 140
    assert summary["high_damage_one_shot"] is True


def test_head_prediction_round_trips_through_reducer():
    pred = EventHeadPrediction(
        event_kind="elimination",
        target_state="active",
        weapon="shotgun",
        weapon_confidence=0.92,
        aim_state="hipfire",
        pov_shot_visible="yes",
        new_damage_visible="yes",
        target_defeat_visible="yes",
        finish_ui_newly_appeared="yes",
    )
    raw = event_summary_from_heads(pred, event_index=0, event_timestamp=10.0)
    attribution = resolve_event_summaries({"events": [raw]})
    assert attribution.labels["shotgun_kill"] == "yes"
    assert attribution.status == "attributed"
