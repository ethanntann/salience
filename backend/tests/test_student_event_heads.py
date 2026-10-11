from salience_api.features.fireworks_teacher import resolve_event_summaries
from salience_api.student.event_heads import EventHeadPrediction, event_summary_from_heads


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
