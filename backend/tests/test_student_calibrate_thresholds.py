import pytest

from salience_api.student.calibrate_thresholds import macro_agreement, sweep_field_threshold


def test_macro_agreement_averages_only_labels_with_teacher_positives():
    labels_summary = {
        "sniper_kill": {
            "agree": 8,
            "clips": 10,
            "teacher_yes": 3,
            "true_positive": 2,
            "false_positive": 1,
            "false_negative": 1,
        },
        "flick_shot": {
            "agree": 10,
            "clips": 10,
            "teacher_yes": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        },  # trivial, excluded
        "shotgun_kill": {
            "agree": 6,
            "clips": 10,
            "teacher_yes": 4,
            "true_positive": 3,
            "false_positive": 1,
            "false_negative": 1,
        },
    }
    result = macro_agreement(labels_summary)
    assert result == pytest.approx(((4 / 6) + (6 / 8)) / 2)


def test_sweep_field_threshold_picks_candidate_maximizing_macro_agreement():
    # Two synthetic clips; higher "weapon" threshold flips clip B's false
    # positive "shotgun_kill" to "no", improving agreement.
    def decode_at(threshold: float, clip_id: str) -> dict[str, str]:
        if clip_id == "a":
            return {"shotgun_kill": "yes"}
        return {"shotgun_kill": "yes" if threshold < 0.6 else "no"}

    cached_clips = ["a", "b"]
    teacher_labels_by_clip = {
        "a": {"shotgun_kill": "yes"},
        "b": {"shotgun_kill": "no"},
    }

    best_threshold, metrics = sweep_field_threshold(
        cached_clips,
        field="weapon",
        candidates=[0.5, 0.6, 0.7],
        decode_fn=decode_at,
        teacher_labels_by_clip=teacher_labels_by_clip,
        label_keys=["shotgun_kill"],
    )

    assert best_threshold == 0.6
    assert metrics["shotgun_kill"]["agree"] == 2
