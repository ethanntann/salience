import pytest

from salience_api.student.calibrate_thresholds import sweep_field_threshold
from salience_api.student.eval_agreement import (
    aggregate_label_metrics,
    clip_locator_diagnostic,
    compare_labels,
    summarize_labels,
    summarize_locator_diagnostics,
)


def test_compare_labels_counts_agreement_and_yes_flags():
    teacher = {
        "sniper_kill": "yes",
        "shotgun_kill": "no",
        "elimination_or_knock": "yes",
    }
    student = {
        "sniper_kill": "yes",
        "shotgun_kill": "yes",
        "elimination_or_knock": "no",
    }
    keys = ["sniper_kill", "shotgun_kill", "elimination_or_knock"]

    metrics = compare_labels(teacher, student, keys)

    assert metrics["sniper_kill"] == {
        "agree": 1,
        "teacher_yes": 1,
        "student_yes": 1,
        "student_uncertain": 0,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert metrics["shotgun_kill"] == {
        "agree": 0,
        "teacher_yes": 0,
        "student_yes": 1,
        "student_uncertain": 0,
        "true_positive": 0,
        "false_positive": 1,
        "false_negative": 0,
    }
    assert metrics["elimination_or_knock"] == {
        "agree": 0,
        "teacher_yes": 1,
        "student_yes": 0,
        "student_uncertain": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 1,
    }


def test_compare_labels_treats_missing_keys_as_uncertain():
    metrics = compare_labels({"combat_visible": "yes"}, {}, ["combat_visible"])

    assert metrics["combat_visible"] == {
        "agree": 0,
        "teacher_yes": 1,
        "student_yes": 0,
        "student_uncertain": 1,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 1,
    }


def test_aggregate_label_metrics_sums_per_clip_results():
    totals = aggregate_label_metrics(
        [
            {
                "sniper_kill": {
                    "agree": 1, "teacher_yes": 1, "student_yes": 1,
                    "student_uncertain": 0, "true_positive": 1,
                    "false_positive": 0, "false_negative": 0,
                }
            },
            {
                "sniper_kill": {
                    "agree": 0, "teacher_yes": 1, "student_yes": 0,
                    "student_uncertain": 0, "true_positive": 0,
                    "false_positive": 0, "false_negative": 1,
                }
            },
        ]
    )

    assert totals["sniper_kill"] == {
        "agree": 1,
        "teacher_yes": 2,
        "student_yes": 1,
        "student_uncertain": 0,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 1,
        "clips": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "uncertain_rate": 0.0,
    }


def test_summarize_labels_excludes_zero_teacher_positive_labels():
    labels = {
        "combat_visible": {
            "agree": 70,
            "teacher_yes": 79,
            "true_positive": 62,
            "false_positive": 1,
            "false_negative": 17,
            "f1": 0.8732,
            "clips": 81,
        },
        "trickshot": {
            "agree": 77,
            "teacher_yes": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "f1": 0.0,
            "clips": 81,
        },
    }

    summary = summarize_labels(labels)

    assert summary["evaluable_labels"] == 1
    assert summary["not_evaluable_labels"] == ["trickshot"]
    # trickshot's undefined 0.0 F1 must not drag the macro down.
    assert summary["macro_f1"] == pytest.approx(0.8732)
    assert summary["mean_agreement"] == pytest.approx(70 / 81)
    assert summary["clips"] == 81
    assert summary["worst_labels_by_f1"] == ["combat_visible"]


def test_clip_locator_diagnostic_flags_zero_event_dropout():
    diagnostic = clip_locator_diagnostic(locator_events=0, attribution_status=None)
    assert diagnostic == {
        "locator_events": 0,
        "attribution_status": "no_attribution",
        "forced_uncertain": True,
    }


def test_clip_locator_diagnostic_passes_through_attributed_clip():
    diagnostic = clip_locator_diagnostic(locator_events=2, attribution_status="attributed")
    assert diagnostic == {
        "locator_events": 2,
        "attribution_status": "attributed",
        "forced_uncertain": False,
    }


def test_summarize_locator_diagnostics_counts_dropout_rate():
    diagnostics = [
        {"locator_events": 0, "attribution_status": "no_attribution", "forced_uncertain": True},
        {"locator_events": 2, "attribution_status": "attributed", "forced_uncertain": False},
        {"locator_events": 1, "attribution_status": "no_event", "forced_uncertain": False},
    ]
    summary = summarize_locator_diagnostics(diagnostics)
    assert summary == {
        "clips": 3,
        "dropout_clips": 1,
        "dropout_rate": pytest.approx(1 / 3),
        "status_counts": {"no_attribution": 1, "attributed": 1, "no_event": 1},
    }


def test_threshold_sweep_looks_up_labels_by_cached_clip_id():
    class CachedClip:
        clip_id = 17

    threshold, metrics = sweep_field_threshold(
        [CachedClip()],
        field="weapon",
        candidates=[0.5, 0.6],
        decode_fn=lambda candidate, _clip: {
            "sniper_kill": "yes" if candidate >= 0.6 else "no"
        },
        teacher_labels_by_clip={17: {"sniper_kill": "yes"}},
        label_keys=["sniper_kill"],
    )

    assert threshold == 0.6
    assert metrics["sniper_kill"]["f1"] == 1.0
