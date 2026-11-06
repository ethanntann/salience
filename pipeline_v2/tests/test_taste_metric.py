"""Pairwise accuracy: what a tie is worth, and what counts as a comparison."""

from __future__ import annotations

from pipeline_v2 import taste


def _example(sha: str, label: int, engagement: str, vector: list[float]) -> taste.Example:
    return taste.Example(
        clip_sha=sha,
        engagement=engagement,
        label=label,
        weight=1.0,
        fields=[],
        embedding=vector,
    )


def test_a_tie_scores_half_not_zero():
    assert taste._pair_credit(1.0, 0.0) == 1.0
    assert taste._pair_credit(0.0, 1.0) == 0.0
    assert taste._pair_credit(0.5, 0.5) == 0.5


def test_a_constant_scorer_reports_chance_rather_than_zero():
    """A model with nothing to say is uninformative, not inverted."""
    examples = [
        _example("a", 1, "eng0", [1.0]),
        _example("b", 0, "eng1", [1.0]),
        _example("c", 1, "eng2", [1.0]),
        _example("d", 0, "eng3", [1.0]),
    ]
    flat = {e.clip_sha: 0.0 for e in examples}

    accuracy, total = taste.fixed_score_accuracy(examples, flat)

    assert total == 4
    assert accuracy == 0.5


def test_a_perfect_fixed_ranking_still_scores_one():
    examples = [
        _example("a", 1, "eng0", [1.0]),
        _example("b", 0, "eng1", [1.0]),
    ]

    accuracy, total = taste.fixed_score_accuracy(examples, {"a": 1.0, "b": 0.0})

    assert (accuracy, total) == (1.0, 1)


def test_clips_without_a_fixed_score_are_skipped_not_counted():
    examples = [
        _example("a", 1, "eng0", [1.0]),
        _example("b", 0, "eng1", [1.0]),
    ]

    accuracy, total = taste.fixed_score_accuracy(examples, {"a": 1.0})

    assert (accuracy, total) == (0.0, 0)


def test_pairs_never_compare_two_clips_of_one_fight():
    same_fight = [
        _example("a", 1, "eng0", [1.0]),
        _example("b", 0, "eng0", [1.0]),
    ]

    assert taste.pairs(same_fight) == []
    assert len(taste.within_engagement_pairs(same_fight)) == 1
