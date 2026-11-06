"""Audio levels from a recording, and the tuner that replays them.

The only audio-trigger result this project has was tuned on its own test data
and could not beat a metronome. These cover the machinery that makes a fair
evaluation possible instead: separate tracks, offline replay, and a refusal to
report recall when nothing was marked.
"""

from __future__ import annotations

import math

import pytest

from pipeline_v2.capture import tune_voice
from pipeline_v2.capture.levels import Level, SILENCE_DB, summarise, to_db


def test_silence_has_a_floor_rather_than_negative_infinity():
    """-inf poisons every average and comparison downstream."""
    assert to_db(0.0) == SILENCE_DB
    assert to_db(-1.0) == SILENCE_DB
    assert math.isfinite(to_db(0.0))


def test_full_scale_is_zero_db():
    assert to_db(1.0) == pytest.approx(0.0)
    assert to_db(0.1) == pytest.approx(-20.0)


def test_a_silent_track_is_distinguishable_from_a_usable_one():
    silent = summarise([Level(i * 0.05, SILENCE_DB) for i in range(100)])
    speech = summarise([Level(i * 0.05, -30.0) for i in range(100)])

    assert silent["audible_fraction"] == 0.0
    assert speech["audible_fraction"] == 1.0


def test_an_empty_track_summarises_without_raising():
    assert summarise([])["windows"] == 0


def _series(quiet_count, loud_count, quiet=-40.0, loud=-25.0, step=0.05):
    levels = [Level(i * step, quiet) for i in range(quiet_count)]
    offset = quiet_count * step
    levels += [Level(offset + i * step, loud) for i in range(loud_count)]
    return levels


def test_replay_finds_a_shout_in_a_recorded_series():
    fires = tune_voice.replay(_series(200, 40), rise_db=8.0, sustain_seconds=0.4)

    assert len(fires) == 1
    assert fires[0] > 10.0  # after the baseline was established


def test_a_higher_threshold_fires_less():
    quiet = tune_voice.replay(_series(200, 40), rise_db=4.0, sustain_seconds=0.2)
    strict = tune_voice.replay(_series(200, 40), rise_db=25.0, sustain_seconds=0.2)

    assert len(quiet) >= len(strict)
    assert strict == []


def test_scoring_credits_a_firing_before_the_mark_not_after():
    """The buffer holds the preceding seconds, so early is a hit and late is not."""
    caught, _ = tune_voice.score(fires=[50.0], marks=[55.0])
    assert caught == 1

    missed, _ = tune_voice.score(fires=[70.0], marks=[55.0])
    assert missed == 0


def test_a_mark_needs_only_one_nearby_firing():
    caught, fires = tune_voice.score(fires=[50.0, 51.0, 52.0], marks=[55.0])

    assert caught == 1
    assert fires == 3  # precision is what makes the extra firings visible


def test_the_metronome_baseline_fires_at_a_fixed_rate():
    """A detector that cannot beat this has learned nothing about the audio."""
    beats = tune_voice.metronome(duration=120.0, every=30.0)

    assert beats == [30.0, 60.0, 90.0, 120.0]


def test_recall_is_undefined_without_marks_rather_than_zero():
    outcome = tune_voice.Outcome(
        rise_db=8.0, sustain_seconds=0.4, fires=3, caught=0, marks=0
    )

    assert math.isnan(outcome.recall)


def test_a_sweep_covers_every_combination():
    results = tune_voice.sweep(
        _series(200, 40), marks=[11.0],
        rises=(6.0, 10.0), sustains=(0.2, 0.4),
    )

    assert len(results) == 4
    assert {(r.rise_db, r.sustain_seconds) for r in results} == {
        (6.0, 0.2), (6.0, 0.4), (10.0, 0.2), (10.0, 0.4)
    }
