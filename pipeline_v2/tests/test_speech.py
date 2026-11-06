"""Speech content, split into categories that behave differently.

Measured on 334 labelled clips: strict excitement 42.9% kept, silent 36.3%,
baseline 31.4%, filler-only 20.5%. The direction reverses with the words, which
is why embedding a whole transcript made ranking worse.
"""

from __future__ import annotations

from pipeline_v2.capture.speech import classify, would_gate


def test_genuine_excitement_is_recognised():
    signal = classify("It was a 3v1. Oh my god. Oh my god.")

    assert "oh my god" in signal.excitement
    assert signal.reacted


def test_conversational_filler_is_not_excitement():
    """"bro"/"damn" clips were kept 20.5% of the time against a 31.4% baseline.

    Counting them as reactions is what made the loose pattern land *below*
    baseline, so the categories must stay separate.
    """
    signal = classify("Three weeks as well, you better enjoy that bro")

    assert signal.excitement == ()
    assert "bro" in signal.filler
    assert not signal.reacted


def test_laughter_counts_as_a_reaction():
    signal = classify("Oh. Oh. Hehe. Hehe.")

    assert signal.laughter
    assert signal.reacted


def test_verbalised_disinterest_is_captured():
    """A direct negative label from the player: those clips were kept 0%."""
    signal = classify("This is a whole lot of bad video bro. It's so boring.")

    assert "boring" in signal.disinterest


def test_silence_produces_no_signal_at_all():
    for text in ("", "   ", None):
        signal = classify(text)
        assert not signal.reacted
        assert signal.words == 0
        assert signal.features == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_opposing_effects_stay_in_separate_features():
    """One merged 'speech' number would let them cancel out.

    Excitement sits above baseline and filler below it, so a single score
    averages away the only useful thing in the signal.
    """
    both = classify("oh my god bro")

    assert both.features[0] == 1.0  # excitement
    assert both.features[3] == 1.0  # filler
    assert len(both.features) == 5


def test_word_count_is_bounded():
    """A long unrelated conversation must not dominate the feature vector."""
    signal = classify(" ".join(["word"] * 500))

    assert signal.features[4] == 1.0


def test_the_gate_only_opens_on_a_rise_over_baseline():
    """Whisper must not run continuously; it competes with the game for CPU."""
    assert would_gate(mic_db=-25.0, baseline_db=-40.0) is True
    assert would_gate(mic_db=-38.0, baseline_db=-40.0) is False
    # A loud talker with a loud baseline should not fire constantly.
    assert would_gate(mic_db=-20.0, baseline_db=-22.0) is False


def test_the_gate_threshold_is_adjustable():
    assert would_gate(-30.0, -40.0, rise_db=15.0) is False
    assert would_gate(-30.0, -40.0, rise_db=5.0) is True
