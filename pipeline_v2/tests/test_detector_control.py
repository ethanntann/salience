"""Comparing detector precision against randomly timed clips.

The point of the control is that a precision figure alone cannot say whether a
detector works. These cover the ways that comparison can quietly mislead: an
interval that runs below zero, a missing control arm, and an arm that is worse
than random being reported as if it were merely unimpressive.
"""

from __future__ import annotations

import math

from pipeline_v2.capture import detector_control as dc


def test_an_interval_never_runs_below_zero():
    """A normal approximation gives negative lower bounds at these rates.

    Printing "-2% to 5%" reads as false precision rather than as uncertainty.
    """
    low, high = dc.wilson(0, 73)

    assert low == 0.0
    assert 0.0 < high < 0.1


def test_an_interval_never_runs_above_one():
    low, high = dc.wilson(73, 73)

    assert high <= 1.0  # lands a float epsilon below, so never clamped in practice
    assert low > 0.9


def test_an_interval_is_undefined_with_no_samples():
    low, high = dc.wilson(0, 0)

    assert math.isnan(low) and math.isnan(high)


def test_a_bigger_sample_gives_a_tighter_interval():
    narrow = dc.wilson(90, 900)
    wide = dc.wilson(9, 90)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_z_is_undefined_when_an_arm_is_empty():
    assert math.isnan(dc.two_proportion_z(0, 0, 9, 73))
    assert math.isnan(dc.two_proportion_z(9, 73, 0, 0))


def test_z_is_undefined_when_nothing_hit_in_either_arm():
    """Pooled rate of zero has no standard error; NaN beats a divide-by-zero."""
    assert math.isnan(dc.two_proportion_z(0, 40, 0, 40))


def test_z_is_positive_when_the_detector_beats_the_baseline():
    assert dc.two_proportion_z(20, 40, 5, 40) > 0


def test_z_is_negative_when_the_detector_is_worse_than_the_baseline():
    """Voice came in at 4.5% precision; if the baseline is higher, that is a
    finding to state, not a weak positive to round up."""
    assert dc.two_proportion_z(2, 44, 9, 73) < 0


def test_arms_are_split_by_filename_marker(tmp_path):
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("create table clips (clip_sha text, path text)")
    conn.execute(
        "create table descriptions (clip_sha text, payload text, prompt_ver text)"
    )
    rows = [
        ("a", "000-control.mkv", '{"outcome": "elimination"}'),
        ("b", "001-control.mkv", '{"outcome": "no_event"}'),
        ("c", "002-input_burst.mkv", '{"outcome": "knock"}'),
        ("d", "003-reaction.mkv", '{"outcome": "no_event"}'),
        ("e", "004-unknownkind.mkv", '{"outcome": "elimination"}'),
    ]
    for sha, name, payload in rows:
        conn.execute("insert into clips values (?, ?)", (sha, name))
        conn.execute("insert into descriptions values (?, ?, 'v')", (sha, payload))

    grouped = dc.arms(conn, "v")

    assert grouped["control (random)"] == [True, False]
    assert grouped["input_burst"] == [True]
    assert grouped["voice_reaction"] == [False]
    assert "004-unknownkind.mkv" not in grouped  # unrecognised arms are dropped


def test_a_malformed_payload_is_skipped_not_counted_as_a_miss(tmp_path):
    """Counting an unparseable row as "no event" would understate a real rate."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("create table clips (clip_sha text, path text)")
    conn.execute(
        "create table descriptions (clip_sha text, payload text, prompt_ver text)"
    )
    conn.execute("insert into clips values ('a', '000-control.mkv')")
    conn.execute("insert into descriptions values ('a', '{tor', 'v')")

    assert dc.arms(conn, "v") == {}


def test_the_report_refuses_to_compare_without_a_control(capsys):
    outcome = dc.report({"input_burst": [True, False, False]})

    assert outcome == 1
    assert "no control arm" in capsys.readouterr().out


def test_the_report_names_an_arm_that_is_worse_than_random(capsys):
    dc.report({
        "control (random)": [True] * 30 + [False] * 30,
        "voice_reaction": [True] * 2 + [False] * 58,
    })

    assert "WORSE than random" in capsys.readouterr().out


def test_the_report_states_its_own_limits(capsys):
    """Correlated samples and small counts must travel with the numbers."""
    dc.report({
        "control (random)": [True] * 9 + [False] * 64,
        "input_burst": [True] * 7 + [False] * 22,
    })
    out = capsys.readouterr().out

    assert "overlap" in out
    assert "correlated" in out
