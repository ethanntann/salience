"""Sweeping input thresholds against marks.

The detector's current settings came from four marks in a session whose journal
no longer exists, so the point of these is that a sweep reports honestly when it
cannot tell anything -- an unmeasured channel, no marks, or no skill over a
timer.
"""

from __future__ import annotations

import json
import math

import pytest

from pipeline_v2.capture import tune_input
from pipeline_v2.capture.input_activity import ActivitySample


def _session(quiet_windows=60, burst_windows=8, quiet=1, loud=6):
    """A quiet stretch then a burst, in the six-column format."""
    samples = [
        ActivitySample(i * 0.5, quiet, 1, 20, 30, 50)
        for i in range(1, quiet_windows)
    ]
    offset = quiet_windows * 0.5
    samples += [
        ActivitySample(offset + i * 0.5, loud, 4, 45, 50, 50)
        for i in range(1, burst_windows)
    ]
    return samples


def test_a_burst_is_found_in_a_recorded_session():
    fires = tune_input.replay(_session(), multiple=2.5, minimum=4.0)

    assert len(fires) == 1
    assert fires[0] > 25.0  # after the rolling baseline was established


def test_a_higher_multiple_fires_less():
    loose = tune_input.replay(_session(), multiple=1.5, minimum=2.0)
    strict = tune_input.replay(_session(), multiple=8.0, minimum=2.0)

    assert len(loose) >= len(strict)
    assert strict == []


def test_a_sweep_covers_every_combination():
    results = tune_input.sweep(
        _session(), marks=[31.0], multiples=(2.0, 3.0), minimums=(2.0, 4.0)
    )

    assert len(results) == 4
    assert {(r.multiple, r.minimum) for r in results} == {
        (2.0, 2.0), (2.0, 4.0), (3.0, 2.0), (3.0, 4.0)
    }


def test_recall_is_undefined_without_marks_rather_than_zero():
    """Zero would rank an unmeasurable sweep alongside a genuinely bad one."""
    outcome = tune_input.Outcome("presses", 2.5, 4.0, fires=3, caught=0, marks=0)

    assert math.isnan(outcome.recall)


def test_precision_is_undefined_when_nothing_fired():
    outcome = tune_input.Outcome("presses", 9.0, 9.0, fires=0, caught=0, marks=4)

    assert math.isnan(outcome.precision)


def test_an_unmeasured_channel_reads_as_absent_not_silent():
    """A three-column journal never measured occupancy; that is not a zero."""
    old = ActivitySample(0.5, 3, 2)

    assert tune_input._channel_value(old, "occupancy") is None
    assert tune_input._channel_value(old, "presses") == 6.0  # 3 per 0.5s window


def test_a_pinned_occupancy_channel_produces_no_burst():
    """Measured on real play: occupancy read 100% in all 23 windows.

    A channel with no dynamic range cannot show a rise over its own median, and
    the sweep should report that rather than appear to find something.
    """
    pinned = [ActivitySample(i * 0.5, 3, 2, 45, 50, 50) for i in range(1, 80)]

    assert tune_input.replay(pinned, multiple=2.5, minimum=4.0, channel="occupancy") == []


def test_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="unknown channel"):
        tune_input._channel_value(ActivitySample(0.5, 3, 2, 1, 1, 50), "loudness")


def test_marks_are_read_from_the_recorder_format(tmp_path):
    path = tmp_path / "marks.json"
    path.write_text(
        json.dumps({
            "marks": [{"at": 12.0, "kind": "keep"}, {"at": 30.0, "kind": "dull"}],
            "keep_seconds": [12.0],
            "dull_seconds": [30.0],
        }),
        encoding="utf-8",
    )

    assert tune_input.read_marks(path) == [12.0]
    assert tune_input.read_dull(path) == [30.0]


def test_a_plain_list_of_seconds_is_also_accepted(tmp_path):
    path = tmp_path / "marks.json"
    path.write_text("[12.0, 44.5]", encoding="utf-8")

    assert tune_input.read_marks(path) == [12.0, 44.5]
    assert tune_input.read_dull(path) == []


def test_the_cli_refuses_a_session_with_no_marks(tmp_path, capsys):
    session = tmp_path / "run"
    session.mkdir()
    (session / "input.tsv").write_text("0.5\t3\t2\t20\t30\t50\n", encoding="utf-8")

    assert tune_input.main([str(session)]) == 1
    assert "nothing to score against" in capsys.readouterr().out


def test_the_cli_refuses_to_sweep_a_channel_the_journal_lacks(tmp_path, capsys):
    session = tmp_path / "run"
    session.mkdir()
    (session / "input.tsv").write_text("0.5\t3\t2\n1.0\t4\t2\n", encoding="utf-8")
    (session / "marks.json").write_text('{"keep_seconds": [1.0]}', encoding="utf-8")

    assert tune_input.main([str(session), "--channel", "occupancy"]) == 1
    assert "never measured" in capsys.readouterr().out


def test_the_cli_reports_no_skill_when_a_timer_does_as_well(tmp_path, capsys):
    """A detector that cannot beat a metronome has learned nothing."""
    session = tmp_path / "run"
    session.mkdir()
    rows = [f"{i * 0.5}\t1\t1\t20\t30\t50" for i in range(1, 400)]
    (session / "input.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (session / "marks.json").write_text('{"keep_seconds": [30.0]}', encoding="utf-8")

    assert tune_input.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "no demonstrated skill" in out
    assert "too few" in out  # one mark cannot separate anything
