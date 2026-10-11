"""Input activity: the only universal signal that works in silence.

Voice detection has nothing to say about someone playing solo without talking,
which is probably the most common session there is. A fight is mechanically
dense and downtime is not, and that is visible without knowing the game.
"""

from __future__ import annotations

from pathlib import Path

from pipeline_v2.capture.input_activity import (
    ActivitySample,
    ActivitySampler,
    InputBurstDetector,
    WATCHED,
)


class FakeUser:
    """Scriptable key state, so no real keyboard is involved."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = -1

    def advance(self):
        self.index += 1

    def GetAsyncKeyState(self, key):  # noqa: N802 - matches the win32 name
        held = self.frames[self.index] if self.index < len(self.frames) else ()
        return 0x8000 if key in held else 0

    def GetLastInputInfo(self, _ref):  # noqa: N802 - matches the win32 name
        return 0  # unavailable, so the presence channel stays out of the way


def test_only_game_relevant_keys_are_watched():
    """This must never resemble a keylogger: a fixed list, no text."""
    names = [name for _, name in WATCHED]

    assert "mouse_left" in names and "w" in names
    assert len(WATCHED) < 25
    # No letter keys beyond movement/build bindings -- nothing that spells words.
    assert "t" not in names and "y" not in names


def test_a_held_key_counts_once_not_every_poll():
    """Holding W to cross the map must not outrank a build fight."""
    sampler = ActivitySampler(started=0.0)
    user = FakeUser([(0x57,)] * 20)

    for index in range(20):
        user.advance()
        sampler.poll_once(now=index * 0.01, user=user)

    sampler.poll_once(now=1.0, user=user)

    assert sampler.samples[0].presses == 1


def test_repeated_taps_count_separately():
    sampler = ActivitySampler(started=0.0)
    # press, release, press, release...
    user = FakeUser([(0x01,), (), (0x01,), (), (0x01,), ()])

    for index in range(6):
        user.advance()
        sampler.poll_once(now=index * 0.01, user=user)
    sampler.poll_once(now=1.0, user=user)

    assert sampler.samples[0].presses == 3


def test_a_window_reports_how_many_distinct_keys_were_used():
    """A build fight uses many keys; spamming one is different."""
    sampler = ActivitySampler(started=0.0)
    user = FakeUser([(0x01,), (0x57,), (0x20,), ()])

    for index in range(4):
        user.advance()
        sampler.poll_once(now=index * 0.01, user=user)
    sampler.poll_once(now=1.0, user=user)

    assert sampler.samples[0].distinct == 3


def test_rate_is_comparable_across_window_sizes():
    sample = ActivitySample(at=1.0, presses=10, distinct=4)

    assert sample.rate == 10 / ActivitySampler.WINDOW_SECONDS


def test_samples_are_journalled_and_recoverable(tmp_path):
    journal = tmp_path / "input.tsv"
    sampler = ActivitySampler(journal=journal, started=0.0)
    sampler._append(ActivitySample(0.5, 12, 5))
    sampler._append(ActivitySample(1.0, 3, 2))

    found = ActivitySampler.read_journal(journal)

    assert [(s.at, s.presses, s.distinct) for s in found] == [
        (0.5, 12, 5), (1.0, 3, 2)
    ]


def test_a_corrupt_journal_line_is_skipped(tmp_path):
    journal = tmp_path / "input.tsv"
    journal.write_text("0.5\t12\t5\ngarbage\n1.0\tx\t2\n1.5\t4\t1\n", encoding="utf-8")

    assert [s.at for s in ActivitySampler.read_journal(journal)] == [0.5, 1.5]


def test_a_journal_failure_does_not_stop_sampling(tmp_path, monkeypatch):
    sampler = ActivitySampler(journal=tmp_path / "x" / "input.tsv", started=0.0)
    monkeypatch.setattr(
        Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
    )

    sampler._append(ActivitySample(0.5, 1, 1))  # must not raise


def _feed(detector, rates, step=0.5):
    fired, moment = [], 0.0
    for rate in rates:
        moment += step
        found = detector.observe(press_rate=rate, now=moment)
        if found is not None:
            fired.append(found)
    return fired


def test_a_burst_over_the_players_own_baseline_fires():
    detector = InputBurstDetector()

    fired = _feed(detector, [2.0] * 40 + [12.0] * 4)

    assert len(fired) == 1
    assert fired[0].kind == "input_burst"
    assert "12.0 presses/s" in fired[0].detail


def test_steady_activity_never_fires():
    """Someone who is always busy has no bursts, and that is correct."""
    assert _feed(InputBurstDetector(), [8.0] * 60) == []


def test_idle_noise_cannot_fire():
    """A median near zero would make any stray click a 'burst'."""
    detector = InputBurstDetector(minimum_rate=4.0)

    fired = _feed(detector, [0.0] * 40 + [2.0] * 4)

    assert fired == []


def test_a_busy_player_and_a_calm_one_both_work():
    """Relative to their own median, not an absolute rate."""
    calm = _feed(InputBurstDetector(), [1.0] * 40 + [10.0] * 4)
    busy = _feed(InputBurstDetector(), [6.0] * 40 + [20.0] * 4)

    assert len(calm) == 1 and len(busy) == 1


def test_no_input_data_means_no_trigger():
    assert InputBurstDetector().observe(press_rate=None, now=1.0) is None


def test_input_is_weaker_evidence_than_an_exact_game_event():
    detector = InputBurstDetector()

    fired = _feed(detector, [2.0] * 40 + [12.0] * 4)

    assert fired[0].confidence < 1.0


def test_a_journal_from_before_movement_was_counted_still_reads(tmp_path):
    """The one real session recorded so far has the three-column format.

    It must stay usable, and its unmeasured presence channel must not be
    mistaken for a measured idle one -- `polls == 0` is what tells them apart.
    """
    journal = tmp_path / "input.tsv"
    journal.write_text("0.5\t12\t5\n1.0\t3\t2\n", encoding="utf-8")

    old_format = ActivitySampler.read_journal(journal)

    assert [s.presses for s in old_format] == [12, 3]
    assert all(s.polls == 0 for s in old_format)
    assert all(s.activity == 0.0 for s in old_format)


def test_mouse_movement_registers_even_with_no_key_presses():
    """The blind spot this channel exists to close.

    Aiming is most of the mechanical work in a shooter and produces no key
    transitions at all, so on presses alone an aim duel reads as idle.
    """
    sampler = ActivitySampler(started=0.0)
    ticks = iter([100, 116, 132, 148, 164])  # a moving mouse advances the tick
    sampler._input_tick = lambda _user: next(ticks)
    user = FakeUser([()] * 5)

    sample = None
    for i in range(5):
        user.advance()
        sample = sampler.poll_once(now=(i + 1) * 0.1, user=user)

    assert sample is not None
    assert sample.presses == 0  # invisible to the presses channel
    assert sample.activity == 0.8  # 4 of 5 polls saw the tick advance


def test_a_still_mouse_and_no_keys_is_genuinely_idle():
    sampler = ActivitySampler(started=0.0)
    sampler._input_tick = lambda _user: 100  # the tick never moves
    user = FakeUser([()] * 5)

    sample = None
    for i in range(5):
        user.advance()
        sample = sampler.poll_once(now=(i + 1) * 0.1, user=user)

    assert sample is not None
    assert sample.activity == 0.0


def test_activity_reads_zero_rather_than_dividing_by_zero():
    assert ActivitySample(at=0.5, presses=0, distinct=0).activity == 0.0


def test_occupancy_sees_the_held_key_that_transitions_miss():
    """Measured mid-match: W held in 45% of polls, one transition counted.

    Walking across the map is the case that made 40% of a real session's windows
    read as zero input. Occupancy is what tells it apart from standing still.
    """
    sampler = ActivitySampler(started=0.0)
    sampler._input_tick = lambda _user: None
    user = FakeUser([(0x57,)] * 5)  # W held the whole window

    sample = None
    for i in range(5):
        user.advance()
        sample = sampler.poll_once(now=(i + 1) * 0.1, user=user)

    assert sample is not None
    assert sample.presses == 1  # one transition, as designed
    assert sample.occupancy == 1.0  # but busy the entire window
    assert sample.distinct == 1  # and mechanically simple: walking, not fighting


def test_an_empty_window_is_idle_on_every_channel():
    sampler = ActivitySampler(started=0.0)
    sampler._input_tick = lambda _user: 100
    user = FakeUser([()] * 5)

    sample = None
    for i in range(5):
        user.advance()
        sample = sampler.poll_once(now=(i + 1) * 0.1, user=user)

    assert sample is not None
    assert (sample.presses, sample.occupancy, sample.activity) == (0, 0.0, 0.0)


def test_a_build_fight_scores_high_on_every_channel():
    """The case worth clipping: dense, varied and continuous at once."""
    sampler = ActivitySampler(started=0.0)
    sampler._input_tick = lambda _user: None
    frames = [(0x57, 0x01), (0x41, 0x02), (0x57, 0x47), (0x20, 0x46), (0x51, 0x01)]
    user = FakeUser(frames)

    sample = None
    for i in range(5):
        user.advance()
        sample = sampler.poll_once(now=(i + 1) * 0.1, user=user)

    assert sample is not None
    assert sample.presses > 5
    assert sample.distinct > 4
    assert sample.occupancy == 1.0


def test_the_five_column_journal_format_still_reads(tmp_path):
    """Written briefly before occupancy existed; occupancy reads as unmeasured."""
    journal = tmp_path / "input.tsv"
    journal.write_text("0.5\t12\t5\t30\t50\n", encoding="utf-8")

    [sample] = ActivitySampler.read_journal(journal)

    assert (sample.presses, sample.active_polls, sample.polls) == (12, 30, 50)
    assert sample.held_polls == 0


def test_the_six_column_journal_round_trips(tmp_path):
    journal = tmp_path / "input.tsv"
    sampler = ActivitySampler(journal=journal, started=0.0)
    sampler._append(ActivitySample(0.5, 12, 5, 30, 44, 50))

    [sample] = ActivitySampler.read_journal(journal)

    assert sample.held_polls == 44
    assert sample.occupancy == 0.88
