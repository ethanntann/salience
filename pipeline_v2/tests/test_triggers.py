"""Detection that is not tied to one game.

The Fortnite log is exact and free and does not generalise: Valorant writes no
equivalent, and no log finds the moment someone says something funny on a call.
These cover the split -- universal detectors everywhere, game adapters as a
bonus -- and the recall-first rule every detector inherits.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline_v2.capture.triggers import (
    GameEventDetector,
    Trigger,
    TriggerBus,
    VoiceReactionDetector,
    detectors_for,
    universal_detectors,
)


@dataclass
class FakeEvent:
    kind: str
    detail: str = ""


def _speak(detector, levels, start=0.0, step=0.1):
    """Feed a level series and return every trigger it produced."""
    fired, moment = [], start
    for level in levels:
        moment += step
        found = detector.observe(mic_db=level, now=moment)
        if found is not None:
            fired.append(found)
    return fired


def test_a_sustained_shout_fires():
    detector = VoiceReactionDetector()

    fired = _speak(detector, [-40.0] * 40 + [-28.0] * 8)

    assert len(fired) == 1
    assert fired[0].kind == "reaction"
    assert "+12.0dB" in fired[0].detail


def test_a_single_spike_does_not_fire():
    """A cough or a chair is not a reaction."""
    detector = VoiceReactionDetector()

    fired = _speak(detector, [-40.0] * 40 + [-20.0] + [-40.0] * 10)

    assert fired == []


def test_the_first_moments_of_a_session_can_still_fire():
    """`_last_fire = 0.0` once read as 'fired at time zero'.

    That silently suppressed every trigger during the first cooldown window --
    which is the start of every session.
    """
    detector = VoiceReactionDetector(cooldown_seconds=20.0)

    fired = _speak(detector, [-40.0] * 40 + [-28.0] * 8, start=0.0)

    assert len(fired) == 1
    assert fired[0].at < 20.0


def test_a_quiet_talker_and_a_loud_one_both_work():
    """The baseline is the player's own rolling median, not a fixed level."""
    quiet = _speak(VoiceReactionDetector(), [-55.0] * 40 + [-43.0] * 8)
    loud = _speak(VoiceReactionDetector(), [-25.0] * 40 + [-13.0] * 8)

    assert len(quiet) == 1 and len(loud) == 1


def test_no_microphone_means_no_voice_triggers():
    detector = VoiceReactionDetector()

    assert detector.observe(mic_db=None, now=1.0) is None
    assert detector.observe(now=1.0) is None


def test_the_cooldown_stops_one_long_reaction_becoming_many():
    detector = VoiceReactionDetector(cooldown_seconds=20.0)

    fired = _speak(detector, [-40.0] * 40 + [-28.0] * 100)

    assert len(fired) == 1


def test_an_exact_game_event_fires_with_full_confidence():
    detector = GameEventDetector()

    found = detector.observe(game_event=FakeEvent("death", "local player died"), now=5.0)

    assert found.kind == "death"
    assert found.confidence == 1.0


def test_uninteresting_game_events_are_ignored():
    """Match start is a boundary, not something worth clipping."""
    detector = GameEventDetector()

    assert detector.observe(game_event=FakeEvent("match_start"), now=1.0) is None
    assert detector.observe(game_event=FakeEvent("viewing"), now=1.0) is None


def test_every_game_gets_the_universal_detectors():
    """Valorant and an unknown title must still be able to clip."""
    for game in ("valorant", "apex", None, ""):
        names = [d.name for d in detectors_for(game)]
        assert "input_burst" in names
        assert "game_events" not in names


def test_voice_is_not_in_the_default_set():
    """Measured 2026-09-28: 2 of 44 voice clips held a real event, 4.5%, against
    8.2% for randomly-timed clips from the same session. The 44 calls it spent
    would have found ~3.6 events drawn uniformly, so it is dominated by chance.
    """
    assert "voice_reaction" not in [d.name for d in universal_detectors()]


def test_input_outranks_voice_so_it_cannot_be_crowded_out():
    """The bus keeps the highest confidence and shares one cooldown.

    At 0.4 against voice's 0.5, the channel measured at 24.1% lost every tie to
    the one measured at 4.5%, and then sat out the cooldown.
    """
    from pipeline_v2.capture.input_activity import InputBurstDetector

    burst = InputBurstDetector()
    # Inside the 60s rolling baseline, or the history is discarded as stale.
    for i in range(12):
        burst.observe(press_rate=1.0, now=float(i))
    fired = burst.observe(press_rate=40.0, now=12.0)

    assert fired is not None
    assert fired.confidence > 0.5  # the value VoiceReactionDetector emits


def test_fortnite_additionally_gets_its_adapter():
    names = [d.name for d in detectors_for("fortnite")]

    assert names == [d.name for d in universal_detectors()] + ["game_events"]


def test_an_exact_event_outranks_an_inference_at_the_same_moment():
    class Always(VoiceReactionDetector):
        def observe(self, **_):
            return Trigger(kind="reaction", at=5.0, source="voice", confidence=0.5)

    bus = TriggerBus([Always(), GameEventDetector()])

    found = bus.observe(game_event=FakeEvent("death"), now=5.0)

    assert found.source == "game_events"
    assert found.confidence == 1.0


def test_the_bus_cooldown_is_shared_across_detectors():
    """Two detectors describing one moment would return overlapping footage."""
    class Always(VoiceReactionDetector):
        def __init__(self, at):
            super().__init__()
            self._at = at

        def observe(self, **_):
            return Trigger(kind="reaction", at=self._at, source="voice")

    bus = TriggerBus([Always(10.0)], cooldown_seconds=20.0)
    assert bus.observe() is not None

    bus.detectors = [Always(15.0)]
    assert bus.observe() is None          # inside the cooldown

    bus.detectors = [Always(31.0)]
    assert bus.observe() is not None      # past it
