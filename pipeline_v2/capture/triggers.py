"""What decides that the last thirty seconds were worth keeping.

The Fortnite log is exact and free, and it is also a dead end as an
architecture: Valorant does not write one, and no amount of log parsing finds
the moment a friend says something funny on a call. A product that only works
for one game because of one game's log file is a Fortnite tool, not a clipping
tool.

So detection is split in two:

**Universal detectors** work in any game and are the floor the product stands
on. They observe the player and the machine, never the game. As measured, that
floor is input bursts alone: voice reaction did not beat randomly-timed clips
and is out of the default set (see `universal_detectors`).

**Game adapters** are a bonus layer. Where a title happens to expose exact
events, an adapter reports them and precision improves; where it does not,
nothing breaks and the universal detectors carry the session alone.

One rule governs every detector, inherited from `sweep.py`: a firing is not a
commitment to keep anything, it is permission to spend real work deciding. The
recorder holds a rolling buffer, so a missed moment is gone forever while a
false firing costs one description. **Optimise recall.**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trigger:
    """A reason to save the buffer."""

    kind: str
    at: float  # time.monotonic()
    source: str  # which detector fired
    confidence: float = 0.5  # only for ordering when several fire at once
    detail: str = ""


class Detector:
    """Something that watches a stream and occasionally says 'now'.

    Deliberately tiny. A detector sees one kind of evidence and answers one
    question; combining them is `TriggerBus`'s job, not theirs.
    """

    name = "detector"

    def observe(self, **signals) -> Trigger | None:
        raise NotImplementedError


@dataclass
class VoiceReactionDetector(Detector):
    """Fires when the microphone gets suddenly, sustainedly louder.

    This is the universal one, and the one the corpus cannot yet judge. The
    earlier finding that voice did not predict a keep came from a feature that
    asked whether the single *loudest* moment of a clip was a voice -- in a
    gunfight a shotgun beats a person shouting, so it was structurally blind to
    exactly the clips where someone reacts. Separate mic and game tracks now
    exist, which is what makes a fair test possible.

    Baseline is a rolling median of the player's own speech level, so a quiet
    talker and a loud one both work without configuration.
    """

    name = "voice_reaction"
    # Swept against a real 24.8-minute session with 4 marks: rise 4.0 /
    # sustain 0.40 caught 4/4 in 31 firings, where a 60s metronome caught 2/4 in
    # 24. The previous guess of 8.0 caught only 1/4, so these are measured
    # rather than chosen -- but on four marks (95% CI 51-100%), so they are a
    # starting point, not a validated setting.
    rise_db: float = 4.0
    sustain_seconds: float = 0.4
    cooldown_seconds: float = 20.0
    baseline_seconds: float = 60.0
    _levels: list[tuple[float, float]] = field(default_factory=list)
    _above_since: float | None = None
    _last_fire: float | None = None

    def observe(self, *, mic_db: float | None = None, now: float | None = None,
                **_) -> Trigger | None:
        if mic_db is None:
            return None
        moment = now if now is not None else time.monotonic()
        self._levels.append((moment, mic_db))
        cutoff = moment - self.baseline_seconds
        self._levels = [(t, v) for t, v in self._levels if t >= cutoff]
        if len(self._levels) < 8:
            return None  # baseline not established; do not fire on the first noise

        values = sorted(v for _, v in self._levels)
        median = values[len(values) // 2]
        if mic_db - median < self.rise_db:
            self._above_since = None
            return None

        # Sustained, not a single spike: a cough or a chair is not a reaction.
        if self._above_since is None:
            self._above_since = moment
            return None
        if moment - self._above_since < self.sustain_seconds:
            return None
        if self._last_fire is not None and (
            moment - self._last_fire < self.cooldown_seconds
        ):
            return None

        self._last_fire = moment
        self._above_since = None
        return Trigger(
            kind="reaction",
            at=moment,
            source=self.name,
            confidence=0.5,
            detail=f"mic {mic_db - median:+.1f}dB over rolling median",
        )


@dataclass
class GameEventDetector(Detector):
    """Adapter layer: exact events a specific title happens to expose.

    Fortnite writes match boundaries, the local player's death, and who is
    being spectated to a log file. Valorant writes nothing comparable. This
    class is where that asymmetry lives, so nothing above it has to care.
    """

    name = "game_events"
    # Match end is included because it means the player died or won; either way
    # the preceding thirty seconds are the end of something.
    interesting: tuple[str, ...] = ("death", "match_end")

    def observe(self, *, game_event=None, now: float | None = None,
                **_) -> Trigger | None:
        if game_event is None or game_event.kind not in self.interesting:
            return None
        return Trigger(
            kind=game_event.kind,
            at=now if now is not None else time.monotonic(),
            source=self.name,
            # Exact, from the game itself -- not an inference like the others.
            confidence=1.0,
            detail=game_event.detail,
        )


class TriggerBus:
    """Runs every detector and applies one shared cooldown.

    The cooldown is here rather than in each detector because the cost being
    protected is shared: two detectors firing three seconds apart describe the
    same moment, and the buffer would hand back overlapping footage -- the
    duplicate-coverage problem `engagements.py` exists to clean up afterwards.
    Better not to create it.
    """

    def __init__(self, detectors: list[Detector], cooldown_seconds: float = 20.0):
        self.detectors = detectors
        self.cooldown_seconds = cooldown_seconds
        self.fired: list[Trigger] = []
        self._last: float | None = None

    def observe(self, **signals) -> Trigger | None:
        candidates = [d.observe(**signals) for d in self.detectors]
        found = [t for t in candidates if t is not None]
        if not found:
            return None
        # An exact game event outranks an inferred one for the same moment.
        best = max(found, key=lambda t: t.confidence)
        if self._last is not None and (
            best.at - self._last < self.cooldown_seconds
        ):
            return None
        self._last = best.at
        self.fired.append(best)
        return best


def universal_detectors() -> list[Detector]:
    """What works with no knowledge of the game at all.

    Input bursts only, as of the 2026-09-28 control. Measured on a 31-minute
    session against 73 randomly-timed clips cut by the same code:

    | arm | contained a real event |
    | --- | --- |
    | input burst | 7 of 29, **24.1%** (2.9x random, z = +2.17) |
    | random windows | 6 of 73, 8.2% |
    | voice reaction | 2 of 44, 4.5% (0.6x random, z = -0.76) |

    Voice is not merely weak, it is **dominated by choosing times at random**:
    the 44 calls it spent would have yielded about 3.6 events drawn uniformly,
    against the 2 it found. It did find those 2 at moments input bursts missed,
    so it adds recall -- just less recall per call than a coin flip, and recall
    is what this file optimises.

    It was also crowding out the better detector. The bus keeps the highest
    confidence when several fire together, voice was 0.5 against input's 0.4,
    and the shared cooldown then blocked input for the next twenty seconds. So
    the weaker channel outranked the stronger one on a number that was a guess.

    `VoiceReactionDetector` is kept and still tunable: its threshold was set
    from four marks, so the *threshold* is as likely to be at fault as the
    channel. Re-enable it by passing it explicitly once a sweep justifies one.
    """
    from pipeline_v2.capture.input_activity import InputBurstDetector

    return [InputBurstDetector()]


def detectors_for(game: str | None = None) -> list[Detector]:
    """Universal detectors, plus an adapter when one exists for this title."""
    detectors = universal_detectors()
    if (game or "").lower() == "fortnite":
        detectors.append(GameEventDetector())
    return detectors
