"""Tune the voice detector against recordings instead of against sessions.

The only audio trigger result this project has was produced on a single
19.4-minute session that was also used to pick the threshold, and it could not
beat a metronome. This exists so that never happens again: thresholds are swept
over a recorded mic track, scored against moments a human marked, and compared
with a detector that has no skill at all.

`sweep.py` does this for the old game-audio trigger. This is the same idea for
`VoiceReactionDetector`, reading the separate microphone track the recorder now
writes.

The number that matters is **recall**. A firing only buys permission to describe
thirty seconds already in the buffer; a missed reaction is gone. Precision is
reported so a detector that fires constantly is visible, not to be maximised.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture.levels import Level, extract_levels
from pipeline_v2.capture.triggers import VoiceReactionDetector


@dataclass(frozen=True)
class Outcome:
    rise_db: float
    sustain_seconds: float
    fires: int
    caught: int
    marks: int

    @property
    def recall(self) -> float:
        return self.caught / self.marks if self.marks else float("nan")

    @property
    def precision(self) -> float:
        return self.caught / self.fires if self.fires else float("nan")


def replay(
    levels: list[Level], rise_db: float, sustain_seconds: float,
    cooldown_seconds: float = 20.0,
) -> list[float]:
    """When the detector would have fired over this recorded track."""
    detector = VoiceReactionDetector(
        rise_db=rise_db,
        sustain_seconds=sustain_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    fires = []
    for level in levels:
        if detector.observe(mic_db=level.db, now=level.at) is not None:
            fires.append(level.at)
    return fires


def score(fires: list[float], marks: list[float], lead: float = 12.0,
          lag: float = 3.0) -> tuple[int, int]:
    """Match firings to marked moments.

    Asymmetric on purpose: the buffer holds the *preceding* seconds, so firing
    well before a mark still captures it, while firing after it does not.
    """
    caught = 0
    for mark in marks:
        if any(mark - lead <= fire <= mark + lag for fire in fires):
            caught += 1
    return caught, len(fires)


def metronome(duration: float, every: float) -> list[float]:
    """A detector with no skill, for comparison.

    Anything that cannot beat firing blindly on a timer at the same rate has
    learned nothing about the audio, whatever its recall looks like.
    """
    return [t * every for t in range(1, int(duration / every) + 1)]


def sweep(
    levels: list[Level], marks: list[float],
    rises: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    sustains: tuple[float, ...] = (0.2, 0.4, 0.8),
) -> list[Outcome]:
    results = []
    for rise in rises:
        for sustain in sustains:
            fires = replay(levels, rise, sustain)
            caught, count = score(fires, marks)
            results.append(
                Outcome(rise, sustain, count, caught, len(marks))
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="a recording with a mic track")
    parser.add_argument("--track", type=int, default=1, help="1 = mic by default")
    parser.add_argument(
        "--marks", type=Path, default=None,
        help="JSON list of seconds where something worth keeping happened",
    )
    args = parser.parse_args()

    levels = extract_levels(args.clip, track=args.track)
    if not levels:
        print("no audio on that track")
        return 1
    duration = levels[-1].at
    marks = json.loads(args.marks.read_text(encoding="utf-8")) if args.marks else []
    print(f"{len(levels)} windows, {duration:.1f}s, {len(marks)} marked moment(s)")

    if not marks:
        # Without marks recall is unknowable, so report firing rate only and say
        # so, rather than printing a number that looks like an accuracy.
        print("\nno marks given -- firing rate only, recall cannot be computed")
        print(f"{'rise':>6}{'sustain':>9}{'fires':>7}{'per min':>9}")
        for outcome in sweep(levels, []):
            rate = outcome.fires / (duration / 60) if duration else 0
            print(f"{outcome.rise_db:>6.1f}{outcome.sustain_seconds:>9.2f}"
                  f"{outcome.fires:>7}{rate:>9.1f}")
        return 0

    print(f"\n{'rise':>6}{'sustain':>9}{'fires':>7}{'recall':>9}{'prec':>8}")
    for outcome in sorted(sweep(levels, marks), key=lambda o: -o.recall):
        print(f"{outcome.rise_db:>6.1f}{outcome.sustain_seconds:>9.2f}"
              f"{outcome.fires:>7}{outcome.recall:>8.0%}{outcome.precision:>8.0%}")

    for every in (30.0, 60.0):
        caught, fires = score(metronome(duration, every), marks)
        recall = caught / len(marks) if marks else float("nan")
        print(f"\n  metronome every {every:.0f}s: {fires} fires, {recall:.0%} recall"
              "   <- beat this or the detector has no skill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
