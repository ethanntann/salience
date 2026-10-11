"""Sweeping the input detector's thresholds against the player's own marks.

`InputBurstDetector` fires when the press rate rises over a rolling median. Its
current settings -- 2.5x the median with a 4/s floor -- came from one 24.8-minute
session with **four** marks, and that session's journal no longer exists. So the
one number anyone might quote for this detector is unreproducible.

This replays a recorded `input.tsv` through the detector at many thresholds, so
a session produces a comparison rather than an anecdote. Scoring and the
no-skill baseline are imported from `tune_voice` rather than rewritten: a metric
that differs between the two channels cannot be used to choose between them.

Recall is the number that matters, for the same reason it is in the audio
sweep. A firing only buys a clip cut from footage already on disk; a missed
moment is gone. Precision is reported so a detector that fires constantly is
visible, not because it should be maximised.

The occupancy channel is swept too, and its measured problem is stated here
rather than discovered again: across 23 windows of real play it read **100% in
every one**. It separates playing from sitting in a menu, not a fight from a
rotate. A sweep will say so numerically; if the best occupancy threshold cannot
beat the metronome, that is the answer and not a bug.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture.input_activity import ActivitySampler, InputBurstDetector
from pipeline_v2.capture.tune_voice import metronome, score


@dataclass(frozen=True)
class Outcome:
    """One threshold combination, scored."""

    channel: str
    multiple: float
    minimum: float
    fires: int
    caught: int
    marks: int

    @property
    def recall(self) -> float:
        """NaN, not zero, when there is nothing to recall.

        Zero would rank an unmeasurable sweep alongside a genuinely bad one.
        """
        return self.caught / self.marks if self.marks else math.nan

    @property
    def precision(self) -> float:
        return self.caught / self.fires if self.fires else math.nan


def replay(
    samples: list,
    *,
    multiple: float,
    minimum: float,
    channel: str = "presses",
    cooldown: float = 20.0,
) -> list[float]:
    """Run a recorded session through the detector and collect firing times.

    Offline replay rather than a live test because a threshold has to be chosen
    against a session that is already recorded, and a live comparison would
    require playing the same match twice.
    """
    detector = InputBurstDetector(
        multiple=multiple, minimum_rate=minimum, cooldown_seconds=cooldown
    )
    fires: list[float] = []
    for sample in samples:
        value = _channel_value(sample, channel)
        if value is None:
            continue
        fired = detector.observe(press_rate=value, now=sample.at)
        if fired is not None:
            fires.append(sample.at)
    return fires


def _channel_value(sample, channel: str) -> float | None:
    """The series being swept, or None where a journal never recorded it.

    Older journals predate the movement and occupancy columns and report
    `polls == 0`. Treating that as a measured zero would score those channels as
    silent for a session that simply never measured them.
    """
    if channel == "presses":
        return sample.rate
    if sample.polls == 0:
        return None
    # Scaled to the same units as the press rate so one floor means the same
    # thing on both, and a sweep over `minimum` is comparable between channels.
    scale = 1.0 / ActivitySampler.WINDOW_SECONDS
    if channel == "occupancy":
        return sample.occupancy * scale
    if channel == "activity":
        return sample.activity * scale
    raise ValueError(f"unknown channel: {channel}")


def sweep(
    samples: list,
    marks: list[float],
    *,
    channel: str = "presses",
    multiples: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0),
    minimums: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
) -> list[Outcome]:
    results: list[Outcome] = []
    for multiple in multiples:
        for minimum in minimums:
            fires = replay(
                samples, multiple=multiple, minimum=minimum, channel=channel
            )
            caught, total = score(fires, marks)
            results.append(
                Outcome(channel, multiple, minimum, total, caught, len(marks))
            )
    return results


def read_marks(path: Path) -> list[float]:
    """Positive marks only. F10 presses are negatives and are counted separately."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [float(v) for v in payload]
    return [float(v) for v in payload.get("keep_seconds", [])]


def read_dull(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return []
    return [float(v) for v in payload.get("dull_seconds", [])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep input detector thresholds against a session's marks."
    )
    parser.add_argument("session", type=Path, help="a recording directory")
    parser.add_argument(
        "--channel", default="presses",
        choices=("presses", "occupancy", "activity"),
        help="which series to sweep (default: presses)",
    )
    args = parser.parse_args(argv)

    samples = ActivitySampler.read_journal(args.session / "input.tsv")
    marks_file = args.session / "marks.json"
    if not samples:
        print(f"no input.tsv in {args.session}")
        return 1
    if not marks_file.is_file():
        print(f"no marks.json in {args.session} -- nothing to score against")
        return 1

    marks, dull = read_marks(marks_file), read_dull(marks_file)
    duration = samples[-1].at
    measured = sum(1 for s in samples if s.polls)
    print(f"{len(samples)} windows over {duration / 60:.1f} min; {len(marks)} keep, "
          f"{len(dull)} dull marks")
    if args.channel != "presses" and not measured:
        print(f"  this journal never measured '{args.channel}' -- "
              "re-record to sweep it")
        return 1

    beats = metronome(duration, every=30.0)
    caught, fired = score(beats, marks)
    baseline = caught / len(marks) if marks else math.nan
    print(f"no-skill baseline (a beat every 30s): {caught}/{len(marks)} "
          f"in {fired} firings, recall {baseline:.0%}")

    results = sweep(samples, marks, channel=args.channel)
    ranked = sorted(
        results,
        key=lambda o: (-(o.caught), o.fires),  # recall first, then fewer firings
    )
    print(f"\nsweep of '{args.channel}' (best first):")
    print("  multiple  floor   fires  caught  recall  precision")
    for outcome in ranked[:12]:
        print(f"  {outcome.multiple:8.1f}  {outcome.minimum:5.1f}  "
              f"{outcome.fires:5d}  {outcome.caught:6d}  "
              f"{outcome.recall:6.0%}  {outcome.precision:9.0%}")

    best = ranked[0]
    if not marks:
        print("\nno keep marks: recall is undefined, so nothing was chosen")
    elif best.caught / len(marks) <= baseline:
        print(f"\n{args.channel} did not beat firing blindly on a timer. "
              "On this session it has no demonstrated skill.")
    else:
        print(f"\nbest: {best.multiple}x over the median with a {best.minimum}/s "
              f"floor -- {best.caught}/{best.marks} marks in {best.fires} firings")
    if len(marks) < 10:
        print(f"{len(marks)} marks is too few to separate these; "
              "the ranking above is indicative only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
