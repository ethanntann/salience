"""Replay a recorded session against many thresholds and both detectors.

The point of the level log is that tuning no longer costs a session per value.
The triggers are deterministic in their input, so feeding them a recorded level
history reproduces exactly what they would have done live -- and the hand-pressed
marks in the same log say what they *should* have done.

Because only the raw level series is stored, detectors invented later can be
tested against sessions already played. That is what let the sustained detector
be evaluated on session-1 without recording anything new.

The number to optimise is **recall**, not accuracy. The ring buffer holds 30
seconds, so a firing is not a commitment to keep anything; it is permission to
spend real work deciding. A missed event is gone forever, a false firing costs
one classifier pass. Tune accordingly.

A metronome baseline is printed alongside: a detector that cannot beat firing
blindly on a timer has no skill, whatever its recall looks like. Session-1's
peak detector could not.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline_v2.capture.trigger import (  # noqa: E402
    LevelTrigger,
    SustainedConfig,
    SustainedTrigger,
    TriggerConfig,
)


@dataclass
class Sample:
    t: float
    level: float
    marks: tuple[str, ...]


def load(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event = row.get("event", "")
            # Session-1 logged bare "MARK"; later logs label the key pressed.
            # Reading both keeps early sessions usable as the format evolves.
            marks = tuple(
                part.split(":", 1)[1] if ":" in part else "clip"
                for part in event.split("+")
                if part.startswith("MARK")
            )
            samples.append(
                Sample(
                    t=float(row["elapsed_s"]),
                    level=float(row["level_db"]),
                    marks=marks,
                )
            )
    return samples


def replay(
    samples: list[Sample], detector: str, threshold_db: float, cooldown_s: float
) -> list[float]:
    """Times at which the detector would fire. Uses the shipped trigger classes
    rather than a reimplementation -- a copy here would drift from real logic and
    quietly invalidate every sweep."""
    if detector == "sustained":
        trigger = SustainedTrigger(
            SustainedConfig(threshold_db=threshold_db, cooldown_seconds=cooldown_s)
        )
    else:
        trigger = LevelTrigger(
            TriggerConfig(threshold_db=threshold_db, cooldown_seconds=cooldown_s)
        )
    return [s.t for s in samples if trigger.observe(s.level, now=s.t)]


def score(fires: list[float], marks: list[float], lead: float, lag: float):
    """Match firings to marks.

    Asymmetric window, deliberately: you press the key *after* recognising the
    moment, while the trigger fires *during* the noise that caused it. So a
    firing well before a mark is a hit; one well after it is not.
    """
    unmatched = list(fires)
    caught = 0
    for mark in marks:
        for fire in unmatched:
            if mark - lead <= fire <= mark + lag:
                unmatched.remove(fire)
                caught += 1
                break
    recall = caught / len(marks) if marks else float("nan")
    precision = (len(fires) - len(unmatched)) / len(fires) if fires else float("nan")
    return caught, recall, precision, len(unmatched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--thresholds",
        default="6,8,10,12,14,16,18",
        help="comma-separated dB values to try",
    )
    parser.add_argument("--detectors", default="peak,sustained")
    parser.add_argument(
        "--mark-type",
        default="clip",
        help="which marks count as ground truth: clip, kill, or any",
    )
    parser.add_argument("--cooldown", type=float, default=25.0)
    parser.add_argument(
        "--lead",
        type=float,
        default=20.0,
        help="seconds a firing may precede a mark and still count",
    )
    parser.add_argument(
        "--lag",
        type=float,
        default=5.0,
        help="seconds a firing may follow a mark and still count",
    )
    args = parser.parse_args()

    samples = load(args.log)
    if not samples:
        raise SystemExit(f"{args.log} has no samples")

    def wanted(sample: Sample) -> bool:
        if args.mark_type == "any":
            return bool(sample.marks)
        return args.mark_type in sample.marks

    marks = [s.t for s in samples if wanted(s)]
    counts: dict[str, int] = {}
    for s in samples:
        for label in s.marks:
            counts[label] = counts.get(label, 0) + 1
    minutes = (samples[-1].t - samples[0].t) / 60.0

    tally = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "none"
    print(f"{args.log.name}: {len(samples)} samples, {minutes:.1f} min, marks: {tally}")
    print(f"scoring against: {args.mark_type} ({len(marks)} marks)")

    if len(marks) < 10:
        print(
            f"\nWARNING: {len(marks)} marks is too few to measure recall. "
            "Numbers below are indicative at best; aim for 15-20."
        )

    # Null model. Any detector that fails to beat this is not detecting.
    if marks:
        span = samples[-1].t
        beat = [t for t in range(20, int(span), int(args.cooldown))]
        caught, recall, _, _ = score(beat, marks, args.lead, args.lag)
        print(
            f"\nmetronome baseline (fires every {args.cooldown:.0f}s): "
            f"{len(beat)} fires, catches {caught}/{len(marks)} "
            f"(recall {recall:.2f}) -- beat this or the signal is doing nothing"
        )

    for detector in args.detectors.split(","):
        print(f"\n=== {detector} ===")
        print(
            f"{'thresh':>7} {'fires':>6} {'/hour':>7} {'caught':>7} "
            f"{'recall':>7} {'precis':>7} {'false':>6}"
        )
        for raw in args.thresholds.split(","):
            threshold = float(raw)
            fires = replay(samples, detector, threshold, args.cooldown)
            caught, recall, precision, false_fires = score(
                fires, marks, args.lead, args.lag
            )
            rate = len(fires) / minutes * 60 if minutes > 0 else 0.0
            print(
                f"{threshold:7.1f} {len(fires):6d} {rate:7.1f} {caught:7d} "
                f"{recall:7.2f} {precision:7.2f} {false_fires:6d}"
            )

    if marks:
        print(
            "\nPrefer recall: a missed event is unrecoverable, a false firing "
            "costs one classifier pass against a 30s buffer."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
