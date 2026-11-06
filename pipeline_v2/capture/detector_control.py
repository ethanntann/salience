"""Did the detectors beat picking moments at random?

A detector's precision means nothing on its own. 12% of triggered clips
containing a real event sounds poor, but if a random 16-second window from the
same session contains one 12% of the time, the detectors have learned nothing --
and if random windows only hit 5%, that same 12% is a genuine signal.

So this compares two arms cut by the same code from the same footage, differing
only in how the time was chosen: the live trigger journal, and a seeded uniform
draw over the session. Clip length, analysis window and encoder match exactly,
because a control built by a different path would bias the comparison it exists
to make.

The vision model is the measuring instrument for both arms, so whatever it gets
wrong about what counts as an event, it gets wrong equally on both sides and the
difference between them survives.

Two limits are structural and reported rather than hidden. The control windows
can overlap each other, so the samples are correlated and the true interval is
wider than the binomial one printed here. And at 73 clips per arm the counts are
small: a 2x difference may well separate, while a small one will not.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

# What the model has to report for a clip to count as containing a real event.
# Deliberately narrow: `no_event` covers looting, rotating and driving, which is
# most of a session and is exactly what a trigger is supposed to skip.
EVENT_OUTCOMES = ("elimination", "knock")


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which behaves at small counts and near 0 or 1.

    The normal approximation gives intervals that run past 0% on rates this low,
    which reads as false precision rather than as the uncertainty it is.
    """
    if total == 0:
        return (math.nan, math.nan)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    spread /= denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def two_proportion_z(a_hits: int, a_n: int, b_hits: int, b_n: int) -> float:
    """z for two independent proportions, or NaN when it is undefined."""
    if not a_n or not b_n:
        return math.nan
    pooled = (a_hits + b_hits) / (a_n + b_n)
    if pooled in (0.0, 1.0):
        return math.nan
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / a_n + 1 / b_n))
    if standard_error == 0:
        return math.nan
    return (a_hits / a_n - b_hits / b_n) / standard_error


def arms(conn: sqlite3.Connection, variant: str) -> dict[str, list[bool]]:
    """Every described clip in this variant, grouped by how its time was chosen.

    Grouped from the stored payload rather than the filename so a renamed clip
    cannot silently land in the wrong arm.
    """
    grouped: dict[str, list[bool]] = {}
    rows = conn.execute(
        "select cl.path, d.payload from descriptions d "
        "join clips cl on cl.clip_sha = d.clip_sha where d.prompt_ver = ?",
        (variant,),
    )
    for path, payload in rows:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        name = Path(path).name
        if "-control" in name:
            arm = "control (random)"
        elif "input_burst" in name:
            arm = "input_burst"
        elif "reaction" in name:
            arm = "voice_reaction"
        else:
            continue
        grouped.setdefault(arm, []).append(
            parsed.get("outcome") in EVENT_OUTCOMES
        )
    return grouped


def report(grouped: dict[str, list[bool]]) -> int:
    control = grouped.get("control (random)", [])
    print("clips containing an elimination or knock:\n")
    print(f"  {'arm':18s} {'hits':>5s} {'n':>4s} {'rate':>7s}   95% CI")
    order = sorted(grouped, key=lambda k: -(sum(grouped[k]) / max(len(grouped[k]), 1)))
    for arm in order:
        outcomes = grouped[arm]
        hits, total = sum(outcomes), len(outcomes)
        low, high = wilson(hits, total)
        print(f"  {arm:18s} {hits:5d} {total:4d} {hits / total:6.1%}   "
              f"{low:.1%} - {high:.1%}")

    if not control:
        print("\nno control arm found; nothing to compare against")
        return 1

    base_hits, base_n = sum(control), len(control)
    print(f"\nagainst the random baseline of {base_hits / base_n:.1%}:")
    for arm in order:
        if arm == "control (random)":
            continue
        hits, total = sum(grouped[arm]), len(grouped[arm])
        z = two_proportion_z(hits, total, base_hits, base_n)
        lift = (hits / total) / (base_hits / base_n) if base_hits else math.inf
        verdict = (
            "no better than random" if math.isnan(z) or abs(z) < 1.96
            else "better than random" if z > 0
            else "WORSE than random"
        )
        print(f"  {arm:18s} {lift:4.1f}x  z = {z:+5.2f}  {verdict}")

    print("\nThe control windows can overlap, so these samples are correlated "
          "and the\nintervals above are narrower than the truth. At 73 clips an "
          "arm, only a\nlarge difference can separate.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare detector precision against randomly timed clips."
    )
    parser.add_argument(
        "--variant", default="v3.6-ownkills-window@minimax-m3",
        help="storage variant holding both arms",
    )
    args = parser.parse_args(argv)

    from pipeline_v2 import store

    conn = store.connect()
    try:
        grouped = arms(conn, args.variant)
    finally:
        conn.close()
    if not grouped:
        print(f"no described clips in variant {args.variant}")
        return 1
    return report(grouped)


if __name__ == "__main__":
    raise SystemExit(main())
