from __future__ import annotations

from math import sqrt


def wilson_interval(
    successes: int, total: int, *, z: float = 1.96
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    phat = successes / total
    denominator = 1.0 + (z * z) / total
    center = phat + (z * z) / (2.0 * total)
    margin = z * sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return max(0.0, lower), min(1.0, upper)


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
