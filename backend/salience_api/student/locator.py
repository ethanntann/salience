from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


def timestamps_from_frame_scores(
    scores: list[float],
    timestamps: list[float],
    *,
    max_events: int,
    min_separation_sec: float = 0.75,
    min_score: float = 0.5,
    always_return_best: bool = False,
) -> list[float]:
    ranked = sorted(
        zip(scores, timestamps, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    chosen: list[float] = []
    for score, ts in ranked:
        if score < min_score:
            break
        if all(abs(ts - prev) >= min_separation_sec for prev in chosen):
            chosen.append(float(ts))
        if len(chosen) >= max_events:
            break
    if not chosen and always_return_best and ranked:
        _best_score, best_ts = ranked[0]
        chosen = [float(best_ts)]
    return sorted(chosen)


def build_locator_net() -> nn.Module:
    from salience_api.student.backbone import build_locator_net as _build_locator_net

    return _build_locator_net()
