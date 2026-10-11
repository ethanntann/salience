"""Prices belong to a model, and a model switch must not silently keep them.

kimi-k2p6 was undeployed and minimax-m3 replaced it; the three prices in `.env`
stayed kimi's, so every reported cost and the budget guard itself described a
bill nobody was being sent.
"""

from __future__ import annotations

import dataclasses

from pipeline_v2 import config, vlm


def _cfg(**changes):
    base = config.load_config()
    return dataclasses.replace(
        base,
        model="accounts/fireworks/models/minimax-m3",
        input_price_per_mtok=0.95,
        output_price_per_mtok=4.0,
        **changes,
    )


def test_prices_for_another_model_are_flagged():
    problem = vlm.price_mismatch(_cfg(price_model="kimi-k2p6"))

    assert problem is not None
    assert "kimi-k2p6" in problem and "minimax-m3" in problem


def test_unlabelled_prices_are_flagged():
    """An unlabelled price list is how the mismatch happened in the first place."""
    assert "V2_PRICE_MODEL" in vlm.price_mismatch(_cfg(price_model=""))


def test_prices_for_this_model_pass():
    assert vlm.price_mismatch(_cfg(price_model="minimax-m3")) is None


def test_a_full_model_path_matches_its_short_tag():
    path = "accounts/fireworks/models/minimax-m3"
    assert vlm.price_mismatch(_cfg(price_model=path)) is None


def test_pricing_switched_off_is_not_a_mismatch():
    """At zero the run reports token counts, which are never wrong."""
    cfg = dataclasses.replace(
        _cfg(price_model=""), input_price_per_mtok=0.0, output_price_per_mtok=0.0
    )
    assert vlm.price_mismatch(cfg) is None
