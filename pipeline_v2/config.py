"""Runtime configuration. Defaults are shared by CLI, storage, and review.
Dollar budgets are recorded-spend thresholds, not prepaid hard caps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

from pipeline_v2 import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / ".data"
RAW_DIR = DATA_DIR / "raw"
FRAME_DIR = DATA_DIR / "frames"
CACHE_DB = DATA_DIR / "cache.sqlite"

DEFAULT_PROMPT_VERSION = schema.DEFAULT_PROMPT_VERSION

# The model that produced every stored description. A prompt version identifies
# the question asked; it says nothing about who answered it. Describing at an
# existing version with a different model would overwrite the corpus in place,
# so `prompt.storage_variant` appends a model tag whenever they differ.
CORPUS_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_RANK_MODE = "embedding"


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    base_url: str
    # Prices are provider- and model-specific and change; they are configuration,
    # not knowledge. Set them from the current pricing page. Left at 0 the run
    # reports token counts returned by the provider.
    input_price_per_mtok: float
    output_price_per_mtok: float
    budget_usd: float
    # Providers bill a re-sent prompt prefix at a reduced rate. Only the text
    # prefix repeats here -- the frames differ per clip -- so this trims a
    # little, and leaving it at 0 just prices everything at full.
    cached_input_price_per_mtok: float = 0.0
    # Which model the three prices above were copied for. Prices are per model,
    # and switching `FIREWORKS_MODEL` without updating them silently prices one
    # model's tokens at another's rates -- which happened when kimi-k2p6 was
    # undeployed and minimax-m3 replaced it, so every spend figure and the budget
    # guard itself described a bill nobody was being sent.
    price_model: str = ""
    # Sent as `reasoning_effort`. "none" stops a reasoning model spending the
    # whole output budget thinking and never reaching the JSON. Some models
    # are thinking-only and reject it outright, so None omits the field and
    # `max_tokens` has to leave room for the thinking as well as the answer.
    reasoning_effort: str | None = "none"
    max_tokens: int = 600
    # Description variants remain queryable after changing the current default.
    prompt_version: str = DEFAULT_PROMPT_VERSION

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def load_config() -> Config:
    return Config(
        api_key=os.environ.get("FIREWORKS_API_KEY", ""),
        model=os.environ.get(
            "FIREWORKS_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
        ),
        base_url=os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        ),
        input_price_per_mtok=float(os.environ.get("V2_INPUT_PRICE_PER_MTOK", "0") or 0),
        output_price_per_mtok=float(
            os.environ.get("V2_OUTPUT_PRICE_PER_MTOK", "0") or 0
        ),
        cached_input_price_per_mtok=float(
            os.environ.get("V2_CACHED_INPUT_PRICE_PER_MTOK", "0") or 0
        ),
        price_model=os.environ.get("V2_PRICE_MODEL", ""),
        reasoning_effort=os.environ.get("V2_REASONING_EFFORT", "none") or None,
        budget_usd=float(os.environ.get("V2_BUDGET_USD", "12") or 12),
        # Retained research variants; the current default is v3.3.
        prompt_version=schema.ABLATIONS.get(
            os.environ.get("V2_ABLATE", ""), DEFAULT_PROMPT_VERSION
        ),
    )


# Conservative estimate from the provider's published model card (2026-09-07).
# It listed conflicting $0.40/$1.60 and $0.50/$3.00 pairs; the higher one is used
# so a run never under-reports. This is an estimate, never the account invoice.
#
# It lived inline in three study modules and not in `index`, so the command that
# describes the whole library was the one recording every call as $0 -- spend
# under-reported and V2_BUDGET_USD unable to stop anything. Set the real prices
# in the environment to override.
_ESTIMATED_PRICES = {
    # model id: (input, output, cached input) per million tokens
    "accounts/fireworks/models/qwen3p7-plus": (0.5, 3.0, 0.0),
    "accounts/fireworks/models/qwen3p8-max": (2.0, 6.0, 0.25),
}


# Models that will not answer without thinking first. Two shapes of this:
# glm-5p3-flash rejects `reasoning_effort="none"` with a 400, while
# muse-glimmer-30b accepts the parameter, ignores it, and spends the entire
# output budget on `reasoning_content` leaving `content` null -- ten calls, zero
# usable results, full price paid. Both need the parameter dropped and enough
# output ceiling for the answer to follow the thinking.
_THINKING_ONLY = {
    "accounts/fireworks/models/glm-5p3-flash",
    "accounts/fireworks/models/glm-5p3",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/muse-glimmer-30b",
}
_THINKING_MAX_TOKENS = 3000


def with_model_defaults(cfg: Config) -> Config:
    """Apply per-model request quirks that would otherwise 400."""
    if cfg.model not in _THINKING_ONLY:
        return cfg
    return replace(cfg, reasoning_effort=None, max_tokens=_THINKING_MAX_TOKENS)


def resolve(cfg: Config) -> Config:
    """Everything model-specific: published prices, plus request quirks."""
    return with_model_defaults(with_estimated_prices(cfg))


def with_estimated_prices(cfg: Config) -> Config:
    """Fill in published prices when none are configured, so spend is recorded."""
    if cfg.input_price_per_mtok or cfg.output_price_per_mtok:
        return cfg
    prices = _ESTIMATED_PRICES.get(cfg.model)
    if prices is None:
        return cfg
    return replace(
        cfg,
        input_price_per_mtok=prices[0],
        output_price_per_mtok=prices[1],
        cached_input_price_per_mtok=cfg.cached_input_price_per_mtok or prices[2],
    )


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, FRAME_DIR):
        path.mkdir(parents=True, exist_ok=True)
