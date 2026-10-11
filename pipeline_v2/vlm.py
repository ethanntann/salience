"""Fireworks client.

Differs from v1's teacher client in two ways that matter here: it returns token
usage (M1 exists to learn the real cost per clip) and it never calls out when a
cached response for the same clip and prompt version already exists.
"""

from __future__ import annotations

import json
import sys
import re
from dataclasses import dataclass
from pathlib import Path
import time
import urllib.error
import urllib.request

from pipeline_v2 import cache
from pipeline_v2.config import Config
from pipeline_v2.facts import ClipFacts
from pipeline_v2.prompt import SYSTEM, build_content
from pipeline_v2.schema import ClipDescription, VlmResult


class BudgetExceeded(RuntimeError):
    """A call limit or already-recorded spend threshold prevents another call."""


@dataclass
class CallBudget:
    """Shared per-invocation allowance; reserve before an ambiguous network attempt."""

    limit: int
    attempted: int = 0

    def reserve(self):
        if self.attempted >= self.limit:
            raise BudgetExceeded("shared call limit reached")
        self.attempted += 1


# A bare word where a string belongs, inside an array: ["a", b, "c"]. Observed
# from qwen3p8-max, which returned one mechanics tag unquoted and took a whole
# 50-clip run down with it. Repair is attempted ONLY after a strict parse fails,
# so a well-formed response is never rewritten, and the raw text is already on
# disk either way. `true`, `false` and `null` are real JSON literals and stay.
_BARE_WORD = re.compile(
    r"(?<=[\[,])(\s*)(?!true\b|false\b|null\b)([A-Za-z_][A-Za-z0-9_]*)(\s*)(?=[,\]])"
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    body = text[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        repaired = _BARE_WORD.sub(r'\1"\2"\3', body)
        if repaired == body:
            raise
        return json.loads(repaired)


_WARNED_PRICES = False


def price_mismatch(cfg: Config) -> str | None:
    """Why the configured prices may not belong to the configured model, or None.

    Unset counts as a mismatch: an unlabelled price list is exactly how one
    model's rates came to be applied to another's tokens.
    """
    current = cfg.model.rsplit("/", 1)[-1]
    priced = cfg.price_model.rsplit("/", 1)[-1]
    if not (cfg.input_price_per_mtok or cfg.output_price_per_mtok):
        return None  # pricing off entirely; token counts are reported honestly
    if not priced:
        return f"prices are not labelled with a model (set V2_PRICE_MODEL={current})"
    if priced != current:
        return (f"prices are for {priced} but the model is {current}; costs and "
                "the budget guard are not your bill")
    return None


def _warn_prices_once(cfg: Config) -> None:
    global _WARNED_PRICES
    problem = price_mismatch(cfg)
    if problem and not _WARNED_PRICES:
        _WARNED_PRICES = True
        print(f"WARNING: {problem}", file=sys.stderr)


def _cost(
    cfg: Config, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float:
    """Estimated dollars for one call, crediting any prefix the provider cached.

    `cached_tokens` is a subset of `input_tokens`, so it is billed at the cheaper
    rate and the remainder at full. When the provider reports nothing, this is
    identical to pricing every input token at full rate -- an upper bound, which
    is the safe direction for a budget guard.
    """
    cached = max(0, min(cached_tokens, input_tokens))
    fresh = input_tokens - cached
    return (
        fresh / 1_000_000 * cfg.input_price_per_mtok
        + cached / 1_000_000 * cfg.cached_input_price_per_mtok
        + output_tokens / 1_000_000 * cfg.output_price_per_mtok
    )


def _request_payload(cfg: Config, content: list[dict]) -> dict:
    """The exact request body. Also what the cache key is built from, so the two
    can never describe different calls."""
    payload = {
        "model": cfg.model,
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
    }
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort
    return payload


def _post(cfg: Config, content: list[dict]) -> dict:
    body = json.dumps(_request_payload(cfg, content)).encode("utf-8")
    for attempt in range(4):
        request = urllib.request.Request(
            f"{cfg.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Only retry an explicit rate rejection. A server error may follow
            # completed inference; replaying it can bill the same clip twice.
            retryable = exc.code == 429
            if not retryable or attempt == 3:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"fireworks request failed: {exc.code} {detail}")
            time.sleep(min(30.0, 3 ** (attempt + 1)))
    raise RuntimeError("unreachable")


def describe(
    cfg: Config,
    clip: Path,
    clip_facts: ClipFacts,
    frame_paths: list[Path],
    *,
    allow_call: bool = True,
) -> VlmResult:
    response, cached = request_raw(
        cfg, clip, build_content(clip_facts, frame_paths), allow_call=allow_call
    )
    payload = _extract_json(response.raw)
    # request_raw persists the whole envelope when `content` is not a string.
    # Surfacing that plainly beats a pydantic "summary: field required" on a dict
    # the caller never sent.
    if "choices" in payload and "summary" not in payload:
        thought = (
            ((payload.get("choices") or [{}])[0].get("message") or {})
        ).get("reasoning_content")
        raise ValueError(
            f"{cfg.model} returned no message content"
            + (
                "; it spent the output budget on reasoning_content. Raise "
                "max_tokens or drop reasoning_effort for this model."
                if thought
                else "."
            )
        )
    return VlmResult(
        description=ClipDescription.model_validate(payload),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        cached=cached,
    )


def request_raw(
    cfg: Config,
    clip: Path,
    content: list[dict],
    *,
    allow_call: bool = True,
    call_budget: CallBudget | None = None,
) -> tuple[cache.CachedResponse, bool]:
    """Shared cached transport; callers validate only after the paid raw response is saved."""
    sha = cache.clip_sha(clip)
    # This dict's exact shape is load-bearing: it hashes to the cache key, and
    # every response ever paid for is filed under it. Restructuring it -- even
    # into something equivalent -- misses every entry and re-bills the corpus.
    # Add fields, never rename or nest existing ones. The defaults below hash
    # identically to the literals this used to carry.
    key = cache.cache_key(
        sha,
        cfg.prompt_version,
        {
            "model": cfg.model,
            "base_url": cfg.base_url,
            "system": SYSTEM,
            "content": content,
            "temperature": 0,
            "max_tokens": cfg.max_tokens,
            "reasoning_effort": cfg.reasoning_effort,
        },
    )

    cached = cache.get(key)
    if cached is not None:
        return cached, True

    if not allow_call:
        raise BudgetExceeded("call limit reached before this clip")
    if not cfg.configured:
        raise BudgetExceeded("FIREWORKS_API_KEY is required for an uncached clip")

    spent = float(cache.spend_summary()["spent"])
    if cfg.budget_usd and spent >= cfg.budget_usd:
        raise BudgetExceeded(f"spent ${spent:.4f} of ${cfg.budget_usd:.2f} budget")

    if call_budget is not None:
        call_budget.reserve()
    _warn_prices_once(cfg)
    response = _post(cfg, content)
    # Even an unexpected provider envelope must be persisted before validation.
    choices = response.get("choices") or []
    raw = (choices[0].get("message") or {}).get("content") if choices else None
    if not isinstance(raw, str):
        raw = json.dumps(response)
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    cost = _cost(cfg, input_tokens, output_tokens, cached_tokens)

    # Persist before parsing. A schema mistake must cost a retry, never a re-call.
    cache.put(
        key,
        clip_path=clip,
        sha=sha,
        prompt_version=cfg.prompt_version,
        raw=raw,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )

    return cache.CachedResponse(raw, input_tokens, output_tokens, cost), False
