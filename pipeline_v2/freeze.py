"""One definition of "the runtime that produced this frozen study".

Every offline study records the configuration it ran under and refuses to add
results under a different one — otherwise a plan accumulates responses from two
models or two prompts and the comparison it exists to make is void.

That check was written four separate times, with four field sets and four error
messages, which is exactly the kind of drift it is supposed to prevent. It lives
here now, and reports *which* field moved instead of only that something did.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(cfg, prompt_text: str, frame_width: int, **extra) -> dict:
    """The runtime identity of a describe/localize pass.

    ``extra`` carries study-specific keys (a source clip, a per-arm prompt
    version). Anything recorded is compared; anything absent is ignored, so an
    older plan written before a field existed stays readable.
    """
    return {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "prompt_version": cfg.prompt_version,
        "prompt_sha": sha256_text(prompt_text),
        "frame_width": frame_width,
        **extra,
    }


def check(recorded: Mapping, current: Mapping, study: str) -> None:
    """Raise when a recorded field disagrees with the current runtime.

    Only keys present on both sides are compared: a plan holds descriptive
    fields (selection notes, window lists) that are not runtime identity.
    """
    drifted = sorted(
        key
        for key in recorded
        if key in current and recorded[key] != current[key]
    )
    if drifted:
        details = ", ".join(
            f"{key}: frozen {recorded[key]!r} != current {current[key]!r}"
            for key in drifted
        )
        raise ValueError(f"Runtime differs from frozen {study} ({details})")
