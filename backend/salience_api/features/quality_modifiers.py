from __future__ import annotations

SNIPER_CONTEXT_CUES = {
    "hunting rifle",
    "no-scope",
    "noscope",
    "sniper",
    "sniped",
}

STATIONARY_TARGET_CUES = {
    "does not react",
    "exposed and stationary",
    "not moving",
    "standing still",
    "stationary target",
    "stationary while",
}


def stationary_target_from_evidence(evidence: list[str]) -> bool:
    if not evidence:
        return False
    text = " ".join(evidence).lower()
    return any(cue in text for cue in STATIONARY_TARGET_CUES)


def stationary_sniper_target_from_evidence(evidence: list[str], tags: set[str]) -> bool:
    if not evidence:
        return False
    text = " ".join(evidence).lower()
    has_sniper_label = bool(tags & {"sniper", "sniper_kill", "no_scope"})
    has_sniper_text = any(cue in text for cue in SNIPER_CONTEXT_CUES)
    return stationary_target_from_evidence(evidence) and (
        has_sniper_label or has_sniper_text
    )
