from __future__ import annotations

import re

KNOWN_BOT_NAME_HINTS = {
    "90cranker",
    "allsmyles:d",
    "bestdoggo83",
    "hippomagician",
    "howaremy90s",
    "ihavefullbrick",
    "maybenot40",
    "quack4bread",
    "sn00tymagician",
    "tallestant4",
}

BOT_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}[A-Z][A-Za-z]{2,}\d{1,3}\b")
ANONYMOUS_PATTERN = re.compile(r"\bAnonymous\s*\[\d+\]", re.IGNORECASE)
BOT_BEHAVIOR_CUES = {
    "bot-like",
    "does not react",
    "easy opponent",
    "misses shots",
    "no building",
    "no evasive movement",
    "no reaction",
    "not moving",
    "standing in the open",
    "standing still",
    "stationary",
    "walks into",
    "walking into",
}


def opponent_likely_bot_from_evidence(evidence: list[str]) -> bool:
    if not evidence:
        return False
    text = " ".join(evidence)
    normalized = re.sub(r"[^a-z0-9:]+", "", text.lower())
    lower_text = text.lower()
    if ANONYMOUS_PATTERN.search(text):
        return False
    if "not a bot" in lower_text or "real player" in lower_text:
        return False
    has_explicit_bot_cue = (
        "bot-like" in lower_text
        or "ai opponent" in lower_text
        or "ai player" in lower_text
    )
    has_known_bot_name = any(name in normalized for name in KNOWN_BOT_NAME_HINTS)
    has_botlike_name = BOT_NAME_PATTERN.search(text) is not None
    has_behavior_cue = any(cue in lower_text for cue in BOT_BEHAVIOR_CUES)
    return (
        has_explicit_bot_cue
        or has_known_bot_name
        or (has_botlike_name and has_behavior_cue)
    )
