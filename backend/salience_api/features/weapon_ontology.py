from __future__ import annotations

import re

WEAPON_ONTOLOGY_VERSION = "fortnite-weapons-v1"

WEAPON_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shotgun", ("shotgun", "pump", "gatekeeper", "thunder shotgun", "drum shotgun")),
    (
        "sniper_or_hunting",
        ("sniper", "hunting rifle", "marksman rifle", "designated marksman", "dmr"),
    ),
    ("pistol", ("pistol", "dualies", "hand cannon", "revolver", "six shooter")),
    (
        "automatic",
        (
            "smg",
            "submachine",
            "assault rifle",
            "burst rifle",
            "minigun",
            "machine gun",
            "lmg",
            "drum gun",
        ),
    ),
    (
        "other",
        (
            "launcher",
            "rocket",
            "grenade",
            "bow",
            "crossbow",
            "sword",
            "blade",
            "pickaxe",
        ),
    ),
)

LEGACY_AIM_STATE_MAP = {"hunting_ads_zoom": "ads_no_scope_overlay"}
AIM_STATES = {
    "hipfire",
    "ads_no_scope_overlay",
    "scope_overlay",
    "other_ads",
    "unknown",
}


def weapon_category_from_text(value: object) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
    text = " ".join(text.split())
    if not text or text == "unknown":
        return "unknown"
    # OCR of stylised HUD fonts routinely loses the space inside a two-word
    # weapon name ("HUNTING RIFLE" -> "HUNTINGRIFLE"), and an exact substring
    # match drops those reads on the floor as "unknown". Whether the space
    # survives depends on crop width and batching, so compare a despaced form
    # too rather than depending on the recogniser to be consistent.
    squeezed = text.replace(" ", "")
    for category, terms in WEAPON_TERMS:
        for term in terms:
            if term in text or term.replace(" ", "") in squeezed:
                return category
    return "unknown"


def normalize_aim_state(value: object) -> str:
    state = str(value or "unknown").strip().lower()
    state = LEGACY_AIM_STATE_MAP.get(state, state)
    return state if state in AIM_STATES else "unknown"
