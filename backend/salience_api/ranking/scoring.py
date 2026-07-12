from dataclasses import dataclass

from salience_api.features.basic import BasicFeatures, clamp, duration_quality
from salience_api.ranking.highlights import HighlightProfile, event_bonus

HIGH_VALUE_TAGS = {
    "combat_visible",
    "enemy_visible",
    "elimination_or_knock",
    "high_damage_hit",
    "flick_shot",
    "sniper",
    "flick",
    "endgame",
    "victory",
    "build_fight",
    "clutch",
    "competitive_context",
    "mechanics",
    "trickshot",
    "no_scope",
    "sniper_kill",
    "shotgun_one_pump",
    "multi_kill",
    "fast_edit",
    "spray_kill",
    "shotgun_kill",
    "pistol_kill",
    "automatic_kill",
}
LOW_VALUE_TAGS = {
    "low_action",
    "cleanup_kill",
    "downed_finish",
    "opponent_likely_bot",
    "rotation",
    "rotation_traversal",
    "looting_or_menu",
    "downtime",
}

# Scored via HighlightProfile / event_bonus when a live event profile exists.
EVENT_OWNED_HIGH_TAGS = {
    "elimination_or_knock",
    "high_damage_hit",
    "no_scope",
    "sniper_kill",
    "shotgun_one_pump",
    "multi_kill",
    "spray_kill",
    "shotgun_kill",
    "pistol_kill",
    "automatic_kill",
}
EVENT_OWNED_LOW_TAGS = {
    "cleanup_kill",
    "downed_finish",
}


@dataclass(frozen=True)
class ClipScore:
    base_score: float
    personal_score: float
    final_score: float
    confidence: float
    explanation: str


def score_clip(
    features: BasicFeatures,
    personal_score: float | None,
    *,
    highlight_profile: HighlightProfile | None = None,
) -> ClipScore:
    tags = set(features.tags)
    high_tags = HIGH_VALUE_TAGS
    low_tags = LOW_VALUE_TAGS
    structured_bonus = 0.0
    using_profile = (
        highlight_profile is not None and highlight_profile.primary is not None
    )

    if using_profile:
        high_tags = HIGH_VALUE_TAGS - EVENT_OWNED_HIGH_TAGS
        low_tags = LOW_VALUE_TAGS - EVENT_OWNED_LOW_TAGS
        # Downed penalties only when the primary highlight is downed-only.
        if highlight_profile.primary.highlight_type == "downed_finish":
            low_tags = low_tags | {"downed_finish", "cleanup_kill"}
        structured_bonus = event_bonus(highlight_profile)

    duration_score = duration_quality(features.duration_sec)
    tag_bonus = min(0.18, 0.045 * len(tags & high_tags))
    tag_penalty = min(0.18, 0.06 * len(tags & low_tags))
    boring_penalty = 0.25 * features.silence_ratio
    base_score = clamp(
        0.32 * features.motion_score
        + 0.23 * features.audio_peak_score
        + 0.18 * features.action_density
        + 0.22 * duration_score
        + tag_bonus
        + structured_bonus
        - tag_penalty
        - boring_penalty
    )
    stored_personal_score = personal_score if personal_score is not None else 0.0
    if personal_score is None:
        final_score = base_score
    else:
        final_score = clamp((0.25 * base_score) + (0.75 * stored_personal_score))
    reasons: list[str] = []
    if using_profile and highlight_profile.primary is not None:
        reasons.append(
            f"primary {highlight_profile.primary.highlight_type.replace('_', ' ')}"
        )
        if highlight_profile.multi_kill:
            reasons.append("multi kill")
    if features.motion_score >= 0.6:
        reasons.append("high motion")
    if features.audio_peak_score >= 0.5:
        reasons.append("audio spikes")
    if features.action_density >= 0.65:
        reasons.append("dense action")
    if duration_score >= 0.8:
        reasons.append("good clip length")
    matched_tags = sorted(tags & high_tags)
    if matched_tags:
        reasons.append("matches " + ", ".join(matched_tags[:3]))
    if "competitive_context" in tags:
        reasons.append("ranked/tournament context")
    if "flick_shot" in tags:
        reasons.append("flick shot")
    if features.silence_ratio >= 0.6:
        reasons.append("quiet/low energy penalty")
    if tags & low_tags:
        reasons.append("low-action tag penalty")
    if "opponent_likely_bot" in tags:
        reasons.append("likely bot opponent")
    if "stationary_sniper_target" in tags:
        reasons.append("stationary sniper target")
    elif "stationary_target" in tags:
        reasons.append("stationary target")
    if stored_personal_score >= 0.65:
        reasons.append("matches your recent feedback")
    if not reasons:
        reasons.append("baseline clip quality")

    return ClipScore(
        base_score=base_score,
        personal_score=stored_personal_score,
        final_score=final_score,
        confidence=features.extraction_confidence,
        explanation=", ".join(reasons),
    )
