from typing import Literal

from pydantic import BaseModel, Field

LabelValue = Literal["yes", "no", "uncertain"]

TEACHER_LABEL_KEYS = [
    "combat_visible",
    "enemy_visible",
    "elimination_or_knock",
    "high_damage_hit",
    "flick_shot",
    "no_scope",
    "trickshot",
    "build_fight",
    "clutch",
    "cleanup_kill",
    "downed_finish",
    "spray_kill",
    "sniper_kill",
    "shotgun_kill",
    "shotgun_one_pump",
    "pistol_kill",
    "automatic_kill",
    "other_weapon_kill",
    "opponent_likely_bot",
    "stationary_target",
    "stationary_sniper_target",
    "competitive_context",
    "victory",
    "multi_kill",
    "fast_edit",
    "rotation_traversal",
    "looting_or_menu",
    "downtime",
]

_EVENT_LABEL_KEYS = frozenset(
    {
        "elimination_or_knock",
        "high_damage_hit",
        "flick_shot",
        "no_scope",
        "trickshot",
        "cleanup_kill",
        "downed_finish",
        "spray_kill",
        "sniper_kill",
        "shotgun_kill",
        "shotgun_one_pump",
        "pistol_kill",
        "automatic_kill",
        "other_weapon_kill",
        "opponent_likely_bot",
        "stationary_target",
        "stationary_sniper_target",
        "multi_kill",
    }
)


class TeacherClipLabels(BaseModel):
    combat_visible: LabelValue
    enemy_visible: LabelValue
    elimination_or_knock: LabelValue
    high_damage_hit: LabelValue
    flick_shot: LabelValue
    no_scope: LabelValue
    trickshot: LabelValue
    build_fight: LabelValue
    clutch: LabelValue
    cleanup_kill: LabelValue
    downed_finish: LabelValue
    spray_kill: LabelValue
    sniper_kill: LabelValue
    shotgun_kill: LabelValue
    shotgun_one_pump: LabelValue
    pistol_kill: LabelValue
    automatic_kill: LabelValue
    other_weapon_kill: LabelValue
    opponent_likely_bot: LabelValue
    stationary_target: LabelValue
    stationary_sniper_target: LabelValue
    competitive_context: LabelValue
    victory: LabelValue
    multi_kill: LabelValue
    fast_edit: LabelValue
    rotation_traversal: LabelValue
    looting_or_menu: LabelValue
    downtime: LabelValue
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]

    def yes_labels(self) -> list[str]:
        return [name for name in TEACHER_LABEL_KEYS if getattr(self, name) == "yes"]


def _label_value(value: object) -> LabelValue:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no", "uncertain"}:
            return normalized  # type: ignore[return-value]
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "uncertain"


def _evidence_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _confidence_value(value: object) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "very high"}:
            return 0.85
        if normalized == "medium":
            return 0.6
        if normalized == "low":
            return 0.3
        try:
            return max(0.0, min(1.0, float(normalized)))
        except ValueError:
            return 0.0
    return 0.0


def normalize_teacher_payload(payload: dict) -> TeacherClipLabels:
    labels = payload.get("labels", payload)
    return TeacherClipLabels(
        combat_visible=_label_value(labels.get("combat_visible", "uncertain")),
        enemy_visible=_label_value(labels.get("enemy_visible", "uncertain")),
        elimination_or_knock=_label_value(
            labels.get(
                "elimination_or_knock", labels.get("knock_or_elimination", "uncertain")
            )
        ),
        high_damage_hit=_label_value(
            labels.get("high_damage_hit", labels.get("high_damage", "uncertain"))
        ),
        flick_shot=_label_value(
            labels.get("flick_shot", labels.get("flick", "uncertain"))
        ),
        no_scope=_label_value(labels.get("no_scope", "uncertain")),
        trickshot=_label_value(labels.get("trickshot", "uncertain")),
        build_fight=_label_value(
            labels.get("build_fight", labels.get("box_fight", "uncertain"))
        ),
        clutch=_label_value(labels.get("clutch", "uncertain")),
        cleanup_kill=_label_value(
            labels.get("cleanup_kill", labels.get("boring_cleanup", "uncertain"))
        ),
        downed_finish=_label_value(labels.get("downed_finish", "uncertain")),
        spray_kill=_label_value(
            labels.get("spray_kill", labels.get("ar_spray", "uncertain"))
        ),
        sniper_kill=_label_value(
            labels.get("sniper_kill", labels.get("sniper_visible", "uncertain"))
        ),
        shotgun_kill=_label_value(labels.get("shotgun_kill", "uncertain")),
        shotgun_one_pump=_label_value(labels.get("shotgun_one_pump", "uncertain")),
        pistol_kill=_label_value(labels.get("pistol_kill", "uncertain")),
        automatic_kill=_label_value(labels.get("automatic_kill", "uncertain")),
        other_weapon_kill=_label_value(labels.get("other_weapon_kill", "uncertain")),
        opponent_likely_bot=_label_value(
            labels.get(
                "opponent_likely_bot", labels.get("likely_bot_kill", "uncertain")
            )
        ),
        stationary_target=_label_value(labels.get("stationary_target", "uncertain")),
        stationary_sniper_target=_label_value(
            labels.get("stationary_sniper_target", "uncertain")
        ),
        competitive_context=_label_value(
            labels.get(
                "competitive_context", labels.get("ranked_or_tournament", "uncertain")
            )
        ),
        victory=_label_value(labels.get("victory", "uncertain")),
        multi_kill=_label_value(labels.get("multi_kill", "uncertain")),
        fast_edit=_label_value(labels.get("fast_edit", "uncertain")),
        rotation_traversal=_label_value(
            labels.get("rotation_traversal", labels.get("rotation", "uncertain"))
        ),
        looting_or_menu=_label_value(labels.get("looting_or_menu", "uncertain")),
        downtime=_label_value(
            labels.get("downtime", labels.get("low_action", "uncertain"))
        ),
        confidence=_confidence_value(
            payload.get("confidence", labels.get("confidence", 0.0))
        ),
        evidence=_evidence_list(payload.get("evidence", labels.get("evidence", []))),
    )


def derive_label_confidences(
    payload: dict,
    *,
    events: list[dict] | None = None,
) -> dict[str, float]:
    """Build per-label reliability weights without replacing label values.

    Clip confidence is the fallback for context labels. Event-owned labels are
    additionally capped by the strongest attributed event confidence, and an
    explicit uncertain label receives zero training weight.
    """
    labels = normalize_teacher_payload(payload)
    clip_confidence = labels.confidence
    attributed_confidences: list[float] = []
    for event in events or []:
        if str(event.get("status", "")).lower() != "attributed":
            continue
        raw = event.get("teacher_confidence", event.get("weapon_confidence"))
        try:
            confidence = float(raw)
        except (TypeError, ValueError):
            confidence = clip_confidence
        attributed_confidences.append(max(0.0, min(1.0, confidence)))
    event_confidence = max(attributed_confidences, default=clip_confidence)

    result: dict[str, float] = {}
    for key in TEACHER_LABEL_KEYS:
        value = str(getattr(labels, key))
        if value == "uncertain":
            result[key] = 0.0
        elif key in _EVENT_LABEL_KEYS:
            result[key] = min(clip_confidence, event_confidence)
        else:
            result[key] = clip_confidence
    return result
