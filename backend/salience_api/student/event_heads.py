from __future__ import annotations

from dataclasses import dataclass

from salience_api.student.schema import AIM_STATES, EVENT_KINDS, TARGET_STATES, WEAPON_CLASSES

_EVIDENCE_STATES = ("yes", "no", "unknown")
_FINISH_KINDS = {"knock", "elimination", "downed_finish"}
# Kill banner/feed UI keeps the event heads firing for ~3s after a finish, so
# finish summaries chained closer than this are the same kill.
SAME_KILL_COLLAPSE_GAP_SEC = 3.0


def _normalize_evidence(value: str) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in _EVIDENCE_STATES else "unknown"


@dataclass
class EventHeadPrediction:
    event_kind: str
    target_state: str
    weapon: str
    weapon_confidence: float
    aim_state: str
    pov_shot_visible: str = "unknown"
    target_reaction_visible: str = "unknown"
    new_damage_visible: str = "unknown"
    target_defeat_visible: str = "unknown"
    finish_ui_newly_appeared: str = "unknown"
    kill_feed_corroborates_pov: str = "unknown"
    visual_action_supported: str = "unknown"
    damaging_shot_count: int = 0
    stationary_target: str = "unknown"
    stationary_duration_supported: str = "unknown"
    flick_shot: str = "unknown"
    trickshot: str = "unknown"
    cleanup_kill: str = "unknown"
    opponent_likely_bot: str = "unknown"
    damage_display_is_cumulative: str = "unknown"
    single_shot_damage_known: str = "unknown"
    single_shot_damage: int | None = None


def event_summary_from_heads(
    pred: EventHeadPrediction,
    *,
    event_index: int,
    event_timestamp: float,
) -> dict:
    event_kind = str(pred.event_kind).strip().lower()
    if event_kind not in EVENT_KINDS:
        event_kind = "none"

    target_state = str(pred.target_state).strip().lower()
    if target_state not in TARGET_STATES:
        target_state = "unknown"

    weapon = str(pred.weapon).strip().lower()
    if weapon not in WEAPON_CLASSES:
        weapon = "unknown"

    aim_state = str(pred.aim_state).strip().lower()
    if aim_state not in AIM_STATES:
        aim_state = "unknown"

    return {
        "event_index": int(event_index),
        "event_timestamp": float(event_timestamp),
        "event_kind": event_kind,
        "target_state": target_state,
        "selected_weapon_before_finish": weapon,
        "weapon_confidence": float(pred.weapon_confidence),
        "selected_weapon_name_text": "unknown",
        "aim_state_at_shot": aim_state,
        "pov_shot_visible": _normalize_evidence(pred.pov_shot_visible),
        "target_reaction_visible": _normalize_evidence(pred.target_reaction_visible),
        "new_damage_visible": _normalize_evidence(pred.new_damage_visible),
        "target_defeat_visible": _normalize_evidence(pred.target_defeat_visible),
        "finish_ui_newly_appeared": _normalize_evidence(pred.finish_ui_newly_appeared),
        "kill_feed_corroborates_pov": _normalize_evidence(pred.kill_feed_corroborates_pov),
        "visual_action_supported": _normalize_evidence(pred.visual_action_supported),
        "damaging_shot_count": max(0, int(pred.damaging_shot_count)),
        "stationary_target": _normalize_evidence(pred.stationary_target),
        "stationary_duration_supported": _normalize_evidence(
            pred.stationary_duration_supported
        ),
        "flick_shot": _normalize_evidence(pred.flick_shot),
        "trickshot": _normalize_evidence(pred.trickshot),
        "cleanup_kill": _normalize_evidence(pred.cleanup_kill),
        "opponent_likely_bot": _normalize_evidence(pred.opponent_likely_bot),
        # A concrete damage-number claim requires the damage to be visible on
        # screen; the regression head alone regresses to ~100+ on every event.
        "single_shot_damage": (
            pred.single_shot_damage
            if pred.single_shot_damage_known == "yes"
            and _normalize_evidence(pred.new_damage_visible) == "yes"
            else None
        ),
        "raw_single_shot_damage": pred.single_shot_damage,
        "damage_display_is_cumulative": (
            _normalize_evidence(pred.damage_display_is_cumulative) == "yes"
        ),
        "high_damage_one_shot": bool(
            pred.single_shot_damage_known == "yes"
            and _normalize_evidence(pred.new_damage_visible) == "yes"
            and (pred.single_shot_damage or 0) >= 100
            and pred.damage_display_is_cumulative != "yes"
        ),
    }


def collapse_same_kill_finishes(
    summaries: list[dict],
    *,
    gap_sec: float = SAME_KILL_COLLAPSE_GAP_SEC,
) -> list[dict]:
    """Merge finish summaries that chain within ``gap_sec`` into one finish.

    Dense locator proposals give the event heads several windows around each
    kill (good for recall), but each window that reads the lingering banner
    becomes another "finish" and fakes multi_kill. Single-linkage chaining on
    the event timestamp merges those; a real second kill needs a quiet gap of
    at least ``gap_sec`` before it.  Non-finish summaries pass through.
    """
    finishes = sorted(
        (s for s in summaries if s.get("event_kind") in _FINISH_KINDS),
        key=lambda s: float(s.get("event_timestamp", 0.0)),
    )
    others = [s for s in summaries if s.get("event_kind") not in _FINISH_KINDS]

    collapsed: list[dict] = []
    chain: list[dict] = []
    for summary in finishes:
        if chain and (
            float(summary.get("event_timestamp", 0.0))
            - float(chain[-1].get("event_timestamp", 0.0))
            > gap_sec
        ):
            collapsed.append(
                max(chain, key=lambda s: float(s.get("weapon_confidence", 0.0)))
            )
            chain = []
        chain.append(summary)
    if chain:
        collapsed.append(
            max(chain, key=lambda s: float(s.get("weapon_confidence", 0.0)))
        )
    return sorted(
        collapsed + others,
        key=lambda s: (int(s.get("event_index", 0)), float(s.get("event_timestamp", 0.0))),
    )
