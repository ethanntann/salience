from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
from math import ceil
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from salience_api.config import Settings
from salience_api.features.hud_ocr import best_weapon_ocr
from salience_api.features.teacher_labels import (
    TeacherClipLabels,
    normalize_teacher_payload,
)
from salience_api.features.weapon_ontology import (
    AIM_STATES,
    normalize_aim_state,
    weapon_category_from_text,
)
from salience_api.ranking.highlights import (
    dedupe_verified_finishes,
    resolve_no_scope_for_events,
)


class FireworksNotConfiguredError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipTeacherInput:
    filename: str
    duration_sec: float | None
    width: int | None
    height: int | None
    fps: float | None
    tags: list[str]
    image_paths: list[Path]
    image_timestamps: list[float] = field(default_factory=list)
    image_views: list[str] = field(default_factory=list)
    image_event_indices: list[int | None] = field(default_factory=list)
    image_event_centers: list[float | None] = field(default_factory=list)
    ocr_observations: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class WeaponAttribution:
    labels: dict[str, str]
    confidence: float
    evidence: list[str]
    status: str
    events: list[dict]
    raw_payload: dict = field(default_factory=dict)


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_teacher_prompt(clip: ClipTeacherInput) -> str:
    labels = ", ".join(
        (
            "combat_visible",
            "enemy_visible",
            "build_fight",
            "clutch",
            "competitive_context",
            "victory",
            "fast_edit",
            "rotation_traversal",
            "looting_or_menu",
            "downtime",
        )
    )
    return (
        "/no_think\n"
        "Analyze these coarse chronological Fortnite frames for clip-level context only. "
        "Return only valid compact JSON with keys labels, confidence, and evidence. Evidence must be an array "
        "of at most 3 short strings. "
        f"labels must include exactly these keys: {labels}. Each value is yes, no, or uncertain. "
        "Do not classify kills, weapons, damage, aim state, target state, flicks, trickshots, bots, "
        "or cleanup quality; a separate event specialist owns those facts. Ignore kill-feed text for "
        "context because it may be stale or belong to another player. Definitions: "
        "combat_visible means the player is actively fighting, aiming at an opponent, shooting, or taking damage; "
        "enemy_visible means an opponent, bot, or enemy player model is visible; "
        "build_fight means a close-range Fortnite build/edit fight; clutch means a high-pressure survival, endgame, or outnumbered win. "
        "In team modes, mark clutch yes when the POV player is the last active teammate or wins a 1v2/1v3 while teammates are knocked, eliminated, or spectating; low health is not required; "
        "competitive_context means a real Ranked mode or tournament is visible. Points-based "
        "tournament/session UI, placement points, matches played, Ranked mode text with a rank "
        "emblem, Ranked Cup, Cash Cup, FNCS, or a competitive leaderboard qualify. XP-based "
        "progress is casual and never qualifies: Survivor I/II/III badges, survivor medals, "
        "accolades, account levels, XP bars, XP gains, quests, milestones, and generic badges are "
        "not competitive context; "
        "victory means a Victory Royale or round win is visible; "
        "fast_edit means fast build edits, piece control, or mechanical edit plays are visible; "
        "rotation_traversal means mostly moving, driving, gliding, rotating, or repositioning without a fight; "
        "looting_or_menu means inventory, map, lobby, menu, looting, or loadout management dominates the clip; "
        "downtime means low-action waiting, wandering, farming, or quiet time dominates the clip. "
        "Only mark rotation_traversal, looting_or_menu, or downtime yes when that low-action state dominates the clip; "
        "do not mark them yes for brief moments after a visible fight, knock, elimination, or high-damage hit. "
        f"Clip metadata: filename={clip.filename}, duration_sec={clip.duration_sec}, "
        f"resolution={clip.width}x{clip.height}, fps={clip.fps}, existing_tags={clip.tags}."
    )


def build_teacher_content(clip: ClipTeacherInput) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": build_teacher_prompt(clip)}]
    for index, image_path in enumerate(clip.image_paths):
        if index < len(clip.image_timestamps):
            view = (
                clip.image_views[index]
                if index < len(clip.image_views)
                else "full gameplay frame"
            )
            event_index = (
                clip.image_event_indices[index]
                if index < len(clip.image_event_indices)
                else None
            )
            event_context = (
                f", event_index={event_index}" if event_index is not None else ""
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame {index + 1} at {clip.image_timestamps[index]:.2f}s"
                        f"{event_context} - {view}"
                    ),
                }
            )
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}}
        )
    return content


def extract_json_value(text: str) -> dict | list:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict | list):
            return payload
        raise ValueError("Teacher response is not a JSON object or array")
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict | list):
                return payload
        raise ValueError("Teacher response does not contain a JSON object or array")


def extract_json_object(text: str) -> dict:
    payload = extract_json_value(text)
    if isinstance(payload, dict):
        return payload
    nested_object = next((item for item in payload if isinstance(item, dict)), None)
    if nested_object is not None:
        return nested_object
    raise ValueError("Teacher response is not a JSON object")


def max_event_candidates(duration_sec: float | None) -> int:
    return min(12, max(4, ceil(max(duration_sec or 0, 20.0) / 10.0)))


def parse_event_timestamps(text: str, *, limit: int = 4) -> list[float]:
    payload = extract_json_value(text)
    if isinstance(payload, dict):
        values = payload.get("event_timestamps", [])
    elif len(payload) == 1 and isinstance(payload[0], dict):
        values = payload[0].get("event_timestamps", [])
    else:
        values = payload
    if not isinstance(values, list):
        return []
    timestamps: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        timestamp = float(value)
        if timestamp >= 0:
            timestamps.append(timestamp)
    return timestamps[:limit]


_EVENT_PHASES = {"none", "damage", "knock_start", "elimination_start", "post_event"}
_WEAPON_CATEGORIES = {
    "pistol",
    "sniper_or_hunting",
    "shotgun",
    "automatic",
    "other",
    "unknown",
}
_AIM_STATES = AIM_STATES
_TARGET_STATES = {"active", "already_downed", "unknown"}
_EVIDENCE_STATES = {"yes", "no", "unknown"}
_WEAPON_LABELS = (
    "sniper_kill",
    "shotgun_kill",
    "shotgun_one_pump",
    "pistol_kill",
    "automatic_kill",
    "other_weapon_kill",
    "spray_kill",
    "no_scope",
)


def _bounded_float(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _timestamp(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _event_index(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _evidence_state(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in _EVIDENCE_STATES else "unknown"


def _damage_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        damage = int(value)
    except (TypeError, ValueError):
        return None
    # Fortnite single-hit markers rarely exceed ~300; larger values are XP/stacks.
    return damage if 0 <= damage <= 300 else None


def _sanitize_event_damage(event: dict) -> None:
    """Drop untrustworthy single-shot damage values."""
    if event.get("damage_display_is_cumulative"):
        event["single_shot_damage"] = None
        event["high_damage_one_shot"] = False
        return
    damage = event.get("single_shot_damage")
    if damage is None:
        return
    try:
        value = int(damage)
    except (TypeError, ValueError):
        event["single_shot_damage"] = None
        event["high_damage_one_shot"] = False
        return
    if value <= 0 or value > 300:
        event["single_shot_damage"] = None
        event["high_damage_one_shot"] = False
        return
    # Multi-hit sprays should not carry a single-shot damage claim.
    if int(event.get("damage_hit_count") or 0) >= 3:
        event["single_shot_damage"] = None
        event["high_damage_one_shot"] = False


def _normalize_event_frame(frame: dict) -> dict | None:
    timestamp = _timestamp(frame.get("timestamp"))
    if timestamp is None:
        return None
    phase = str(frame.get("event_phase", "none")).strip().lower()
    weapon = str(frame.get("weapon_category", "unknown")).strip().lower()
    target_state = str(frame.get("target_state", "unknown")).strip().lower()
    aim_state = normalize_aim_state(frame.get("aim_state"))
    if aim_state == "unknown":
        if _boolean(frame.get("hipfire")):
            aim_state = "hipfire"
        elif "hipfire" in frame:
            aim_state = "other_ads"
        else:
            aim_state = "unknown"
    return {
        "event_index": _event_index(frame.get("event_index", 0)),
        "timestamp": timestamp,
        "event_phase": phase if phase in _EVENT_PHASES else "none",
        "weapon_category": weapon if weapon in _WEAPON_CATEGORIES else "unknown",
        "weapon_confidence": _bounded_float(frame.get("weapon_confidence")),
        "high_damage_one_shot": _boolean(frame.get("high_damage_one_shot")),
        "single_shot_damage": _damage_value(frame.get("single_shot_damage")),
        "damage_display_is_cumulative": _boolean(
            frame.get("damage_display_is_cumulative")
        ),
        "target_state": target_state if target_state in _TARGET_STATES else "unknown",
        "hipfire": _boolean(frame.get("hipfire")),
        "aim_state": aim_state,
        "pov_shot_visible": _evidence_state(frame.get("pov_shot_visible")),
        "target_reaction_visible": _evidence_state(
            frame.get("target_reaction_visible")
        ),
        "new_damage_visible": _evidence_state(frame.get("new_damage_visible")),
        "finish_ui_newly_appeared": _evidence_state(
            frame.get("finish_ui_newly_appeared")
        ),
        "target_defeat_visible": _evidence_state(frame.get("target_defeat_visible")),
        "kill_feed_corroborates_pov": _evidence_state(
            frame.get("kill_feed_corroborates_pov")
        ),
        "selected_weapon_before_shot": (
            str(frame.get("selected_weapon_before_shot", "unknown")).strip().lower()
            if str(frame.get("selected_weapon_before_shot", "unknown")).strip().lower()
            in _WEAPON_CATEGORIES
            else "unknown"
        ),
        "damage_instance_id": (
            str(frame["damage_instance_id"])
            if frame.get("damage_instance_id") not in {None, ""}
            else None
        ),
    }


def _resolve_event(
    event_index: int, frames: list[dict], *, strict_action_evidence: bool = False
) -> dict:
    ordered = sorted(frames, key=lambda frame: frame["timestamp"])
    finish = next(
        (
            frame
            for frame in ordered
            if frame["event_phase"] in {"knock_start", "elimination_start"}
        ),
        None,
    )
    if finish is None:
        return {
            "event_index": event_index,
            "status": "no_finish",
            "resolved_weapon": "unknown",
            "frames": ordered,
        }

    recent_damage_frames = [
        frame
        for frame in ordered
        if frame["event_phase"] == "damage"
        and frame["timestamp"] <= finish["timestamp"]
        and finish["timestamp"] - frame["timestamp"] <= 1.0
    ]
    recent_damage = recent_damage_frames[-1] if recent_damage_frames else None
    finish_weapon = finish["selected_weapon_before_shot"]
    if finish_weapon == "unknown":
        finish_weapon = finish["weapon_category"]
    damage_weapon = "unknown"
    if recent_damage:
        damage_weapon = recent_damage["selected_weapon_before_shot"]
        if damage_weapon == "unknown":
            damage_weapon = recent_damage["weapon_category"]
    known_finish = finish_weapon != "unknown"
    known_damage = damage_weapon != "unknown"

    action_frames = [
        frame
        for frame in ordered
        if frame["timestamp"] <= finish["timestamp"]
        and finish["timestamp"] - frame["timestamp"] <= 1.0
    ]
    visual_action = any(
        frame["pov_shot_visible"] == "yes"
        and (
            frame["target_reaction_visible"] == "yes"
            or frame["new_damage_visible"] == "yes"
        )
        for frame in action_frames
    )
    finish_onset = finish["finish_ui_newly_appeared"] == "yes"
    visible_defeat = any(
        frame["target_defeat_visible"] == "yes" for frame in action_frames
    )

    if strict_action_evidence and not (
        visual_action and (finish_onset or visible_defeat)
    ):
        status = "no_visual_causality"
        resolved_weapon = "unknown"
        evidence_frame = finish
    elif known_finish and known_damage and finish_weapon != damage_weapon:
        status = "weapon_conflict"
        resolved_weapon = "unknown"
        evidence_frame = finish
    elif known_damage and recent_damage["weapon_confidence"] >= 0.7:
        status = "attributed"
        resolved_weapon = damage_weapon
        evidence_frame = recent_damage
    elif known_finish and finish["weapon_confidence"] >= 0.7:
        status = "finish_only"
        resolved_weapon = finish_weapon
        evidence_frame = finish
    else:
        status = "low_confidence"
        resolved_weapon = "unknown"
        evidence_frame = finish

    matching_damage_frames = [
        frame
        for frame in recent_damage_frames
        if frame["weapon_category"] == resolved_weapon
    ]
    single_shot_values = [
        frame["single_shot_damage"]
        for frame in matching_damage_frames
        if frame["single_shot_damage"] is not None
        and not frame["damage_display_is_cumulative"]
    ]
    damage_instances = {
        frame["damage_instance_id"]
        for frame in matching_damage_frames
        if frame["damage_instance_id"] is not None
    }
    damage_hit_count = (
        len(damage_instances) if strict_action_evidence else len(matching_damage_frames)
    )
    target_states = {
        frame["target_state"] for frame in [*matching_damage_frames, finish]
    }
    return {
        "event_index": event_index,
        "status": status,
        "finish_timestamp": finish["timestamp"],
        "damage_timestamp": recent_damage["timestamp"] if recent_damage else None,
        "resolved_weapon": resolved_weapon,
        "high_damage_one_shot": evidence_frame["high_damage_one_shot"],
        "single_shot_damage": max(single_shot_values) if single_shot_values else None,
        "damage_hit_count": damage_hit_count,
        "target_was_downed": "already_downed" in target_states,
        "target_was_active": "active" in target_states,
        "visual_action_supported": visual_action,
        "finish_onset_supported": finish_onset,
        "visible_defeat_supported": visible_defeat,
        "kill_feed_corroborates_pov": any(
            frame["kill_feed_corroborates_pov"] == "yes" for frame in action_frames
        ),
        "hipfire": evidence_frame["hipfire"],
        "damage_aim_state": recent_damage["aim_state"] if recent_damage else "unknown",
        "finish_aim_state": finish["aim_state"],
        "frames": ordered,
    }


def resolve_legacy_frame_attribution(payload: dict) -> WeaponAttribution:
    """Read historical frame-table payloads; current inference uses event summaries."""
    raw_frames = payload.get("frames", [])
    normalized_frames: list[dict] = []
    if isinstance(raw_frames, list):
        for frame in raw_frames:
            if not isinstance(frame, dict):
                continue
            normalized = _normalize_event_frame(frame)
            if normalized is not None:
                normalized_frames.append(normalized)
    grouped: dict[int, list[dict]] = {}
    for frame in normalized_frames:
        grouped.setdefault(frame["event_index"], []).append(frame)
    strict_action_evidence = int(payload.get("action_evidence_version", 0) or 0) >= 2
    events = [
        _resolve_event(index, frames, strict_action_evidence=strict_action_evidence)
        for index, frames in sorted(grouped.items())
    ]

    attributable = [
        event for event in events if event["status"] in {"attributed", "finish_only"}
    ]
    valid_weapon_events = [
        event
        for event in events
        if event["status"] == "attributed" and not event.get("target_was_downed", False)
    ]
    labels = {key: "no" for key in _WEAPON_LABELS}
    if not attributable:
        labels = {key: "uncertain" for key in _WEAPON_LABELS}
    for event in valid_weapon_events:
        weapon = event["resolved_weapon"]
        if weapon == "sniper_or_hunting":
            labels["sniper_kill"] = "yes"
            if (
                event["status"] == "attributed"
                and event["damage_aim_state"] == "hipfire"
                and event["finish_aim_state"] == "hipfire"
            ):
                labels["no_scope"] = "yes"
        elif weapon == "shotgun":
            labels["shotgun_kill"] = "yes"
            one_shot_damage = event.get("single_shot_damage")
            if event["damage_hit_count"] == 1 and (
                (one_shot_damage is not None and one_shot_damage >= 100)
                or event["high_damage_one_shot"]
            ):
                labels["shotgun_one_pump"] = "yes"
        elif weapon == "pistol":
            labels["pistol_kill"] = "yes"
        elif weapon == "automatic":
            labels["automatic_kill"] = "yes"
            if event["damage_hit_count"] >= 3:
                labels["spray_kill"] = "yes"
        elif weapon == "other":
            labels["other_weapon_kill"] = "yes"

    evidence = payload.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    if attributable:
        status = "attributed"
    elif any(event["status"] == "weapon_conflict" for event in events):
        status = "weapon_conflict"
    else:
        status = "no_event"
    return WeaponAttribution(
        labels=labels,
        confidence=_bounded_float(payload.get("confidence")),
        evidence=[str(item) for item in evidence][:3],
        status=status,
        events=events,
        raw_payload=payload,
    )


def resolve_event_summaries(payload: dict) -> WeaponAttribution:
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []
    events: list[dict] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        event_index = _event_index(raw_event.get("event_index"))
        event_kind = str(raw_event.get("event_kind", "none")).strip().lower()
        target_state = str(raw_event.get("target_state", "unknown")).strip().lower()
        if target_state not in _TARGET_STATES:
            target_state = "unknown"
        weapon_name_text = str(
            raw_event.get("selected_weapon_name_text", "unknown")
        ).strip()[:120]
        weapon_from_name = weapon_category_from_text(weapon_name_text)
        weapon = (
            str(
                raw_event.get(
                    "selected_weapon_before_finish",
                    raw_event.get("weapon_category", "unknown"),
                )
            )
            .strip()
            .lower()
        )
        if weapon not in _WEAPON_CATEGORIES:
            weapon = "unknown"
        weapon_confidence = _bounded_float(raw_event.get("weapon_confidence"))
        if weapon_from_name != "unknown":
            # Prefer exact HUD / OCR weapon name over silhouette category.
            weapon = weapon_from_name
            weapon_confidence = max(weapon_confidence, 0.85)
        pov_shot_visible = _evidence_state(raw_event.get("pov_shot_visible")) == "yes"
        target_reaction_visible = (
            _evidence_state(raw_event.get("target_reaction_visible")) == "yes"
        )
        new_damage_visible = (
            _evidence_state(raw_event.get("new_damage_visible")) == "yes"
        )
        finish_onset = (
            _evidence_state(raw_event.get("finish_ui_newly_appeared")) == "yes"
        )
        visible_defeat = (
            _evidence_state(raw_event.get("target_defeat_visible")) == "yes"
        )
        kill_feed_corroborates = (
            _evidence_state(raw_event.get("kill_feed_corroborates_pov")) == "yes"
        )
        # Component evidence owns causality. The VLM's aggregate flag alone is not enough,
        # and requiring it AND reaction/damage drops distant sniper finishes after swaps.
        component_visual_action = pov_shot_visible and (
            target_reaction_visible or new_damage_visible or visible_defeat
        )
        visual_action = component_visual_action or (
            _evidence_state(raw_event.get("visual_action_supported")) == "yes"
            and pov_shot_visible
            and (target_reaction_visible or new_damage_visible)
        )
        is_finish = event_kind in {"knock", "elimination", "downed_finish"}
        target_was_downed = (
            target_state == "already_downed" or event_kind == "downed_finish"
        )
        target_identity = str(raw_event.get("target_identity", "unknown")).strip()[:80]
        has_visual_causality = visual_action and (finish_onset or visible_defeat)
        if not is_finish:
            status = "no_finish"
            resolved_weapon = "unknown"
        elif not has_visual_causality:
            status = "no_visual_causality"
            resolved_weapon = "unknown"
        elif target_state == "unknown" and not target_was_downed:
            status = "target_state_unknown"
            resolved_weapon = "unknown"
        elif weapon == "unknown" or weapon_confidence < 0.7:
            status = "low_confidence"
            resolved_weapon = "unknown"
        else:
            status = "attributed"
            resolved_weapon = weapon
        aim_state = normalize_aim_state(raw_event.get("aim_state_at_shot"))
        try:
            damaging_shot_count = max(0, int(raw_event.get("damaging_shot_count", 0)))
        except (TypeError, ValueError):
            damaging_shot_count = 0
        stationary_report = _evidence_state(raw_event.get("stationary_target"))
        stationary_duration = _evidence_state(
            raw_event.get("stationary_duration_supported")
        )
        if stationary_report == "yes" and stationary_duration == "yes":
            stationary_target = "yes"
        elif "no" in {stationary_report, stationary_duration}:
            stationary_target = "no"
        else:
            stationary_target = "uncertain"
        events.append(
            {
                "event_index": event_index,
                "status": status,
                "event_kind": event_kind,
                "teacher_confidence": weapon_confidence,
                "finish_timestamp": _timestamp(raw_event.get("event_timestamp")),
                "resolved_weapon": resolved_weapon,
                "selected_weapon_name_text": weapon_name_text,
                "local_ocr": raw_event.get("local_ocr"),
                "high_damage_one_shot": _boolean(raw_event.get("high_damage_one_shot")),
                "single_shot_damage": _damage_value(
                    raw_event.get("single_shot_damage")
                ),
                "damage_display_is_cumulative": _boolean(
                    raw_event.get("damage_display_is_cumulative")
                ),
                "damage_hit_count": damaging_shot_count,
                "target_was_downed": target_was_downed,
                "target_was_active": target_state == "active",
                "target_identity": target_identity,
                "visual_action_supported": visual_action,
                "pov_shot_visible": pov_shot_visible,
                "target_reaction_visible": target_reaction_visible,
                "new_damage_visible": new_damage_visible,
                "finish_onset_supported": finish_onset,
                "visible_defeat_supported": visible_defeat,
                "kill_feed_corroborates_pov": kill_feed_corroborates,
                "flick_shot": _evidence_state(raw_event.get("flick_shot")),
                "trickshot": _evidence_state(raw_event.get("trickshot")),
                "cleanup_kill": _evidence_state(raw_event.get("cleanup_kill")),
                "opponent_likely_bot": _evidence_state(
                    raw_event.get("opponent_likely_bot")
                ),
                "stationary_target": stationary_target,
                "stationary_duration_supported": stationary_duration,
                "hipfire": aim_state == "hipfire",
                "damage_aim_state": aim_state,
                "finish_aim_state": aim_state,
                "summary": str(raw_event.get("summary", ""))[:240],
            }
        )

    # Event windows are independent. Never transfer finish credit between them;
    # an unknown target or a nearby weapon swap is not a causal identity link.
    events = _collapse_duplicate_target_finishes(events)
    for event in events:
        _sanitize_event_damage(event)

    labels = {key: "no" for key in _WEAPON_LABELS}
    attributable = [event for event in events if event["status"] == "attributed"]
    if not attributable:
        labels = {key: "uncertain" for key in _WEAPON_LABELS}
    for event in attributable:
        if event["target_was_downed"] or not event["target_was_active"]:
            continue
        weapon = event["resolved_weapon"]
        if weapon == "sniper_or_hunting":
            labels["sniper_kill"] = "yes"
        elif weapon == "shotgun":
            labels["shotgun_kill"] = "yes"
            damage = event["single_shot_damage"]
            if (
                event["damage_hit_count"] == 1
                and damage is not None
                and damage >= 100
                and not event["damage_display_is_cumulative"]
            ):
                labels["shotgun_one_pump"] = "yes"
        elif weapon == "pistol":
            labels["pistol_kill"] = "yes"
        elif weapon == "automatic":
            labels["automatic_kill"] = "yes"
            if event["damage_hit_count"] >= 3:
                labels["spray_kill"] = "yes"
        elif weapon == "other":
            labels["other_weapon_kill"] = "yes"
    if labels.get("sniper_kill") == "yes" and resolve_no_scope_for_events(events):
        labels["no_scope"] = "yes"

    evidence = payload.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    if attributable:
        status = "attributed"
    elif any(event["status"] == "no_visual_causality" for event in events):
        status = "no_visual_causality"
    else:
        status = "no_event"
    return WeaponAttribution(
        labels=labels,
        confidence=_bounded_float(payload.get("confidence")),
        evidence=[str(item) for item in evidence][:3],
        status=status,
        events=events,
        raw_payload=payload,
    )


_DEFERRED_FINISH_BIND_WINDOW_SEC = 2.0
_SAME_TARGET_FINISH_WINDOW_SEC = 2.0
_BINDABLE_DAMAGE_WEAPONS = {
    "sniper_or_hunting",
    "shotgun",
    "pistol",
    "automatic",
    "other",
}


def _collapse_duplicate_target_finishes(events: list[dict]) -> list[dict]:
    """Keep the latest attributed active finish per named target in a short window."""
    attributed = [
        event
        for event in events
        if event.get("status") == "attributed"
        and event.get("event_kind") in {"knock", "elimination"}
        and event.get("target_was_active")
        and not event.get("target_was_downed")
    ]
    by_target: dict[str, list[dict]] = {}
    for event in attributed:
        name = str(event.get("target_identity", "unknown")).strip().lower()
        if name in {"", "unknown"}:
            continue
        by_target.setdefault(name, []).append(event)

    for group in by_target.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item.get("finish_timestamp") is None,
                float(item["finish_timestamp"])
                if item.get("finish_timestamp") is not None
                else float("inf"),
                int(item.get("event_index", 0)),
            ),
        )
        kept: list[dict] = []
        for event in ordered:
            if not kept:
                kept.append(event)
                continue
            prior = kept[-1]
            prior_ts = prior.get("finish_timestamp")
            event_ts = event.get("finish_timestamp")
            if (
                prior_ts is not None
                and event_ts is not None
                and abs(float(event_ts) - float(prior_ts)) <= _SAME_TARGET_FINISH_WINDOW_SEC
                and prior.get("resolved_weapon") != event.get("resolved_weapon")
                and prior.get("resolved_weapon") not in {None, "unknown"}
                and event.get("resolved_weapon") not in {None, "unknown"}
            ):
                # Conflicting weapons on the same named finish: keep the later window.
                prior["status"] = "duplicate_target_finish"
                prior["resolved_weapon"] = "unknown"
                prior["duplicate_of_event_index"] = int(event.get("event_index", 0))
                kept[-1] = event
            else:
                kept.append(event)
    return events


def ocr_should_override_weapon(
    event: dict, ocr_evidence: dict, *, prefer_ocr: bool = False
) -> bool:
    """Apply OCR weapon when it fills a gap, agrees, or corrects silhouette-only VLM."""
    ocr_category = str(ocr_evidence.get("category", "unknown"))
    if ocr_category == "unknown":
        return False
    vlm_category = str(
        event.get("selected_weapon_before_finish", event.get("weapon_category", "unknown"))
    ).strip().lower()
    if vlm_category not in _WEAPON_CATEGORIES:
        vlm_category = "unknown"
    vlm_name_category = weapon_category_from_text(event.get("selected_weapon_name_text"))
    vlm_confidence = _bounded_float(event.get("weapon_confidence"))
    ocr_confidence = _bounded_float(ocr_evidence.get("confidence"))
    if prefer_ocr and ocr_confidence >= 0.9:
        # Strong local HUD text is authoritative over a visual family guess.
        return True
    if vlm_category == "unknown" and vlm_name_category == "unknown":
        return True
    if ocr_category in {vlm_category, vlm_name_category}:
        return True
    # Low-confidence silhouette with no HUD name: allow OCR correction.
    if vlm_confidence < 0.7 and vlm_name_category == "unknown":
        return True
    # Strong HUD text beats silhouette/category guesses when VLM never named a
    # known weapon family (e.g. invented AR names vs clear shotgun OCR).
    if ocr_confidence >= 0.9 and vlm_name_category == "unknown":
        return True
    # Two concrete named families disagree: keep the VLM name.
    return False


def apply_weapon_ocr_to_event(
    event: dict, ocr_evidence: dict, *, prefer_ocr: bool = False
) -> bool:
    """Mutate event with OCR weapon evidence when safe. Returns True if applied."""
    if ocr_evidence.get("ambiguous"):
        # Mid-swap window: do not invent a kill weapon from conflicting HUD reads.
        event["selected_weapon_before_finish"] = "unknown"
        event["selected_weapon_name_text"] = "unknown"
        event["weapon_confidence"] = 0.0
        event["local_ocr"] = {**ocr_evidence, "applied": False}
        return False
    if not ocr_should_override_weapon(
        event, ocr_evidence, prefer_ocr=prefer_ocr
    ):
        event["local_ocr"] = {
            **ocr_evidence,
            "applied": False,
            "reason": "conflicts_with_vlm_weapon",
        }
        return False
    event["selected_weapon_name_text"] = ocr_evidence["text"]
    event["selected_weapon_before_finish"] = ocr_evidence["category"]
    event["weapon_confidence"] = max(
        _bounded_float(event.get("weapon_confidence")),
        float(ocr_evidence["confidence"]),
    )
    event["local_ocr"] = {**ocr_evidence, "applied": True}
    return True


def _event_named_weapon(event: dict) -> str:
    named = weapon_category_from_text(event.get("selected_weapon_name_text"))
    if named != "unknown":
        return named
    resolved = str(event.get("resolved_weapon", "unknown"))
    return resolved if resolved in _BINDABLE_DAMAGE_WEAPONS else "unknown"


def _compatible_event_targets(left: dict, right: dict) -> bool:
    left_name = str(left.get("target_identity", "unknown")).strip().lower()
    right_name = str(right.get("target_identity", "unknown")).strip().lower()
    if left_name in {"", "unknown"} or right_name in {"", "unknown"}:
        return True
    return left_name == right_name


def _bind_deferred_finishes(events: list[dict]) -> list[dict]:
    """Attach a nearby finish UI to an earlier serious weapon hit after a swap.

    Specialist windows often mark the damaging shot as damage_only, then attribute
    the knock/elim banner in a later window to the swapped weapon. This is not
    weapon-specific: any named weapon with a serious hit can reclaim that finish.
    """
    if len(events) < 2:
        return events

    bound_donor_indices: set[int] = set()
    for damage in events:
        if (
            damage.get("status") != "no_finish"
            or damage.get("event_kind") != "damage_only"
        ):
            continue
        if damage.get("target_was_downed") or not damage.get("target_was_active"):
            continue
        if not damage.get("pov_shot_visible"):
            continue
        named_weapon = _event_named_weapon(damage)
        if named_weapon not in _BINDABLE_DAMAGE_WEAPONS:
            continue
        damage_value = damage.get("single_shot_damage")
        serious_hit = bool(damage.get("high_damage_one_shot")) or (
            damage_value is not None and int(damage_value) >= 100
        )
        # Automatic sprays can qualify via multiple distinct damaging shots.
        if named_weapon == "automatic" and not serious_hit:
            serious_hit = int(damage.get("damage_hit_count") or 0) >= 3
        if not serious_hit:
            continue
        damage_ts = damage.get("finish_timestamp")
        if damage_ts is None:
            continue

        donor: dict | None = None
        donor_gap = float("inf")
        for finish in events:
            if finish is damage or finish.get("status") != "attributed":
                continue
            if finish.get("event_kind") not in {"knock", "elimination"}:
                continue
            if finish.get("target_was_downed") or not finish.get("target_was_active"):
                continue
            finish_ts = finish.get("finish_timestamp")
            if finish_ts is None:
                continue
            gap = float(finish_ts) - float(damage_ts)
            if gap <= 0 or gap > _DEFERRED_FINISH_BIND_WINDOW_SEC:
                continue
            if not _compatible_event_targets(damage, finish):
                continue
            # Prefer swap-shaped donors; same-weapon donors are left alone.
            if finish.get("resolved_weapon") == named_weapon:
                continue
            if gap < donor_gap:
                donor = finish
                donor_gap = gap
        if donor is None:
            continue

        damage["event_kind"] = donor["event_kind"]
        damage["status"] = "attributed"
        damage["resolved_weapon"] = named_weapon
        damage["finish_onset_supported"] = True
        damage["visible_defeat_supported"] = bool(
            donor.get("visible_defeat_supported")
            or damage.get("visible_defeat_supported")
        )
        damage["kill_feed_corroborates_pov"] = bool(
            donor.get("kill_feed_corroborates_pov")
            or damage.get("kill_feed_corroborates_pov")
        )
        damage["visual_action_supported"] = True
        damage["deferred_finish_bound"] = True
        damage["bound_from_event_index"] = int(donor.get("event_index", 0))
        if donor.get("target_identity") and (
            str(damage.get("target_identity", "unknown")).lower() in {"", "unknown"}
        ):
            damage["target_identity"] = donor["target_identity"]

        donor_index = int(donor.get("event_index", -1))
        if donor_index in bound_donor_indices:
            continue
        donor["status"] = "deferred_finish_donor"
        donor["resolved_weapon"] = "unknown"
        donor["deferred_finish_donor"] = True
        bound_donor_indices.add(donor_index)

    return events


def _evidence_single_shot_damage(evidence: list[str]) -> int | None:
    values: list[int] = []
    for item in evidence:
        lowered = item.lower()
        for match in re.finditer(
            r"\b(\d{2,3})\b[^.;]{0,14}\b(?:damage|dmg)\b", lowered
        ):
            context = lowered[max(0, match.start() - 8) : match.end() + 8]
            if "xp" not in context:
                values.append(int(match.group(1)))
    return max(values) if values else None


def _evidence_indicates_downed_finish(evidence: list[str]) -> bool:
    text = " ".join(evidence).lower()
    return any(
        cue in text
        for cue in (
            "already downed",
            "already knocked",
            "downed opponent",
            "dbno",
            "crawling opponent",
            "finish on a knocked",
            "finish on knocked",
            "finishing a knocked",
            "finishing the knocked",
        )
    )


def merge_legacy_teacher_labels(
    base_labels: TeacherClipLabels,
    attribution: WeaponAttribution | None,
) -> TeacherClipLabels:
    """Reproduce historical canonical labels without affecting current inference."""
    merged = base_labels.model_dump()
    weapon_values = (
        attribution.labels
        if attribution is not None
        else {key: "uncertain" for key in _WEAPON_LABELS}
    )

    global_event_values = (
        base_labels.combat_visible,
        base_labels.enemy_visible,
        base_labels.elimination_or_knock,
    )
    attribution_events = attribution.events if attribution else []
    direct_weapons = {
        str(event.get("resolved_weapon"))
        for event in attribution_events
        if event.get("status") == "attributed"
        and not event.get("target_was_downed", False)
    }
    has_explicit_active_kill = any(
        event.get("status") == "attributed"
        and event.get("target_was_active", False)
        and not event.get("target_was_downed", False)
        for event in attribution_events
    )
    event_downed_finish = any(
        event.get("target_was_downed", False) for event in attribution_events
    )
    evidence_downed_finish = _evidence_indicates_downed_finish(base_labels.evidence)
    downed_finish = event_downed_finish or evidence_downed_finish
    # A downed finish in one event must not suppress a valid kill in another event.
    suppress_weapon_kills = (event_downed_finish and not direct_weapons) or (
        evidence_downed_finish and not has_explicit_active_kill
    )
    merged["downed_finish"] = "yes" if downed_finish else "no"

    if any(value != "yes" for value in global_event_values):
        fallback = "no" if "no" in global_event_values else "uncertain"
        weapon_values = {key: fallback for key in _WEAPON_LABELS}
    else:
        weapon_values = dict(weapon_values)
        evidence_items = [item.lower() for item in base_labels.evidence]
        strong_v8_sniper_evidence = any(
            "kill feed" in item
            and any(cue in item for cue in ("sniped", "sniper", "hunting rifle"))
            and any(cue in item for cue in ("knock", "elimin", "banner", "xp"))
            for item in evidence_items
        )
        if base_labels.multi_kill == "yes" and base_labels.sniper_kill == "yes":
            weapon_values["sniper_kill"] = "yes"
        elif strong_v8_sniper_evidence and base_labels.sniper_kill == "yes":
            weapon_values["sniper_kill"] = "yes"
            for key in _WEAPON_LABELS:
                if key not in {"sniper_kill", "no_scope"}:
                    weapon_values[key] = "no"
        elif base_labels.sniper_kill == "yes" and not direct_weapons:
            weapon_values["sniper_kill"] = "yes"

        if suppress_weapon_kills:
            weapon_values = {key: "no" for key in _WEAPON_LABELS}

    event_single_shot_damage = max(
        (
            int(event["single_shot_damage"])
            for event in attribution_events
            if event.get("single_shot_damage") is not None
            and not event.get("target_was_downed", False)
        ),
        default=0,
    )
    evidence_single_shot_damage = (
        _evidence_single_shot_damage(base_labels.evidence) or 0
    )
    if "no" in global_event_values or suppress_weapon_kills:
        merged["high_damage_hit"] = "no"
    elif weapon_values.get("shotgun_one_pump") == "yes":
        merged["high_damage_hit"] = "yes"
    elif max(event_single_shot_damage, evidence_single_shot_damage) >= 100:
        merged["high_damage_hit"] = "yes"
    else:
        merged["high_damage_hit"] = "no"

    evidence_text = f" {' '.join(base_labels.evidence).lower()} "
    ads_cues = (
        " aiming down sights ",
        " ads ",
        " scoped ",
        " scope overlay ",
        " hunting rifle ads ",
    )
    if weapon_values.get("no_scope") == "yes" and any(
        cue in evidence_text for cue in ads_cues
    ):
        weapon_values["no_scope"] = "no"
    if weapon_values.get("sniper_kill") != "yes":
        weapon_values["no_scope"] = "no"

    for key in _WEAPON_LABELS:
        merged[key] = weapon_values[key]
    return normalize_teacher_payload(merged)


_EVENT_OWNED_LABELS = {
    "elimination_or_knock",
    "high_damage_hit",
    "no_scope",
    "downed_finish",
    "spray_kill",
    "sniper_kill",
    "shotgun_kill",
    "shotgun_one_pump",
    "pistol_kill",
    "automatic_kill",
    "other_weapon_kill",
    "multi_kill",
    "flick_shot",
    "trickshot",
    "cleanup_kill",
    "opponent_likely_bot",
    "stationary_target",
    "stationary_sniper_target",
}


def merge_event_labels(
    context_labels: TeacherClipLabels,
    attribution: WeaponAttribution | None,
) -> TeacherClipLabels:
    """Combine context with event facts without allowing context to override causality."""
    merged = context_labels.model_dump()
    if attribution is None or attribution.status == "incomplete":
        for key in _EVENT_OWNED_LABELS:
            merged[key] = "uncertain"
        return normalize_teacher_payload(merged)

    events = attribution.events
    if not events:
        for key in _EVENT_OWNED_LABELS:
            merged[key] = "uncertain"
        return normalize_teacher_payload(merged)

    verified_finishes = dedupe_verified_finishes(events)
    active_finishes = [
        event
        for event in verified_finishes
        if event.get("target_was_active", False)
        and not event.get("target_was_downed", False)
    ]
    downed_finishes = [
        event for event in verified_finishes if event.get("target_was_downed", False)
    ]
    active_action_events = [
        event
        for event in events
        if event.get("visual_action_supported", False)
        and event.get("target_was_active", False)
        and not event.get("target_was_downed", False)
    ]
    unresolved = any(
        event.get("status") in {"low_confidence", "target_state_unknown"}
        for event in events
    )

    # Active-target knocks/eliminations only. Downed finishes are tracked separately.
    merged["elimination_or_knock"] = (
        "yes" if active_finishes else "uncertain" if unresolved else "no"
    )
    merged["downed_finish"] = "yes" if downed_finishes else "no"
    merged["multi_kill"] = "yes" if len(active_finishes) >= 2 else "no"

    def aggregate_event_state(key: str, candidates: list[dict]) -> str:
        values = {str(event.get(key, "unknown")) for event in candidates}
        if "yes" in values:
            return "yes"
        return "uncertain" if "unknown" in values else "no"

    def aggregate_negative_state(key: str, candidates: list[dict]) -> str:
        values = [str(event.get(key, "unknown")) for event in candidates]
        if not values:
            return "no"
        if all(value == "yes" for value in values):
            return "yes"
        if any(value == "no" for value in values):
            return "no"
        return "uncertain"

    merged["flick_shot"] = aggregate_event_state("flick_shot", active_action_events)
    merged["trickshot"] = aggregate_event_state("trickshot", active_finishes)
    merged["cleanup_kill"] = aggregate_negative_state("cleanup_kill", active_finishes)
    merged["opponent_likely_bot"] = aggregate_negative_state(
        "opponent_likely_bot", active_finishes
    )
    merged["stationary_target"] = aggregate_negative_state(
        "stationary_target", active_action_events
    )

    for key in _WEAPON_LABELS:
        merged[key] = attribution.labels.get(key, "uncertain")

    high_damage = any(
        (event.get("single_shot_damage") or 0) >= 100
        and not event.get("damage_display_is_cumulative", False)
        and not event.get("target_was_downed", False)
        and (
            event.get("visual_action_supported", False)
            or event.get("pov_shot_visible", False)
        )
        for event in events
    )
    merged["high_damage_hit"] = "yes" if high_damage else "no"

    if not active_finishes:
        for key in _WEAPON_LABELS:
            if merged[key] == "yes":
                merged[key] = "no"
    if merged["sniper_kill"] != "yes":
        merged["no_scope"] = "no"
        merged["stationary_sniper_target"] = "no"
    else:
        merged["no_scope"] = "yes" if resolve_no_scope_for_events(events) else "no"
        sniper_events = [
            event
            for event in active_action_events
            if event.get("resolved_weapon") == "sniper_or_hunting"
        ]
        merged["stationary_sniper_target"] = aggregate_negative_state(
            "stationary_target", sniper_events
        )
    return normalize_teacher_payload(merged)


def prepare_weapon_payload(payload: dict, clip: ClipTeacherInput) -> dict:
    """Fill deterministic frame metadata the model may omit from its JSON."""
    prepared = dict(payload)
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raw_frames = payload.get("frame_analyses", payload.get("frame_analysis", []))
    if isinstance(raw_frames, dict):
        raw_frames = list(raw_frames.values())
    if not isinstance(raw_frames, list):
        raw_frames = []

    frames: list[dict] = []
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            continue
        frame = dict(raw_frame)
        if _timestamp(frame.get("timestamp")) is None and index < len(
            clip.image_timestamps
        ):
            frame["timestamp"] = clip.image_timestamps[index]
        if "event_index" not in frame and index < len(clip.image_event_indices):
            frame["event_index"] = clip.image_event_indices[index]
        frames.append(frame)
    prepared["frames"] = frames
    return prepared


class OpenAICompatibleTeacherClient:
    def __init__(
        self, *, provider: str, api_key: str | None, base_url: str | None, model: str
    ):
        if not api_key:
            raise FireworksNotConfiguredError(
                f"{provider} API key is required for VLM feature extraction."
            )
        if not base_url:
            raise FireworksNotConfiguredError(
                f"{provider} base URL is required for VLM feature extraction."
            )
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def label_clip(self, clip: ClipTeacherInput) -> TeacherClipLabels:
        return normalize_teacher_payload(
            extract_json_object(
                self._complete(build_teacher_content(clip), max_tokens=2000)
            )
        )

    def locate_event_timestamps(self, clip: ClipTeacherInput) -> list[float]:
        event_limit = max_event_candidates(clip.duration_sec)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    '/no_think\nReturn only JSON: {"event_timestamps": [number]}. '
                    f"These are chronological coarse Fortnite frames. Find up to {event_limit} timestamps "
                    "where a POV damage, knock, or elimination FIRST appears. Ignore persistent "
                    "banners after their event."
                ),
            }
        ]
        for index, path in enumerate(clip.image_paths):
            timestamp = (
                clip.image_timestamps[index]
                if index < len(clip.image_timestamps)
                else 0.0
            )
            content.extend(
                [
                    {"type": "text", "text": f"Frame at {timestamp:.2f}s"},
                    {"type": "image_url", "image_url": {"url": _data_url(path)}},
                ]
            )
        return parse_event_timestamps(
            self._complete(content, max_tokens=max(300, event_limit * 70)),
            limit=event_limit,
        )

    def label_weapon_event(self, clip: ClipTeacherInput) -> WeaponAttribution:
        content = build_teacher_content(clip)
        expected_event_indices = sorted(
            {int(index) for index in clip.image_event_indices if index is not None}
        )
        expected_event_count = len(expected_event_indices)
        content[0] = {
            "type": "text",
            "text": (
                "/no_think\nReturn only compact JSON with events, confidence, and evidence. "
                f"Return exactly one event summary for each event_index in {expected_event_indices}. "
                "Each event summary must contain: event_index; event_timestamp; event_kind "
                "(none, damage_only, knock, elimination, downed_finish); target_state "
                "(active, already_downed, unknown); visual_action_supported (yes, no, unknown); "
                "pov_shot_visible; target_reaction_visible; new_damage_visible; "
                "finish_ui_newly_appeared; target_defeat_visible; kill_feed_corroborates_pov; "
                "flick_shot; trickshot; cleanup_kill; opponent_likely_bot; stationary_target; "
                "stationary_duration_supported; "
                "selected_weapon_before_finish (pistol, sniper_or_hunting, shotgun, automatic, "
                "other, unknown); selected_weapon_name_text (exact visible HUD weapon name or "
                "unknown); weapon_confidence (0 to 1); aim_state_at_shot (hipfire, "
                "ads_no_scope_overlay, scope_overlay, other_ads, unknown); single_shot_damage "
                "(integer or null); damage_display_is_cumulative (boolean); target_identity "
                "(visible target name or unknown); high_damage_one_shot (boolean); damaging_shot_count "
                "(integer); and summary (one short sentence). All evidence fields use yes, no, "
                "or unknown. Keep numbered events completely independent. Determine the weapon "
                "selected at the damaging shot immediately before that event's finish; ignore "
                "weapons selected afterward. If the HUD weapon changes between the shot and the "
                "finish banner, use the weapon from the shot frames, not the swapped weapon. "
                "If two different weapons are both clearly selected in the same event window, "
                "set selected_weapon_before_finish to unknown and weapon_confidence below 0.7. "
                "Transcribe the on-screen weapon name exactly when visible; never invent or "
                "paraphrase names (for example do not turn Extending Focus Shotgun into a made-up "
                "rifle name). Explicit HUD name text is stronger than a silhouette guess. "
                "An already knocked, DBNO, or crawling target is a "
                "downed_finish, not a new weapon kill. The center panel emphasizes the physical "
                "action. visual_action_supported is yes only when the POV shot plus a target "
                "reaction or newly appearing damage is visible. target_defeat_visible means the "
                "target model visibly enters a knocked, defeated, or disappearing state; it does "
                "not mean a banner appeared. A kill-feed line, elimination "
                "banner, reward, or other UI alone never proves the POV action. UI can corroborate "
                "only when it newly appears during the visible action. Any ADS without a scope "
                "overlay, including Hunting Rifle ADS, uses ads_no_scope_overlay, not hipfire. "
                "Count distinct damaging shots, not repeated frames "
                "showing the same persistent number. Do not treat accumulated hit-marker, XP, "
                "shield-crack, or headshot totals as single_shot_damage. If the single-hit number "
                "is unclear, set single_shot_damage to null rather than guessing. "
                "flick_shot requires a "
                "visible fast crosshair/camera snap onto the target immediately before impact. "
                "trickshot requires a visibly intentional stylish or difficult action. "
                "cleanup_kill applies only to an active but already weak, exposed, easy opponent; "
                "never use it for an already-down target. stationary_target is yes only when the "
                "target remains barely moving or non-reactive across multiple consecutive "
                "pre-impact frames or a clearly sustained interval. A single frozen frame, brief "
                "pause, or the instant of impact is not stationary; set stationary_duration_supported "
                "yes only when that sustained evidence is visible. opponent_likely_bot requires "
                "explicit AI evidence or bot-like behavior plus a bot-like name pattern; standing "
                "still alone is not enough."
            ),
        }
        attempts: list[dict] = []
        prepared: dict = {"events": []}
        for attempt in range(2):
            value = extract_json_value(self._complete(content, max_tokens=1200))
            payload = value if isinstance(value, dict) else {"events": value}
            attempts.append(payload)
            raw_events = payload.get("events", [])
            if isinstance(raw_events, dict):
                raw_events = list(raw_events.values())
            prepared = dict(payload)
            prepared["events"] = raw_events if isinstance(raw_events, list) else []
            event_centers: dict[int, float] = {}
            for event_index in expected_event_indices:
                centers = [
                    center
                    for center, index in zip(
                        clip.image_event_centers,
                        clip.image_event_indices,
                        strict=False,
                    )
                    if index == event_index and center is not None
                ]
                if centers:
                    event_centers[event_index] = float(centers[0])
            for event in prepared["events"]:
                if not isinstance(event, dict):
                    continue
                event_index = _event_index(event.get("event_index"))
                if _timestamp(event.get("event_timestamp")) is None:
                    event["event_timestamp"] = event_centers.get(event_index)
            returned_indices = [
                _event_index(event.get("event_index"))
                for event in prepared["events"]
                if isinstance(event, dict)
            ]
            if len(returned_indices) == expected_event_count and set(
                returned_indices
            ) == set(expected_event_indices):
                break
            if attempt == 0:
                content[0] = {
                    "type": "text",
                    "text": (
                        content[0]["text"]
                        + " Your previous format was incomplete. Return the complete events array "
                        "with one summary per requested event_index."
                    ),
                }

        returned_indices = [
            _event_index(event.get("event_index"))
            for event in prepared.get("events", [])
            if isinstance(event, dict)
        ]
        response_complete = len(returned_indices) == expected_event_count and set(
            returned_indices
        ) == set(expected_event_indices)
        if not response_complete and expected_event_count > 1:
            focused_events: list[dict] = []
            full_prompt = str(content[0]["text"])
            for event_index in expected_event_indices:
                positions = [
                    index
                    for index, value in enumerate(clip.image_event_indices)
                    if value == event_index
                ]
                focused_clip = ClipTeacherInput(
                    filename=clip.filename,
                    duration_sec=clip.duration_sec,
                    width=clip.width,
                    height=clip.height,
                    fps=clip.fps,
                    tags=clip.tags,
                    image_paths=[clip.image_paths[index] for index in positions],
                    image_timestamps=[
                        clip.image_timestamps[index] for index in positions
                    ],
                    image_views=[
                        clip.image_views[index]
                        if index < len(clip.image_views)
                        else "event composite"
                        for index in positions
                    ],
                    image_event_indices=[event_index for _ in positions],
                    image_event_centers=[
                        clip.image_event_centers[index]
                        if index < len(clip.image_event_centers)
                        else None
                        for index in positions
                    ],
                )
                focused_content = build_teacher_content(focused_clip)
                focused_content[0] = {
                    "type": "text",
                    "text": (
                        full_prompt.replace(
                            str(expected_event_indices), str([event_index]), 1
                        )
                        + " Focus only on this one event and return its complete summary."
                    ),
                }
                value = extract_json_value(
                    self._complete(focused_content, max_tokens=700)
                )
                payload = value if isinstance(value, dict) else {"events": value}
                attempts.append(payload)
                raw_events = payload.get("events", [])
                if isinstance(raw_events, dict):
                    raw_events = list(raw_events.values())
                if not isinstance(raw_events, list) or not raw_events:
                    if "event_index" in payload and "event_kind" in payload:
                        raw_events = [payload]
                    else:
                        continue
                matching = next(
                    (
                        event
                        for event in raw_events
                        if isinstance(event, dict)
                        and _event_index(event.get("event_index")) == event_index
                    ),
                    None,
                )
                if matching is not None:
                    focused_events.append(matching)
            prepared = {"events": focused_events}
            for event in prepared["events"]:
                event_index = _event_index(event.get("event_index"))
                if _timestamp(event.get("event_timestamp")) is None:
                    event["event_timestamp"] = event_centers.get(event_index)

        applied_ocr: list[dict] = []
        rejected_ocr: list[dict] = []
        for event in prepared.get("events", []):
            if not isinstance(event, dict):
                continue
            event_index = _event_index(event.get("event_index"))
            ocr_evidence = best_weapon_ocr(
                clip.ocr_observations,
                event_index,
                event_timestamp=_timestamp(event.get("event_timestamp")),
            )
            if ocr_evidence is None:
                continue
            if apply_weapon_ocr_to_event(event, ocr_evidence):
                applied_ocr.append({"event_index": event_index, **ocr_evidence})
            else:
                rejected_ocr.append(
                    {
                        "event_index": event_index,
                        **ocr_evidence,
                        "reason": (
                            "ambiguous_weapons_in_window"
                            if ocr_evidence.get("ambiguous")
                            else "conflicts_with_vlm_weapon"
                        ),
                    }
                )

        resolved = resolve_event_summaries(prepared)
        returned_indices = [
            _event_index(event.get("event_index"))
            for event in prepared.get("events", [])
            if isinstance(event, dict)
        ]
        response_complete = len(returned_indices) == expected_event_count and set(
            returned_indices
        ) == set(expected_event_indices)
        return WeaponAttribution(
            labels=(
                resolved.labels
                if response_complete
                else {key: "uncertain" for key in _WEAPON_LABELS}
            ),
            confidence=resolved.confidence,
            evidence=resolved.evidence,
            status=resolved.status if response_complete else "incomplete",
            events=resolved.events,
            raw_payload={
                "action_evidence_version": 2,
                "attempts": attempts,
                "prepared_event_count": len(prepared.get("events", [])),
                "expected_event_count": expected_event_count,
                "ocr_observations": clip.ocr_observations,
                "applied_ocr": applied_ocr,
                "rejected_ocr": rejected_ocr,
            },
        )

    def _complete(self, content: list[dict], *, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "system",
                    "content": "You output only parseable JSON. No reasoning text or markdown.",
                },
                {"role": "user", "content": content},
            ],
        }
        for attempt in range(4):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < 3:
                    retry_after = (
                        exc.headers.get("Retry-After") if exc.headers else None
                    )
                    try:
                        delay = (
                            float(retry_after)
                            if retry_after is not None
                            else 2 ** (attempt + 1)
                        )
                    except ValueError:
                        delay = 2 ** (attempt + 1)
                    time.sleep(max(1.0, min(30.0, delay)))
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{self.provider} VLM request failed: {exc.code} {body}"
                ) from exc

        return response_payload["choices"][0]["message"]["content"]


class FireworksTeacherClient(OpenAICompatibleTeacherClient):
    def __init__(self, settings: Settings):
        super().__init__(
            provider="fireworks",
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
            model=settings.fireworks_model,
        )


class AmdDeveloperCloudTeacherClient(OpenAICompatibleTeacherClient):
    def __init__(self, settings: Settings):
        super().__init__(
            provider="amd-developer-cloud",
            api_key=settings.amd_developer_cloud_api_key,
            base_url=settings.amd_developer_cloud_base_url,
            model=settings.amd_developer_cloud_model,
        )
