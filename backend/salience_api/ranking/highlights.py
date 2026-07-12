from __future__ import annotations

from dataclasses import dataclass

HIGHLIGHT_HIERARCHY = (
    "no_scope",
    "shotgun_one_pump",
    "spray_kill",
    "high_damage_hit",
    "active_finish",
    "downed_finish",
)

_WEAPON_DISPLAY = {
    "sniper_or_hunting": "sniper-class weapon",
    "shotgun": "shotgun",
    "pistol": "pistol",
    "automatic": "automatic weapon",
    "other": "weapon",
    "unknown": "weapon",
}

_AIM_ADS = {
    "ads_no_scope_overlay",
    "hunting_ads_zoom",
    "scope_overlay",
    "other_ads",
}


@dataclass(frozen=True)
class RankedHighlightEvent:
    event_index: int
    timestamp: float | None
    highlight_type: str
    labels: tuple[str, ...]
    active: bool
    downed: bool
    resolved_weapon: str
    target_state: str
    damage_aim_state: str
    finish_aim_state: str
    single_shot_damage: int | None
    damage_hit_count: int
    summary: str
    accepted_reasons: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    event_kind: str = "none"


@dataclass(frozen=True)
class HighlightProfile:
    primary: RankedHighlightEvent | None
    secondary: tuple[RankedHighlightEvent, ...] = ()
    rejected: tuple[RankedHighlightEvent, ...] = ()
    active_finish_count: int = 0
    multi_kill: bool = False


def dedupe_verified_finishes(events: list[dict]) -> list[dict]:
    verified = [
        event
        for event in events
        if event.get("status") == "attributed"
        and event.get("event_kind") in {"knock", "elimination", "downed_finish"}
    ]
    deduplicated: list[dict] = []
    for event in sorted(
        verified,
        key=lambda item: (
            item.get("finish_timestamp") is None,
            item.get("finish_timestamp")
            if item.get("finish_timestamp") is not None
            else float("inf"),
            int(item.get("event_index", 0)),
        ),
    ):
        duplicate = False
        for prior in deduplicated:
            if _events_are_duplicates(prior, event):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(event)
    return deduplicated


def _events_are_duplicates(left: dict, right: dict) -> bool:
    left_ts = left.get("finish_timestamp")
    right_ts = right.get("finish_timestamp")
    if left_ts is None or right_ts is None:
        # Fall back to named-target proximity when timestamps are missing.
        same_named_target = _same_named_target(left, right)
        if not same_named_target:
            return False
        return (
            abs(int(left.get("event_index", 0)) - int(right.get("event_index", 0))) <= 1
        )
    close_in_time = abs(float(left_ts) - float(right_ts)) <= 1.1
    if not close_in_time:
        return False
    if _same_named_target(left, right):
        return True
    return (
        left.get("resolved_weapon") == right.get("resolved_weapon")
        and left.get("event_kind") == right.get("event_kind")
        and left.get("single_shot_damage") == right.get("single_shot_damage")
    )


def _same_named_target(left: dict, right: dict) -> bool:
    left_name = str(left.get("target_identity", "unknown")).lower()
    right_name = str(right.get("target_identity", "unknown")).lower()
    return left_name not in {"", "unknown"} and left_name == right_name


def _event_labels(event: dict) -> list[str]:
    labels: list[str] = []
    weapon = str(event.get("resolved_weapon", "unknown"))
    downed = bool(event.get("target_was_downed", False))
    active = bool(event.get("target_was_active", False)) and not downed
    if downed:
        labels.append("downed_finish")
        return labels
    if not active:
        return labels
    if weapon == "sniper_or_hunting":
        labels.append("sniper_kill")
        if (
            event.get("damage_aim_state") == "hipfire"
            and event.get("finish_aim_state") == "hipfire"
            and not _sniper_has_ads_contradiction(event, [event])
        ):
            labels.append("no_scope")
    elif weapon == "shotgun":
        labels.append("shotgun_kill")
        damage = event.get("single_shot_damage")
        if (
            int(event.get("damage_hit_count") or 0) == 1
            and damage is not None
            and int(damage) >= 100
            and not event.get("damage_display_is_cumulative", False)
        ):
            labels.append("shotgun_one_pump")
    elif weapon == "pistol":
        labels.append("pistol_kill")
    elif weapon == "automatic":
        labels.append("automatic_kill")
        if int(event.get("damage_hit_count") or 0) >= 3:
            labels.append("spray_kill")
    elif weapon == "other":
        labels.append("other_weapon_kill")
    else:
        labels.append("active_finish")
    damage = event.get("single_shot_damage")
    if (
        damage is not None
        and int(damage) >= 100
        and not event.get("damage_display_is_cumulative", False)
    ):
        labels.append("high_damage_hit")
    return labels


def _sniper_has_ads_contradiction(event: dict, cluster: list[dict]) -> bool:
    if event.get("resolved_weapon") != "sniper_or_hunting":
        return False
    for candidate in cluster:
        if candidate.get("resolved_weapon") != "sniper_or_hunting":
            continue
        if not _events_are_duplicates(event, candidate) and candidate is not event:
            # Also treat near-duplicate same-target sniper shots as the same cluster.
            if not _same_named_target(event, candidate):
                left_ts = event.get("finish_timestamp")
                right_ts = candidate.get("finish_timestamp")
                if left_ts is None or right_ts is None:
                    continue
                if abs(float(left_ts) - float(right_ts)) > 1.1:
                    continue
        aim = str(candidate.get("damage_aim_state", "unknown"))
        if aim in _AIM_ADS:
            return True
    return False


def _highlight_type(labels: list[str], *, downed: bool, active: bool) -> str:
    if downed:
        return "downed_finish"
    for item in HIGHLIGHT_HIERARCHY:
        if item in labels:
            return item
    if active:
        return "active_finish"
    return "downed_finish"


def _hierarchy_rank(highlight_type: str) -> int:
    try:
        return HIGHLIGHT_HIERARCHY.index(highlight_type)
    except ValueError:
        return len(HIGHLIGHT_HIERARCHY)


def _target_state(event: dict) -> str:
    if event.get("target_was_downed", False):
        return "already_downed"
    if event.get("target_was_active", False):
        return "active"
    return "unknown"


def _weapon_label(event: dict) -> str:
    name = str(event.get("selected_weapon_name_text") or "").strip()
    if name and name.lower() != "unknown":
        cleaned = " ".join(name.split())
        # Prefer a short readable HUD name.
        return cleaned.title() if cleaned.isupper() else cleaned
    return _WEAPON_DISPLAY.get(str(event.get("resolved_weapon", "unknown")), "weapon")


def _format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "?:??"
    total = max(0, int(round(float(timestamp))))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}:{seconds:02d}"


def _accepted_reasons(event: dict, labels: list[str]) -> list[str]:
    reasons = ["verified attributed finish"]
    if event.get("target_was_active", False) and not event.get(
        "target_was_downed", False
    ):
        reasons.append("active target")
    if event.get("visual_action_supported", False):
        reasons.append("visible POV action")
    if "no_scope" in labels:
        reasons.append("hipfire sniper shot")
    if "shotgun_one_pump" in labels:
        reasons.append("single shotgun hit >= 100 damage")
    if "spray_kill" in labels:
        reasons.append("automatic weapon with >= 3 damaging shots")
    return reasons


def _rejected_reasons(event: dict) -> list[str]:
    reasons: list[str] = []
    status = str(event.get("status", ""))
    if status != "attributed":
        reasons.append(f"rejected status: {status or 'unknown'}")
    if event.get("target_was_downed", False):
        reasons.append("downed target cannot create new weapon-kill credit")
    if not event.get("visual_action_supported", False):
        reasons.append("missing visible POV action")
    if event.get("event_kind") not in {"knock", "elimination", "downed_finish"}:
        reasons.append("not a finish event")
    return reasons or ["not selected as highlight"]


def _to_ranked_event(
    event: dict, *, labels: list[str] | None = None, rejected: bool = False
) -> RankedHighlightEvent:
    downed = bool(event.get("target_was_downed", False))
    active = bool(event.get("target_was_active", False)) and not downed
    resolved_labels = labels if labels is not None else _event_labels(event)
    highlight_type = _highlight_type(resolved_labels, downed=downed, active=active)
    return RankedHighlightEvent(
        event_index=int(event.get("event_index", 0)),
        timestamp=(
            float(event["finish_timestamp"])
            if event.get("finish_timestamp") is not None
            else None
        ),
        highlight_type=highlight_type,
        labels=tuple(resolved_labels),
        active=active,
        downed=downed,
        resolved_weapon=str(event.get("resolved_weapon", "unknown")),
        target_state=_target_state(event),
        damage_aim_state=str(event.get("damage_aim_state", "unknown")),
        finish_aim_state=str(event.get("finish_aim_state", "unknown")),
        single_shot_damage=(
            int(event["single_shot_damage"])
            if event.get("single_shot_damage") is not None
            else None
        ),
        damage_hit_count=int(event.get("damage_hit_count") or 0),
        summary=str(event.get("summary", "")),
        accepted_reasons=tuple(
            [] if rejected else _accepted_reasons(event, resolved_labels)
        ),
        rejected_reasons=tuple(
            _rejected_reasons(event)
            if rejected
            else (
                ("downed target cannot create new weapon-kill credit",)
                if downed
                else ()
            )
        ),
        event_kind=str(event.get("event_kind", "none")),
    )


def resolve_no_scope_for_events(events: list[dict]) -> bool:
    """Return True only when a sniper finish is hipfire with no ADS contradiction."""
    finishes = dedupe_verified_finishes(events)
    sniper_finishes = [
        event
        for event in finishes
        if event.get("resolved_weapon") == "sniper_or_hunting"
        and event.get("target_was_active", False)
        and not event.get("target_was_downed", False)
    ]
    if not sniper_finishes:
        return False
    for event in sniper_finishes:
        if event.get("damage_aim_state") != "hipfire":
            continue
        if _sniper_has_ads_contradiction(event, events):
            continue
        return True
    return False


def build_highlight_profile(events: list[dict]) -> HighlightProfile:
    finishes = dedupe_verified_finishes(events)
    active_finishes = [
        event
        for event in finishes
        if event.get("target_was_active", False)
        and not event.get("target_was_downed", False)
    ]
    downed_finishes = [
        event for event in finishes if event.get("target_was_downed", False)
    ]
    ranked_active = [_to_ranked_event(event) for event in active_finishes]
    # Apply clip-level no-scope veto across near-duplicate sniper shots.
    adjusted: list[RankedHighlightEvent] = []
    for ranked, source in zip(ranked_active, active_finishes, strict=True):
        labels = list(ranked.labels)
        if "no_scope" in labels and _sniper_has_ads_contradiction(source, events):
            labels = [label for label in labels if label != "no_scope"]
            ranked = RankedHighlightEvent(
                event_index=ranked.event_index,
                timestamp=ranked.timestamp,
                highlight_type=_highlight_type(labels, downed=False, active=True),
                labels=tuple(labels),
                active=True,
                downed=False,
                resolved_weapon=ranked.resolved_weapon,
                target_state=ranked.target_state,
                damage_aim_state=ranked.damage_aim_state,
                finish_aim_state=ranked.finish_aim_state,
                single_shot_damage=ranked.single_shot_damage,
                damage_hit_count=ranked.damage_hit_count,
                summary=ranked.summary,
                accepted_reasons=tuple(_accepted_reasons(source, labels)),
                rejected_reasons=(),
                event_kind=ranked.event_kind,
            )
        adjusted.append(ranked)
    ranked_active = adjusted
    ranked_downed = [
        RankedHighlightEvent(
            event_index=int(event.get("event_index", 0)),
            timestamp=(
                float(event["finish_timestamp"])
                if event.get("finish_timestamp") is not None
                else None
            ),
            highlight_type="downed_finish",
            labels=("downed_finish",),
            active=False,
            downed=True,
            resolved_weapon=str(event.get("resolved_weapon", "unknown")),
            target_state="already_downed",
            damage_aim_state=str(event.get("damage_aim_state", "unknown")),
            finish_aim_state=str(event.get("finish_aim_state", "unknown")),
            single_shot_damage=(
                int(event["single_shot_damage"])
                if event.get("single_shot_damage") is not None
                else None
            ),
            damage_hit_count=int(event.get("damage_hit_count") or 0),
            summary=str(event.get("summary", "")),
            accepted_reasons=("verified downed finish",),
            rejected_reasons=("downed target cannot create new weapon-kill credit",),
            event_kind=str(event.get("event_kind", "downed_finish")),
        )
        for event in downed_finishes
    ]

    candidates = ranked_active + ranked_downed
    if not candidates:
        rejected = [
            RankedHighlightEvent(
                event_index=int(event.get("event_index", 0)),
                timestamp=(
                    float(event["finish_timestamp"])
                    if event.get("finish_timestamp") is not None
                    else None
                ),
                highlight_type="active_finish",
                labels=(),
                active=False,
                downed=bool(event.get("target_was_downed", False)),
                resolved_weapon=str(event.get("resolved_weapon", "unknown")),
                target_state=_target_state(event),
                damage_aim_state=str(event.get("damage_aim_state", "unknown")),
                finish_aim_state=str(event.get("finish_aim_state", "unknown")),
                single_shot_damage=(
                    int(event["single_shot_damage"])
                    if event.get("single_shot_damage") is not None
                    else None
                ),
                damage_hit_count=int(event.get("damage_hit_count") or 0),
                summary=str(event.get("summary", "")),
                accepted_reasons=(),
                rejected_reasons=tuple(_rejected_reasons(event)),
                event_kind=str(event.get("event_kind", "none")),
            )
            for event in events
        ]
        return HighlightProfile(primary=None, rejected=tuple(rejected))

    candidates.sort(
        key=lambda item: (
            0 if item.active else 1,
            _hierarchy_rank(item.highlight_type),
            item.timestamp if item.timestamp is not None else float("inf"),
            item.event_index,
        )
    )
    primary = candidates[0]
    secondary = tuple(candidates[1:])
    rejected = tuple(
        RankedHighlightEvent(
            event_index=int(event.get("event_index", 0)),
            timestamp=(
                float(event["finish_timestamp"])
                if event.get("finish_timestamp") is not None
                else None
            ),
            highlight_type="active_finish",
            labels=(),
            active=False,
            downed=bool(event.get("target_was_downed", False)),
            resolved_weapon=str(event.get("resolved_weapon", "unknown")),
            target_state=_target_state(event),
            damage_aim_state=str(event.get("damage_aim_state", "unknown")),
            finish_aim_state=str(event.get("finish_aim_state", "unknown")),
            single_shot_damage=(
                int(event["single_shot_damage"])
                if event.get("single_shot_damage") is not None
                else None
            ),
            damage_hit_count=int(event.get("damage_hit_count") or 0),
            summary=str(event.get("summary", "")),
            accepted_reasons=(),
            rejected_reasons=tuple(_rejected_reasons(event)),
            event_kind=str(event.get("event_kind", "none")),
        )
        for event in events
        if event.get("status") != "attributed"
        or event.get("event_kind") not in {"knock", "elimination", "downed_finish"}
    )
    return HighlightProfile(
        primary=primary,
        secondary=secondary,
        rejected=rejected,
        active_finish_count=len(ranked_active),
        multi_kill=len(ranked_active) >= 2,
    )


def _describe_event(event: RankedHighlightEvent) -> str:
    weapon = _WEAPON_DISPLAY.get(event.resolved_weapon, "weapon")
    result = "eliminates" if event.event_kind == "elimination" else "knocks"
    if event.downed:
        return f"finishes an already-downed opponent with a {weapon}"
    bits = [weapon]
    if event.highlight_type == "no_scope":
        bits.insert(0, "no-scope")
    if event.single_shot_damage is not None and event.single_shot_damage >= 100:
        if "headshot" in event.summary.lower():
            action = f"headshot {result} an active opponent for {event.single_shot_damage} damage"
        else:
            action = (
                f"{result} an active opponent for {event.single_shot_damage} damage"
            )
    elif event.highlight_type == "spray_kill":
        action = f"spray {result} an active opponent"
    elif event.highlight_type == "shotgun_one_pump":
        action = f"one-pump {result} an active opponent"
    else:
        action = f"{result} an active opponent"
    return f"{' '.join(bits)} {action}"


def format_highlight_description(profile: HighlightProfile) -> str | None:
    if profile.primary is None:
        return None
    primary = profile.primary
    lead = f"{_format_timestamp(primary.timestamp)} - {_describe_event(primary).rstrip('.')}."
    if not profile.secondary:
        return lead
    secondary = profile.secondary[0]
    if (
        secondary.highlight_type == "spray_kill"
        or secondary.resolved_weapon == "automatic"
    ):
        result = "elimination" if secondary.event_kind == "elimination" else "knock"
        follow = f"A later automatic spray secures a second {result}."
    elif secondary.downed:
        follow = (
            "A later downed finish cleans up with a "
            f"{_WEAPON_DISPLAY.get(secondary.resolved_weapon, 'weapon')}."
        )
    else:
        follow = f"A later {_describe_event(secondary).rstrip('.')}."
    return f"{lead} {follow}"


def _serialize_ranked_event(event: RankedHighlightEvent) -> dict:
    return {
        "event_index": event.event_index,
        "timestamp_sec": event.timestamp,
        "event_kind": event.event_kind,
        "highlight_type": event.highlight_type,
        "labels": list(event.labels),
        "summary": event.summary,
        "resolved_weapon": event.resolved_weapon,
        "target_state": event.target_state,
        "damage_aim_state": event.damage_aim_state,
        "finish_aim_state": event.finish_aim_state,
        "single_shot_damage": event.single_shot_damage,
        "damage_hit_count": event.damage_hit_count,
        "accepted_reasons": list(event.accepted_reasons),
        "rejected_reasons": list(event.rejected_reasons),
    }


def build_event_audit(events: list[dict]) -> dict:
    profile = build_highlight_profile(events)
    return {
        "decision_schema_version": "highlight-audit-v1",
        "highlight_description": format_highlight_description(profile),
        "primary_event": (
            _serialize_ranked_event(profile.primary) if profile.primary else None
        ),
        "secondary_events": [
            _serialize_ranked_event(event) for event in profile.secondary
        ],
        "rejected_events": [
            _serialize_ranked_event(event) for event in profile.rejected
        ],
        "multi_kill": profile.multi_kill,
        "active_finish_count": profile.active_finish_count,
    }


def attach_event_audit(event_data: dict, attribution) -> dict:
    """Return event_data with additive audit fields; preserve existing evidence."""
    audited = dict(event_data)
    events = list(getattr(attribution, "events", []) or audited.get("events") or [])
    audit = build_event_audit(events)
    audited.update(audit)
    # Keep raw specialist evidence untouched when present.
    if "specialist_evidence" in event_data:
        audited["specialist_evidence"] = list(event_data["specialist_evidence"])
    return audited


def event_audit_summary(event_data: dict | None) -> dict:
    """Return UI-safe audit payload; never invent descriptions for legacy rows."""
    if not isinstance(event_data, dict):
        return {
            "available": False,
            "highlight_description": None,
            "primary_event": None,
            "secondary_events": [],
            "rejected_events": [],
            "multi_kill": None,
            "active_finish_count": None,
        }
    has_audit = bool(
        event_data.get("decision_schema_version")
        or event_data.get("highlight_description")
        or event_data.get("primary_event") is not None
    )
    if not has_audit:
        return {
            "available": False,
            "highlight_description": None,
            "primary_event": None,
            "secondary_events": [],
            "rejected_events": [],
            "multi_kill": None,
            "active_finish_count": None,
        }
    description = event_data.get("highlight_description")
    return {
        "available": True,
        "highlight_description": str(description) if description else None,
        "primary_event": event_data.get("primary_event"),
        "secondary_events": list(event_data.get("secondary_events") or []),
        "rejected_events": list(event_data.get("rejected_events") or []),
        "multi_kill": event_data.get("multi_kill"),
        "active_finish_count": event_data.get("active_finish_count"),
    }


def event_bonus(profile: HighlightProfile | None) -> float:
    if profile is None or profile.primary is None:
        return 0.0
    primary_bonus = {
        "no_scope": 0.18,
        "shotgun_one_pump": 0.17,
        "spray_kill": 0.10,
        "high_damage_hit": 0.09,
        "active_finish": 0.07,
        "downed_finish": 0.0,
    }.get(profile.primary.highlight_type, 0.0)
    secondary_bonus = min(
        0.04,
        0.02
        * sum(1 for event in profile.secondary if event.active and not event.downed),
    )
    multi_kill_bonus = 0.03 if profile.multi_kill else 0.0
    return min(0.25, primary_bonus + secondary_bonus + multi_kill_bonus)
