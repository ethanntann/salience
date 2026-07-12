from salience_api.features.fireworks_teacher import (
    merge_event_labels,
    resolve_event_summaries,
)
from salience_api.features.teacher_labels import normalize_teacher_payload
from salience_api.ranking.highlights import (
    attach_event_audit,
    build_event_audit,
    build_highlight_profile,
    event_audit_summary,
    format_highlight_description,
)


def _active_sniper(
    *, index=0, aim="hipfire", timestamp=14.0, damage=237, identity="Enemy1"
):
    return {
        "event_index": index,
        "event_kind": "knock",
        "event_timestamp": timestamp,
        "target_state": "active",
        "target_identity": identity,
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "new_damage_visible": "yes",
        "target_defeat_visible": "yes",
        "selected_weapon_before_finish": "sniper_or_hunting",
        "selected_weapon_name_text": "Mythic Hunting Rifle",
        "weapon_confidence": 1,
        "aim_state_at_shot": aim,
        "single_shot_damage": damage,
        "damaging_shot_count": 1,
        "summary": "Hunting Rifle knock",
    }


def _active_spray(*, index=1, timestamp=18.0, identity="Enemy2"):
    return {
        "event_index": index,
        "event_kind": "elimination",
        "event_timestamp": timestamp,
        "target_state": "active",
        "target_identity": identity,
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "target_reaction_visible": "yes",
        "finish_ui_newly_appeared": "yes",
        "selected_weapon_before_finish": "automatic",
        "selected_weapon_name_text": "Stinger SMG",
        "weapon_confidence": 1,
        "aim_state_at_shot": "other_ads",
        "damaging_shot_count": 4,
        "summary": "Automatic spray elimination",
    }


def _downed_finish(*, index=0, timestamp=10.0, weapon="pistol"):
    return {
        "event_index": index,
        "event_kind": "downed_finish",
        "event_timestamp": timestamp,
        "target_state": "already_downed",
        "target_identity": "Downed1",
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "target_reaction_visible": "yes",
        "finish_ui_newly_appeared": "yes",
        "selected_weapon_before_finish": weapon,
        "weapon_confidence": 1,
        "damaging_shot_count": 2,
        "summary": "Finish on already downed opponent",
    }


def test_highlight_description_names_primary_and_secondary_events():
    attribution = resolve_event_summaries(
        {"events": [_active_sniper(), _active_spray()]}
    )
    profile = build_highlight_profile(attribution.events)

    description = format_highlight_description(profile)

    assert description.startswith("0:14 -")
    assert "Hunting Rifle" in description or "sniper" in description.lower()
    assert "237" in description
    assert "spray" in description.lower() or "automatic" in description.lower()
    assert "second" in description.lower() or "later" in description.lower()


def test_primary_highlight_prefers_distinctive_mechanic_over_weapon_family():
    attribution = resolve_event_summaries(
        {
            "events": [
                _active_spray(index=0, timestamp=10.0),
                _active_sniper(index=1, timestamp=14.0, aim="hunting_ads_zoom"),
            ]
        }
    )
    profile = build_highlight_profile(attribution.events)

    assert profile.primary is not None
    assert profile.primary.highlight_type == "spray_kill"
    assert any("sniper_kill" in item.labels for item in profile.secondary)
    assert profile.multi_kill is True


def test_no_scope_vetoed_when_same_target_has_ads_contradiction():
    attribution = resolve_event_summaries(
        {
            "events": [
                _active_sniper(
                    index=0,
                    aim="hunting_ads_zoom",
                    timestamp=14.0,
                    identity="Buffering5",
                ),
                _active_sniper(
                    index=1, aim="hipfire", timestamp=14.4, identity="Buffering5"
                ),
            ]
        }
    )
    merged = merge_event_labels(
        normalize_teacher_payload(
            {"labels": {"combat_visible": "yes", "enemy_visible": "yes"}}
        ),
        attribution,
    )

    assert merged.sniper_kill == "yes"
    assert merged.no_scope == "no"
    assert merged.multi_kill == "no"


def test_downed_finish_kept_when_separate_active_kill_exists():
    context = normalize_teacher_payload(
        {"labels": {"combat_visible": "yes", "enemy_visible": "yes"}}
    )
    attribution = resolve_event_summaries(
        {
            "events": [
                _downed_finish(),
                _active_sniper(index=1, aim="hunting_ads_zoom", timestamp=16.0),
            ]
        }
    )
    merged = merge_event_labels(context, attribution)

    assert merged.downed_finish == "yes"
    assert merged.elimination_or_knock == "yes"
    assert merged.sniper_kill == "yes"
    assert merged.pistol_kill == "no"
    assert merged.multi_kill == "no"


def test_downed_only_clip_is_not_elimination_or_knock():
    context = normalize_teacher_payload(
        {"labels": {"combat_visible": "yes", "enemy_visible": "yes"}}
    )
    attribution = resolve_event_summaries({"events": [_downed_finish()]})
    merged = merge_event_labels(context, attribution)

    assert merged.downed_finish == "yes"
    assert merged.elimination_or_knock == "no"
    assert merged.pistol_kill == "no"


def test_event_audit_preserves_raw_evidence_and_rejects_downed_weapon_credit():
    attribution = resolve_event_summaries(
        {
            "events": [_downed_finish()],
            "evidence": ["Finish on already downed opponent with Hunting Rifle"],
        }
    )
    event_data = {
        "events": attribution.events,
        "specialist_evidence": ["Finish on already downed opponent with Hunting Rifle"],
    }
    audited = attach_event_audit(event_data, attribution)

    assert audited["specialist_evidence"] == [
        "Finish on already downed opponent with Hunting Rifle"
    ]
    assert audited["highlight_description"]
    assert audited["primary_event"] is not None
    assert audited["primary_event"]["highlight_type"] == "downed_finish"
    denial_reasons = [
        *audited["primary_event"]["rejected_reasons"],
        *(
            reason
            for event in audited["rejected_events"]
            for reason in event["rejected_reasons"]
        ),
    ]
    assert any(
        "weapon" in reason.lower() or "downed" in reason.lower()
        for reason in denial_reasons
    )


def test_build_event_audit_marks_accepted_active_finish():
    attribution = resolve_event_summaries(
        {"events": [_active_sniper(aim="hunting_ads_zoom")]}
    )
    audit = build_event_audit(attribution.events)

    assert audit["primary_event"]["accepted_reasons"]
    assert audit["primary_event"]["resolved_weapon"] == "sniper_or_hunting"
    assert audit["primary_event"]["target_state"] == "active"
    assert audit["primary_event"]["damage_aim_state"] == "ads_no_scope_overlay"


def test_event_audit_summary_does_not_invent_legacy_descriptions():
    unavailable = event_audit_summary(
        {"events": [], "specialist_evidence": ["raw note"]}
    )
    assert unavailable["available"] is False
    assert unavailable["highlight_description"] is None

    available = event_audit_summary(
        {
            "decision_schema_version": "highlight-audit-v1",
            "highlight_description": "0:14 - Hunting Rifle headshot",
            "primary_event": {"highlight_type": "sniper_kill"},
            "secondary_events": [],
            "rejected_events": [],
            "multi_kill": False,
            "active_finish_count": 1,
        }
    )
    assert available["available"] is True
    assert available["highlight_description"] == "0:14 - Hunting Rifle headshot"
