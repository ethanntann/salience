from salience_api.features.fireworks_teacher import (
    merge_event_labels,
    resolve_event_summaries,
)
from salience_api.features.teacher_labels import normalize_teacher_payload
from salience_api.ranking.highlights import (
    attach_event_audit,
    build_event_audit,
    build_highlight_profile,
    dedupe_verified_finishes,
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


def test_dedupe_collapses_conflicting_weapon_windows_from_one_lingering_finish():
    """One shotgun finish must not become shotgun + SMG + sniper after swaps."""
    common = {
        "status": "attributed",
        "event_kind": "elimination",
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
    }
    events = [
        {
            **common,
            "event_index": 1,
            "finish_timestamp": 14.44,
            "resolved_weapon": "shotgun",
            "selected_weapon_name_text": "STRIKER PUMP SHOTGUN",
            "single_shot_damage": 190,
            "damage_hit_count": 1,
            "target_identity": "unknown",
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "shotgun",
                "confidence": 0.998,
            },
        },
        {
            **common,
            "event_index": 2,
            "finish_timestamp": 15.40,
            "resolved_weapon": "automatic",
            "selected_weapon_name_text": "Stinger SMG",
            "single_shot_damage": None,
            "damage_hit_count": 1,
            "target_identity": "unknown",
        },
        {
            **common,
            "event_index": 3,
            "finish_timestamp": 16.37,
            "resolved_weapon": "sniper_or_hunting",
            "selected_weapon_name_text": "unknown",
            "single_shot_damage": 190,
            "damage_hit_count": 1,
            "target_identity": "Agraphone-8",
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert len(finishes) == 1
    assert finishes[0]["event_index"] == 1
    assert finishes[0]["resolved_weapon"] == "shotgun"


def test_dedupe_preserves_rapid_finishes_for_distinct_named_targets():
    common = {
        "status": "attributed",
        "event_kind": "knock",
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
    }
    events = [
        {
            **common,
            "event_index": 0,
            "finish_timestamp": 10.0,
            "resolved_weapon": "sniper_or_hunting",
            "target_identity": "FirstOpponent",
        },
        {
            **common,
            "event_index": 1,
            "finish_timestamp": 10.8,
            "resolved_weapon": "pistol",
            "target_identity": "SecondOpponent",
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert [event["event_index"] for event in finishes] == [0, 1]


def test_dedupe_credits_the_later_finishing_weapon_not_setup_damage_weapon():
    """A sniper setup hit followed by a shotgun finish is a shotgun kill."""
    common = {
        "status": "attributed",
        "event_kind": "knock",
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
        "target_identity": "SameOpponent",
        "single_shot_damage": 150,
        "damage_hit_count": 1,
    }
    events = [
        {
            **common,
            "event_index": 0,
            "finish_timestamp": 2.15,
            "resolved_weapon": "sniper_or_hunting",
            "selected_weapon_name_text": "HUNTING RIFLE",
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "sniper_or_hunting",
                "confidence": 0.99,
            },
        },
        {
            **common,
            "event_index": 1,
            "finish_timestamp": 3.59,
            "resolved_weapon": "shotgun",
            "selected_weapon_name_text": "STRIKER PUMP SHOTGUN",
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "shotgun",
                "confidence": 0.99,
            },
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert len(finishes) == 1
    assert finishes[0]["event_index"] == 1
    assert finishes[0]["resolved_weapon"] == "shotgun"


def test_dedupe_does_not_let_an_ungrounded_later_weapon_guess_steal_finish():
    common = {
        "status": "attributed",
        "event_kind": "knock",
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
        "target_identity": "SameOpponent",
        "single_shot_damage": None,
    }
    events = [
        {
            **common,
            "event_index": 0,
            "finish_timestamp": 14.6,
            "resolved_weapon": "automatic",
            "selected_weapon_name_text": "CHAOS EXPLORER RIFLE",
            "damage_hit_count": 3,
        },
        {
            **common,
            "event_index": 1,
            "finish_timestamp": 16.5,
            "resolved_weapon": "sniper_or_hunting",
            "selected_weapon_name_text": "CHAOS EXPLORER RIFLE",
            "damage_hit_count": 1,
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert len(finishes) == 1
    assert finishes[0]["event_index"] == 0
    assert finishes[0]["resolved_weapon"] == "automatic"


def test_dedupe_keeps_active_knock_separate_from_later_downed_cleanup():
    common = {
        "status": "attributed",
        "target_identity": "SameOpponent",
        "finish_onset_supported": True,
        "visual_action_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
    }
    events = [
        {
            **common,
            "event_index": 0,
            "event_kind": "knock",
            "finish_timestamp": 10.0,
            "resolved_weapon": "pistol",
            "selected_weapon_name_text": "RANGER PISTOL",
            "target_was_active": True,
            "target_was_downed": False,
        },
        {
            **common,
            "event_index": 1,
            "event_kind": "downed_finish",
            "finish_timestamp": 11.5,
            "resolved_weapon": "shotgun",
            "selected_weapon_name_text": "STRIKER PUMP SHOTGUN",
            "target_was_active": False,
            "target_was_downed": True,
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert [event["event_index"] for event in finishes] == [0, 1]


def test_dedupe_credits_later_finish_for_same_named_target_across_clip():
    common = {
        "status": "attributed",
        "event_kind": "elimination",
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
        "target_identity": "Forbidden80",
        "single_shot_damage": 91,
        "damage_hit_count": 1,
    }
    events = [
        {
            **common,
            "event_index": 0,
            "finish_timestamp": 9.7,
            "resolved_weapon": "sniper_or_hunting",
            "selected_weapon_name_text": "HUNTING RIFLE",
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "sniper_or_hunting",
                "confidence": 0.99,
            },
        },
        {
            **common,
            "event_index": 1,
            "finish_timestamp": 15.5,
            "resolved_weapon": "pistol",
            "selected_weapon_name_text": "RANGER PISTOL",
            "target_identity": "Forbidden60",
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "pistol",
                "confidence": 0.99,
            },
        },
    ]

    finishes = dedupe_verified_finishes(events)

    assert len(finishes) == 1
    assert finishes[0]["event_index"] == 1
    assert finishes[0]["resolved_weapon"] == "pistol"
