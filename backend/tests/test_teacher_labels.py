import io

import salience_api.features.fireworks_teacher as fireworks_teacher
from salience_api.features.fireworks_teacher import (
    ClipTeacherInput,
    OpenAICompatibleTeacherClient,
    build_teacher_content,
    build_teacher_prompt,
    extract_json_object,
    merge_event_labels,
    merge_legacy_teacher_labels as merge_teacher_labels,
    parse_event_timestamps,
    prepare_weapon_payload,
    resolve_event_summaries,
    resolve_legacy_frame_attribution as resolve_weapon_attribution,
)
from salience_api.features.teacher_labels import (
    TeacherClipLabels,
    derive_label_confidences,
    normalize_teacher_payload,
)


def test_teacher_labels_accept_constrained_values():
    labels = TeacherClipLabels(
        combat_visible="yes",
        enemy_visible="yes",
        elimination_or_knock="yes",
        high_damage_hit="uncertain",
        flick_shot="yes",
        no_scope="yes",
        trickshot="uncertain",
        build_fight="no",
        clutch="yes",
        cleanup_kill="no",
        downed_finish="no",
        spray_kill="no",
        sniper_kill="yes",
        shotgun_kill="yes",
        shotgun_one_pump="yes",
        pistol_kill="no",
        automatic_kill="no",
        other_weapon_kill="no",
        opponent_likely_bot="no",
        stationary_target="yes",
        stationary_sniper_target="yes",
        competitive_context="yes",
        victory="no",
        multi_kill="no",
        fast_edit="uncertain",
        rotation_traversal="no",
        looting_or_menu="no",
        downtime="no",
        confidence=0.75,
        evidence=["scope visible", "audio spike near action"],
    )

    assert labels.no_scope == "yes"
    assert labels.flick_shot == "yes"
    assert labels.competitive_context == "yes"
    assert labels.stationary_target == "yes"
    assert labels.stationary_sniper_target == "yes"
    assert labels.confidence == 0.75


def test_normalize_teacher_payload_accepts_nested_labels():
    labels = normalize_teacher_payload(
        {
            "labels": {
                "no_scope": "no",
                "flick": "yes",
                "trickshot": "yes",
                "build_fight": "uncertain",
                "clutch": "yes",
                "cleanup_kill": "no",
                "spray_kill": "no",
                "sniper_kill": "no",
                "shotgun_one_pump": "no",
                "victory": "no",
                "multi_kill": "no",
                "fast_edit": "no",
            },
            "confidence": 0.7,
            "evidence": ["airborne camera movement"],
        }
    )

    assert labels.trickshot == "yes"
    assert labels.yes_labels() == ["flick_shot", "trickshot", "clutch"]


def test_normalize_teacher_payload_accepts_legacy_label_names():
    labels = normalize_teacher_payload(
        {
            "labels": {
                "box_fight": "yes",
                "boring_cleanup": "yes",
                "ar_spray": "yes",
                "sniper_visible": "yes",
            }
        }
    )

    assert labels.build_fight == "yes"
    assert labels.cleanup_kill == "yes"
    assert labels.spray_kill == "yes"
    assert labels.sniper_kill == "yes"


def test_normalize_teacher_payload_accepts_string_evidence_and_bool_labels():
    labels = normalize_teacher_payload(
        {
            "labels": {
                "no_scope": True,
                "trickshot": False,
                "shotgun_one_pump": "YES",
            },
            "confidence": 0.6,
            "evidence": "single sentence evidence",
        }
    )

    assert labels.no_scope == "yes"
    assert labels.trickshot == "no"
    assert labels.shotgun_one_pump == "yes"
    assert labels.evidence == ["single sentence evidence"]


def test_normalize_teacher_payload_accepts_word_confidence():
    labels = normalize_teacher_payload({"confidence": "high"})

    assert labels.confidence == 0.85


def test_derive_label_confidences_keeps_labels_and_weights_events():
    confidences = derive_label_confidences(
        {
            "labels": {
                "sniper_kill": "yes",
                "pistol_kill": "no",
                "clutch": "yes",
                "shotgun_kill": "uncertain",
            },
            "confidence": 0.9,
        },
        events=[{"status": "attributed", "teacher_confidence": 0.65}],
    )

    assert confidences["sniper_kill"] == 0.65
    assert confidences["pistol_kill"] == 0.65
    assert confidences["clutch"] == 0.9
    assert confidences["shotgun_kill"] == 0.0


def test_fireworks_prompt_requests_constrained_json():
    prompt = build_teacher_prompt(
        ClipTeacherInput(
            filename="clip.mp4",
            duration_sec=20,
            width=1920,
            height=1080,
            fps=60,
            tags=["sniper"],
            image_paths=[],
        )
    )

    assert "Return only valid compact JSON" in prompt
    assert "clip-level context only" in prompt
    assert "separate event specialist owns those facts" in prompt
    assert "Ignore kill-feed text" in prompt
    assert "no_scope" not in prompt
    assert "shotgun_one_pump" not in prompt
    assert "competitive_context" in prompt
    assert "rank emblem" in prompt
    assert "Survivor I/II/III" in prompt
    assert "XP-based progress is casual" in prompt
    assert "combat_visible" in prompt
    assert "last active teammate" in prompt
    assert "low health is not required" in prompt


def test_teacher_content_provides_frame_timestamps_for_temporal_attribution(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    content = build_teacher_content(
        ClipTeacherInput(
            filename="clip.mp4",
            duration_sec=20,
            width=1920,
            height=1080,
            fps=60,
            tags=[],
            image_paths=[frame, frame],
            image_timestamps=[1.5, 18.5],
            image_views=["full gameplay frame", "weapon timeline composite"],
        )
    )

    frame_context = [item["text"] for item in content if item["type"] == "text"]

    assert "Frame 1 at 1.50s - full gameplay frame" in frame_context
    assert "Frame 2 at 18.50s - weapon timeline composite" in frame_context


def test_extract_json_object_handles_wrapped_response():
    payload = extract_json_object(
        'Here is the JSON: {"confidence": 0.5, "evidence": []}'
    )

    assert payload["confidence"] == 0.5


def test_event_locator_accepts_bare_numeric_array():
    assert parse_event_timestamps("[4.25, 12, -1, true]") == [4.25, 12.0]


def test_weapon_attribution_ignores_post_finish_weapon_swap():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 4.0,
                    "event_phase": "damage",
                    "weapon_category": "pistol",
                    "weapon_confidence": 0.95,
                },
                {
                    "event_index": 0,
                    "timestamp": 4.3,
                    "event_phase": "elimination_start",
                    "weapon_category": "pistol",
                    "weapon_confidence": 0.9,
                },
                {
                    "event_index": 0,
                    "timestamp": 4.6,
                    "event_phase": "post_event",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.99,
                },
            ],
            "confidence": 0.9,
        }
    )

    assert attribution.status == "attributed"
    assert attribution.labels["pistol_kill"] == "yes"
    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.events[0]["resolved_weapon"] == "pistol"


def test_weapon_attribution_keeps_sniper_finish_despite_later_pistol():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.9,
                    "hipfire": True,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "knock_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.85,
                    "hipfire": True,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.5,
                    "event_phase": "post_event",
                    "weapon_category": "pistol",
                    "weapon_confidence": 0.95,
                },
            ]
        }
    )

    assert attribution.labels["sniper_kill"] == "yes"
    assert attribution.labels["no_scope"] == "yes"


def test_no_scope_rejects_finish_only_hipfire_claim():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "knock_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.95,
                    "aim_state": "hipfire",
                }
            ]
        }
    )

    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.labels["no_scope"] == "no"


def test_shotgun_one_pump_rejects_finish_only_claim():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 0.95,
                    "high_damage_one_shot": True,
                }
            ]
        }
    )

    assert attribution.labels["shotgun_one_pump"] == "no"


def test_shotgun_family_does_not_require_one_pump_damage():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                    "high_damage_one_shot": False,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "knock_start",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                },
            ]
        }
    )

    assert attribution.labels["shotgun_kill"] == "yes"
    assert attribution.labels["shotgun_one_pump"] == "no"


def test_spray_kill_requires_repeated_automatic_damage():
    single_hit = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
            ]
        }
    )
    repeated_hits = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 7.6,
                    "event_phase": "damage",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
                {
                    "event_index": 0,
                    "timestamp": 7.8,
                    "event_phase": "damage",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "automatic",
                    "weapon_confidence": 1,
                },
            ]
        }
    )

    assert single_hit.labels["automatic_kill"] == "yes"
    assert single_hit.labels["spray_kill"] == "no"
    assert repeated_hits.labels["automatic_kill"] == "yes"
    assert repeated_hits.labels["spray_kill"] == "yes"


def test_already_downed_target_does_not_create_weapon_kill():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "pistol",
                    "weapon_confidence": 1,
                    "target_state": "already_downed",
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "pistol",
                    "weapon_confidence": 1,
                    "target_state": "already_downed",
                },
            ]
        }
    )

    assert attribution.events[0]["target_was_downed"] is True
    assert all(value == "no" for value in attribution.labels.values())


def test_no_scope_rejects_hunting_rifle_ads():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.95,
                    "aim_state": "hunting_ads_zoom",
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "knock_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.95,
                    "aim_state": "hunting_ads_zoom",
                },
            ]
        }
    )

    assert attribution.labels["sniper_kill"] == "yes"
    assert attribution.labels["no_scope"] == "no"


def test_weapon_labels_require_visible_pov_combat_and_elimination():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "no",
                "enemy_visible": "no",
                "elimination_or_knock": "yes",
            }
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state": "hipfire",
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state": "hipfire",
                },
            ]
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.sniper_kill == "no"
    assert merged.no_scope == "no"


def test_global_ads_evidence_vetoes_no_scope():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
            },
            "evidence": ["Hunting Rifle ADS view before the knock"],
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 8.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state": "hipfire",
                },
                {
                    "event_index": 0,
                    "timestamp": 8.2,
                    "event_phase": "knock_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state": "hipfire",
                },
            ]
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.sniper_kill == "yes"
    assert merged.no_scope == "no"


def test_hybrid_prefers_explicit_sniper_kill_feed_over_conflicting_shotgun_event():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
                "sniper_kill": "yes",
            },
            "evidence": [
                "Knock banner and kill feed confirm POV knocked the target with a Sniper"
            ],
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 5.0,
                    "event_phase": "damage",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                    "high_damage_one_shot": True,
                },
                {
                    "event_index": 0,
                    "timestamp": 5.2,
                    "event_phase": "knock_start",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                    "high_damage_one_shot": True,
                },
            ]
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.sniper_kill == "yes"
    assert merged.shotgun_one_pump == "no"


def test_hybrid_retains_both_weapons_for_multi_kill_clip():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
                "sniper_kill": "yes",
                "multi_kill": "yes",
            }
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 1,
                    "timestamp": 10.0,
                    "event_phase": "damage",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                    "high_damage_one_shot": True,
                },
                {
                    "event_index": 1,
                    "timestamp": 10.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "shotgun",
                    "weapon_confidence": 1,
                    "high_damage_one_shot": True,
                },
            ]
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.sniper_kill == "yes"
    assert merged.shotgun_one_pump == "yes"


def test_hybrid_falls_back_to_clip_understanding_when_timing_is_unresolved():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
                "sniper_kill": "yes",
            }
        }
    )

    merged = merge_teacher_labels(base, resolve_weapon_attribution({"frames": []}))

    assert merged.sniper_kill == "yes"


def test_downed_finish_evidence_suppresses_weapon_and_high_damage_labels():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
                "high_damage_hit": "yes",
                "sniper_kill": "yes",
            },
            "evidence": ["POV finishes an already downed opponent"],
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 5.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                },
                {
                    "event_index": 0,
                    "timestamp": 5.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                },
            ]
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.downed_finish == "yes"
    assert merged.sniper_kill == "no"
    assert merged.high_damage_hit == "no"


def test_downed_finish_does_not_suppress_a_separate_active_sniper_kill():
    base = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
            },
            "evidence": ["POV finishes a downed target, then gets another elimination"],
        }
    )
    attribution = resolve_weapon_attribution(
        {
            "action_evidence_version": 2,
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 2.0,
                    "event_phase": "damage",
                    "weapon_category": "pistol",
                    "weapon_confidence": 1,
                    "target_state": "already_downed",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "damage_instance_id": "0-a",
                },
                {
                    "event_index": 0,
                    "timestamp": 2.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "pistol",
                    "weapon_confidence": 1,
                    "target_state": "already_downed",
                    "finish_ui_newly_appeared": "yes",
                },
                {
                    "event_index": 1,
                    "timestamp": 7.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "target_state": "active",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "damage_instance_id": "1-a",
                },
                {
                    "event_index": 1,
                    "timestamp": 7.2,
                    "event_phase": "knock_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "target_state": "active",
                    "finish_ui_newly_appeared": "yes",
                },
            ],
        }
    )

    merged = merge_teacher_labels(base, attribution)

    assert merged.downed_finish == "yes"
    assert merged.sniper_kill == "yes"
    assert merged.pistol_kill == "no"
    assert attribution.events[0]["target_was_downed"] is True
    assert attribution.events[1]["target_was_active"] is True


def test_strict_action_evidence_rejects_kill_feed_only_finish():
    attribution = resolve_weapon_attribution(
        {
            "action_evidence_version": 2,
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 5.0,
                    "event_phase": "elimination_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "finish_ui_newly_appeared": "yes",
                    "kill_feed_corroborates_pov": "yes",
                }
            ],
        }
    )

    assert attribution.status == "no_event"
    assert attribution.events[0]["status"] == "no_visual_causality"
    assert attribution.labels["sniper_kill"] == "uncertain"


def test_high_damage_requires_single_shot_value_not_accumulated_ui_text():
    common_labels = {
        "combat_visible": "yes",
        "enemy_visible": "yes",
        "elimination_or_knock": "yes",
    }
    accumulated = normalize_teacher_payload(
        {
            "labels": {**common_labels, "high_damage_hit": "yes"},
            "evidence": ["Shield cracked, headshot, and XP totals build up on screen"],
        }
    )
    single_shot = normalize_teacher_payload(
        {
            "labels": {**common_labels, "high_damage_hit": "yes"},
            "evidence": ["A visible 114 damage number comes from one pistol shot"],
        }
    )
    no_events = resolve_weapon_attribution({"frames": []})

    assert merge_teacher_labels(accumulated, no_events).high_damage_hit == "no"
    assert merge_teacher_labels(single_shot, no_events).high_damage_hit == "yes"


def test_weapon_attribution_rejects_damage_finish_disagreement():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 3.0,
                    "event_phase": "damage",
                    "weapon_category": "pistol",
                    "weapon_confidence": 0.95,
                },
                {
                    "event_index": 0,
                    "timestamp": 3.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.95,
                },
            ]
        }
    )

    assert attribution.status == "weapon_conflict"
    assert all(value == "uncertain" for value in attribution.labels.values())


def test_weapon_attribution_does_not_cross_event_windows():
    attribution = resolve_weapon_attribution(
        {
            "frames": [
                {
                    "event_index": 0,
                    "timestamp": 2.0,
                    "event_phase": "damage",
                    "weapon_category": "sniper_or_hunting",
                    "weapon_confidence": 0.95,
                },
                {
                    "event_index": 1,
                    "timestamp": 2.2,
                    "event_phase": "elimination_start",
                    "weapon_category": "pistol",
                    "weapon_confidence": 0.95,
                },
            ]
        }
    )

    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.events[1]["resolved_weapon"] == "pistol"


def test_prepare_weapon_payload_restores_omitted_frame_metadata():
    clip = ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=20,
        width=1920,
        height=1080,
        fps=60,
        tags=[],
        image_paths=[],
        image_timestamps=[4.0, 4.3],
        image_event_indices=[1, 1],
    )

    payload = prepare_weapon_payload(
        {
            "frame_analyses": [
                {"event_phase": "damage", "weapon_category": "pistol"},
                {"event_phase": "elimination_start", "weapon_category": "pistol"},
            ]
        },
        clip,
    )

    assert payload["frames"][0]["timestamp"] == 4.0
    assert payload["frames"][1]["event_index"] == 1


def test_current_event_fusion_rejects_stale_ui_and_old_sniper_fallback():
    context = normalize_teacher_payload(
        {
            "labels": {
                "combat_visible": "yes",
                "enemy_visible": "yes",
                "elimination_or_knock": "yes",
                "sniper_kill": "yes",
            },
            "evidence": ["A persistent kill feed says the POV player sniped someone"],
        }
    )
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "active",
                    "visual_action_supported": "no",
                    "finish_ui_newly_appeared": "yes",
                    "kill_feed_corroborates_pov": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "weapon_confidence": 1,
                }
            ]
        }
    )

    merged = merge_event_labels(context, attribution)

    assert merged.elimination_or_knock == "no"
    assert merged.sniper_kill != "yes"
    assert merged.no_scope == "no"


def test_ui_only_distant_finish_does_not_override_visible_later_action():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "event_timestamp": 8.67,
                    "target_state": "active",
                    "visual_action_supported": "no",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "no",
                    "new_damage_visible": "no",
                    "target_defeat_visible": "no",
                    "finish_ui_newly_appeared": "yes",
                    "kill_feed_corroborates_pov": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "selected_weapon_name_text": "HUNTING RIFLE",
                    "aim_state_at_shot": "hipfire",
                    "damaging_shot_count": 1,
                },
                {
                    "event_index": 1,
                    "event_kind": "elimination",
                    "event_timestamp": 15.41,
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "no",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "kill_feed_corroborates_pov": "yes",
                    "selected_weapon_before_finish": "automatic",
                    "selected_weapon_name_text": "STRYKER 0.50 CAL",
                    "weapon_confidence": 0.95,
                    "aim_state_at_shot": "hipfire",
                    "damaging_shot_count": 3,
                },
            ],
            "confidence": 0.9,
        }
    )

    assert attribution.events[0]["status"] == "no_visual_causality"
    assert attribution.events[0]["resolved_weapon"] == "unknown"
    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.labels["automatic_kill"] == "yes"
    assert attribution.labels["spray_kill"] == "yes"


def test_separate_hunting_damage_does_not_rewrite_later_pistol_finish():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "event_timestamp": 13.58,
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "no",
                    "selected_weapon_name_text": "HUNTING RIFLE",
                    "single_shot_damage": 100,
                    "high_damage_one_shot": True,
                    "aim_state_at_shot": "hunting_ads_zoom",
                    "damaging_shot_count": 1,
                    "target_identity": "Forbidden60",
                },
                {
                    "event_index": 1,
                    "event_kind": "elimination",
                    "event_timestamp": 15.52,
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "kill_feed_corroborates_pov": "yes",
                    "selected_weapon_before_finish": "pistol",
                    "selected_weapon_name_text": "Ranger Pistol",
                    "weapon_confidence": 0.95,
                    "aim_state_at_shot": "hipfire",
                    "target_identity": "Forbidden60",
                },
            ],
            "confidence": 0.9,
        }
    )

    assert attribution.events[0]["status"] == "no_finish"
    assert attribution.events[1]["status"] == "attributed"
    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.labels["pistol_kill"] == "yes"


def test_separate_shotgun_damage_does_not_rewrite_later_pistol_finish():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "event_timestamp": 10.0,
                    "target_state": "active",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "selected_weapon_name_text": "Striker Pump Shotgun",
                    "single_shot_damage": 150,
                    "high_damage_one_shot": True,
                    "damaging_shot_count": 1,
                    "target_identity": "TargetOne",
                },
                {
                    "event_index": 1,
                    "event_kind": "knock",
                    "event_timestamp": 11.2,
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "pistol",
                    "selected_weapon_name_text": "Ranger Pistol",
                    "weapon_confidence": 0.95,
                    "target_identity": "TargetOne",
                },
            ]
        }
    )

    assert attribution.events[0]["status"] == "no_finish"
    assert attribution.events[1]["status"] == "attributed"
    assert attribution.labels["shotgun_kill"] == "no"
    assert attribution.labels["shotgun_one_pump"] == "no"
    assert attribution.labels["pistol_kill"] == "yes"


def test_deferred_finish_bind_skips_weak_damage_and_different_named_targets():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "event_timestamp": 13.5,
                    "target_state": "active",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "selected_weapon_name_text": "HUNTING RIFLE",
                    "single_shot_damage": 40,
                    "target_identity": "Alpha",
                },
                {
                    "event_index": 1,
                    "event_kind": "elimination",
                    "event_timestamp": 14.4,
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "shotgun",
                    "selected_weapon_name_text": "Striker Pump Shotgun",
                    "weapon_confidence": 0.95,
                    "target_identity": "Bravo",
                },
            ]
        }
    )

    assert attribution.events[0]["status"] == "no_finish"
    assert attribution.events[1]["status"] == "attributed"
    assert attribution.labels["sniper_kill"] == "no"
    assert attribution.labels["shotgun_kill"] == "yes"


def test_hud_weapon_name_supplies_confidence_for_named_finish():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "knock",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "unknown",
                    "selected_weapon_name_text": "Mythic Hunting Rifle",
                    "weapon_confidence": 0.2,
                }
            ]
        }
    )

    assert attribution.events[0]["status"] == "attributed"
    assert attribution.events[0]["resolved_weapon"] == "sniper_or_hunting"
    assert attribution.labels["sniper_kill"] == "yes"


def test_current_event_reducer_does_not_trust_visual_summary_without_action_fields():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "no",
                    "target_reaction_visible": "no",
                    "new_damage_visible": "no",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "weapon_confidence": 1,
                }
            ]
        }
    )

    assert attribution.events[0]["status"] == "no_visual_causality"
    assert attribution.labels["sniper_kill"] == "uncertain"


def test_current_event_reducer_requires_known_active_target_for_weapon_credit():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "unknown",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "pistol",
                    "weapon_confidence": 1,
                }
            ]
        }
    )

    assert attribution.events[0]["status"] == "target_state_unknown"
    assert attribution.labels["pistol_kill"] == "uncertain"


def test_current_event_fusion_keeps_mixed_events_independent():
    context = normalize_teacher_payload(
        {"labels": {"combat_visible": "yes", "enemy_visible": "yes"}}
    )
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "downed_finish",
                    "target_state": "already_downed",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "pistol",
                    "weapon_confidence": 1,
                    "damaging_shot_count": 2,
                },
                {
                    "event_index": 1,
                    "event_kind": "knock",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state_at_shot": "hunting_ads_zoom",
                    "single_shot_damage": 105,
                    "damaging_shot_count": 1,
                },
            ]
        }
    )

    merged = merge_event_labels(context, attribution)

    assert merged.elimination_or_knock == "yes"
    assert merged.multi_kill == "no"
    assert merged.downed_finish == "yes"
    assert merged.pistol_kill == "no"
    assert merged.sniper_kill == "yes"
    assert merged.no_scope == "no"
    assert merged.high_damage_hit == "yes"


def test_current_event_reducer_derives_styles_from_explicit_event_facts():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "shotgun",
                    "weapon_confidence": 1,
                    "single_shot_damage": 120,
                    "high_damage_one_shot": True,
                    "damaging_shot_count": 1,
                },
                {
                    "event_index": 1,
                    "event_kind": "knock",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "automatic",
                    "weapon_confidence": 1,
                    "damaging_shot_count": 4,
                },
                {
                    "event_index": 2,
                    "event_kind": "knock",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "aim_state_at_shot": "hunting_ads_zoom",
                    "damaging_shot_count": 1,
                },
            ]
        }
    )

    assert attribution.labels["shotgun_one_pump"] == "yes"
    assert attribution.labels["spray_kill"] == "yes"
    assert attribution.labels["sniper_kill"] == "yes"
    assert attribution.labels["no_scope"] == "no"


def test_visible_weapon_name_overrides_silhouette_category_guess():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "knock",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "pistol",
                    "selected_weapon_name_text": "LEGENDARY STRIKER PUMP SHOTGUN",
                    "weapon_confidence": 1,
                    "single_shot_damage": 112,
                    "damaging_shot_count": 1,
                }
            ]
        }
    )

    assert attribution.labels["shotgun_kill"] == "yes"
    assert attribution.labels["shotgun_one_pump"] == "yes"
    assert attribution.labels["pistol_kill"] == "no"


def test_current_one_pump_requires_numeric_single_shot_damage():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "finish_ui_newly_appeared": "yes",
                    "selected_weapon_before_finish": "shotgun",
                    "weapon_confidence": 1,
                    "single_shot_damage": None,
                    "high_damage_one_shot": True,
                    "damaging_shot_count": 1,
                }
            ]
        }
    )
    context = normalize_teacher_payload(
        {"labels": {"combat_visible": "yes", "enemy_visible": "yes"}}
    )
    merged = merge_event_labels(context, attribution)

    assert merged.shotgun_kill == "yes"
    assert merged.shotgun_one_pump == "no"
    assert merged.high_damage_hit == "no"


def test_high_damage_survives_fast_flick_after_visible_shot_and_number():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "target_state": "unknown",
                    "visual_action_supported": "no",
                    "pov_shot_visible": "yes",
                    "target_reaction_visible": "no",
                    "new_damage_visible": "yes",
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "weapon_confidence": 1,
                    "single_shot_damage": 147,
                    "damaging_shot_count": 1,
                }
            ]
        }
    )
    context = normalize_teacher_payload({"labels": {}})

    merged = merge_event_labels(context, attribution)

    assert merged.high_damage_hit == "yes"
    assert merged.sniper_kill != "yes"


def test_cumulative_damage_total_is_not_a_single_high_damage_hit():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "single_shot_damage": 245,
                    "damage_display_is_cumulative": True,
                }
            ]
        }
    )

    merged = merge_event_labels(normalize_teacher_payload({"labels": {}}), attribution)

    assert merged.high_damage_hit == "no"


def test_duplicate_locator_summaries_do_not_create_a_multi_kill():
    common = {
        "event_kind": "knock",
        "target_state": "active",
        "target_identity": "Buffering5",
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "new_damage_visible": "yes",
        "target_defeat_visible": "yes",
        "selected_weapon_before_finish": "sniper_or_hunting",
        "weapon_confidence": 1,
        "single_shot_damage": 91,
    }
    attribution = resolve_event_summaries(
        {
            "events": [
                {**common, "event_index": 0, "event_timestamp": 15.61},
                {**common, "event_index": 1, "event_timestamp": 16.58},
            ]
        }
    )

    merged = merge_event_labels(normalize_teacher_payload({"labels": {}}), attribution)

    assert merged.sniper_kill == "yes"
    assert merged.multi_kill == "no"
    assert attribution.events[0]["status"] == "attributed"
    assert attribution.events[1]["status"] == "attributed"


def test_same_target_keeps_latest_finish_weapon_only():
    """Clip-371 style: HR tag then shotgun finish on the same name → shotgun only."""
    common = {
        "event_kind": "elimination",
        "target_state": "active",
        "target_identity": "DoctorLobby92",
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "new_damage_visible": "yes",
        "target_defeat_visible": "yes",
        "finish_ui_newly_appeared": "yes",
        "weapon_confidence": 0.95,
    }
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    **common,
                    "event_index": 0,
                    "event_timestamp": 14.45,
                    "selected_weapon_before_finish": "sniper_or_hunting",
                    "selected_weapon_name_text": "Hunting Rifle",
                    "single_shot_damage": 150,
                },
                {
                    **common,
                    "event_index": 1,
                    "event_timestamp": 15.41,
                    "selected_weapon_before_finish": "shotgun",
                    "selected_weapon_name_text": "Extending Focus Shotgun",
                    "single_shot_damage": 120,
                    "damaging_shot_count": 1,
                },
            ]
        }
    )
    merged = merge_event_labels(normalize_teacher_payload({"labels": {}}), attribution)

    assert attribution.events[0]["status"] == "duplicate_target_finish"
    assert attribution.events[1]["status"] == "attributed"
    assert merged.sniper_kill == "no"
    assert merged.shotgun_kill == "yes"
    assert merged.multi_kill == "no"


def test_ocr_does_not_override_named_conflicting_vlm_weapon():
    event = {
        "selected_weapon_before_finish": "automatic",
        "selected_weapon_name_text": "Assault Rifle",
        "weapon_confidence": 0.92,
    }
    ocr = {
        "text": "HUNTING-RIFLE",
        "category": "sniper_or_hunting",
        "confidence": 0.99,
    }
    assert fireworks_teacher.ocr_should_override_weapon(event, ocr) is False
    assert fireworks_teacher.apply_weapon_ocr_to_event(event, ocr) is False
    assert event["selected_weapon_before_finish"] == "automatic"
    assert event["local_ocr"]["applied"] is False


def test_local_runtime_can_prefer_strong_ocr_over_visual_weapon_family():
    event = {
        "selected_weapon_before_finish": "automatic",
        "selected_weapon_name_text": "Assault Rifle",
        "weapon_confidence": 0.92,
    }
    ocr = {
        "text": "HUNTING-RIFLE",
        "category": "sniper_or_hunting",
        "confidence": 0.99,
    }

    assert fireworks_teacher.apply_weapon_ocr_to_event(
        event, ocr, prefer_ocr=True
    ) is True
    assert event["selected_weapon_before_finish"] == "sniper_or_hunting"
    assert event["local_ocr"]["applied"] is True


def test_ocr_corrects_vlm_when_vlm_name_is_unmapped():
    event = {
        "selected_weapon_before_finish": "automatic",
        "selected_weapon_name_text": "Extreme Force Distro",
        "weapon_confidence": 0.92,
    }
    ocr = {
        "text": "EXTENDING FOCUS SHOTGUN",
        "category": "shotgun",
        "confidence": 0.99,
    }
    assert fireworks_teacher.apply_weapon_ocr_to_event(event, ocr) is True
    assert event["selected_weapon_before_finish"] == "shotgun"
    assert event["selected_weapon_name_text"] == "EXTENDING FOCUS SHOTGUN"


def test_ocr_fills_unknown_vlm_weapon():
    event = {
        "selected_weapon_before_finish": "unknown",
        "selected_weapon_name_text": "unknown",
        "weapon_confidence": 0.4,
    }
    ocr = {
        "text": "HUNTING-RIFLE",
        "category": "sniper_or_hunting",
        "confidence": 0.9,
    }
    assert fireworks_teacher.apply_weapon_ocr_to_event(event, ocr) is True
    assert event["selected_weapon_before_finish"] == "sniper_or_hunting"
    assert event["selected_weapon_name_text"] == "HUNTING-RIFLE"


def test_ocr_ambiguous_window_clears_weapon_credit():
    event = {
        "selected_weapon_before_finish": "shotgun",
        "selected_weapon_name_text": "Pump Shotgun",
        "weapon_confidence": 0.95,
    }
    ocr = {
        "ambiguous": True,
        "category": "unknown",
        "text": "unknown",
        "confidence": 0.0,
        "categories": ["shotgun", "automatic"],
        "reason": "conflicting_weapons_in_event_window",
    }
    assert fireworks_teacher.apply_weapon_ocr_to_event(event, ocr) is False
    assert event["selected_weapon_before_finish"] == "unknown"
    assert event["weapon_confidence"] == 0.0


def test_absurd_single_shot_damage_is_cleared():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "damage_only",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "single_shot_damage": 850,
                    "damage_display_is_cumulative": False,
                }
            ]
        }
    )
    merged = merge_event_labels(normalize_teacher_payload({"labels": {}}), attribution)

    assert attribution.events[0]["single_shot_damage"] is None
    assert merged.high_damage_hit == "no"


def test_spray_damage_count_clears_single_shot_claim():
    attribution = resolve_event_summaries(
        {
            "events": [
                {
                    "event_index": 0,
                    "event_kind": "elimination",
                    "target_state": "active",
                    "visual_action_supported": "yes",
                    "pov_shot_visible": "yes",
                    "new_damage_visible": "yes",
                    "target_defeat_visible": "yes",
                    "selected_weapon_before_finish": "automatic",
                    "weapon_confidence": 0.95,
                    "single_shot_damage": 180,
                    "damaging_shot_count": 5,
                }
            ]
        }
    )
    assert attribution.events[0]["single_shot_damage"] is None


def test_stationary_target_requires_sustained_duration_evidence():
    common = {
        "event_index": 0,
        "event_kind": "damage_only",
        "target_state": "active",
        "visual_action_supported": "yes",
        "pov_shot_visible": "yes",
        "target_reaction_visible": "yes",
        "stationary_target": "yes",
    }
    momentary = resolve_event_summaries(
        {"events": [{**common, "stationary_duration_supported": "no"}]}
    )
    sustained = resolve_event_summaries(
        {"events": [{**common, "stationary_duration_supported": "yes"}]}
    )
    context = normalize_teacher_payload({"labels": {}})

    assert merge_event_labels(context, momentary).stationary_target == "no"
    assert merge_event_labels(context, sustained).stationary_target == "yes"


def test_weapon_specialist_retries_incomplete_event_summaries(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")

    class StubTeacher(OpenAICompatibleTeacherClient):
        def __init__(self):
            self.responses = iter(
                [
                    '{"events":[]}',
                    '{"events":[{"event_index":0,"event_kind":"elimination",'
                    '"target_state":"active","visual_action_supported":"yes",'
                    '"pov_shot_visible":"yes","target_reaction_visible":"yes",'
                    '"finish_ui_newly_appeared":"yes",'
                    '"selected_weapon_before_finish":"pistol","weapon_confidence":1}]}',
                ]
            )
            self.calls = 0

        def _complete(self, content, *, max_tokens):
            self.calls += 1
            return next(self.responses)

    client = StubTeacher()
    attribution = client.label_weapon_event(
        ClipTeacherInput(
            filename="clip.mp4",
            duration_sec=20,
            width=1920,
            height=1080,
            fps=60,
            tags=[],
            image_paths=[frame, frame],
            image_timestamps=[4.0, 4.3],
            image_event_indices=[0, 0],
        )
    )

    assert client.calls == 2
    assert attribution.status == "attributed"
    assert attribution.raw_payload["prepared_event_count"] == 1


def test_weapon_specialist_recovers_duplicate_ids_with_focused_event_calls(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    duplicate = (
        '{"events":['
        '{"event_index":0,"event_kind":"none"},'
        '{"event_index":0,"event_kind":"none"}'
        "]} "
    )

    class StubTeacher(OpenAICompatibleTeacherClient):
        def __init__(self):
            self.responses = iter(
                [
                    duplicate,
                    duplicate,
                    '{"event_index":0,"event_kind":"none"}',
                    '{"event_index":1,"event_kind":"none"}',
                ]
            )

        def _complete(self, content, *, max_tokens):
            return next(self.responses)

    attribution = StubTeacher().label_weapon_event(
        ClipTeacherInput(
            filename="clip.mp4",
            duration_sec=20,
            width=1920,
            height=1080,
            fps=60,
            tags=[],
            image_paths=[frame, frame],
            image_timestamps=[4.0, 8.0],
            image_event_indices=[0, 1],
        )
    )

    assert attribution.status == "no_event"
    assert attribution.raw_payload["prepared_event_count"] == 2
    assert all(value == "uncertain" for value in attribution.labels.values())


def test_teacher_client_retries_rate_limits(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise fireworks_teacher.urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "0"},
                io.BytesIO(b'{"error":"rate limited"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(fireworks_teacher.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fireworks_teacher.time, "sleep", sleeps.append)
    client = OpenAICompatibleTeacherClient(
        provider="test",
        api_key="key",
        base_url="https://example.test/v1",
        model="model",
    )

    result = client._complete([], max_tokens=10)

    assert result == '{"ok":true}'
    assert calls == 2
    assert sleeps == [1.0]
