from salience_api.features.basic import BasicFeatures
from salience_api.ranking.highlights import (
    HIGHLIGHT_HIERARCHY,
    build_highlight_profile,
    event_bonus,
)
from salience_api.ranking.scoring import score_clip


def _active_sniper(*, index=0, aim="hunting_ads_zoom", timestamp=14.0, damage=237):
    return {
        "event_index": index,
        "status": "attributed",
        "event_kind": "knock",
        "target_was_active": True,
        "target_was_downed": False,
        "resolved_weapon": "sniper_or_hunting",
        "damage_aim_state": aim,
        "finish_aim_state": aim,
        "finish_timestamp": timestamp,
        "single_shot_damage": damage,
        "damage_hit_count": 1,
        "summary": "Hunting Rifle knock",
        "target_identity": "Enemy1",
    }


def _active_spray(*, index=1, timestamp=18.0):
    return {
        "event_index": index,
        "status": "attributed",
        "event_kind": "elimination",
        "target_was_active": True,
        "target_was_downed": False,
        "resolved_weapon": "automatic",
        "damage_aim_state": "hipfire",
        "finish_aim_state": "hipfire",
        "finish_timestamp": timestamp,
        "single_shot_damage": 40,
        "damage_hit_count": 4,
        "summary": "Automatic spray elimination",
        "target_identity": "Enemy2",
        "damaging_shot_count": 4,
    }


def _downed_finish(*, index=2, timestamp=20.0):
    return {
        "event_index": index,
        "status": "attributed",
        "event_kind": "downed_finish",
        "target_was_active": False,
        "target_was_downed": True,
        "resolved_weapon": "sniper_or_hunting",
        "damage_aim_state": "hunting_ads_zoom",
        "finish_aim_state": "hunting_ads_zoom",
        "finish_timestamp": timestamp,
        "single_shot_damage": 100,
        "damage_hit_count": 1,
        "summary": "Downed finish",
        "target_identity": "Enemy3",
    }


def _features(tags: list[str]) -> BasicFeatures:
    return BasicFeatures(
        duration_sec=20,
        motion_score=0.7,
        audio_peak_score=0.6,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.75,
        tags=tags,
    )


def test_hierarchy_invariants_prefer_mechanics_and_quality_over_weapon_family():
    no_scope = build_highlight_profile([_active_sniper(aim="hipfire")])
    ads_sniper = build_highlight_profile([_active_sniper(aim="hunting_ads_zoom")])
    downed = build_highlight_profile([_downed_finish()])
    mixed = build_highlight_profile(
        [_active_sniper(aim="hunting_ads_zoom"), _downed_finish()]
    )

    assert no_scope.primary.highlight_type == "no_scope"
    assert ads_sniper.primary.highlight_type == "high_damage_hit"
    assert downed.primary.highlight_type == "downed_finish"
    assert mixed.primary.highlight_type == "high_damage_hit"
    assert HIGHLIGHT_HIERARCHY.index("no_scope") < HIGHLIGHT_HIERARCHY.index(
        "active_finish"
    )
    assert "sniper_kill" not in HIGHLIGHT_HIERARCHY


def test_spray_and_sniper_coexist_without_weapon_family_priority():
    profile = build_highlight_profile(
        [_active_sniper(aim="hunting_ads_zoom"), _active_spray()]
    )

    assert profile.primary.highlight_type == "spray_kill"
    assert any("sniper_kill" in event.labels for event in profile.secondary)
    assert profile.multi_kill is True


def test_multi_kill_bonus_increases_event_aware_score():
    single = build_highlight_profile([_active_sniper(aim="hunting_ads_zoom")])
    multi = build_highlight_profile(
        [_active_sniper(aim="hunting_ads_zoom"), _active_spray()]
    )
    tags = [
        "combat_visible",
        "elimination_or_knock",
        "sniper_kill",
        "spray_kill",
        "multi_kill",
    ]

    single_score = score_clip(_features(tags), None, highlight_profile=single)
    multi_score = score_clip(_features(tags), None, highlight_profile=multi)

    assert event_bonus(multi) > event_bonus(single)
    assert multi_score.base_score > single_score.base_score


def test_event_profile_stops_double_counting_event_owned_tags():
    profile = build_highlight_profile([_active_sniper(aim="hunting_ads_zoom")])
    tags = ["combat_visible", "elimination_or_knock", "sniper_kill", "flick_shot"]
    with_profile = score_clip(_features(tags), None, highlight_profile=profile)
    without_profile = score_clip(_features(tags), None)

    # Profile replaces event-owned tag stacking with bounded structured bonus.
    assert with_profile.base_score != without_profile.base_score
    assert "primary high damage hit" in with_profile.explanation
    assert "flick_shot" in with_profile.explanation
    assert "matches sniper_kill" not in with_profile.explanation
    assert "matches elimination_or_knock" not in with_profile.explanation


def test_downed_penalty_only_when_primary_is_downed_only():
    downed_only = build_highlight_profile([_downed_finish()])
    mixed = build_highlight_profile(
        [_active_sniper(aim="hunting_ads_zoom"), _downed_finish()]
    )
    tags = ["combat_visible", "elimination_or_knock", "sniper_kill", "downed_finish"]

    downed_score = score_clip(_features(tags), None, highlight_profile=downed_only)
    mixed_score = score_clip(_features(tags), None, highlight_profile=mixed)

    assert mixed_score.base_score > downed_score.base_score
    assert "low-action tag penalty" in downed_score.explanation
    assert "low-action tag penalty" not in mixed_score.explanation


def test_taste_cannot_change_primary_highlight():
    events = [_active_sniper(aim="hunting_ads_zoom"), _active_spray()]
    profile = build_highlight_profile(events)
    primary_before = profile.primary.highlight_type

    # Taste only affects personal blend, not which event is primary.
    loved = score_clip(
        _features(["sniper_kill", "spray_kill"]),
        personal_score=0.95,
        highlight_profile=profile,
    )
    hated = score_clip(
        _features(["sniper_kill", "spray_kill"]),
        personal_score=0.05,
        highlight_profile=profile,
    )
    profile_after = build_highlight_profile(events)

    assert primary_before == "spray_kill"
    assert profile_after.primary.highlight_type == primary_before
    assert loved.final_score > hated.final_score


def test_demo_rows_without_event_profile_keep_legacy_scoring():
    features = _features(["combat_visible", "elimination_or_knock", "sniper_kill"])
    legacy = score_clip(features, None)
    empty_profile = score_clip(features, None, highlight_profile=None)

    assert legacy.base_score == empty_profile.base_score
    assert "primary" not in legacy.explanation
