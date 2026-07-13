from scripts.reconcile_teacher_weapons import _reconciled_labels


def _finish(index: int, timestamp: float, weapon: str, target: str) -> dict:
    return {
        "event_index": index,
        "status": "attributed",
        "event_kind": "elimination",
        "finish_timestamp": timestamp,
        "resolved_weapon": weapon,
        "selected_weapon_name_text": "unknown",
        "target_identity": target,
        "target_was_active": True,
        "target_was_downed": False,
        "visual_action_supported": True,
        "finish_onset_supported": True,
        "visible_defeat_supported": True,
        "new_damage_visible": True,
        "single_shot_damage": None,
        "damage_hit_count": 1,
        "damage_display_is_cumulative": False,
        "damage_aim_state": "unknown",
        "finish_aim_state": "unknown",
    }


def test_reconcile_removes_weapons_from_lingering_finish_windows():
    shotgun = _finish(0, 10.0, "shotgun", "unknown")
    shotgun.update(
        {
            "selected_weapon_name_text": "STRIKER PUMP SHOTGUN",
            "single_shot_damage": 180,
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "shotgun",
                "confidence": 0.99,
            },
        }
    )
    events = [
        shotgun,
        _finish(1, 10.9, "automatic", "unknown"),
        _finish(2, 11.8, "sniper_or_hunting", "OpponentName"),
    ]

    labels = _reconciled_labels({}, events)

    assert labels["shotgun_kill"] == "yes"
    assert labels["automatic_kill"] == "no"
    assert labels["sniper_kill"] == "no"
    assert labels["multi_kill"] == "no"


def test_reconcile_keeps_distinct_named_rapid_finishes():
    events = [
        _finish(0, 10.0, "sniper_or_hunting", "FirstOpponent"),
        _finish(1, 10.8, "pistol", "SecondOpponent"),
    ]

    labels = _reconciled_labels({}, events)

    assert labels["sniper_kill"] == "yes"
    assert labels["pistol_kill"] == "yes"
    assert labels["multi_kill"] == "yes"
