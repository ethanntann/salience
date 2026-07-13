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


def test_reconcile_uses_finishing_weapon_after_setup_damage():
    sniper = _finish(0, 2.15, "sniper_or_hunting", "SameOpponent")
    sniper.update(
        {
            "selected_weapon_name_text": "HUNTING RIFLE",
            "single_shot_damage": 150,
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "sniper_or_hunting",
                "confidence": 0.99,
            },
        }
    )
    shotgun = _finish(1, 3.59, "shotgun", "SameOpponent")
    shotgun.update(
        {
            "selected_weapon_name_text": "STRIKER PUMP SHOTGUN",
            "single_shot_damage": 150,
            "local_ocr": {
                "applied": True,
                "ambiguous": False,
                "category": "shotgun",
                "confidence": 0.99,
            },
        }
    )

    labels = _reconciled_labels({}, [sniper, shotgun])

    assert labels["sniper_kill"] == "no"
    assert labels["shotgun_kill"] == "yes"
    assert labels["multi_kill"] == "no"


def test_reconcile_does_not_credit_weapon_used_only_for_downed_cleanup():
    knock = _finish(0, 10.0, "pistol", "SameOpponent")
    cleanup = _finish(1, 11.5, "shotgun", "SameOpponent")
    cleanup.update(
        {
            "event_kind": "downed_finish",
            "target_was_active": False,
            "target_was_downed": True,
        }
    )

    labels = _reconciled_labels({}, [knock, cleanup])

    assert labels["pistol_kill"] == "yes"
    assert labels["shotgun_kill"] == "no"
    assert labels["downed_finish"] == "yes"
