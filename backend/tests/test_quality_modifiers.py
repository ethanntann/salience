from salience_api.features.quality_modifiers import (
    stationary_sniper_target_from_evidence,
    stationary_target_from_evidence,
)


def test_stationary_target_is_detected_as_standalone_feature():
    assert stationary_target_from_evidence(
        ["Opponent is standing still before a shotgun elimination"]
    )


def test_stationary_without_sniper_context_is_not_stationary_sniper_target():
    assert not stationary_sniper_target_from_evidence(
        ["Opponent is standing still before a shotgun elimination"],
        {"shotgun_one_pump"},
    )


def test_stationary_with_sniper_context_is_stationary_sniper_target():
    assert stationary_sniper_target_from_evidence(
        ["Player hits a Hunting Rifle no-scope on an opponent standing still"],
        {"no_scope"},
    )
