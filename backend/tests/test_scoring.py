from salience_api.features.basic import BasicFeatures
from salience_api.ranking.scoring import score_clip


def test_score_clip_rewards_action_and_penalizes_silence():
    features = BasicFeatures(
        duration_sec=25,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
    )

    score = score_clip(features, personal_score=None)

    assert score.final_score > 0.55
    assert "motion" in score.explanation
    assert score.confidence == 0.9


def test_score_clip_downranks_long_quiet_low_motion_clip():
    features = BasicFeatures(
        duration_sec=90,
        motion_score=0.1,
        audio_peak_score=0.0,
        silence_ratio=0.9,
        extraction_confidence=0.8,
        action_density=0.1,
    )

    score = score_clip(features, personal_score=None)

    assert score.final_score < 0.25


def test_score_clip_penalizes_likely_bot_opponent():
    base_features = BasicFeatures(
        duration_sec=20,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
        tags=["combat_visible", "elimination_or_knock", "shotgun_one_pump"],
    )
    bot_features = BasicFeatures(
        duration_sec=20,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
        tags=[
            "combat_visible",
            "elimination_or_knock",
            "shotgun_one_pump",
            "opponent_likely_bot",
        ],
    )

    base_score = score_clip(base_features, personal_score=None)
    bot_score = score_clip(bot_features, personal_score=None)

    assert bot_score.final_score < base_score.final_score
    assert "likely bot opponent" in bot_score.explanation


def test_score_clip_rewards_flick_and_competitive_context():
    plain_features = BasicFeatures(
        duration_sec=20,
        motion_score=0.65,
        audio_peak_score=0.6,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.7,
        tags=["combat_visible", "elimination_or_knock"],
    )
    skilled_features = BasicFeatures(
        duration_sec=20,
        motion_score=0.65,
        audio_peak_score=0.6,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.7,
        tags=[
            "combat_visible",
            "elimination_or_knock",
            "flick_shot",
            "competitive_context",
        ],
    )

    plain_score = score_clip(plain_features, personal_score=None)
    skilled_score = score_clip(skilled_features, personal_score=None)

    assert skilled_score.final_score > plain_score.final_score
    assert "flick shot" in skilled_score.explanation
    assert "ranked/tournament context" in skilled_score.explanation


def test_score_clip_describes_stationary_sniper_target_without_hard_veto():
    moving_target = BasicFeatures(
        duration_sec=20,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
        tags=["combat_visible", "elimination_or_knock", "sniper_kill"],
    )
    stationary_target = BasicFeatures(
        duration_sec=20,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
        tags=[
            "combat_visible",
            "elimination_or_knock",
            "sniper_kill",
            "stationary_sniper_target",
        ],
    )

    moving_score = score_clip(moving_target, personal_score=0.8)
    stationary_score = score_clip(stationary_target, personal_score=0.8)

    assert stationary_score.final_score >= moving_score.final_score
    assert "stationary sniper target" in stationary_score.explanation


def test_score_clip_blends_trained_zero_but_preserves_no_signal_baseline():
    features = BasicFeatures(
        duration_sec=20,
        motion_score=0.8,
        audio_peak_score=0.7,
        silence_ratio=0.1,
        extraction_confidence=0.9,
        action_density=0.8,
    )

    baseline = score_clip(features, personal_score=None)
    negative_match = score_clip(features, personal_score=0.0)

    assert baseline.personal_score == 0.0
    assert baseline.final_score == baseline.base_score
    assert negative_match.personal_score == 0.0
    assert negative_match.final_score == 0.25 * negative_match.base_score
