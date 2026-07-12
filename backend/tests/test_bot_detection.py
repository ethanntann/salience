from salience_api.features.bot_detection import opponent_likely_bot_from_evidence


def test_standing_still_alone_does_not_imply_bot():
    assert not opponent_likely_bot_from_evidence(
        ["Enemy is standing still in the open before being eliminated"]
    )


def test_botlike_name_with_behavior_implies_likely_bot():
    assert opponent_likely_bot_from_evidence(
        [
            "Kill feed shows TallestAnt4 eliminated",
            "Opponent is standing still and does not react",
        ]
    )


def test_anonymous_name_is_not_likely_bot():
    assert not opponent_likely_bot_from_evidence(
        ["Anonymous[313] stands still and does not react"]
    )
