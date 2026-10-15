"""Speech counts only as much as it predicts this player's own ratings."""

from __future__ import annotations

import random

from pipeline_v2 import ranking
from pipeline_v2 import taste as taste_mod


def _player(keeps_follow: str, n: int = 80, seed: int = 0):
    """Clips whose keep depends on what was seen, or on what was said."""
    rng = random.Random(seed)
    training, speech = [], {}
    for i in range(n):
        seen, said = rng.random() < 0.4, rng.random() < 0.4
        label = int(said if keeps_follow == "speech" else seen)
        sha = f"c{i}"
        noise = [rng.gauss(0, 0.3) for _ in range(6)]
        training.append(taste_mod.Example(
            clip_sha=sha, engagement=f"g{i % 16}", label=label, weight=1.0, fields=[],
            embedding=[1.0 if seen else -1.0, *noise],
        ))
        speech[sha] = [1.0 if said else -1.0, *[rng.gauss(0, 0.3) for _ in range(4)]]
    groups = {e.clip_sha: e.engagement for e in training}
    return training, speech, groups


def test_speech_is_weighed_in_when_keeps_follow_what_was_said():
    training, speech, groups = _player("speech")

    weight, tried = ranking.choose_speech_weight(training, speech, 5, groups, "embedding")

    assert weight > 0
    assert tried[weight] > tried[0.0]


def test_speech_is_ignored_when_keeps_follow_what_was_seen():
    training, speech, groups = _player("seen")

    weight, _ = ranking.choose_speech_weight(training, speech, 5, groups, "embedding")

    assert weight == 0.0


def test_too_little_speech_is_not_judged():
    training, speech, groups = _player("speech")

    weight, tried = ranking.choose_speech_weight(
        training, {k: v for k, v in list(speech.items())[:5]}, 5, groups, "embedding")

    assert (weight, tried) == (0.0, {})


def test_silent_clip_gets_zeros_not_missing_data():
    assert taste_mod.with_speech([3.0, 4.0], None, 0.5, 2) == [0.6, 0.8, 0.0, 0.0]
    assert taste_mod.with_speech([3.0, 4.0], [1.0, 0.0], 0.0, 2) == [3.0, 4.0]
