"""Shared fight ranking for the CLI and browser.
Keep siblings accessible: reliable within-fight ordering is not established."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_v2 import engagements
from pipeline_v2 import facts as facts_mod
from pipeline_v2 import feedback as feedback_mod
from pipeline_v2 import store
from pipeline_v2 import taste as taste_mod
from pipeline_v2 import prompt
from pipeline_v2 import config
from pipeline_v2 import schema


SPEECH_VARIANT = "speech"
# Shares speech is tried at, against the description. 0.3 measured neutral on
# a player whose talk was callouts; 1.0 is for a player whose keeps are mostly
# what was said.
SPEECH_WEIGHTS = (0.0, 0.3, 1.0)
# Speech is on unless the player's ratings show it hurts. It used to be off
# unless they showed it helped, which a player who says six things a session
# can never show -- and what was said is the one signal the frames cannot
# carry. 0.3 was neutral where speech did not matter (92.8% vs 92.6%).
DEFAULT_SPEECH_WEIGHT = 0.3
# A difference smaller than this on held-out minutes is noise: it can neither
# switch speech off nor turn it up.
MIN_SPEECH_GAIN = 0.01
# Held-out blocks for choosing the weight: neighbouring windows of one session
# are near-duplicates, so they must fall in the same fold.
CV_BLOCK_SECONDS = 120.0
# A keep the player marked as loved counts as two. Not yet measured: there
# were no loved ratings to measure it on when it was added.
LOVE_WEIGHT = 2.0


def _cv_group(clip, fallback: str) -> str:
    """The session minute-block a capture clip belongs to, from its sidecar."""
    import json

    try:
        sidecar = json.loads(
            Path(clip.path).with_suffix(".capture.json").read_text(encoding="utf-8"))
        block = int(float(sidecar["segment_start_sec"]) // CV_BLOCK_SECONDS)
    except (OSError, ValueError, KeyError, TypeError):
        return fallback
    return f"{Path(clip.path).parent.parent}:{block}"


def choose_speech_weight(
    training: list, speech: dict, dims: int, groups: dict, mode: str
) -> tuple[float, dict]:
    """How much what was said should count, judged on this player's own ratings.

    Returns (weight, held-out accuracy per weight tried). `DEFAULT_SPEECH_WEIGHT`
    unless the ratings say otherwise: off if it measurably hurts, full if full
    measurably beats it. Without enough speech to judge, the default stands.
    """
    from dataclasses import replace

    if not dims:
        return 0.0, {}
    voiced = sum(e.clip_sha in speech for e in training)
    if voiced < 10:
        return DEFAULT_SPEECH_WEIGHT, {}
    regrouped = [replace(e, engagement=groups.get(e.clip_sha, e.engagement)) for e in training]
    tried: dict[float, float] = {}
    for weight in SPEECH_WEIGHTS:
        examples = [
            replace(e, embedding=taste_mod.with_speech(
                e.embedding, speech.get(e.clip_sha), weight, dims))
            for e in regrouped
        ]
        accuracy, pairs = taste_mod.pairwise_accuracy(
            examples, mode, folds=5, model="logistic")
        if not pairs:
            return DEFAULT_SPEECH_WEIGHT, {}
        tried[weight] = accuracy
    default = DEFAULT_SPEECH_WEIGHT
    if tried[0.0] > tried[default] + MIN_SPEECH_GAIN:
        return 0.0, tried
    if tried[1.0] > tried[default] + MIN_SPEECH_GAIN:
        return 1.0, tried
    return default, tried


@dataclass(frozen=True)
class RankedFight:
    score: float
    best: store.StoredClip
    siblings: list[store.StoredClip]


def ranked_fights(
    conn,
    *,
    db_path: Path | None = None,
    variant: str | None = None,
    mode: str = config.DEFAULT_RANK_MODE,
    info: dict | None = None,
) -> tuple[list[RankedFight], int]:
    """Returns (fights ordered best first, number of labels trained on).

    `info`, when given, is filled with how speech was weighed and why.
    """
    cfg = config.load_config()
    variant = variant or prompt.active_variant(cfg)
    mechanics = schema.mechanics_for(variant)
    stored = store.all_clips(conn, variant)
    vectors = store.embeddings(conn, variant)
    images = store.embeddings(conn, "image")
    prefs = feedback_mod.PreferenceIndex(
        feedback_mod.load(db_path, require_media=False)
    )
    own = {
        row["clip_sha"]: int(row["label"])
        for row in conn.execute("select clip_sha, label from preferences")
    }
    loved = {row[0] for row in conn.execute("select clip_sha from rating_notes where love = 1")}

    groups = engagements.group(
        [c.path for c in stored], event_of=facts_mod.event_from_filename
    )
    engagement_of = {
        clip: f"eng{index}"
        for index, group in enumerate(groups)
        for clip in group.clips
    }

    def example(clip, label: int, weight: float) -> taste_mod.Example:
        return taste_mod.Example(
            clip_sha=clip.clip_sha,
            engagement=engagement_of.get(clip.path, clip.clip_sha),
            label=label,
            weight=weight,
            fields=taste_mod.field_vector(clip.payload, clip.facts, mechanics),
            embedding=vectors.get(clip.clip_sha, []),
            image=images.get(clip.clip_sha),
        )

    training = []
    for clip in stored:
        own_label = own.get(clip.clip_sha)
        pref = prefs.get(clip.path)
        if own_label is None and pref is None:
            continue
        candidate = example(
            clip,
            own_label if own_label is not None else pref.label,
            (LOVE_WEIGHT if clip.clip_sha in loved else 1.0)
            if own_label is not None else pref.weight,
        )
        if taste_mod.has_features(candidate, mode):
            training.append(candidate)
    if len(training) < 10 or len({e.label for e in training}) < 2:
        return [], len(training)

    speech = store.embeddings(conn, SPEECH_VARIANT)
    dims = len(next(iter(speech.values()), []))
    by_sha = {clip.clip_sha: clip for clip in stored}
    blocks = {
        e.clip_sha: _cv_group(by_sha[e.clip_sha], e.engagement or e.clip_sha)
        for e in training
    }
    weight, tried = choose_speech_weight(training, speech, dims, blocks, mode)
    if info is not None:
        info.update(speech_weight=weight, speech_cv=tried)

    def heard(e: taste_mod.Example) -> taste_mod.Example:
        if not weight:
            return e
        from dataclasses import replace

        return replace(e, embedding=taste_mod.with_speech(
            e.embedding, speech.get(e.clip_sha), weight, dims))

    training = [heard(e) for e in training]
    ranker = taste_mod.LogisticRanker().fit(training, mode)
    candidates = ((clip, heard(example(clip, 0, 1.0))) for clip in stored)
    scores = {
        clip.clip_sha: ranker.score(candidate, mode)
        for clip, candidate in candidates
        if taste_mod.has_features(candidate, mode)
    }

    by_path = {clip.path: clip for clip in stored}
    fights: list[RankedFight] = []
    for group in groups:
        members = [
            by_path[path]
            for path in group.clips
            if path in by_path and by_path[path].clip_sha in scores
        ]
        if not members:
            continue
        # The highest score represents the fight; siblings keep their original
        # chronological order. There is no validated within-fight ordering.
        best = max(members, key=lambda c: scores[c.clip_sha])
        fights.append(
            RankedFight(
                score=scores[best.clip_sha],
                best=best,
                siblings=[c for c in members if c.clip_sha != best.clip_sha],
            )
        )
    fights.sort(key=lambda fight: fight.score, reverse=True)
    return fights, len(training)
