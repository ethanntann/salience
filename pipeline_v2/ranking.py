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
) -> tuple[list[RankedFight], int]:
    """Returns (fights ordered best first, number of labels trained on)."""
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
            1.0 if own_label is not None else pref.weight,
        )
        if taste_mod.has_features(candidate, mode):
            training.append(candidate)
    if len(training) < 10 or len({e.label for e in training}) < 2:
        return [], len(training)

    ranker = taste_mod.LogisticRanker().fit(training, mode)
    candidates = ((clip, example(clip, 0, 1.0)) for clip in stored)
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
