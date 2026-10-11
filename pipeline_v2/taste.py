"""Preference models and fight-grouped cross-validation.
The logistic and centroid models fit weighted keep/skip examples. Pairwise
accuracy evaluates held-out ordering; the models are not trained on pairs."""

from __future__ import annotations

from dataclasses import dataclass

OUTCOMES = ("elimination", "knock", "death", "no_event", "other")
HEALTH = ("healthy", "damaged", "critical", "unknown")


def _unit(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector] if norm else list(vector)


@dataclass(frozen=True)
class Example:
    clip_sha: str
    engagement: str | None
    label: int
    weight: float
    fields: list[float]
    embedding: list[float]
    image: list[float] | None = None
    audio: list[float] | None = None
    # Reaction scalars: [peak/median loudness, loud fraction, has-speech].
    # Deliberately separate from the transcript embedding -- "people got loud"
    # is a different claim from "people were discussing X", and the transcript
    # test measured only the second.
    reaction: list[float] | None = None
    # Loudness split by what made the noise: [voice-peak strength,
    # effect-peak strength, voice fraction]. Whichever of the first two carries
    # weight says what `peak_ratio` was actually detecting.
    peak_split: list[float] | None = None

    def features(self, mode: str) -> list[float]:
        if mode == "fields":
            return _unit(self.fields)
        if mode == "embedding":
            return _unit(self.embedding)
        if mode == "image":
            return _unit(self.image or [])
        if mode == "peak_split":
            return list(self.peak_split or [])
        if mode == "text+peak_split":
            return _unit(self.embedding) + list(self.peak_split or [])
        if mode == "image+text+peak_split":
            return (
                _unit(self.embedding)
                + _unit(self.image or [])
                + list(self.peak_split or [])
            )
        if mode == "reaction":
            return list(self.reaction or [])
        if mode == "text+reaction":
            return _unit(self.embedding) + list(self.reaction or [])
        if mode == "image+text+reaction":
            return (
                _unit(self.embedding)
                + _unit(self.image or [])
                + list(self.reaction or [])
            )
        if mode == "audio":
            return _unit(self.audio or [])
        if mode == "text+audio":
            return _unit(self.embedding) + _unit(self.audio or [])
        if mode == "all":
            return (
                _unit(self.embedding)
                + _unit(self.image or [])
                + _unit(self.audio or [])
            )
        if mode == "image+text":
            # Two views of the same clip: what it looked like and what it was
            # described as. Normalised per block so neither wins on dimension
            # count alone.
            return _unit(self.image or []) + _unit(self.embedding)
        # Each block is normalised *before* concatenation. Without this the
        # comparison is rigged by dimension count: a 500-dim embedding next to a
        # 30-dim field vector contributes most of the norm regardless of whether
        # it carries any signal.
        if mode == "both":
            return _unit(self.fields) + _unit(self.embedding)
        raise ValueError(f"unknown feature mode: {mode}")


def has_features(example: Example, mode: str) -> bool:
    """Do not train on variable-length vectors when an optional block is absent."""
    blocks = {
        "fields": ("fields",),
        "embedding": ("embedding",),
        "both": ("fields", "embedding"),
        "image": ("image",),
        "audio": ("audio",),
        "reaction": ("reaction",),
        "peak_split": ("peak_split",),
        "image+text": ("image", "embedding"),
        "text+audio": ("embedding", "audio"),
        "all": ("embedding", "image", "audio"),
        "text+reaction": ("embedding", "reaction"),
        "image+text+reaction": ("image", "embedding", "reaction"),
        "text+peak_split": ("embedding", "peak_split"),
        "image+text+peak_split": ("image", "embedding", "peak_split"),
    }
    if mode not in blocks:
        raise ValueError(f"unknown feature mode: {mode}")
    return all(bool(getattr(example, block)) for block in blocks[mode])


def field_vector(
    payload: dict, facts: dict, mechanics_vocab: tuple[str, ...]
) -> list[float]:
    """Structured fields as numbers -- the v1-style representation, for comparison.

    No weapon block. It read `weapon_category`, which was a derived property and
    so never survived `model_dump()` -- absent from all 348 stored payloads and
    every `facts_json`, leaving six dimensions pinned to "unknown" for every clip
    since the table was created. The v1 OCR ontology that produced it went back
    to `backend/` with the rest of the retired pipeline.
    """
    outcome = str(payload.get("outcome") or "other")
    health = str(payload.get("player_health_state") or "unknown")
    mechanics = {str(m) for m in (payload.get("mechanics") or [])}
    census = facts.get("event_census") or {}
    return [
        *(1.0 if outcome == o else 0.0 for o in OUTCOMES),
        *(1.0 if health == h else 0.0 for h in HEALTH),
        *(1.0 if m in mechanics else 0.0 for m in mechanics_vocab),
        min(float(census.get("elimination", 0)), 4.0) / 4.0,
        min(float(census.get("knock", 0)), 4.0) / 4.0,
        1.0 if facts.get("victory_banner") else 0.0,
    ]


def _centroid(rows: list[list[float]], weights: list[float]) -> list[float]:
    total = sum(weights) or 1.0
    dims = len(rows[0])
    return [
        sum(row[i] * weight for row, weight in zip(rows, weights, strict=True)) / total
        for i in range(dims)
    ]


class LogisticRanker:
    """Logistic regression on the same features, fitted with sample weights.

    The centroid draws one line through the middle of the classes and treats
    every dimension as equally informative. Regression learns per-dimension
    weights, which is what `both` needs: an equal-weighted concatenation of a
    30-dim field block and a 384-dim embedding lets dimension count decide the
    balance, and measured, that made `both` worse than the embedding alone.

    Chosen at 175 labels, not at 24 -- with a few dozen examples this would have
    memorised the training set and reported a flattering number.
    """

    def __init__(self, c: float = 1.0) -> None:
        self.model = None
        self.c = c

    def fit(self, examples: list[Example], mode: str) -> "LogisticRanker":
        from sklearn.linear_model import LogisticRegression

        labels = [e.label for e in examples]
        if len(set(labels)) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(
            C=self.c, max_iter=2000, class_weight="balanced"
        ).fit(
            [e.features(mode) for e in examples],
            labels,
            sample_weight=[e.weight for e in examples],
        )
        return self

    def score(self, example: Example, mode: str) -> float:
        if self.model is None:
            return 0.0
        return float(self.model.decision_function([example.features(mode)])[0])


class CentroidRanker:
    def __init__(self) -> None:
        self.direction: list[float] | None = None

    def fit(self, examples: list[Example], mode: str) -> "CentroidRanker":
        positives = [e for e in examples if e.label == 1]
        negatives = [e for e in examples if e.label == 0]
        if not positives or not negatives:
            self.direction = None
            return self
        pos = _centroid(
            [e.features(mode) for e in positives], [e.weight for e in positives]
        )
        neg = _centroid(
            [e.features(mode) for e in negatives], [e.weight for e in negatives]
        )
        self.direction = [p - n for p, n in zip(pos, neg, strict=True)]
        return self

    def score(self, example: Example, mode: str) -> float:
        """Cosine, not raw dot product.

        A raw projection rewards a clip for having a large feature norm -- more
        mechanics tags, longer summary -- independently of whether it points the
        liked way. Cosine measures only direction, which is the thing the model
        actually learned.
        """
        if self.direction is None:
            return 0.0
        vector = example.features(mode)
        denominator = (
            sum(v * v for v in vector) ** 0.5
            * sum(d * d for d in self.direction) ** 0.5
        )
        if not denominator:
            return 0.0
        return (
            sum(v * d for v, d in zip(vector, self.direction, strict=True))
            / denominator
        )


def _pair_credit(good_score: float, bad_score: float) -> float:
    """Credit for one held-out comparison, ties counted as half.

    A strict ``>`` scored every tie as a loss, so a model with nothing to say
    reported 0% rather than chance -- printed under a banner reading "below that
    is worse than shuffling". A feature block that is absent or constant ties
    every pair, which is uninformative, not inverted. Half credit on ties is the
    usual convention for a pairwise ranking metric (it is what makes this equal
    the area under the ROC curve).
    """
    if good_score > bad_score:
        return 1.0
    return 0.5 if good_score == bad_score else 0.0


def pairs(examples: list[Example]) -> list[tuple[Example, Example]]:
    """Every (liked, disliked) pair drawn from *different* engagements.

    Two clips of one fight are not an independent comparison -- they show the
    same moment, so ordering them says nothing about taste.
    """
    liked = [e for e in examples if e.label == 1]
    disliked = [e for e in examples if e.label == 0]
    return [
        (good, bad)
        for good in liked
        for bad in disliked
        if good.engagement is None
        or bad.engagement is None
        or good.engagement != bad.engagement
    ]


def fixed_score_accuracy(
    examples: list[Example], scores: dict[str, float]
) -> tuple[float, int]:
    """Pairwise accuracy for a ranking that already exists.

    v1's `final_score` needs no training and no folds -- it is a finished
    ranking. Scoring it on the same clips and the same pairs is the only
    apples-to-apples comparison between the two pipelines, because both are then
    judged by the product question rather than by their own internal metrics.
    """
    correct = 0.0
    total = 0
    for good, bad in pairs(examples):
        if good.clip_sha not in scores or bad.clip_sha not in scores:
            continue
        total += 1
        correct += _pair_credit(scores[good.clip_sha], scores[bad.clip_sha])
    return (correct / total if total else 0.0), total


def within_engagement_pairs(
    examples: list[Example],
) -> list[tuple[Example, Example]]:
    """(liked, disliked) pairs from the *same* fight -- the hard comparison."""
    return [
        (good, bad)
        for good in examples
        if good.label == 1
        for bad in examples
        if bad.label == 0
        and good.engagement is not None
        and good.engagement == bad.engagement
    ]


def pairwise_accuracy(
    examples: list[Example],
    mode: str,
    *,
    folds: int = 5,
    within: bool = False,
    model: str = "centroid",
) -> tuple[float, int]:
    """Cross-validated pairwise accuracy, and the number of pairs judged.

    Folds split by *fight*, and pairs are formed only among held-out clips, so no
    clip contributes to both the direction vector and the comparison it is
    judged in.
    """
    if folds < 2:
        raise ValueError("evaluation needs at least two folds")
    # Folds split by ENGAGEMENT, not by clip. Splitting by clip put siblings of
    # one fight on both sides: the model trained on a near-duplicate of the clip
    # it was then tested on, which leaks and inflates the across-fight number.
    # It also scattered siblings across folds, leaving almost no within-fight
    # pairs in any single test set (15 pairs from 334 clips).
    ordered = sorted(examples, key=lambda e: e.clip_sha)
    group_of = {}
    for example in ordered:
        key = example.engagement or example.clip_sha
        group_of.setdefault(key, len(group_of))

    pair_fn = within_engagement_pairs if within else pairs
    correct = 0.0
    total = 0
    for fold in range(folds):

        def in_fold(example, fold=fold):
            return group_of[example.engagement or example.clip_sha] % folds == fold

        test = [e for e in ordered if in_fold(e)]
        train = [e for e in ordered if not in_fold(e)]
        ranker = (
            LogisticRanker().fit(train, mode)
            if model == "logistic"
            else CentroidRanker().fit(train, mode)
        )
        if (
            getattr(ranker, "direction", True) is None
            or getattr(ranker, "model", True) is None
        ):
            continue
        for good, bad in pair_fn(test):
            total += 1
            correct += _pair_credit(
                ranker.score(good, mode), ranker.score(bad, mode)
            )
    return (correct / total if total else 0.0), total
