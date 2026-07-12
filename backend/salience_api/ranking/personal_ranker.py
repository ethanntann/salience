from dataclasses import dataclass
import math


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


@dataclass
class EmptyPersonalRanker:
    def predict(self, rows: list[list[float]]) -> list[float]:
        return [0.0 for _ in rows]


@dataclass
class CentroidPersonalRanker:
    positive_centroid: list[float]
    negative_centroid: list[float]

    def predict(self, rows: list[list[float]]) -> list[float]:
        scores: list[float] = []
        for row in rows:
            positive_similarity = _cosine(row, self.positive_centroid)
            negative_similarity = _cosine(row, self.negative_centroid)
            scores.append(
                max(
                    0.0,
                    min(1.0, (positive_similarity - negative_similarity + 1.0) / 2.0),
                )
            )
        return scores


@dataclass
class OneSidedPersonalRanker:
    centroid: list[float]
    positive: bool

    def predict(self, rows: list[list[float]]) -> list[float]:
        scores: list[float] = []
        for row in rows:
            similarity = max(0.0, _cosine(row, self.centroid))
            scores.append(similarity if self.positive else 1.0 - similarity)
        return scores


@dataclass
class LinearPersonalRanker:
    weights: list[float]
    bias: float

    def predict(self, rows: list[list[float]]) -> list[float]:
        scores: list[float] = []
        for row in rows:
            raw = (
                sum(
                    value * weight
                    for value, weight in zip(row, self.weights, strict=True)
                )
                + self.bias
            )
            scores.append(_sigmoid(raw))
        return scores


def _weighted_centroid(rows: list[list[float]], weights: list[float]) -> list[float]:
    width = len(rows[0])
    total = sum(abs(weight) for weight in weights)
    if total == 0:
        return [0.0] * width
    return [
        sum(row[index] * abs(weight) for row, weight in zip(rows, weights, strict=True))
        / total
        for index in range(width)
    ]


def train_personal_ranker(
    feature_rows: list[list[float]],
    feedback_weights: list[float],
) -> EmptyPersonalRanker | OneSidedPersonalRanker | LinearPersonalRanker:
    positives = [
        (row, weight)
        for row, weight in zip(feature_rows, feedback_weights, strict=True)
        if weight > 0
    ]
    negatives = [
        (row, weight)
        for row, weight in zip(feature_rows, feedback_weights, strict=True)
        if weight < 0
    ]
    if not positives and not negatives:
        return EmptyPersonalRanker()
    if positives and not negatives:
        positive_rows, positive_weights = zip(*positives, strict=True)
        return OneSidedPersonalRanker(
            centroid=_weighted_centroid(list(positive_rows), list(positive_weights)),
            positive=True,
        )
    if negatives and not positives:
        negative_rows, negative_weights = zip(*negatives, strict=True)
        return OneSidedPersonalRanker(
            centroid=_weighted_centroid(list(negative_rows), list(negative_weights)),
            positive=False,
        )

    positive_rows, positive_weights = zip(*positives, strict=True)
    negative_rows, negative_weights = zip(*negatives, strict=True)
    positive_centroid = _weighted_centroid(list(positive_rows), list(positive_weights))
    negative_centroid = _weighted_centroid(list(negative_rows), list(negative_weights))
    weights = [
        4.0 * (positive_value - negative_value)
        for positive_value, negative_value in zip(
            positive_centroid, negative_centroid, strict=True
        )
    ]
    midpoint = [
        (positive_value + negative_value) / 2.0
        for positive_value, negative_value in zip(
            positive_centroid, negative_centroid, strict=True
        )
    ]
    bias = -sum(value * weight for value, weight in zip(midpoint, weights, strict=True))
    return LinearPersonalRanker(weights=weights, bias=bias)
