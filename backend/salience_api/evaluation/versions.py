from __future__ import annotations

REQUIRED_PROMOTION_LABELS = (
    "sniper_kill",
    "shotgun_one_pump",
    "no_scope",
    "elimination_or_knock",
    "spray_kill",
)

DEFAULT_PRECISION_GATE = 0.90
DEFAULT_RECALL_GATE = 0.80
DEFAULT_MIN_POSITIVES = 30
DEFAULT_MIN_NEGATIVES = 30
DEFAULT_MAX_INCOMPLETE_RATE = 0.02
DEFAULT_MIN_PREDICTION_COVERAGE = 0.95
DEFAULT_MIN_HIGHLIGHT_REVIEWS = 30
DEFAULT_HIGHLIGHT_ACCURACY_GATE = 0.90

HIGHLIGHT_REVIEW_ASPECTS = (
    "primary",
    "timestamp",
    "weapon",
    "target_state",
    "description",
)

BATCH_STATUS_OPEN = "open"
BATCH_STATUS_REVIEW_READY = "review_ready"
BATCH_STATUS_PROMOTING = "promoting"
BATCH_STATUS_PROMOTED = "promoted"
BATCH_STATUS_FAILED = "failed"
