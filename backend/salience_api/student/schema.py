from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STUDENT_DATASET_VERSION_PREFIX = "student-dataset-v2-temporal"
EVENT_WINDOW_FRAMES = 7
# Event composites are full(640)+action(640)+hud(400). Crop from the HUD panel start.
COMPOSITE_HUD_LEFT_RATIO = 1280 / 1680
# Window offsets: [-0.75, -0.375, 0, ...]; index 1 is the last pre-finish sample.
WEAPON_EVIDENCE_FRAME_INDEX = 1

WEAPON_CLASSES = (
    "unknown",
    "pistol",
    "sniper_or_hunting",
    "shotgun",
    "automatic",
    "other",
)
EVENT_KINDS = ("none", "damage_only", "knock", "elimination", "downed_finish")
TARGET_STATES = ("active", "already_downed", "unknown")
AIM_STATES = ("hipfire", "ads_no_scope_overlay", "scope_overlay", "other_ads", "unknown")


def select_window_indices(total: int, window: int = EVENT_WINDOW_FRAMES) -> list[int]:
    """Evenly sample ``window`` indices covering ``[0, total)`` (train/serve shared)."""
    if total <= 0:
        return []
    if total <= window:
        return list(range(total))
    last = total - 1
    return [round(index * last / (window - 1)) for index in range(window)]


def hud_crop_left(width: int) -> int:
    return max(0, int(width * COMPOSITE_HUD_LEFT_RATIO))


@dataclass
class ClipStudentRecord:
    clip_id: int
    path: str
    filename: str
    duration_sec: float | None
    locator_timestamps: list[float]
    label_json: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    teacher_confidence: float = 1.0
    label_confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    version: str
    train_clips: list[ClipStudentRecord]
    eval_clips: list[ClipStudentRecord]
    target_coverage: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "train_clips": [asdict(c) for c in self.train_clips],
            "eval_clips": [asdict(c) for c in self.eval_clips],
            "target_coverage": self.target_coverage,
        }
