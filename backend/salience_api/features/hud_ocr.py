from __future__ import annotations

from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from salience_api.clips.keyframes import Keyframe

from salience_api.features.weapon_ontology import weapon_category_from_text

# Prefer HUD reads from the damaging-shot portion of the window, not the
# finish banner where a post-shot swap often already shows a different weapon.
_SHOT_PHASE_END_SEC = 0.15
_OCR_LOOKBACK_SEC = 0.8
_OCR_LOOKAHEAD_SEC = 0.05


class RapidHudOcr:
    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._engine = None

    @property
    def available(self) -> bool:
        return self.enabled and find_spec("rapidocr") is not None

    def _get_engine(self):
        if not self.available:
            return None
        if self._engine is None:
            from rapidocr import RapidOCR

            # RapidOCR's default models ship in the wheel, so production does not
            # need network access or a first-run model download.
            self._engine = RapidOCR()
        return self._engine

    def recognize_event_frames(self, frames: list[Keyframe]) -> list[dict]:
        engine = self._get_engine()
        if engine is None:
            return []

        grouped: dict[int, list[Keyframe]] = defaultdict(list)
        for frame in frames:
            if frame.event_index is not None:
                grouped[int(frame.event_index)].append(frame)

        observations: list[dict] = []
        for event_index, event_frames in grouped.items():
            ordered = sorted(event_frames, key=lambda item: item.timestamp_sec)
            event_center = next(
                (
                    frame.event_center_sec
                    for frame in ordered
                    if frame.event_center_sec is not None
                ),
                None,
            )
            eligible = [
                frame
                for frame in ordered
                if event_center is None or frame.timestamp_sec <= event_center + 0.05
            ]
            if event_center is None:
                selected = eligible[-2:] if len(eligible) > 2 else eligible
            elif len(eligible) <= 2:
                selected = eligible
            else:
                shot_phase = [
                    frame
                    for frame in eligible
                    if frame.timestamp_sec
                    <= float(event_center) - _SHOT_PHASE_END_SEC
                ]
                if len(shot_phase) >= 2:
                    selected = shot_phase[-2:]
                elif shot_phase:
                    selected = [shot_phase[-1], eligible[-1]]
                else:
                    selected = eligible[-2:]
            for frame in selected:
                frame_observations = self._recognize_weapon_panel(
                    engine,
                    frame.ocr_path or frame.path,
                    event_index=event_index,
                    timestamp=frame.timestamp_sec,
                    dedicated_crop=frame.ocr_path is not None,
                )
                observations.extend(frame_observations)
                # Most HUDs are stable across neighboring frames. Only pay for
                # the fallback frame when the first pass found no usable weapon.
                if any(
                    item.get("weapon_category") != "unknown"
                    and float(item.get("confidence", 0.0)) >= 0.5
                    for item in frame_observations
                ):
                    break
        return observations

    def recognize_full_frame_text(self, path: Path) -> list[str]:
        engine = self._get_engine()
        if engine is None:
            return []
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return []
        result = engine(image)
        texts = tuple(getattr(result, "txts", ()) or ())
        return [" ".join(str(text).split()) for text in texts if str(text).strip()]

    @staticmethod
    def _recognize_weapon_panel(
        engine,
        path: Path,
        *,
        event_index: int,
        timestamp: float,
        dedicated_crop: bool,
    ) -> list[dict]:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return []
        height, width = image.shape[:2]
        if dedicated_crop:
            panel = image
        else:
            panel_width = max(1, min(width, max(400, int(width * 0.24))))
            panel = image[0:height, width - panel_width : width]
            panel = cv2.resize(
                panel, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
        result = engine(panel)
        texts = tuple(getattr(result, "txts", ()) or ())
        scores = tuple(getattr(result, "scores", ()) or ())
        observations: list[dict] = []
        for text, score in zip(texts, scores, strict=False):
            normalized = " ".join(str(text).split())
            if not normalized:
                continue
            observations.append(
                {
                    "event_index": event_index,
                    "timestamp": round(float(timestamp), 3),
                    "region": "weapon_hud",
                    "text": normalized,
                    "confidence": max(0.0, min(1.0, float(score))),
                    "weapon_category": weapon_category_from_text(normalized),
                }
            )
        return observations


def best_weapon_ocr(
    observations: list[dict],
    event_index: int,
    *,
    event_timestamp: float | None = None,
) -> dict | None:
    """Pick a weapon from pre-finish HUD OCR, preferring the shot-phase frames.

    Returns None when there is no usable evidence. Returns an ambiguous result
    (category unknown, ambiguous=True) when two weapon families are both strong
    in the same window — callers should not invent a kill weapon in that case.
    """
    per_category: dict[str, dict[float, dict]] = defaultdict(dict)
    for observation in observations:
        if int(observation.get("event_index", -1)) != event_index:
            continue
        category = weapon_category_from_text(observation.get("text"))
        confidence = float(observation.get("confidence", 0.0))
        if category == "unknown" or confidence < 0.5:
            continue
        timestamp = float(observation.get("timestamp", 0.0))
        if event_timestamp is not None and not (
            event_timestamp - _OCR_LOOKBACK_SEC
            <= timestamp
            <= event_timestamp + _OCR_LOOKAHEAD_SEC
        ):
            continue
        previous = per_category[category].get(timestamp)
        if previous is None or confidence > float(previous.get("confidence", 0.0)):
            per_category[category][timestamp] = observation

    def _phase_score(items: list[dict]) -> float:
        score = 0.0
        for item in items:
            confidence = float(item["confidence"])
            timestamp = float(item["timestamp"])
            if event_timestamp is None:
                score += confidence
            elif timestamp <= event_timestamp - _SHOT_PHASE_END_SEC:
                # Damaging-shot HUD outweighs finish-time swap HUD.
                score += confidence * 2.0
            else:
                score += confidence * 0.5
        return score

    candidates: list[tuple[float, float, str, list[dict]]] = []
    for category, by_timestamp in per_category.items():
        items = list(by_timestamp.values())
        maximum = max(float(item["confidence"]) for item in items)
        if len(items) >= 2 or maximum >= 0.85:
            raw = sum(float(item["confidence"]) for item in items)
            candidates.append((_phase_score(items), raw, category, items))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    top_score, _raw, category, items = candidates[0]
    if len(candidates) > 1:
        second_score, second_raw, _second_cat, _second_items = candidates[1]
        # Two families both look real in this window (typical mid-swap).
        if second_score >= top_score * 0.75 and second_raw >= 0.85:
            return {
                "ambiguous": True,
                "category": "unknown",
                "text": "unknown",
                "confidence": 0.0,
                "supporting_frames": 0,
                "categories": [candidates[0][2], candidates[1][2]],
                "reason": "conflicting_weapons_in_event_window",
            }

    # Prefer the earliest strong read in the shot phase for the HUD name.
    shot_items = [
        item
        for item in items
        if event_timestamp is None
        or float(item["timestamp"]) <= event_timestamp - _SHOT_PHASE_END_SEC
    ]
    name_pool = shot_items or items
    best_text = max(name_pool, key=lambda item: float(item["confidence"]))
    return {
        "ambiguous": False,
        "category": category,
        "text": str(best_text["text"]),
        "confidence": max(float(item["confidence"]) for item in items),
        "supporting_frames": len(items),
    }
