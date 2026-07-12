"""Local ONNX student teacher matching the Fireworks teacher client interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from salience_api.features.fireworks_teacher import (
    ClipTeacherInput,
    WeaponAttribution,
    apply_weapon_ocr_to_event,
    max_event_candidates,
    resolve_event_summaries,
)
from salience_api.features.hud_ocr import RapidHudOcr, best_weapon_ocr
from salience_api.features.screen_state_ocr import matches_menu_screen, matches_victory_banner
from salience_api.features.teacher_labels import TeacherClipLabels, normalize_teacher_payload
from salience_api.student.event_heads import (
    collapse_same_kill_finishes,
    event_summary_from_heads,
)
from salience_api.student.locator import timestamps_from_frame_scores
from salience_api.student.onnx_runtime import IMAGE_SIZE, StudentOnnxModels, preprocess_frame_nchw
from salience_api.student.schema import hud_crop_left, select_window_indices


def _timestamp(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _event_index(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def load_frames_nchw(image_paths: list[Path]) -> list[np.ndarray]:
    """Load clip frame images as ImageNet-normalized NCHW tensors."""
    import cv2

    frames: list[np.ndarray] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"unable to read frame image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            rgb = cv2.resize(
                rgb,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        frames.append(preprocess_frame_nchw(rgb))
    return frames


def load_hud_frames_nchw(image_paths: list[Path]) -> list[np.ndarray]:
    """Load the enlarged right-hand weapon HUD panel from event composites."""
    import cv2

    frames: list[np.ndarray] = []
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"unable to read frame image: {path}")
        hud = bgr[:, hud_crop_left(bgr.shape[1]) :]
        rgb = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        frames.append(preprocess_frame_nchw(rgb))
    return frames


def _event_timestamp(
    clip: ClipTeacherInput,
    *,
    event_index: int,
    frame_indices: list[int],
) -> float:
    centers = [
        center
        for center, index in zip(
            clip.image_event_centers,
            clip.image_event_indices,
            strict=False,
        )
        if index == event_index and center is not None
    ]
    if centers:
        return float(centers[0])
    timestamps = [
        clip.image_timestamps[index]
        for index in frame_indices
        if index < len(clip.image_timestamps)
    ]
    if timestamps:
        return float(sum(timestamps) / len(timestamps))
    return 0.0


class LocalTeacherClient:
    provider: str = "local"

    def __init__(self, *, models: StudentOnnxModels, model: str) -> None:
        self.provider = "local"
        self.model = model
        self._models = models

    @classmethod
    def from_sessions(
        cls,
        *,
        models: StudentOnnxModels,
        model: str,
    ) -> LocalTeacherClient:
        return cls(models=models, model=model)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> LocalTeacherClient:
        root = Path(artifacts_dir)
        model = "local-student"
        meta_path = root / "artifact_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model = str(meta.get("version", model))
        models = StudentOnnxModels.from_artifacts(root)
        return cls(models=models, model=model)

    def locate_event_timestamps(self, clip: ClipTeacherInput) -> list[float]:
        if not clip.image_paths:
            return []
        images = load_frames_nchw(clip.image_paths)
        scores = self._models.score_frames(images)
        timestamps = (
            list(clip.image_timestamps)
            if len(clip.image_timestamps) == len(clip.image_paths)
            else [float(index) for index in range(len(clip.image_paths))]
        )
        return timestamps_from_frame_scores(
            scores,
            timestamps,
            max_events=max_event_candidates(clip.duration_sec),
            always_return_best=True,
        )

    def label_weapon_event(self, clip: ClipTeacherInput) -> WeaponAttribution:
        expected_event_indices = sorted(
            {int(index) for index in clip.image_event_indices if index is not None}
        )
        events: list[dict] = []
        confidences: list[float] = []

        for event_index in expected_event_indices:
            frame_indices = [
                index
                for index, value in enumerate(clip.image_event_indices)
                if value == event_index
            ]
            images = load_frames_nchw(
                [clip.image_paths[index] for index in frame_indices]
            )
            hud_images = load_hud_frames_nchw(
                [clip.image_paths[index] for index in frame_indices]
            )
            prediction = self._models.predict_event(images, hud_images)
            summary = event_summary_from_heads(
                prediction,
                event_index=event_index,
                event_timestamp=_event_timestamp(
                    clip,
                    event_index=event_index,
                    frame_indices=frame_indices,
                ),
            )
            events.append(summary)

        events = collapse_same_kill_finishes(events)
        confidences = [float(summary["weapon_confidence"]) for summary in events]

        prepared: dict = {
            "events": events,
            "confidence": float(np.mean(confidences)) if confidences else 0.0,
            "evidence": ["local student event heads"],
        }

        for event in prepared["events"]:
            event_index = _event_index(event.get("event_index"))
            ocr_evidence = best_weapon_ocr(
                clip.ocr_observations,
                event_index,
                event_timestamp=_timestamp(event.get("event_timestamp")),
            )
            if ocr_evidence is not None:
                apply_weapon_ocr_to_event(event, ocr_evidence, prefer_ocr=True)

        resolved = resolve_event_summaries(prepared)
        return WeaponAttribution(
            labels=resolved.labels,
            confidence=resolved.confidence,
            evidence=resolved.evidence,
            status=resolved.status,
            events=resolved.events,
            raw_payload=prepared,
        )

    def label_clip(
        self, clip: ClipTeacherInput, *, ocr: RapidHudOcr | None = None
    ) -> TeacherClipLabels:
        labels = self._models.predict_context(
            load_frames_nchw(clip.image_paths)
        ) if clip.image_paths else {}

        if ocr is not None and clip.image_paths:
            labels = dict(labels)
            total = len(clip.image_paths)
            sample_count = min(5, total)
            # Victory banners and post-match menus live in the final seconds,
            # so always OCR the last few frames alongside the even samples.
            tail_indices = range(max(0, total - 3), total)
            sample_indices = sorted(
                set(select_window_indices(total, sample_count)) | set(tail_indices)
            )
            for index in sample_indices:
                texts = ocr.recognize_full_frame_text(clip.image_paths[index])
                if not texts:
                    continue
                if matches_victory_banner(texts):
                    labels["victory"] = "yes"
                if matches_menu_screen(texts):
                    labels["looting_or_menu"] = "yes"
                if labels.get("victory") == "yes" and labels.get("looting_or_menu") == "yes":
                    break

        return normalize_teacher_payload(
            {
                "labels": labels,
                "confidence": 0.5,
                "evidence": ["local student coarse context"],
            }
        )
