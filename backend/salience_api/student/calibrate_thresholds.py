"""Sweep student decision thresholds against the eval split without touching the teacher.

Usage::

    python -m salience_api.student.calibrate_thresholds \\
        --dataset .local-data/student/dataset \\
        --artifacts .local-data/student/artifacts-v6 \\
        --out .local-data/student/thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from salience_api.clips.keyframes import (
    cleanup_keyframes,
    extract_event_keyframes,
    extract_timeline_keyframes,
)
from salience_api.features.fireworks_teacher import (
    ClipTeacherInput,
    apply_weapon_ocr_to_event,
    max_event_candidates,
    merge_event_labels,
    resolve_event_summaries,
)
from salience_api.features.hud_ocr import RapidHudOcr, best_weapon_ocr
from salience_api.features.teacher_labels import TEACHER_LABEL_KEYS, normalize_teacher_payload
from salience_api.student.backbone import CONTEXT_HEADS, load_manifest
from salience_api.student.event_heads import (
    collapse_same_kill_finishes,
    event_summary_from_heads,
)
from salience_api.student.eval_agreement import (
    aggregate_label_metrics,
    compare_labels,
    teacher_labels_for_clip,
)
from salience_api.student.local_teacher import load_frames_nchw, load_hud_frames_nchw
from salience_api.student.locator import timestamps_from_frame_scores
from salience_api.student.onnx_runtime import (
    DEFAULT_SINGLE_SHOT_DAMAGE_CUTOFF,
    DEFAULT_THRESHOLDS,
    StudentOnnxModels,
    decode_context_logits,
    decode_event_logits,
)

# Fields swept independently (coordinate-wise), chosen from the diagnosed
# failure modes: "weapon" drives sniper/shotgun/pistol/automatic precision,
# "single_shot_damage_known" gates the regression feeding high_damage_hit,
# "single_shot_damage_cutoff" drops regression-to-the-mean damage claims,
# "target_state" separates elimination_or_knock from downed_finish, and the
# context heads control the uncertain-fallback rate of coarse labels like
# combat_visible and rotation_traversal.
SWEEP_FIELDS = (
    "weapon",
    "single_shot_damage_known",
    "single_shot_damage_cutoff",
    "target_state",
    *CONTEXT_HEADS,
)
CANDIDATES = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
# Raw damage scale, not a probability: the swept decision boundary for
# regressed single-shot damage claims.
DAMAGE_CUTOFF_CANDIDATES = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 240.0]


def field_candidates(field: str) -> list[float]:
    if field == "single_shot_damage_cutoff":
        return list(DAMAGE_CUTOFF_CANDIDATES)
    return list(CANDIDATES)


def macro_agreement(labels_summary: dict[str, dict[str, int | float]]) -> float:
    """Average positive-class F1, excluding labels with no teacher positives."""
    scores = [
        2.0 * float(metric.get("true_positive", 0))
        / max(
            2.0 * float(metric.get("true_positive", 0))
            + float(metric.get("false_positive", 0))
            + float(metric.get("false_negative", 0)),
            1.0,
        )
        for metric in labels_summary.values()
        if int(metric.get("teacher_yes", 0)) > 0
    ]
    return sum(scores) / len(scores) if scores else 0.0


def sweep_field_threshold(
    cached_clips: list[Any],
    *,
    field: str,
    candidates: list[float],
    decode_fn: Callable[[float, Any], dict[str, str]],
    teacher_labels_by_clip: dict[Any, dict[str, str]],
    label_keys: list[str],
) -> tuple[float, dict[str, dict[str, int | float]]]:
    """Pick the candidate threshold for ``field`` maximizing macro agreement.

    ``decode_fn(threshold, clip_ref) -> student_labels`` must be a pure function
    over already-cached raw logits (no re-inference per candidate).
    """
    best_threshold = candidates[0]
    best_score = -1.0
    best_metrics: dict[str, dict[str, int | float]] = {}
    for candidate in candidates:
        per_clip = [
            compare_labels(
                teacher_labels_by_clip[
                    getattr(clip_ref, "clip_id", clip_ref)
                ],
                decode_fn(candidate, clip_ref),
                label_keys,
            )
            for clip_ref in cached_clips
        ]
        metrics = aggregate_label_metrics(per_clip)
        score = macro_agreement(metrics)
        if score > best_score:
            best_score = score
            best_threshold = candidate
            best_metrics = metrics
    return best_threshold, best_metrics


class _CachedClip:
    __slots__ = ("clip_id", "context_raw", "events_raw", "ocr_observations", "event_meta")

    def __init__(self, clip_id: int) -> None:
        self.clip_id = clip_id
        self.context_raw: dict[str, Any] = {}
        self.events_raw: list[dict[str, Any]] = []
        self.ocr_observations: list[dict] = []
        self.event_meta: list[dict[str, Any]] = []


def _clip_event_timestamp(clip: Any, event_index: int, fallback: float) -> float:
    """Use the canonical event timestamp when the frame cache has no timing sidecar."""
    for event in getattr(clip, "events", []) or []:
        if isinstance(event, dict):
            candidate_index = event.get("event_index", -1)
            values = {field: event.get(field) for field in ("event_timestamp", "finish_timestamp")}
        else:
            candidate_index = getattr(event, "event_index", -1)
            values = {
                field: getattr(event, field, None)
                for field in ("event_timestamp", "finish_timestamp")
            }
        if int(candidate_index) != event_index:
            continue
        for value in values.values():
            if value is not None:
                return float(value)
    return float(fallback)


def _cache_clip_from_dataset(
    client_models: StudentOnnxModels,
    dataset_dir: Path,
    clip,
) -> _CachedClip | None:
    """Load model inputs from the exported dataset instead of decoding source video."""
    clip_dir = Path(dataset_dir) / "frames" / str(clip.clip_id)
    timeline_dir = clip_dir / "timeline"
    events_dir = clip_dir / "events"
    timeline_meta_path = timeline_dir / "meta.json"
    events_meta_path = events_dir / "meta.json"
    if not timeline_meta_path.is_file() and not events_meta_path.is_file():
        return None

    cached = _CachedClip(clip.clip_id)
    if timeline_meta_path.is_file():
        timeline_meta = json.loads(timeline_meta_path.read_text(encoding="utf-8"))
        timeline_paths = [
            timeline_dir / f"{index:04d}.jpg"
            for index, _timestamp in enumerate(timeline_meta.get("timestamps", []))
        ]
        timeline_paths = [path for path in timeline_paths if path.is_file()]
        if timeline_paths:
            timeline_images = load_frames_nchw(timeline_paths)
            cached.context_raw = client_models.raw_context_logits(timeline_images)

    if events_meta_path.is_file():
        events_meta = json.loads(events_meta_path.read_text(encoding="utf-8"))
        grouped_paths: dict[int, list[Path]] = {}
        for entry in events_meta.get("frames", []):
            filename = str(entry.get("filename", ""))
            path = events_dir / filename
            if not filename or not path.is_file():
                continue
            event_index = int(entry.get("event_index", 0))
            grouped_paths.setdefault(event_index, []).append(path)

        for event_index in sorted(grouped_paths):
            paths = grouped_paths[event_index]
            if not paths:
                continue
            images = load_frames_nchw(paths)
            hud_images = load_hud_frames_nchw(paths)
            cached.events_raw.append(client_models.raw_event_logits(images, hud_images))
            fallback = float(event_index)
            cached.event_meta.append(
                {
                    "event_index": event_index,
                    "event_timestamp": _clip_event_timestamp(clip, event_index, fallback),
                }
            )
    return cached


def _cache_clip(
    client_models: StudentOnnxModels,
    ocr: RapidHudOcr,
    clip,
    *,
    dataset_dir: Path | None = None,
) -> _CachedClip | None:
    if dataset_dir is not None:
        cached = _cache_clip_from_dataset(client_models, dataset_dir, clip)
        if cached is not None:
            return cached

    video_path = Path(clip.path)
    if not video_path.is_file():
        return None
    cached = _CachedClip(clip.clip_id)

    coarse = extract_timeline_keyframes(video_path, clip.duration_sec)
    if coarse:
        coarse_paths = [frame.path for frame in coarse]
        cached.context_raw = client_models.raw_context_logits(load_frames_nchw(coarse_paths))

        scores = client_models.score_frames(load_frames_nchw(coarse_paths))
        event_timestamps = timestamps_from_frame_scores(
            scores,
            [frame.timestamp_sec for frame in coarse],
            max_events=max_event_candidates(clip.duration_sec),
            always_return_best=True,
        )
        cleanup_keyframes(coarse)

        if event_timestamps:
            event_frames = extract_event_keyframes(video_path, clip.duration_sec, event_timestamps)
            if event_frames:
                cached.ocr_observations = ocr.recognize_event_frames(event_frames)
                by_index: dict[int, list] = {}
                for frame in event_frames:
                    by_index.setdefault(int(frame.event_index or 0), []).append(frame)
                for event_index, frames in by_index.items():
                    images = load_frames_nchw([frame.path for frame in frames])
                    hud_images = load_hud_frames_nchw([frame.path for frame in frames])
                    raw = client_models.raw_event_logits(images, hud_images)
                    centers = [f.event_center_sec for f in frames if f.event_center_sec is not None]
                    timestamps = [f.timestamp_sec for f in frames]
                    event_timestamp = (
                        float(centers[0]) if centers else float(sum(timestamps) / len(timestamps))
                    )
                    cached.events_raw.append(raw)
                    cached.event_meta.append(
                        {"event_index": event_index, "event_timestamp": event_timestamp}
                    )
                cleanup_keyframes(event_frames)
    return cached


def _decode_cached_clip(cached: _CachedClip, thresholds: dict[str, float]) -> dict[str, str]:
    base_labels_dict = (
        decode_context_logits(cached.context_raw, thresholds=thresholds)
        if cached.context_raw
        else {}
    )
    base_labels = normalize_teacher_payload(
        {"labels": base_labels_dict, "confidence": 0.5, "evidence": []}
    )

    events = []
    for raw, meta in zip(cached.events_raw, cached.event_meta, strict=True):
        prediction = decode_event_logits(raw, thresholds=thresholds)
        summary = event_summary_from_heads(
            prediction,
            event_index=meta["event_index"],
            event_timestamp=meta["event_timestamp"],
        )
        events.append(summary)

    events = collapse_same_kill_finishes(events)

    for event in events:
        ocr_evidence = best_weapon_ocr(
            cached.ocr_observations,
            int(event.get("event_index", 0)),
            event_timestamp=float(event.get("event_timestamp", 0.0)),
        )
        if ocr_evidence is not None:
            apply_weapon_ocr_to_event(event, ocr_evidence, prefer_ocr=True)

    prepared = {
        "events": events,
        "confidence": 0.0,
        "evidence": ["local student event heads (calibration)"],
    }
    resolved = resolve_event_summaries(prepared) if events else None
    merged = merge_event_labels(base_labels, resolved)
    return {key: str(getattr(merged, key)) for key in TEACHER_LABEL_KEYS}


def run_calibration(*, dataset_dir: Path, artifacts_dir: Path) -> dict[str, float]:
    manifest = load_manifest(dataset_dir)
    client_models = StudentOnnxModels.from_artifacts(Path(artifacts_dir))
    ocr = RapidHudOcr(enabled=True)

    cached_clips: list[_CachedClip] = []
    teacher_labels_by_clip: dict[int, dict[str, str]] = {}
    for clip in manifest.eval_clips:
        cached = _cache_clip(client_models, ocr, clip, dataset_dir=dataset_dir)
        if cached is None:
            continue
        cached_clips.append(cached)
        teacher_labels_by_clip[clip.clip_id] = teacher_labels_for_clip(clip)

    thresholds: dict[str, float] = dict(DEFAULT_THRESHOLDS)
    for field in SWEEP_FIELDS:

        def decode_fn(
            candidate: float, cached_clip: _CachedClip, *, field=field
        ) -> dict[str, str]:
            trial_thresholds = {**thresholds, field: candidate}
            return _decode_cached_clip(cached_clip, trial_thresholds)

        best_threshold, _metrics = sweep_field_threshold(
            cached_clips,
            field=field,
            candidates=field_candidates(field),
            decode_fn=decode_fn,
            teacher_labels_by_clip={
                cached.clip_id: teacher_labels_by_clip[cached.clip_id] for cached in cached_clips
            },
            label_keys=list(TEACHER_LABEL_KEYS),
        )
        thresholds[field] = best_threshold

    return thresholds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate student decision thresholds against the eval split"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    thresholds = run_calibration(dataset_dir=args.dataset, artifacts_dir=args.artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(json.dumps(thresholds, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
