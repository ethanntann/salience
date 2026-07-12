"""Student-vs-teacher label agreement eval harness.

Usage::

    python -m salience_api.student.eval_agreement \\
        --dataset .local-data/student/dataset \\
        --artifacts .local-data/student/artifacts \\
        --report .local-data/student/agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from salience_api.clips.keyframes import (
    cleanup_keyframes,
    extract_event_keyframes,
    extract_timeline_keyframes,
)
from salience_api.features.fireworks_teacher import (
    ClipTeacherInput,
    merge_event_labels,
)
from salience_api.features.hud_ocr import RapidHudOcr
from salience_api.features.teacher_labels import TEACHER_LABEL_KEYS, normalize_teacher_payload
from salience_api.student.backbone import load_manifest
from salience_api.student.local_teacher import LocalTeacherClient
from salience_api.student.schema import ClipStudentRecord


def _normalize_label_value(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no", "uncertain"}:
            return normalized
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "uncertain"


def clip_locator_diagnostic(
    *, locator_events: int, attribution_status: str | None
) -> dict[str, object]:
    """Record why a clip's event-owned labels were (or weren't) forced uncertain.

    ``attribution_status`` is ``None`` when ``LocalTeacherClient.locate_event_timestamps``
    found zero candidate events, so ``label_weapon_event`` never ran and
    ``merge_event_labels`` forces every event-owned label to "uncertain".
    """
    status = attribution_status or "no_attribution"
    return {
        "locator_events": int(locator_events),
        "attribution_status": status,
        "forced_uncertain": locator_events == 0,
    }


def summarize_locator_diagnostics(
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    dropout_clips = 0
    for entry in diagnostics:
        status = str(entry["attribution_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if entry["forced_uncertain"]:
            dropout_clips += 1
    clips = len(diagnostics)
    return {
        "clips": clips,
        "dropout_clips": dropout_clips,
        "dropout_rate": dropout_clips / clips if clips else 0.0,
        "status_counts": status_counts,
    }


def compare_labels(
    teacher: dict[str, str],
    student: dict[str, str],
    keys: list[str],
) -> dict[str, dict[str, int]]:
    """Return explicit agreement, uncertainty, and binary confusion counts."""
    metrics: dict[str, dict[str, int]] = {}
    for key in keys:
        teacher_value = _normalize_label_value(teacher.get(key))
        student_value = _normalize_label_value(student.get(key))
        metrics[key] = {
            "agree": 1 if teacher_value == student_value else 0,
            "teacher_yes": 1 if teacher_value == "yes" else 0,
            "student_yes": 1 if student_value == "yes" else 0,
            "student_uncertain": 1 if student_value == "uncertain" else 0,
            "true_positive": 1 if teacher_value == student_value == "yes" else 0,
            "false_positive": 1 if teacher_value == "no" and student_value == "yes" else 0,
            "false_negative": 1 if teacher_value == "yes" and student_value != "yes" else 0,
        }
    return metrics


def aggregate_label_metrics(
    per_clip_metrics: list[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int | float]]:
    totals: dict[str, dict[str, int | float]] = {}
    for clip_metrics in per_clip_metrics:
        for key, metric in clip_metrics.items():
            bucket = totals.setdefault(
                key,
                {
                    "agree": 0,
                    "teacher_yes": 0,
                    "student_yes": 0,
                    "student_uncertain": 0,
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "clips": 0,
                },
            )
            for name, value in metric.items():
                bucket[name] += int(value)
            bucket["clips"] += 1
    for bucket in totals.values():
        true_positive = int(bucket["true_positive"])
        false_positive = int(bucket["false_positive"])
        false_negative = int(bucket["false_negative"])
        clips = int(bucket["clips"])
        bucket["precision"] = true_positive / max(true_positive + false_positive, 1)
        bucket["recall"] = true_positive / max(true_positive + false_negative, 1)
        bucket["f1"] = (
            2.0 * true_positive
            / max(2.0 * true_positive + false_positive + false_negative, 1)
        )
        bucket["uncertain_rate"] = int(bucket["student_uncertain"]) / max(clips, 1)
    return totals


def summarize_labels(
    labels: dict[str, dict[str, int | float]],
) -> dict[str, Any]:
    """Roll up per-label metrics into a judge-friendly summary.

    Labels the teacher never marked "yes" on the eval split cannot show
    precision/recall, so they are listed as not evaluable instead of
    dragging the macro numbers down as false zeros.
    """
    evaluable = {
        key: metric
        for key, metric in labels.items()
        if int(metric.get("teacher_yes", 0)) > 0
    }
    not_evaluable = sorted(key for key in labels if key not in evaluable)
    f1_scores = {key: float(metric.get("f1", 0.0)) for key, metric in evaluable.items()}
    clips = max(
        (int(metric.get("clips", 0)) for metric in labels.values()), default=0
    )
    agree_rates = [
        float(metric.get("agree", 0)) / max(int(metric.get("clips", 0)), 1)
        for metric in evaluable.values()
    ]
    return {
        "evaluable_labels": len(evaluable),
        "not_evaluable_labels": not_evaluable,
        "macro_f1": (
            sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
        ),
        "mean_agreement": (
            sum(agree_rates) / len(agree_rates) if agree_rates else 0.0
        ),
        "clips": clips,
        "worst_labels_by_f1": sorted(f1_scores, key=f1_scores.get)[:5],
    }


def _labels_dict(labels: object) -> dict[str, str]:
    return {
        key: _normalize_label_value(getattr(labels, key))
        for key in TEACHER_LABEL_KEYS
    }


def _teacher_labels_for_clip(clip: ClipStudentRecord) -> dict[str, str]:
    return _labels_dict(normalize_teacher_payload({"labels": clip.label_json or {}}))


def teacher_labels_for_clip(clip: ClipStudentRecord) -> dict[str, str]:
    """Public alias of ``_teacher_labels_for_clip`` for reuse by calibration tooling."""
    return _teacher_labels_for_clip(clip)


def _student_labels_for_clip(
    client: LocalTeacherClient,
    clip: ClipStudentRecord,
    *,
    ocr: RapidHudOcr,
) -> tuple[dict[str, str], dict[str, object]]:
    video_path = Path(clip.path)
    coarse = extract_timeline_keyframes(video_path, clip.duration_sec)
    coarse_input = ClipTeacherInput(
        filename=clip.filename,
        duration_sec=clip.duration_sec,
        width=None,
        height=None,
        fps=None,
        tags=[],
        image_paths=[frame.path for frame in coarse],
        image_timestamps=[frame.timestamp_sec for frame in coarse],
        image_views=[frame.view for frame in coarse],
    )
    base_labels = client.label_clip(coarse_input, ocr=ocr)
    event_timestamps = (
        client.locate_event_timestamps(coarse_input) if coarse else []
    )
    cleanup_keyframes(coarse)
    coarse = []

    attribution = None
    ocr_observations: list[dict] = []
    if event_timestamps:
        event_frames = extract_event_keyframes(
            video_path,
            clip.duration_sec,
            event_timestamps,
        )
        if event_frames:
            ocr_observations = ocr.recognize_event_frames(event_frames)
            attribution = client.label_weapon_event(
                ClipTeacherInput(
                    filename=clip.filename,
                    duration_sec=clip.duration_sec,
                    width=None,
                    height=None,
                    fps=None,
                    tags=[],
                    image_paths=[frame.path for frame in event_frames],
                    image_timestamps=[frame.timestamp_sec for frame in event_frames],
                    image_views=[frame.view for frame in event_frames],
                    image_event_indices=[frame.event_index for frame in event_frames],
                    image_event_centers=[frame.event_center_sec for frame in event_frames],
                    ocr_observations=ocr_observations,
                )
            )
            cleanup_keyframes(event_frames)

    merged = merge_event_labels(base_labels, attribution)
    diagnostic = clip_locator_diagnostic(
        locator_events=len(event_timestamps),
        attribution_status=attribution.status if attribution else None,
    )
    # Raw (pre-gate) event summaries so damage-cutoff choices can be analyzed
    # offline from the report instead of re-running the 40-minute eval.
    event_summaries = (
        list(attribution.raw_payload.get("events", [])) if attribution else []
    )
    return _labels_dict(merged), diagnostic, event_summaries


def run_agreement_eval(
    *,
    dataset_dir: Path,
    artifacts_dir: Path,
    label_keys: list[str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    keys = label_keys or list(TEACHER_LABEL_KEYS)
    client = LocalTeacherClient.from_artifacts(Path(artifacts_dir))
    ocr = RapidHudOcr(enabled=True)

    per_clip: list[dict[str, dict[str, int]]] = []
    locator_diagnostics: list[dict[str, object]] = []
    skipped: list[dict[str, Any]] = []
    clip_details: list[dict[str, Any]] = []

    for clip in manifest.eval_clips:
        video_path = Path(clip.path)
        if not video_path.is_file():
            skipped.append(
                {
                    "clip_id": clip.clip_id,
                    "path": clip.path,
                    "reason": "missing_video",
                }
            )
            continue
        teacher_labels = _teacher_labels_for_clip(clip)
        student_labels, diagnostic, event_summaries = _student_labels_for_clip(
            client, clip, ocr=ocr
        )
        diagnostic["clip_id"] = clip.clip_id
        locator_diagnostics.append(diagnostic)
        per_clip.append(compare_labels(teacher_labels, student_labels, keys))
        clip_details.append(
            {
                "clip_id": clip.clip_id,
                "teacher": teacher_labels,
                "student": student_labels,
                "events": event_summaries,
            }
        )

    labels = aggregate_label_metrics(per_clip)
    return {
        "dataset_version": manifest.version,
        "model": client.model,
        "eval_clips_total": len(manifest.eval_clips),
        "eval_clips_scored": len(per_clip),
        "eval_clips_skipped": len(skipped),
        "skipped_clips": skipped,
        "summary": summarize_labels(labels),
        "labels": labels,
        "locator_diagnostics": locator_diagnostics,
        "locator_diagnostics_summary": summarize_locator_diagnostics(locator_diagnostics),
        "clip_details": clip_details,
    }


def write_agreement_report(report: dict[str, Any], report_path: Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare local student labels to teacher labels on eval split"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Student dataset directory containing manifest.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory with locator.onnx and event_heads.onnx",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Output JSON report path",
    )
    args = parser.parse_args(argv)
    report = run_agreement_eval(
        dataset_dir=args.dataset,
        artifacts_dir=args.artifacts,
    )
    write_agreement_report(report, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
