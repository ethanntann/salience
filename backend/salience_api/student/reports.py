"""Load offline student speed/agreement reports for the eval UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def student_reports_dir(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir).resolve().parent


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_agreement_path(root: Path) -> Path | None:
    preferred = [
        root / "agreement-v6.json",
        root / "agreement.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(
        root.glob("agreement*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_student_reports(artifacts_dir: Path) -> dict[str, Any]:
    root = student_reports_dir(artifacts_dir)
    speed_path = root / "speed-bench.json"
    agreement_path = _latest_agreement_path(root)
    speed_payload = _read_json(speed_path)
    agreement_payload = _read_json(agreement_path) if agreement_path else None

    model = None
    if speed_payload:
        model = speed_payload.get("model")
    if not model and agreement_payload:
        model = agreement_payload.get("model")

    speed = None
    if speed_payload and isinstance(speed_payload.get("summary"), dict):
        summary = speed_payload["summary"]
        phases = {}
        raw_phases = summary.get("phases") or {}
        if isinstance(raw_phases, dict):
            for name, stats in raw_phases.items():
                if not isinstance(stats, dict):
                    continue
                phases[str(name)] = {
                    "mean": float(stats.get("mean", 0.0)),
                    "median": float(stats.get("median", 0.0)),
                    "p90": float(stats.get("p90", 0.0)),
                    "max": float(stats.get("max", 0.0)),
                }
        speed = {
            "clips": int(summary.get("clips", 0)),
            "median_sec": float(summary.get("median_sec", 0.0)),
            "mean_sec": float(summary.get("mean_sec", 0.0)),
            "p90_sec": float(summary.get("p90_sec", 0.0)),
            "max_sec": float(summary.get("max_sec", 0.0)),
            "phases": phases,
        }

    agreement_labels: list[dict[str, Any]] = []
    labels = (agreement_payload or {}).get("labels") or {}
    if isinstance(labels, dict):
        for label_key, metric in labels.items():
            if not isinstance(metric, dict):
                continue
            clips = max(int(metric.get("clips", 0)), 1)
            agreement_labels.append(
                {
                    "label_key": str(label_key),
                    "agree": float(metric.get("agree", 0)) / clips,
                    "precision": float(metric.get("precision", 0.0)),
                    "recall": float(metric.get("recall", 0.0)),
                    "teacher_yes": int(metric.get("teacher_yes", 0)),
                    "student_yes": int(metric.get("student_yes", 0)),
                    "uncertain_rate": float(metric.get("uncertain_rate", 0.0)),
                    "clips": int(metric.get("clips", 0)),
                }
            )
        agreement_labels.sort(
            key=lambda item: (-item["teacher_yes"], -item["agree"], item["label_key"])
        )

    example_clips: list[dict[str, Any]] = []
    raw_clips = (speed_payload or {}).get("clips") or []
    if isinstance(raw_clips, list):
        for item in raw_clips:
            if not isinstance(item, dict) or "clip_id" not in item:
                continue
            labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
            yes_labels = item.get("yes_labels")
            if not isinstance(yes_labels, list):
                yes_labels = sorted(
                    key for key, value in labels.items() if str(value).lower() == "yes"
                )
            events = []
            for event in item.get("events") or []:
                if isinstance(event, dict):
                    events.append(event)
            example_clips.append(
                {
                    "clip_id": int(item["clip_id"]),
                    "filename": str(item.get("filename") or f"clip-{item['clip_id']}"),
                    "duration_sec": float(item.get("duration_sec") or 0.0),
                    "total_sec": float(item.get("total_sec") or 0.0),
                    "locator_timestamps": [
                        float(value)
                        for value in (item.get("locator_timestamps") or [])
                    ],
                    "locator_events": int(item.get("locator_events") or 0),
                    "attribution_status": (
                        str(item["attribution_status"])
                        if item.get("attribution_status") is not None
                        else None
                    ),
                    "yes_labels": [str(value) for value in yes_labels],
                    "labels": {str(key): str(value) for key, value in labels.items()},
                    "events": events,
                    "video_url": (
                        str(item["video_url"])
                        if item.get("video_url")
                        else f"/clips/{int(item['clip_id'])}/video"
                    ),
                }
            )

    available = speed is not None or bool(agreement_labels) or bool(example_clips)
    message = None
    if not available:
        message = (
            f"No student reports found under {root}. "
            "Run bench_speed and eval_agreement first."
        )

    return {
        "available": available,
        "model": str(model) if model else None,
        "speed": speed,
        "agreement_labels": agreement_labels,
        "example_clips": example_clips,
        "speed_report_path": str(speed_path) if speed_path.is_file() else None,
        "agreement_report_path": str(agreement_path) if agreement_path else None,
        "message": message,
    }
