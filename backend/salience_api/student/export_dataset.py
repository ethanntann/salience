from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from salience_api.student.schema import (
    STUDENT_DATASET_VERSION_PREFIX,
    ClipStudentRecord,
    DatasetManifest,
)
from salience_api.features.teacher_labels import derive_label_confidences

_EXPORT_QUERY = """
select
  c.id as clip_id,
  c.path,
  c.filename,
  c.duration_sec,
  tl.label_json,
  te.event_json
from live_teacher_assignments lta
join clips c on c.id = lta.clip_id
left join teacher_labels tl on tl.id = lta.label_row_id
left join teacher_events te on te.id = lta.event_row_id
order by c.id
"""


def canonicalize_event(event: dict) -> dict:
    """Map reducer output fields back to the student event-head contract."""
    canonical = dict(event)
    canonical.setdefault("event_timestamp", event.get("finish_timestamp"))
    if not canonical.get("target_state"):
        if event.get("target_was_downed"):
            canonical["target_state"] = "already_downed"
        elif event.get("target_was_active"):
            canonical["target_state"] = "active"
        else:
            canonical["target_state"] = "unknown"
    canonical.setdefault(
        "selected_weapon_before_finish", event.get("resolved_weapon", "unknown")
    )
    aim_state = event.get("damage_aim_state") or event.get("finish_aim_state")
    if not aim_state and event.get("hipfire"):
        aim_state = "hipfire"
    canonical.setdefault("aim_state_at_shot", aim_state or "unknown")
    canonical.setdefault("damaging_shot_count", event.get("damage_hit_count"))
    canonical.setdefault(
        "finish_ui_newly_appeared", event.get("finish_onset_supported")
    )
    canonical.setdefault("target_defeat_visible", event.get("visible_defeat_supported"))
    if canonical.get("teacher_confidence") is None:
        raw_confidence = canonical.get("weapon_confidence")
        if raw_confidence is None:
            raw_confidence = 1.0 if canonical.get("status") == "attributed" else 0.25
        try:
            canonical["teacher_confidence"] = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            canonical["teacher_confidence"] = 0.25
    if not isinstance(canonical.get("label_confidences"), dict):
        target_fields = (
            "event_kind",
            "target_state",
            "weapon",
            "aim_state",
            "damaging_shot_count",
            "single_shot_damage",
            "pov_shot_visible",
            "target_reaction_visible",
            "new_damage_visible",
            "target_defeat_visible",
            "finish_ui_newly_appeared",
            "kill_feed_corroborates_pov",
            "visual_action_supported",
            "stationary_target",
            "stationary_duration_supported",
            "flick_shot",
            "trickshot",
            "cleanup_kill",
            "opponent_likely_bot",
            "damage_display_is_cumulative",
            "single_shot_damage_known",
        )
        canonical["label_confidences"] = {
            field: canonical["teacher_confidence"] for field in target_fields
        }
    return canonical


def _target_coverage(records: list[ClipStudentRecord]) -> dict[str, dict[str, int]]:
    fields = (
        "event_kind",
        "target_state",
        "selected_weapon_before_finish",
        "aim_state_at_shot",
        "damaging_shot_count",
        "finish_ui_newly_appeared",
        "target_defeat_visible",
        "flick_shot",
        "trickshot",
    )
    counters = {field: Counter() for field in fields}
    for record in records:
        for event in record.events:
            for field in fields:
                value = event.get(field)
                normalized = str(value).lower() if value is not None else "missing"
                counters[field][normalized] += 1
    return {field: dict(counter) for field, counter in counters.items()}


def _is_eval_clip(clip_id: int, seed: int, eval_fraction: float) -> bool:
    if eval_fraction <= 0:
        return False
    if eval_fraction >= 1:
        return True
    # Stable, well-distributed bucket in [0, 1). Avoid raw (id % N) — small
    # clip ids all fall below eval_fraction when divided by 10_000.
    digest = hashlib.md5(f"{seed}:{clip_id}".encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < eval_fraction


def _row_to_record(row: sqlite3.Row) -> ClipStudentRecord:
    label_json: dict[str, str] = {}
    if row["label_json"]:
        label_json = json.loads(row["label_json"])

    locator_timestamps: list[float] = []
    events: list[dict] = []
    if row["event_json"]:
        event_payload = json.loads(row["event_json"])
        locator_timestamps = list(event_payload.get("locator_timestamps") or [])
        events = [
            canonicalize_event(event)
            for event in (event_payload.get("events") or [])
            if isinstance(event, dict)
        ]

    try:
        teacher_confidence = max(
            0.0, min(1.0, float(label_json.get("confidence", 1.0)))
        )
    except (TypeError, ValueError):
        teacher_confidence = 0.0
    raw_label_confidences = label_json.get("label_confidences")
    if isinstance(raw_label_confidences, dict):
        label_confidences = {}
        for key, value in raw_label_confidences.items():
            try:
                label_confidences[str(key)] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
    else:
        confidence_payload = dict(label_json)
        confidence_payload.setdefault("confidence", teacher_confidence)
        label_confidences = derive_label_confidences(
            confidence_payload, events=events
        )

    return ClipStudentRecord(
        clip_id=int(row["clip_id"]),
        path=str(row["path"]),
        filename=str(row["filename"]),
        duration_sec=row["duration_sec"],
        locator_timestamps=locator_timestamps,
        label_json=label_json,
        events=events,
        teacher_confidence=teacher_confidence,
        label_confidences=label_confidences,
    )


def export_live_teacher_dataset(
    conn: sqlite3.Connection,
    *,
    output_dir: Path,
    eval_fraction: float = 0.15,
    seed: int = 7,
) -> DatasetManifest:
    train_clips: list[ClipStudentRecord] = []
    eval_clips: list[ClipStudentRecord] = []

    for row in conn.execute(_EXPORT_QUERY):
        record = _row_to_record(row)
        if _is_eval_clip(record.clip_id, seed, eval_fraction):
            eval_clips.append(record)
        else:
            train_clips.append(record)

    manifest = DatasetManifest(
        version=STUDENT_DATASET_VERSION_PREFIX,
        train_clips=train_clips,
        eval_clips=eval_clips,
        target_coverage=_target_coverage(train_clips),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_json(), indent=2),
        encoding="utf-8",
    )

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export promoted live teacher labels into a student dataset"
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to salience.db (must contain live_teacher_assignments)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for manifest.json",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        manifest = export_live_teacher_dataset(
            conn,
            output_dir=args.out,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "version": manifest.version,
                "train_clips": len(manifest.train_clips),
                "eval_clips": len(manifest.eval_clips),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
