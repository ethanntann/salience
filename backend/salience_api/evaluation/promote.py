from __future__ import annotations

from datetime import UTC, datetime
import json
import sqlite3

from salience_api.clips.indexer import (
    EVENT_VALIDATION_SCHEMA,
    TEACHER_SCHEMA_VERSION,
    record_teacher_events,
    record_teacher_labels,
    rescore_all_clips,
)
from salience_api.evaluation.batches import batch_metrics, get_batch, mark_batch_status
from salience_api.evaluation.versions import (
    BATCH_STATUS_FAILED,
    BATCH_STATUS_PROMOTED,
    BATCH_STATUS_PROMOTING,
    BATCH_STATUS_REVIEW_READY,
)
from salience_api.features.teacher_labels import normalize_teacher_payload


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def promote_batch(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    force: bool = False,
    precision_min: float | None = None,
    recall_min: float | None = None,
    min_positives: int | None = None,
    min_negatives: int | None = None,
    max_incomplete_rate: float | None = None,
    min_highlight_reviews: int | None = None,
    highlight_accuracy_min: float | None = None,
) -> dict:
    """Atomically promote a complete review-ready batch into live teacher assignments."""
    gate_kwargs = {}
    if precision_min is not None:
        gate_kwargs["precision_min"] = precision_min
    if recall_min is not None:
        gate_kwargs["recall_min"] = recall_min
    if min_positives is not None:
        gate_kwargs["min_positives"] = min_positives
    if min_negatives is not None:
        gate_kwargs["min_negatives"] = min_negatives
    if max_incomplete_rate is not None:
        gate_kwargs["max_incomplete_rate"] = max_incomplete_rate
    if min_highlight_reviews is not None:
        gate_kwargs["min_highlight_reviews"] = min_highlight_reviews
    if highlight_accuracy_min is not None:
        gate_kwargs["highlight_accuracy_min"] = highlight_accuracy_min
    metrics = batch_metrics(conn, batch_id, **gate_kwargs)
    batch = get_batch(conn, batch_id)
    if batch["candidate_version"] != EVENT_VALIDATION_SCHEMA:
        raise ValueError(
            f"Batch version {batch['candidate_version']} is not promotable"
        )
    if batch["status"] != BATCH_STATUS_REVIEW_READY:
        raise ValueError(f"Batch status must be review_ready, got {batch['status']}")
    if not force and not metrics["gates"]["promotable"]:
        raise ValueError(
            "Batch is not promotable: required label gates or status checks failed"
        )
    mark_batch_status(conn, batch_id, BATCH_STATUS_PROMOTING)

    items = conn.execute(
        """
        select
            i.clip_id,
            i.complete,
            i.prediction_snapshot_id,
            s.label_json,
            s.event_json,
            s.pipeline_manifest_id,
            m.provider,
            m.model
        from evaluation_batch_items i
        join prediction_snapshots s on s.id = i.prediction_snapshot_id
        join teacher_pipeline_manifests m on m.id = s.pipeline_manifest_id
        where i.batch_id = ?
        """,
        (batch_id,),
    ).fetchall()

    promoted = 0
    skipped_incomplete = 0
    try:
        for row in items:
            if not int(row["complete"]):
                skipped_incomplete += 1
                continue
            clip_id = int(row["clip_id"])
            labels = normalize_teacher_payload(json.loads(row["label_json"]))
            event_data = json.loads(row["event_json"] or "{}")
            provider = str(row["provider"])
            model = str(row["model"])
            record_teacher_labels(
                conn,
                clip_id=clip_id,
                provider=provider,
                labels=labels,
                model=model,
            )
            record_teacher_events(
                conn,
                clip_id=clip_id,
                provider=provider,
                model=model,
                status="promoted",
                event_data=event_data,
            )
            label_row = conn.execute(
                """
                select id from teacher_labels
                where clip_id = ? and schema_version like ?
                order by id desc limit 1
                """,
                (clip_id, f"{TEACHER_SCHEMA_VERSION}:%"),
            ).fetchone()
            event_row = conn.execute(
                """
                select id from teacher_events
                where clip_id = ? and schema_version = ?
                order by id desc limit 1
                """,
                (clip_id, TEACHER_SCHEMA_VERSION),
            ).fetchone()
            conn.execute(
                """
                insert into live_teacher_assignments(
                    clip_id, label_row_id, event_row_id, prediction_snapshot_id,
                    pipeline_manifest_id, promoted_from_batch_id, assigned_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(clip_id) do update set
                    label_row_id = excluded.label_row_id,
                    event_row_id = excluded.event_row_id,
                    prediction_snapshot_id = excluded.prediction_snapshot_id,
                    pipeline_manifest_id = excluded.pipeline_manifest_id,
                    promoted_from_batch_id = excluded.promoted_from_batch_id,
                    assigned_at = excluded.assigned_at
                """,
                (
                    clip_id,
                    int(label_row["id"]) if label_row else None,
                    int(event_row["id"]) if event_row else None,
                    int(row["prediction_snapshot_id"]),
                    int(row["pipeline_manifest_id"]),
                    batch_id,
                    _now(),
                ),
            )
            promoted += 1

        conn.execute(
            """
            insert into active_teacher_pipeline(id, pipeline_manifest_id, batch_id, activated_at)
            values (1, ?, ?, ?)
            on conflict(id) do update set
                pipeline_manifest_id = excluded.pipeline_manifest_id,
                batch_id = excluded.batch_id,
                activated_at = excluded.activated_at
            """,
            (int(batch["pipeline_manifest_id"]), batch_id, _now()),
        )
        rescore_all_clips(conn)
        mark_batch_status(conn, batch_id, BATCH_STATUS_PROMOTED)
        conn.execute(
            """
            insert into evaluation_promotions(
                batch_id, promoted_at, item_count, status, details_json
            )
            values (?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                _now(),
                promoted,
                BATCH_STATUS_PROMOTED,
                json.dumps(
                    {
                        "promoted": promoted,
                        "skipped_incomplete": skipped_incomplete,
                        "force": force,
                        "gates": metrics["gates"],
                    }
                ),
            ),
        )
    except Exception:
        mark_batch_status(conn, batch_id, BATCH_STATUS_FAILED)
        raise

    return {
        "batch_id": batch_id,
        "status": BATCH_STATUS_PROMOTED,
        "promoted": promoted,
        "skipped_incomplete": skipped_incomplete,
        "force": force,
        "gates": metrics["gates"],
    }


def legacy_candidates_promotable() -> bool:
    """Unversioned/legacy candidate reviews remain readable but non-promotable."""
    return False
