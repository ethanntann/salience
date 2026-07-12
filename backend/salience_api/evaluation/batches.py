from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import sqlite3

from salience_api.evaluation.metrics import ratio, wilson_interval
from salience_api.evaluation.versions import (
    BATCH_STATUS_OPEN,
    BATCH_STATUS_REVIEW_READY,
    DEFAULT_MAX_INCOMPLETE_RATE,
    DEFAULT_HIGHLIGHT_ACCURACY_GATE,
    DEFAULT_MIN_HIGHLIGHT_REVIEWS,
    DEFAULT_MIN_NEGATIVES,
    DEFAULT_MIN_PREDICTION_COVERAGE,
    DEFAULT_MIN_POSITIVES,
    DEFAULT_PRECISION_GATE,
    DEFAULT_RECALL_GATE,
    HIGHLIGHT_REVIEW_ASPECTS,
    REQUIRED_PROMOTION_LABELS,
)
from salience_api.features.teacher_labels import (
    TEACHER_LABEL_KEYS,
    normalize_teacher_payload,
)
from salience_api.clips.indexer import EVENT_VALIDATION_SCHEMA


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_pipeline_manifest(
    conn: sqlite3.Connection,
    *,
    pipeline_key: str,
    provider: str,
    model: str,
    revision: str,
    prompt_hash: str,
    config: dict | None = None,
) -> int:
    existing = conn.execute(
        "select id from teacher_pipeline_manifests where pipeline_key = ?",
        (pipeline_key,),
    ).fetchone()
    if existing:
        return int(existing["id"])
    cursor = conn.execute(
        """
        insert into teacher_pipeline_manifests(
            pipeline_key, provider, model, prompt_hash, revision, config_json, created_at
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pipeline_key,
            provider,
            model,
            prompt_hash,
            revision,
            json.dumps(config or {}),
            _now(),
        ),
    )
    return int(cursor.lastrowid)


def create_batch_from_candidates(
    conn: sqlite3.Connection,
    *,
    batch_key: str,
    candidate_version: str = EVENT_VALIDATION_SCHEMA,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    existing = conn.execute(
        "select id, status from evaluation_batches where batch_key = ?",
        (batch_key,),
    ).fetchone()
    if existing:
        raise ValueError(f"Batch already exists: {batch_key}")

    candidates = conn.execute(
        """
        select candidate.*
        from teacher_event_candidates candidate
        join (
            select clip_id, max(id) as id
            from teacher_event_candidates
            where candidate_version = ?
            group by clip_id
        ) latest on latest.id = candidate.id
        """,
        (candidate_version,),
    ).fetchall()
    if not candidates:
        raise ValueError(f"No candidates found for version {candidate_version}")
    providers = {str(row["provider"]) for row in candidates}
    models = {str(row["model"]) for row in candidates}
    if len(providers) != 1 or len(models) != 1:
        raise ValueError("Candidate version contains multiple providers or models")
    actual_provider = providers.pop()
    actual_model = models.pop()
    if provider not in {None, "manual", actual_provider} or model not in {
        None,
        "manual",
        actual_model,
    }:
        raise ValueError("Requested provider/model does not match candidate data")
    manifest_payload = {
        "candidate_version": candidate_version,
        "provider": actual_provider,
        "model": actual_model,
    }
    prompt_hash = _stable_hash(manifest_payload)
    pipeline_id = ensure_pipeline_manifest(
        conn,
        pipeline_key=f"{candidate_version}:{actual_provider}:{actual_model}:{prompt_hash}",
        provider=actual_provider,
        model=actual_model,
        revision=candidate_version,
        prompt_hash=prompt_hash,
        config=manifest_payload,
    )
    now = _now()
    cursor = conn.execute(
        """
        insert into evaluation_batches(
            batch_key, pipeline_manifest_id, candidate_version, status, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?)
        """,
        (batch_key, pipeline_id, candidate_version, BATCH_STATUS_OPEN, now, now),
    )
    batch_id = int(cursor.lastrowid)

    incomplete = 0
    for row in candidates:
        label_json = row["label_json"]
        event_json = row["event_json"]
        event_data = json.loads(event_json or "{}")
        frames_hash = _stable_hash(
            event_data.get("frame_refs") or event_data.get("locator_timestamps") or []
        )
        ocr_hash = _stable_hash(
            event_data.get("ocr_observations") or event_data.get("ocr") or []
        )
        complete = (
            1 if str(row["status"]) not in {"failed", "incomplete", "error"} else 0
        )
        if not complete:
            incomplete += 1
        snapshot = conn.execute(
            """
            insert into prediction_snapshots(
                clip_id, batch_id, pipeline_manifest_id, candidate_id,
                label_json, event_json, frames_hash, ocr_hash, frozen_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["clip_id"]),
                batch_id,
                pipeline_id,
                int(row["id"]),
                label_json,
                event_json,
                frames_hash,
                ocr_hash,
                now,
            ),
        )
        conn.execute(
            """
            insert into evaluation_batch_items(
                batch_id, clip_id, candidate_id, prediction_snapshot_id, complete
            )
            values (?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                int(row["clip_id"]),
                int(row["id"]),
                int(snapshot.lastrowid),
                complete,
            ),
        )

    status = BATCH_STATUS_REVIEW_READY
    conn.execute(
        "update evaluation_batches set status = ?, updated_at = ? where id = ?",
        (status, now, batch_id),
    )
    return get_batch(conn, batch_id)


def list_batches(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        select
            b.*,
            count(i.id) as item_count,
            sum(case when i.complete = 0 then 1 else 0 end) as incomplete_count
        from evaluation_batches b
        left join evaluation_batch_items i on i.batch_id = b.id
        group by b.id
        order by b.id desc
        """
    ).fetchall()
    return [_batch_dict(row) for row in rows]


def get_batch(conn: sqlite3.Connection, batch_id: int) -> dict:
    row = conn.execute(
        """
        select
            b.*,
            count(i.id) as item_count,
            sum(case when i.complete = 0 then 1 else 0 end) as incomplete_count
        from evaluation_batches b
        left join evaluation_batch_items i on i.batch_id = b.id
        where b.id = ?
        group by b.id
        """,
        (batch_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Batch not found: {batch_id}")
    return _batch_dict(row)


def _batch_dict(row: sqlite3.Row) -> dict:
    item_count = int(row["item_count"] or 0)
    incomplete_count = int(row["incomplete_count"] or 0)
    return {
        "id": int(row["id"]),
        "batch_key": str(row["batch_key"]),
        "pipeline_manifest_id": int(row["pipeline_manifest_id"]),
        "candidate_version": str(row["candidate_version"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "item_count": item_count,
        "incomplete_count": incomplete_count,
        "incomplete_rate": ratio(incomplete_count, item_count),
        "promotable": str(row["candidate_version"]) == EVENT_VALIDATION_SCHEMA
        and str(row["status"]) == BATCH_STATUS_REVIEW_READY,
    }


def list_batch_clips(conn: sqlite3.Connection, batch_id: int) -> list[dict]:
    rows = conn.execute(
        """
        select
            i.clip_id,
            i.complete,
            i.prediction_snapshot_id,
            i.candidate_id,
            s.label_json,
            s.event_json
        from evaluation_batch_items i
        join prediction_snapshots s on s.id = i.prediction_snapshot_id
        where i.batch_id = ?
        order by i.clip_id
        """,
        (batch_id,),
    ).fetchall()
    clips: list[dict] = []
    for row in rows:
        labels = normalize_teacher_payload(json.loads(row["label_json"]))
        event_data = json.loads(row["event_json"] or "{}")
        clips.append(
            {
                "clip_id": int(row["clip_id"]),
                "complete": bool(row["complete"]),
                "prediction_snapshot_id": int(row["prediction_snapshot_id"]),
                "candidate_id": int(row["candidate_id"])
                if row["candidate_id"] is not None
                else None,
                "candidate_labels": {
                    key: getattr(labels, key) for key in TEACHER_LABEL_KEYS
                },
                "event_json": event_data,
            }
        )
    return clips


def record_snapshot_label_review(
    conn: sqlite3.Connection,
    *,
    prediction_snapshot_id: int,
    label_key: str,
    expected_value: str,
    notes: str | None = None,
) -> None:
    if label_key not in set(TEACHER_LABEL_KEYS):
        raise ValueError(f"Unknown label: {label_key}")
    if expected_value not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Invalid expected_value: {expected_value}")
    snapshot = conn.execute(
        "select id, clip_id from prediction_snapshots where id = ?",
        (prediction_snapshot_id,),
    ).fetchone()
    if snapshot is None:
        raise KeyError(f"Prediction snapshot not found: {prediction_snapshot_id}")
    conn.execute(
        """
        insert into evaluation_label_reviews(
            prediction_snapshot_id, clip_id, label_key, expected_value, notes, created_at
        )
        values (?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_snapshot_id,
            int(snapshot["clip_id"]),
            label_key,
            expected_value,
            notes,
            _now(),
        ),
    )


def record_highlight_review(
    conn: sqlite3.Connection,
    *,
    prediction_snapshot_id: int,
    aspect: str,
    expected_value: str,
    notes: str | None = None,
) -> None:
    if aspect not in HIGHLIGHT_REVIEW_ASPECTS:
        raise ValueError(f"Unknown highlight aspect: {aspect}")
    if expected_value not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Invalid expected_value: {expected_value}")
    snapshot = conn.execute(
        "select id, clip_id from prediction_snapshots where id = ?",
        (prediction_snapshot_id,),
    ).fetchone()
    if snapshot is None:
        raise KeyError(f"Prediction snapshot not found: {prediction_snapshot_id}")
    conn.execute(
        """
        insert into evaluation_highlight_reviews(
            prediction_snapshot_id, clip_id, aspect, expected_value, notes, created_at
        )
        values (?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_snapshot_id,
            int(snapshot["clip_id"]),
            aspect,
            expected_value,
            notes,
            _now(),
        ),
    )


def latest_snapshot_label_reviews(
    conn: sqlite3.Connection, snapshot_ids: list[int]
) -> dict[int, dict[str, str]]:
    if not snapshot_ids:
        return {}
    placeholders = ",".join("?" for _ in snapshot_ids)
    rows = conn.execute(
        f"""
        select r.prediction_snapshot_id, r.label_key, r.expected_value
        from evaluation_label_reviews r
        join (
            select prediction_snapshot_id, label_key, max(id) as id
            from evaluation_label_reviews
            where prediction_snapshot_id in ({placeholders})
            group by prediction_snapshot_id, label_key
        ) latest on latest.id = r.id
        """,
        snapshot_ids,
    ).fetchall()
    reviews: dict[int, dict[str, str]] = {}
    for row in rows:
        reviews.setdefault(int(row["prediction_snapshot_id"]), {})[
            str(row["label_key"])
        ] = str(row["expected_value"])
    return reviews


def latest_snapshot_highlight_reviews(
    conn: sqlite3.Connection, snapshot_ids: list[int]
) -> dict[int, dict[str, str]]:
    if not snapshot_ids:
        return {}
    placeholders = ",".join("?" for _ in snapshot_ids)
    rows = conn.execute(
        f"""
        select r.prediction_snapshot_id, r.aspect, r.expected_value
        from evaluation_highlight_reviews r
        join (
            select prediction_snapshot_id, aspect, max(id) as id
            from evaluation_highlight_reviews
            where prediction_snapshot_id in ({placeholders})
            group by prediction_snapshot_id, aspect
        ) latest on latest.id = r.id
        """,
        snapshot_ids,
    ).fetchall()
    reviews: dict[int, dict[str, str]] = {}
    for row in rows:
        reviews.setdefault(int(row["prediction_snapshot_id"]), {})[
            str(row["aspect"])
        ] = str(row["expected_value"])
    return reviews


def batch_metrics(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    precision_min: float = DEFAULT_PRECISION_GATE,
    recall_min: float = DEFAULT_RECALL_GATE,
    min_positives: int = DEFAULT_MIN_POSITIVES,
    min_negatives: int = DEFAULT_MIN_NEGATIVES,
    max_incomplete_rate: float = DEFAULT_MAX_INCOMPLETE_RATE,
    min_prediction_coverage: float = DEFAULT_MIN_PREDICTION_COVERAGE,
    min_highlight_reviews: int = DEFAULT_MIN_HIGHLIGHT_REVIEWS,
    highlight_accuracy_min: float = DEFAULT_HIGHLIGHT_ACCURACY_GATE,
) -> dict:
    batch = get_batch(conn, batch_id)
    rows = conn.execute(
        """
        select
            r.label_key,
            r.expected_value,
            s.label_json,
            i.complete
        from evaluation_batch_items i
        join prediction_snapshots s on s.id = i.prediction_snapshot_id
        left join (
            select prediction_snapshot_id, label_key, max(id) as id
            from evaluation_label_reviews
            group by prediction_snapshot_id, label_key
        ) latest on latest.prediction_snapshot_id = i.prediction_snapshot_id
        left join evaluation_label_reviews r on r.id = latest.id
        where i.batch_id = ?
        """,
        (batch_id,),
    ).fetchall()

    metrics = {
        label: {
            "label_key": label,
            "reviewed": 0,
            "teacher_yes": 0,
            "expected_yes": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "support_positive": 0,
            "support_negative": 0,
            "abstention": 0,
            "prediction_abstention": 0,
            "prediction_coverage": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "precision_ci": [None, None],
            "recall_ci": [None, None],
            "gate_pass": False,
            "required": label in REQUIRED_PROMOTION_LABELS,
        }
        for label in TEACHER_LABEL_KEYS
    }

    for row in rows:
        label_key = row["label_key"]
        if label_key is None:
            continue
        label_key = str(label_key)
        if label_key not in metrics:
            continue
        expected = str(row["expected_value"])
        metric = metrics[label_key]
        if expected == "uncertain":
            metric["abstention"] += 1
            continue
        if not row["label_json"]:
            continue
        teacher_labels = normalize_teacher_payload(json.loads(row["label_json"]))
        prediction = getattr(teacher_labels, label_key)
        if prediction == "uncertain":
            metric["prediction_abstention"] += 1
            continue
        predicted_yes = prediction == "yes"
        expected_yes = expected == "yes"
        metric["reviewed"] += 1
        metric["teacher_yes"] += 1 if predicted_yes else 0
        metric["expected_yes"] += 1 if expected_yes else 0
        metric["support_positive"] += 1 if expected_yes else 0
        metric["support_negative"] += 0 if expected_yes else 1
        if predicted_yes and expected_yes:
            metric["true_positive"] += 1
        elif predicted_yes and not expected_yes:
            metric["false_positive"] += 1
        elif not predicted_yes and expected_yes:
            metric["false_negative"] += 1
        else:
            metric["true_negative"] += 1

    for metric in metrics.values():
        tp = metric["true_positive"]
        fp = metric["false_positive"]
        fn = metric["false_negative"]
        tn = metric["true_negative"]
        reviewed = metric["reviewed"]
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        metric["precision"] = precision
        metric["recall"] = recall
        metric["accuracy"] = ratio(tp + tn, reviewed)
        total_predictions = reviewed + metric["prediction_abstention"]
        metric["prediction_coverage"] = ratio(reviewed, total_predictions)
        metric["precision_ci"] = list(wilson_interval(tp, tp + fp))
        metric["recall_ci"] = list(wilson_interval(tp, tp + fn))
        metric["gate_pass"] = _label_gate_pass(
            metric,
            precision_min=precision_min,
            recall_min=recall_min,
            min_positives=min_positives,
            min_negatives=min_negatives,
            min_prediction_coverage=min_prediction_coverage,
        )

    incomplete_rate = batch["incomplete_rate"] or 0.0
    required = [metrics[label] for label in REQUIRED_PROMOTION_LABELS]
    highlight_rows = conn.execute(
        """
        select r.aspect, r.expected_value
        from evaluation_batch_items i
        join (
            select prediction_snapshot_id, aspect, max(id) as id
            from evaluation_highlight_reviews
            group by prediction_snapshot_id, aspect
        ) latest on latest.prediction_snapshot_id = i.prediction_snapshot_id
        join evaluation_highlight_reviews r on r.id = latest.id
        where i.batch_id = ?
        """,
        (batch_id,),
    ).fetchall()
    highlight_metrics = {
        aspect: {"aspect": aspect, "reviewed": 0, "correct": 0, "accuracy": None}
        for aspect in HIGHLIGHT_REVIEW_ASPECTS
    }
    for row in highlight_rows:
        value = str(row["expected_value"])
        if value == "uncertain":
            continue
        metric = highlight_metrics[str(row["aspect"])]
        metric["reviewed"] += 1
        metric["correct"] += 1 if value == "yes" else 0
    for metric in highlight_metrics.values():
        metric["accuracy"] = ratio(metric["correct"], metric["reviewed"])
    highlights_pass = all(
        min_highlight_reviews == 0
        or (
            metric["reviewed"] >= min_highlight_reviews
            and (metric["accuracy"] or 0.0) >= highlight_accuracy_min
        )
        for metric in highlight_metrics.values()
    )
    gates = {
        "precision_min": precision_min,
        "recall_min": recall_min,
        "min_positives": min_positives,
        "min_negatives": min_negatives,
        "max_incomplete_rate": max_incomplete_rate,
        "min_prediction_coverage": min_prediction_coverage,
        "incomplete_rate": incomplete_rate,
        "incomplete_pass": incomplete_rate < max_incomplete_rate,
        "required_labels_pass": all(item["gate_pass"] for item in required),
        "required_highlights_pass": highlights_pass,
        "min_highlight_reviews": min_highlight_reviews,
        "highlight_accuracy_min": highlight_accuracy_min,
        "batch_status_ok": batch["status"] == BATCH_STATUS_REVIEW_READY,
        "promotable": (
            batch["status"] == BATCH_STATUS_REVIEW_READY
            and incomplete_rate < max_incomplete_rate
            and all(item["gate_pass"] for item in required)
            and highlights_pass
            and batch["candidate_version"] == EVENT_VALIDATION_SCHEMA
        ),
    }
    return {
        "batch": batch,
        "labels": list(metrics.values()),
        "gates": gates,
        "required_labels": list(REQUIRED_PROMOTION_LABELS),
        "highlights": list(highlight_metrics.values()),
    }


def _label_gate_pass(
    metric: dict,
    *,
    precision_min: float,
    recall_min: float,
    min_positives: int,
    min_negatives: int,
    min_prediction_coverage: float,
) -> bool:
    precision = metric["precision"]
    recall = metric["recall"]
    if precision is None or recall is None:
        return False
    return (
        precision >= precision_min
        and recall >= recall_min
        and metric["support_positive"] >= min_positives
        and metric["support_negative"] >= min_negatives
        and (metric["prediction_coverage"] or 0.0) >= min_prediction_coverage
    )


def mark_batch_status(conn: sqlite3.Connection, batch_id: int, status: str) -> None:
    conn.execute(
        "update evaluation_batches set status = ?, updated_at = ? where id = ?",
        (status, _now(), batch_id),
    )
