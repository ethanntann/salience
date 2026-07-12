import json
import sqlite3

from salience_api.clips.indexer import (
    EVENT_VALIDATION_SCHEMA,
    index_demo_clip,
    list_ranked_clips,
    record_teacher_candidate,
    record_teacher_events,
    record_teacher_labels,
)
from salience_api.db import init_db
from salience_api.evaluation import (
    REQUIRED_PROMOTION_LABELS,
    batch_metrics,
    create_batch_from_candidates,
    promote_batch,
    record_snapshot_label_review,
)
from salience_api.features.teacher_labels import normalize_teacher_payload
from salience_api.evaluation.versions import BATCH_STATUS_PROMOTED


def _seed_clip(conn: sqlite3.Connection, filename: str) -> int:
    return index_demo_clip(
        conn,
        {
            "filename": filename,
            "duration_sec": 20,
            "motion_score": 0.8,
            "audio_peak_score": 0.7,
            "silence_ratio": 0.05,
            "action_density": 0.85,
            "tags": [],
        },
    )


def _yes_labels(*keys: str) -> dict:
    payload = {
        key: "no"
        for key in (
            "combat_visible",
            "enemy_visible",
            "elimination_or_knock",
            "high_damage_hit",
            "flick_shot",
            "trickshot",
            "no_scope",
            "sniper_kill",
            "shotgun_kill",
            "shotgun_one_pump",
            "pistol_kill",
            "automatic_kill",
            "other_weapon_kill",
            "spray_kill",
            "multi_kill",
            "downed_finish",
            "build_fight",
            "clutch",
            "fast_edit",
            "competitive_context",
            "victory",
            "rotation_traversal",
            "looting_or_menu",
            "downtime",
            "cleanup_kill",
            "opponent_likely_bot",
            "stationary_target",
            "stationary_sniper_target",
        )
    }
    for key in keys:
        payload[key] = "yes"
    return payload


def test_create_batch_freezes_prediction_snapshots():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = _seed_clip(conn, "a.mp4")
    labels = normalize_teacher_payload(
        {
            "labels": _yes_labels("sniper_kill", "elimination_or_knock"),
            "confidence": 0.9,
            "evidence": ["raw"],
        }
    )
    record_teacher_candidate(
        conn,
        clip_id=clip_id,
        provider="test",
        model="model",
        status="ok",
        labels=labels,
        event_data={
            "decision_schema_version": "highlight-audit-v1",
            "highlight_description": "0:14 — Hunting Rifle knock",
            "events": [],
        },
    )

    batch = create_batch_from_candidates(
        conn, batch_key="batch-1", provider="test", model="model"
    )
    snapshots = conn.execute("select * from prediction_snapshots").fetchall()
    items = conn.execute("select * from evaluation_batch_items").fetchall()

    assert batch["status"] == "review_ready"
    assert batch["item_count"] == 1
    assert len(snapshots) == 1
    assert len(items) == 1
    assert json.loads(snapshots[0]["label_json"])["sniper_kill"] == "yes"


def test_snapshot_reviews_drive_batch_gates_and_atomic_promotion():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    positive_ids = []
    negative_ids = []
    for index in range(2):
        clip_id = _seed_clip(conn, f"pos-{index}.mp4")
        positive_ids.append(clip_id)
        labels = normalize_teacher_payload(
            {
                "labels": _yes_labels(*REQUIRED_PROMOTION_LABELS),
                "confidence": 0.9,
                "evidence": [],
            }
        )
        record_teacher_candidate(
            conn,
            clip_id=clip_id,
            provider="test",
            model="model",
            status="ok",
            labels=labels,
            event_data={"events": [{"event_index": 0, "status": "attributed"}]},
        )
    for index in range(2):
        clip_id = _seed_clip(conn, f"neg-{index}.mp4")
        negative_ids.append(clip_id)
        labels = normalize_teacher_payload(
            {"labels": _yes_labels(), "confidence": 0.9, "evidence": []}
        )
        record_teacher_candidate(
            conn,
            clip_id=clip_id,
            provider="test",
            model="model",
            status="ok",
            labels=labels,
            event_data={"events": []},
        )
    incomplete_id = _seed_clip(conn, "incomplete.mp4")
    record_teacher_candidate(
        conn,
        clip_id=incomplete_id,
        provider="test",
        model="model",
        status="failed",
        labels=normalize_teacher_payload(
            {"labels": _yes_labels("sniper_kill"), "confidence": 0.2, "evidence": []}
        ),
        event_data={"events": []},
    )

    batch = create_batch_from_candidates(
        conn, batch_key="promo-batch", provider="test", model="model"
    )
    items = conn.execute(
        "select clip_id, prediction_snapshot_id, complete from evaluation_batch_items"
    ).fetchall()
    snapshot_by_clip = {
        int(row["clip_id"]): int(row["prediction_snapshot_id"]) for row in items
    }

    for clip_id in positive_ids:
        for label in REQUIRED_PROMOTION_LABELS:
            record_snapshot_label_review(
                conn,
                prediction_snapshot_id=snapshot_by_clip[clip_id],
                label_key=label,
                expected_value="yes",
            )
    for clip_id in negative_ids:
        for label in REQUIRED_PROMOTION_LABELS:
            record_snapshot_label_review(
                conn,
                prediction_snapshot_id=snapshot_by_clip[clip_id],
                label_key=label,
                expected_value="no",
            )

    metrics = batch_metrics(
        conn,
        int(batch["id"]),
        precision_min=0.9,
        recall_min=0.8,
        min_positives=2,
        min_negatives=2,
        max_incomplete_rate=0.5,
        min_highlight_reviews=0,
    )
    assert metrics["gates"]["promotable"] is True

    result = promote_batch(
        conn,
        int(batch["id"]),
        precision_min=0.9,
        recall_min=0.8,
        min_positives=2,
        min_negatives=2,
        max_incomplete_rate=0.5,
        min_highlight_reviews=0,
    )
    assert result["status"] == BATCH_STATUS_PROMOTED
    assert result["promoted"] == 4
    assert result["skipped_incomplete"] == 1

    live = {clip["id"]: clip for clip in list_ranked_clips(conn)}
    assert live[positive_ids[0]]["teacher_labels"]["sniper_kill"] == "yes"
    assert live[negative_ids[0]]["teacher_labels"]["sniper_kill"] == "no"
    # Incomplete stays without a promoted live assignment from this batch.
    assignment = conn.execute(
        "select clip_id from live_teacher_assignments where clip_id = ?",
        (incomplete_id,),
    ).fetchone()
    assert assignment is None
    active = conn.execute(
        "select batch_id from active_teacher_pipeline where id = 1"
    ).fetchone()
    assert int(active["batch_id"]) == int(batch["id"])

    # Later experimental writes cannot bypass the promoted live pointer.
    later = normalize_teacher_payload(
        {"labels": _yes_labels(), "confidence": 0.9, "evidence": []}
    )
    record_teacher_labels(
        conn,
        clip_id=positive_ids[0],
        provider="experiment",
        labels=later,
        model="later-model",
    )
    record_teacher_events(
        conn,
        clip_id=positive_ids[0],
        provider="experiment",
        model="later-model",
        status="attributed",
        event_data={"events": []},
    )
    still_live = {clip["id"]: clip for clip in list_ranked_clips(conn)}
    assert still_live[positive_ids[0]]["teacher_labels"]["sniper_kill"] == "yes"


def test_unversioned_legacy_batch_is_not_promotable():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = _seed_clip(conn, "legacy.mp4")
    record_teacher_candidate(
        conn,
        clip_id=clip_id,
        provider="test",
        model="model",
        status="ok",
        labels=normalize_teacher_payload(
            {"labels": _yes_labels("sniper_kill"), "confidence": 0.9, "evidence": []}
        ),
        event_data={"events": []},
    )
    # Force a legacy version by rewriting after freeze path isn't used.
    conn.execute("update teacher_event_candidates set candidate_version = 'legacy'")
    batch = create_batch_from_candidates(
        conn,
        batch_key="legacy-batch",
        candidate_version="legacy",
        provider="test",
        model="model",
    )
    metrics = batch_metrics(
        conn,
        int(batch["id"]),
        min_positives=0,
        min_negatives=0,
        precision_min=0.0,
        recall_min=0.0,
    )
    assert metrics["gates"]["promotable"] is False
    assert EVENT_VALIDATION_SCHEMA != "legacy"
