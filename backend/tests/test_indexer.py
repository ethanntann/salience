import sqlite3
from pathlib import Path

from salience_api.clips.indexer import (
    TEACHER_FALLBACK_SCHEMA_VERSION,
    _training_rows,
    index_clip_path,
    index_demo_clip,
    index_known_clip_path,
    list_ranked_clips,
    record_teacher_events,
    record_teacher_labels,
    rescore_all_clips,
    training_status,
)
from salience_api.db import init_db
from salience_api.feedback.service import record_feedback
from salience_api.features.teacher_labels import normalize_teacher_payload
from salience_api.taste import save_taste_preferences


def demo_seed(filename: str, tags: list[str] | None = None) -> dict:
    return {
        "filename": filename,
        "duration_sec": 20,
        "motion_score": 0.6,
        "audio_peak_score": 0.6,
        "silence_ratio": 0.1,
        "action_density": 0.6,
        "tags": tags or [],
    }


def test_index_clip_path_inserts_clip(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    clip_id = index_clip_path(conn, clip)

    row = conn.execute("select id, path, filename from clips").fetchone()
    assert row["id"] == clip_id
    assert row["path"] == str(clip)
    assert row["filename"] == "clip.mp4"


def test_ranked_local_clip_exposes_video_url(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_clip_path(conn, clip)

    ranked = list_ranked_clips(conn)

    assert ranked[0]["id"] == clip_id
    assert ranked[0]["video_url"] == f"/clips/{clip_id}/video"


def test_rescore_preserves_frozen_submission_snapshot_score():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    index_demo_clip(
        conn,
        {
            "filename": "teacher-ranked-001.mp4",
            "path": "snapshot://teacher-ranked-001.mp4",
            "source": "supervised",
            "duration_sec": 20,
            "motion_score": 0.6,
            "audio_peak_score": 0.6,
            "silence_ratio": 0.1,
            "action_density": 0.6,
            "tags": ["sniper_kill"],
            "final_score": 0.8929,
            "base_score": 0.5717,
            "confidence": 0.8,
        },
    )

    rescore_all_clips(conn)

    ranked = list_ranked_clips(conn)
    assert ranked[0]["final_score"] == 0.8929
    assert ranked[0]["base_score"] == 0.5717
    assert ranked[0]["video_url"] is None


def test_feedback_changes_ranked_scores():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    high_id = index_demo_clip(
        conn,
        {
            "filename": "high.mp4",
            "duration_sec": 20,
            "motion_score": 0.9,
            "audio_peak_score": 0.8,
            "silence_ratio": 0.05,
            "action_density": 0.9,
            "tags": ["sniper"],
        },
    )
    low_id = index_demo_clip(
        conn,
        {
            "filename": "low.mp4",
            "duration_sec": 20,
            "motion_score": 0.1,
            "audio_peak_score": 0.1,
            "silence_ratio": 0.9,
            "action_density": 0.1,
            "tags": ["low_action"],
        },
    )

    record_feedback(conn, clip_id=high_id, action="favorite", label=None)
    record_feedback(conn, clip_id=low_id, action="boring", label=None)
    record_feedback(conn, clip_id=high_id, action="keep", label=None)
    rescore_all_clips(conn)

    ranked = list_ranked_clips(conn)
    assert ranked[0]["filename"] == "high.mp4"
    assert ranked[0]["personal_score"] > ranked[1]["personal_score"]


def test_single_negative_exact_match_is_downranked_at_zero_personal_score():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("negative.mp4"))

    record_feedback(conn, clip_id=clip_id, action="skip", label=None)
    rescore_all_clips(conn)

    score = conn.execute(
        "select base_score, personal_score, final_score from clip_scores where clip_id = ?",
        (clip_id,),
    ).fetchone()
    assert score["personal_score"] == 0.0
    assert score["final_score"] == 0.25 * score["base_score"]


def test_rescore_without_personalization_keeps_non_null_zero_and_base_score():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("baseline.mp4"))

    rescore_all_clips(conn)

    score = conn.execute(
        "select base_score, personal_score, final_score from clip_scores where clip_id = ?",
        (clip_id,),
    ).fetchone()
    assert score["personal_score"] == 0.0
    assert score["final_score"] == score["base_score"]


def test_reimport_does_not_hide_teacher_tags_from_ranker():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_known_clip_path(
        conn, Path("C:/clips/keeper.mp4"), source="supervised"
    )
    record_teacher_labels(
        conn,
        clip_id=clip_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {"labels": {"sniper_kill": "yes"}, "confidence": 0.8, "evidence": []}
        ),
    )
    index_known_clip_path(conn, Path("C:/clips/keeper.mp4"), source="supervised")
    record_feedback(conn, clip_id=clip_id, action="supervised_keep", label="test")
    rescore_all_clips(conn)

    ranked = list_ranked_clips(conn)

    assert "sniper_kill" in ranked[0]["tags"]


def test_list_ranked_clips_batches_teacher_label_lookup():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    for index in range(2):
        clip_id = index_demo_clip(
            conn,
            {
                "filename": f"clip-{index}.mp4",
                "duration_sec": 20,
                "motion_score": 0.8,
                "audio_peak_score": 0.7,
                "silence_ratio": 0.05,
                "action_density": 0.85,
                "tags": [],
            },
        )
        record_teacher_labels(
            conn,
            clip_id=clip_id,
            provider="test",
            model="test-model",
            labels=normalize_teacher_payload(
                {"labels": {"sniper_kill": "yes"}, "confidence": 0.9, "evidence": []}
            ),
        )

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    list_ranked_clips(conn)
    conn.set_trace_callback(None)

    teacher_reads = [
        statement
        for statement in statements
        if "from teacher_labels" in statement.lower()
    ]
    assert len(teacher_reads) == 1


def test_taste_hate_directly_downranks_matching_teacher_tag():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    sniper_id = index_demo_clip(
        conn,
        {
            "filename": "sniper.mp4",
            "duration_sec": 20,
            "motion_score": 0.8,
            "audio_peak_score": 0.7,
            "silence_ratio": 0.05,
            "action_density": 0.85,
            "tags": [],
        },
    )
    shotgun_id = index_demo_clip(
        conn,
        {
            "filename": "shotgun.mp4",
            "duration_sec": 20,
            "motion_score": 0.8,
            "audio_peak_score": 0.7,
            "silence_ratio": 0.05,
            "action_density": 0.85,
            "tags": [],
        },
    )
    record_teacher_labels(
        conn,
        clip_id=sniper_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {"labels": {"sniper_kill": "yes"}, "confidence": 0.9, "evidence": []}
        ),
    )
    record_teacher_labels(
        conn,
        clip_id=shotgun_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {"labels": {"shotgun_one_pump": "yes"}, "confidence": 0.9, "evidence": []}
        ),
    )

    save_taste_preferences(conn, {"sniper_kill": -2})
    rescore_all_clips(conn)

    ranked = {clip["filename"]: clip for clip in list_ranked_clips(conn)}
    assert ranked["sniper.mp4"]["personal_score"] <= 0.1
    assert ranked["sniper.mp4"]["final_score"] < ranked["shotgun.mp4"]["final_score"]


def test_sniper_taste_downranks_hunting_rifle_no_scope_tag():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    hunting_id = index_demo_clip(
        conn,
        {
            "filename": "hunting-rifle.mp4",
            "duration_sec": 20,
            "motion_score": 0.8,
            "audio_peak_score": 0.7,
            "silence_ratio": 0.05,
            "action_density": 0.85,
            "tags": [],
        },
    )
    shotgun_id = index_demo_clip(
        conn,
        {
            "filename": "shotgun.mp4",
            "duration_sec": 20,
            "motion_score": 0.8,
            "audio_peak_score": 0.7,
            "silence_ratio": 0.05,
            "action_density": 0.85,
            "tags": [],
        },
    )
    record_teacher_labels(
        conn,
        clip_id=hunting_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {
                "labels": {"no_scope": "yes", "high_damage_hit": "yes"},
                "confidence": 0.9,
                "evidence": [],
            }
        ),
    )
    record_teacher_labels(
        conn,
        clip_id=shotgun_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {"labels": {"shotgun_one_pump": "yes"}, "confidence": 0.9, "evidence": []}
        ),
    )

    save_taste_preferences(conn, {"sniper_kill": -2, "shotgun_one_pump": 2})
    rescore_all_clips(conn)

    ranked = {clip["filename"]: clip for clip in list_ranked_clips(conn)}
    assert (
        ranked["hunting-rifle.mp4"]["final_score"]
        < ranked["shotgun.mp4"]["final_score"]
    )


def test_latest_manual_feedback_is_the_only_training_decision():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))

    record_feedback(conn, clip_id=clip_id, action="supervised_skip", label="import")
    record_feedback(conn, clip_id=clip_id, action="keep", label=None)
    _, weights = _training_rows(conn)
    assert len(weights) == 1
    assert weights[0] > 0

    record_feedback(conn, clip_id=clip_id, action="skip", label=None)
    _, weights = _training_rows(conn)
    assert len(weights) == 1
    assert weights[0] < 0


def test_manual_feedback_overrides_liked_import():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))

    record_feedback(conn, clip_id=clip_id, action="keep", label="liked_import")
    record_feedback(conn, clip_id=clip_id, action="skip", label=None)

    _, weights = _training_rows(conn)
    assert len(weights) == 1
    assert weights[0] < 0
    assert list_ranked_clips(conn)[0]["feedback"] == ["skip"]


def test_taste_preferences_do_not_create_synthetic_training_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    index_demo_clip(conn, demo_seed("clip.mp4", ["sniper_kill"]))
    save_taste_preferences(conn, {"sniper_kill": -2})

    features, weights = _training_rows(conn)

    assert features == []
    assert weights == []


def test_training_status_counts_only_current_teacher_schema():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))
    conn.execute("delete from teacher_labels where clip_id = ?", (clip_id,))
    conn.execute(
        """
        insert into teacher_labels(
            clip_id, provider, schema_version, label_json, confidence, created_at
        ) values (?, 'test', ?, '{}', 0.5, '2026-07-10T00:00:00Z')
        """,
        (clip_id, f"{TEACHER_FALLBACK_SCHEMA_VERSION}:model"),
    )

    status = training_status(conn)

    assert status["teacher_labeled"] == 0
    assert status["teacher_pending"] == 1


def test_teacher_event_attribution_is_persisted_for_audit():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))

    record_teacher_events(
        conn,
        clip_id=clip_id,
        provider="test",
        model="model",
        status="weapon_conflict",
        event_data={"locator_timestamps": [4.0], "events": [{"event_index": 0}]},
    )

    row = conn.execute("select status, event_json from teacher_events").fetchone()
    assert row["status"] == "weapon_conflict"
    assert '"event_index": 0' in row["event_json"]


def test_list_ranked_clips_exposes_highlight_description_from_live_events():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))
    record_teacher_labels(
        conn,
        clip_id=clip_id,
        provider="test",
        model="test-model",
        labels=normalize_teacher_payload(
            {"labels": {"sniper_kill": "yes"}, "confidence": 0.9, "evidence": ["raw"]}
        ),
    )
    record_teacher_events(
        conn,
        clip_id=clip_id,
        provider="test",
        model="model",
        status="ok",
        event_data={
            "decision_schema_version": "highlight-audit-v1",
            "highlight_description": "0:14 — Hunting Rifle headshot knocks an active opponent.",
            "primary_event": {"highlight_type": "sniper_kill"},
            "events": [],
        },
    )

    ranked = list_ranked_clips(conn)

    assert ranked[0]["highlight_description"] == (
        "0:14 — Hunting Rifle headshot knocks an active opponent."
    )
    assert ranked[0]["teacher_evidence"] == ["raw"]


def test_list_ranked_clips_leaves_legacy_events_without_invented_description():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    clip_id = index_demo_clip(conn, demo_seed("clip.mp4"))
    record_teacher_events(
        conn,
        clip_id=clip_id,
        provider="test",
        model="model",
        status="ok",
        event_data={"events": [], "specialist_evidence": ["legacy only"]},
    )

    ranked = list_ranked_clips(conn)

    assert ranked[0]["highlight_description"] is None
