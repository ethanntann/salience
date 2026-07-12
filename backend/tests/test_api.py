from contextlib import contextmanager
from fastapi.testclient import TestClient
import json
from pathlib import Path
import time

import pytest
import salience_api.app as app_module
from salience_api.app import create_app
from salience_api.clips.indexer import (
    index_known_clip_path,
    record_teacher_candidate,
)
from salience_api.clips.keyframes import Keyframe
from salience_api.config import Settings
from salience_api.db import connect, init_db
from salience_api.features.fireworks_teacher import WeaponAttribution
from salience_api.features.teacher_labels import normalize_teacher_payload


def wait_for_background_run(client: TestClient, path: str) -> dict:
    for _ in range(100):
        status = client.get(path).json()
        if not status["running"]:
            return status
        time.sleep(0.01)
    pytest.fail(f"Background run did not finish: {path}")


def test_clips_endpoint_returns_seeded_demo_clips(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)

    response = client.get("/clips")

    assert response.status_code == 200
    clips = response.json()
    assert len(clips) == 553
    assert clips[0]["source"] == "supervised"
    assert clips[0]["path"].startswith("snapshot://")
    assert clips[0]["video_url"] is None
    assert clips[0]["teacher_provider"] != "local"
    assert clips[0]["teacher_labels"]


def test_feedback_rejects_unknown_clip(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=False))
    client = TestClient(app)

    response = client.post(
        "/feedback",
        json={"clip_id": 999, "action": "favorite", "label": None},
    )

    assert response.status_code == 404


def test_feedback_returns_updated_clips(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip_id = client.get("/clips").json()[0]["id"]

    response = client.post(
        "/feedback",
        json={"clip_id": clip_id, "action": "favorite", "label": None},
    )

    assert response.status_code == 200
    assert response.json()[0]["feedback"]


def test_ai_status_reports_fireworks_configuration(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, fireworks_api_key="test-key"))
    client = TestClient(app)

    response = client.get("/ai/status")

    assert response.status_code == 200
    assert response.json()["fireworks_configured"] is True
    assert (
        response.json()["fireworks_model"] == "accounts/fireworks/models/qwen3p7-plus"
    )
    assert response.json()["vlm_provider"] == "fireworks"


def test_ai_status_reports_amd_developer_cloud_configuration(tmp_path):
    app = create_app(
        Settings(
            app_data_dir=tmp_path,
            vlm_provider="amd-developer-cloud",
            amd_developer_cloud_api_key="test-key",
            amd_developer_cloud_base_url="https://amd.example/v1",
        )
    )
    client = TestClient(app)

    response = client.get("/ai/status")

    assert response.status_code == 200
    assert response.json()["amd_developer_cloud_configured"] is True
    assert response.json()["vlm_provider"] == "amd-developer-cloud"


def test_enrich_requires_fireworks_key(tmp_path):
    app = create_app(
        Settings(app_data_dir=tmp_path, demo_mode=True, fireworks_api_key=None)
    )
    client = TestClient(app)

    response = client.post("/ai/enrich", json={"clip_id": None, "limit": 10})

    assert response.status_code == 400
    assert "FIREWORKS_API_KEY" in response.text


def test_fresh_clip_uses_full_timeline_then_event_specialist(monkeypatch, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    clip_path = clips_dir / "clip.mp4"
    clip_path.write_bytes(b"fake")
    coarse_dir = tmp_path / "coarse"
    event_dir = tmp_path / "event"
    coarse_dir.mkdir()
    event_dir.mkdir()
    coarse_paths = [coarse_dir / "1.jpg", coarse_dir / "2.jpg"]
    event_path = event_dir / "1.jpg"
    for path in [*coarse_paths, event_path]:
        path.write_bytes(b"frame")

    calls: dict[str, object] = {}

    class FakeTeacher:
        provider = "test"
        model = "test-model"

        def __init__(self, settings):
            pass

        def label_clip(self, clip):
            calls["general_frame_count"] = len(clip.image_paths)
            return normalize_teacher_payload(
                {
                    "labels": {
                        "combat_visible": "yes",
                        "enemy_visible": "yes",
                        "elimination_or_knock": "yes",
                        "sniper_kill": "yes",
                    },
                    "confidence": 0.8,
                    "evidence": ["whole clip combat"],
                }
            )

        def locate_event_timestamps(self, clip):
            calls["locator_frame_count"] = len(clip.image_paths)
            return [5.0]

        def label_weapon_event(self, clip):
            calls["event_indices"] = clip.image_event_indices
            return WeaponAttribution(
                labels={
                    "sniper_kill": "no",
                    "shotgun_kill": "no",
                    "shotgun_one_pump": "no",
                    "pistol_kill": "yes",
                    "automatic_kill": "no",
                    "other_weapon_kill": "no",
                    "spray_kill": "no",
                    "no_scope": "no",
                },
                confidence=0.55,
                evidence=["pistol finish"],
                status="attributed",
                events=[
                    {
                        "event_index": 0,
                        "status": "attributed",
                        "event_kind": "elimination",
                        "resolved_weapon": "pistol",
                        "target_was_active": True,
                        "target_was_downed": False,
                    }
                ],
            )

    monkeypatch.setattr(app_module, "FireworksTeacherClient", FakeTeacher)
    monkeypatch.setattr(
        app_module,
        "extract_timeline_keyframes",
        lambda *args, **kwargs: [
            Keyframe(path=coarse_paths[0], timestamp_sec=1.0),
            Keyframe(path=coarse_paths[1], timestamp_sec=9.0),
        ],
    )
    monkeypatch.setattr(
        app_module,
        "extract_event_keyframes",
        lambda *args, **kwargs: [
            Keyframe(path=event_path, timestamp_sec=5.0, event_index=0)
        ],
    )
    monkeypatch.setattr(app_module, "cleanup_keyframes", lambda frames: None)

    settings = Settings(app_data_dir=tmp_path / "data", fireworks_api_key="test-key")
    client = TestClient(create_app(settings))
    scan = client.post("/folders/scan", json={"path": str(clips_dir), "enrich": False})
    clip_id = scan.json()["clips"][0]["id"]
    response = client.post("/ai/enrich", json={"clip_id": clip_id, "limit": 1})

    assert response.status_code == 200
    assert response.json()["enriched"] == 1
    assert calls == {
        "general_frame_count": 2,
        "locator_frame_count": 2,
        "event_indices": [0],
    }
    enriched = response.json()["clips"][0]
    assert enriched["teacher_labels"]["combat_visible"] == "yes"
    assert enriched["teacher_labels"]["sniper_kill"] == "no"
    assert enriched["teacher_confidence"] == 0.8
    assert enriched["teacher_evidence"] == ["whole clip combat"]
    with connect(settings.resolved_database_path()) as conn:
        event = conn.execute("select status, event_json from teacher_events").fetchone()
        assert event["status"] == "attributed"
        assert (
            json.loads(event["event_json"])["events"][0]["resolved_weapon"] == "pistol"
        )

    validation = client.post(
        "/ai/event-validation/background",
        json={"clip_ids": [clip_id, 999999]},
    )
    assert validation.status_code == 200
    status = wait_for_background_run(client, "/ai/event-validation/background")
    assert status["requested"] == 2
    assert status["enriched"] == 1
    assert status["failed"] == 1
    assert "clip 999999" in status["last_error"]
    assert status["enriched"] + status["failed"] == status["requested"]
    with connect(settings.resolved_database_path()) as conn:
        candidate = conn.execute(
            "select label_json from teacher_event_candidates"
        ).fetchone()
        assert json.loads(candidate["label_json"])["pistol_kill"] == "yes"

    candidate_response = client.get(
        f"/eval/teacher-clips?mode=candidate&clip_id={clip_id}"
    )
    assert candidate_response.status_code == 200
    candidate_clip = candidate_response.json()["clips"][0]
    assert candidate_response.json()["mode"] == "candidate"
    assert candidate_clip["candidate_labels"]["pistol_kill"] == "yes"
    assert candidate_clip["candidate_status"] == "attributed"
    assert candidate_clip["teacher_labels"]["pistol_kill"] == "yes"

    batch_response = client.post(
        "/eval/batches", json={"batch_key": "frozen-test-batch"}
    )
    assert batch_response.status_code == 200
    batch_id = batch_response.json()["batch"]["id"]
    frozen_response = client.get(f"/eval/batches/{batch_id}/clips")
    assert frozen_response.status_code == 200
    frozen_clip = frozen_response.json()["clips"][0]
    assert frozen_clip["prediction_snapshot_id"] is not None
    assert frozen_clip["candidate_labels"]["pistol_kill"] == "yes"
    review_response = client.post(
        "/eval/snapshot-review",
        json={
            "prediction_snapshot_id": frozen_clip["prediction_snapshot_id"],
            "label_key": "pistol_kill",
            "expected_value": "yes",
        },
    )
    pistol_metric = next(
        item
        for item in review_response.json()["labels"]
        if item["label_key"] == "pistol_kill"
    )
    assert pistol_metric["reviewed"] == 1


def test_training_status_reports_teacher_and_feedback_counts(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip_id = client.get("/clips").json()[0]["id"]
    client.post("/feedback", json={"clip_id": clip_id, "action": "keep", "label": None})

    response = client.get("/training/status")

    assert response.status_code == 200
    status = response.json()
    assert status["clips"] >= 3
    assert status["teacher_labeled"] >= 3
    assert status["feedback_count"] == 1
    assert status["positive_count"] == 1


def test_eval_teacher_review_updates_summary(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip = client.get("/clips").json()[0]
    label_key = next(
        key for key, value in clip["teacher_labels"].items() if value == "yes"
    )

    list_response = client.get("/eval/teacher-clips")
    review_response = client.post(
        "/eval/teacher-review",
        json={"clip_id": clip["id"], "label_key": label_key, "expected_value": "no"},
    )

    assert list_response.status_code == 200
    assert list_response.json()["clips"][0]["label_reviews"] == {}
    assert review_response.status_code == 200
    metric = next(
        item
        for item in review_response.json()["labels"]
        if item["label_key"] == label_key
    )
    assert metric["reviewed"] == 1
    assert metric["false_positive"] == 1
    assert metric["precision"] == 0


def test_eval_summary_excludes_latest_uncertain_review_from_metrics(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip = client.get("/clips").json()[0]
    label_key = next(
        key for key, value in clip["teacher_labels"].items() if value == "yes"
    )
    review = {"clip_id": clip["id"], "label_key": label_key}
    client.post(
        "/eval/teacher-review",
        json={**review, "expected_value": "no"},
    )

    response = client.post(
        "/eval/teacher-review",
        json={**review, "expected_value": "uncertain"},
    )

    assert response.status_code == 200
    metric = next(
        item for item in response.json()["labels"] if item["label_key"] == label_key
    )
    assert metric["reviewed"] == 0
    assert metric["teacher_yes"] == 0
    assert metric["expected_yes"] == 0
    assert metric["true_positive"] == 0
    assert metric["false_positive"] == 0
    assert metric["false_negative"] == 0
    assert metric["true_negative"] == 0
    assert metric["accuracy"] is None


def test_eval_teacher_clips_can_find_exact_clip_number(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip = client.get("/clips").json()[1]

    response = client.get(
        f"/eval/teacher-clips?label_key=sniper_kill&clip_id={clip['id']}"
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["clips"]] == [clip["id"]]


def test_candidate_eval_includes_clip_without_live_teacher(tmp_path):
    settings = Settings(app_data_dir=tmp_path, demo_mode=False)
    app = create_app(settings)
    client = TestClient(app)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, Path("C:/clips/candidate-only.mp4"))
        record_teacher_candidate(
            conn,
            clip_id=clip_id,
            provider="test",
            model="test-model",
            status="attributed",
            labels=normalize_teacher_payload(
                {
                    "labels": {"sniper_kill": "yes"},
                    "confidence": 0.8,
                    "evidence": ["candidate only"],
                }
            ),
            event_data={},
        )

    candidate_response = client.get(
        f"/eval/teacher-clips?mode=candidate&clip_id={clip_id}"
    )
    live_response = client.get(f"/eval/teacher-clips?mode=live&clip_id={clip_id}")

    assert candidate_response.status_code == 200
    candidate = candidate_response.json()["clips"][0]
    assert candidate["id"] == clip_id
    assert candidate["teacher_provider"] is None
    assert candidate["candidate_labels"]["sniper_kill"] == "yes"
    assert live_response.status_code == 404


def test_eval_teacher_clips_rejects_unknown_mode(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)

    response = client.get("/eval/teacher-clips?mode=ambiguous")

    assert response.status_code == 400


def test_eval_teacher_review_rejects_unknown_label(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip = client.get("/clips").json()[0]

    response = client.post(
        "/eval/teacher-review",
        json={
            "clip_id": clip["id"],
            "label_key": "not_a_label",
            "expected_value": "yes",
        },
    )

    assert response.status_code == 400


def test_background_enrich_status_defaults_to_idle(tmp_path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=False))
    client = TestClient(app)

    response = client.get("/ai/enrich/background")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_background_enrich_records_outer_failure(monkeypatch, tmp_path):
    app = create_app(
        Settings(
            app_data_dir=tmp_path,
            demo_mode=False,
            fireworks_api_key="test-key",
        )
    )

    @contextmanager
    def failing_connect(path):
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(app_module, "connect", failing_connect)
    client = TestClient(app)

    response = client.post("/ai/enrich/background", json={"limit": 10})
    status = wait_for_background_run(client, "/ai/enrich/background")

    assert response.status_code == 200
    assert status["running"] is False
    assert status["requested"] == 0
    assert status["failed"] == 0
    assert status["enriched"] + status["failed"] == status["requested"]
    assert "database unavailable" in status["last_error"]
    assert status["finished_at"] is not None


def test_background_enrich_counts_false_result_as_failed(monkeypatch, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip.mp4").write_bytes(b"fake video")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        fireworks_api_key="test-key",
    )
    client = TestClient(create_app(settings))
    clip_id = client.post(
        "/folders/scan", json={"path": str(clips_dir), "enrich": False}
    ).json()["clips"][0]["id"]
    monkeypatch.setattr(
        app_module, "clip_teacher_input_row", lambda conn, clip_id: None
    )

    response = client.post("/ai/enrich/background", json={"limit": 1})
    status = wait_for_background_run(client, "/ai/enrich/background")

    assert response.status_code == 200
    assert status["requested"] == 1
    assert status["enriched"] == 0
    assert status["failed"] == 1
    assert status["enriched"] + status["failed"] == status["requested"]
    assert f"clip {clip_id}" in status["last_error"]


def test_background_enrich_late_outer_failure_does_not_overcount(monkeypatch, tmp_path):
    settings = Settings(
        app_data_dir=tmp_path,
        demo_mode=False,
        fireworks_api_key="test-key",
    )
    app = create_app(settings)
    monkeypatch.setattr(
        app_module, "clip_ids_for_enrichment", lambda conn, clip_id, limit: [404]
    )

    def fail_rescore(conn):
        raise RuntimeError("rescore failed")

    monkeypatch.setattr(app_module, "rescore_all_clips", fail_rescore)
    client = TestClient(app)

    response = client.post("/ai/enrich/background", json={"limit": 1})
    status = wait_for_background_run(client, "/ai/enrich/background")

    assert response.status_code == 200
    assert status["requested"] == 1
    assert status["enriched"] == 0
    assert status["failed"] == 1
    assert status["enriched"] + status["failed"] == status["requested"]
    assert "rescore failed" in status["last_error"]


def test_event_validation_records_outer_failure_for_all_requested_ids(
    monkeypatch, tmp_path
):
    app = create_app(
        Settings(
            app_data_dir=tmp_path,
            demo_mode=False,
            fireworks_api_key="test-key",
        )
    )

    @contextmanager
    def failing_connect(path):
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(app_module, "connect", failing_connect)
    client = TestClient(app)

    response = client.post(
        "/ai/event-validation/background",
        json={"clip_ids": [101, 202]},
    )
    status = wait_for_background_run(client, "/ai/event-validation/background")

    assert response.status_code == 200
    assert status["running"] is False
    assert status["requested"] == 2
    assert status["enriched"] == 0
    assert status["failed"] == 2
    assert status["enriched"] + status["failed"] == status["requested"]
    assert "database unavailable" in status["last_error"]
    assert status["finished_at"] is not None


def test_event_validation_late_outer_failure_does_not_overcount(monkeypatch, tmp_path):
    settings = Settings(
        app_data_dir=tmp_path,
        demo_mode=False,
        fireworks_api_key="test-key",
    )
    app = create_app(settings)
    real_connect = app_module.connect

    @contextmanager
    def late_failing_connect(path):
        with real_connect(path) as conn:
            yield conn
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(app_module, "connect", late_failing_connect)
    client = TestClient(app)

    response = client.post(
        "/ai/event-validation/background",
        json={"clip_ids": [404]},
    )
    status = wait_for_background_run(client, "/ai/event-validation/background")

    assert response.status_code == 200
    assert status["requested"] == 1
    assert status["enriched"] == 0
    assert status["failed"] == 1
    assert status["enriched"] + status["failed"] == status["requested"]
    assert "teardown failed" in status["last_error"]


def test_event_validation_counts_missing_clip_as_failed(tmp_path):
    app = create_app(
        Settings(
            app_data_dir=tmp_path,
            demo_mode=False,
            fireworks_api_key="test-key",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/ai/event-validation/background",
        json={"clip_ids": [404]},
    )
    status = wait_for_background_run(client, "/ai/event-validation/background")

    assert response.status_code == 200
    assert status["requested"] == 1
    assert status["enriched"] == 0
    assert status["failed"] == 1
    assert status["enriched"] + status["failed"] == status["requested"]
    assert "clip 404" in status["last_error"]


def test_enrich_false_result_is_returned_as_failure(monkeypatch, tmp_path):
    settings = Settings(
        app_data_dir=tmp_path,
        demo_mode=True,
        fireworks_api_key="test-key",
    )
    monkeypatch.setattr(
        app_module, "clip_teacher_input_row", lambda conn, clip_id: None
    )
    client = TestClient(create_app(settings))

    response = client.post("/ai/enrich", json={"clip_id": 1, "limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["enriched"] == 0
    assert payload["failed"] == 1
    assert payload["errors"] == [
        {"clip_id": 1, "error": "Clip not found during enrichment"}
    ]


def test_enrichment_rollback_preserves_fresh_demo_seed(monkeypatch, tmp_path):
    settings = Settings(
        app_data_dir=tmp_path,
        demo_mode=True,
        fireworks_api_key="test-key",
    )
    real_record_teacher_labels = app_module.record_teacher_labels

    class FakeTeacher:
        provider = "test"
        model = "test-model"

        def __init__(self, settings):
            pass

    def fail_enrichment_label_write(conn, *, clip_id, provider, labels, model):
        if provider == "test":
            raise RuntimeError("enrichment persistence failed")
        real_record_teacher_labels(
            conn,
            clip_id=clip_id,
            provider=provider,
            labels=labels,
            model=model,
        )

    monkeypatch.setattr(app_module, "FireworksTeacherClient", FakeTeacher)
    monkeypatch.setattr(
        app_module, "record_teacher_labels", fail_enrichment_label_write
    )
    client = TestClient(create_app(settings))

    response = client.post("/ai/enrich", json={"clip_id": 1, "limit": 1})

    assert response.status_code == 200
    assert response.json()["failed"] == 1
    assert len(response.json()["clips"]) >= 3
    with connect(settings.resolved_database_path()) as conn:
        snapshot_count = conn.execute(
            "select count(*) from clips where path like 'snapshot://%'"
        ).fetchone()[0]
        assert snapshot_count == 553


def test_scan_folder_indexes_without_teacher_by_default(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip.mp4").write_bytes(b"fake video")
    app = create_app(
        Settings(
            app_data_dir=tmp_path / "data",
            demo_mode=False,
            fireworks_api_key="test-key",
        )
    )
    client = TestClient(app)

    response = client.post("/folders/scan", json={"path": str(clips_dir)})

    assert response.status_code == 200
    clip = response.json()["clips"][0]
    assert clip["filename"] == "clip.mp4"
    assert clip["teacher_provider"] is None


def test_scan_folder_commits_indexes_before_enrichment_and_isolates_failures(
    monkeypatch, tmp_path: Path
):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp4").write_bytes(b"fake video")
    (clips_dir / "b.mp4").write_bytes(b"fake video")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        fireworks_api_key="test-key",
    )
    indexed_connections = []
    indexed_names = {}
    real_index_clip_path = app_module.index_clip_path
    real_record_teacher_labels = app_module.record_teacher_labels

    def track_index_transaction(conn, path):
        clip_id = real_index_clip_path(conn, path)
        indexed_connections.append(conn)
        indexed_names[clip_id] = path.name
        return clip_id

    def fail_first_label_write(conn, *, clip_id, provider, labels, model):
        if indexed_names[clip_id] == "a.mp4":
            raise RuntimeError("enrichment persistence failed")
        real_record_teacher_labels(
            conn,
            clip_id=clip_id,
            provider=provider,
            labels=labels,
            model=model,
        )

    class FakeTeacher:
        provider = "test"
        model = "test-model"

        def __init__(self, settings):
            pass

        def label_clip(self, clip):
            assert indexed_connections[-1].in_transaction is False
            return normalize_teacher_payload(
                {
                    "labels": {"combat_visible": "yes"},
                    "confidence": 0.8,
                    "evidence": ["test"],
                }
            )

    monkeypatch.setattr(app_module, "index_clip_path", track_index_transaction)
    monkeypatch.setattr(app_module, "record_teacher_labels", fail_first_label_write)
    monkeypatch.setattr(app_module, "FireworksTeacherClient", FakeTeacher)
    monkeypatch.setattr(app_module, "extract_timeline_keyframes", lambda *args: [])
    client = TestClient(create_app(settings), raise_server_exceptions=False)

    response = client.post(
        "/folders/scan",
        json={"path": str(clips_dir), "enrich": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"indexed", "total_found", "clips"}
    assert payload["indexed"] == 2
    assert payload["total_found"] == 2
    assert {clip["filename"] for clip in payload["clips"]} == {"a.mp4", "b.mp4"}
    labeled = {clip["filename"]: clip["teacher_provider"] for clip in payload["clips"]}
    assert labeled == {"a.mp4": None, "b.mp4": "test"}
    with connect(settings.resolved_database_path()) as conn:
        assert conn.execute("select count(*) from clips").fetchone()[0] == 2
        assert conn.execute("select count(*) from teacher_labels").fetchone()[0] == 1
        event_filenames = conn.execute(
            """
            select clips.filename
            from teacher_events
            join clips on clips.id = teacher_events.clip_id
            """
        ).fetchall()
        assert [row["filename"] for row in event_filenames] == ["b.mp4"]


def test_export_copies_keeper_clips(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    export_dir = tmp_path / "exports"
    clips_dir.mkdir()
    (clips_dir / "keeper.mp4").write_bytes(b"fake video")
    app = create_app(Settings(app_data_dir=tmp_path / "data", demo_mode=False))
    client = TestClient(app)
    clip = client.post("/folders/scan", json={"path": str(clips_dir)}).json()["clips"][
        0
    ]
    client.post(
        "/feedback", json={"clip_id": clip["id"], "action": "keep", "label": None}
    )

    response = client.post(
        "/clips/export",
        json={"destination": str(export_dir), "mode": "keepers", "limit": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["exported"] == 1
    assert Path(payload["files"][0]).exists()


def test_taste_profile_updates_rankings(tmp_path: Path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)

    response = client.post(
        "/taste/profile",
        json={"preferences": {"shotgun_one_pump": 2, "cleanup_kill": -2}},
    )

    assert response.status_code == 200
    assert response.json()["saved"] == 2
    status = client.get("/training/status").json()
    assert status["feedback_count"] == 0


def test_reset_taste_profile_keeps_feedback_count(tmp_path: Path):
    app = create_app(Settings(app_data_dir=tmp_path, demo_mode=True))
    client = TestClient(app)
    clip = client.get("/clips").json()[0]
    client.post(
        "/feedback", json={"clip_id": clip["id"], "action": "keep", "label": None}
    )
    client.post(
        "/taste/profile",
        json={"preferences": {"sniper_kill": -2, "shotgun_one_pump": 2}},
    )

    response = client.post("/taste/reset")

    assert response.status_code == 200
    assert response.json()["saved"] == 2
    status = client.get("/training/status").json()
    assert status["feedback_count"] == 1


def test_import_liked_folder_marks_clips_positive(tmp_path: Path):
    clips_dir = tmp_path / "favorites"
    clips_dir.mkdir()
    (clips_dir / "favorite.mp4").write_bytes(b"fake video")
    app = create_app(Settings(app_data_dir=tmp_path / "data", demo_mode=False))
    client = TestClient(app)

    response = client.post("/taste/import-liked-folder", json={"path": str(clips_dir)})

    assert response.status_code == 200
    assert response.json()["imported"] == 1
    status = client.get("/training/status").json()
    assert status["positive_count"] == 1


def test_frontend_fallback_rejects_encoded_traversal(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("index", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    client = TestClient(
        create_app(
            Settings(
                app_data_dir=tmp_path / "data",
                demo_mode=False,
                static_dir=static_dir,
            )
        )
    )

    response = client.get("/%2e%2e%2fsecret.txt")

    assert response.status_code == 404


def test_frontend_fallback_rejects_symlink_escape(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("index", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (static_dir / "escape.txt").symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks are not supported: {exc}")
    client = TestClient(
        create_app(
            Settings(
                app_data_dir=tmp_path / "data",
                demo_mode=False,
                static_dir=static_dir,
            )
        )
    )

    response = client.get("/escape.txt")

    assert response.status_code == 404


def test_frontend_index_rejects_symlink_escape_for_root_and_spa_fallback(
    tmp_path: Path,
):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    outside = tmp_path / "outside-index.html"
    outside.write_text("outside", encoding="utf-8")
    try:
        (static_dir / "index.html").symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks are not supported: {exc}")
    client = TestClient(
        create_app(
            Settings(
                app_data_dir=tmp_path / "data",
                demo_mode=False,
                static_dir=static_dir,
            )
        )
    )

    responses = [client.get("/"), client.get("/missing-spa-route")]

    assert [response.status_code for response in responses] == [404, 404]


def test_assets_mount_rejects_resolved_directory_outside_static_root(
    monkeypatch, tmp_path: Path
):
    static_dir = tmp_path / "static"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("index", encoding="utf-8")
    (assets_dir / "secret.txt").write_text("secret", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    real_resolve = Path.resolve

    def resolve_assets_outside(self: Path, strict: bool = False) -> Path:
        if self == assets_dir or assets_dir in self.parents:
            escaped = outside_dir / self.relative_to(assets_dir)
            return real_resolve(escaped, strict=strict)
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_assets_outside)
    client = TestClient(
        create_app(
            Settings(
                app_data_dir=tmp_path / "data",
                demo_mode=False,
                static_dir=static_dir,
            )
        )
    )

    response = client.get("/assets/secret.txt")

    assert response.status_code == 404


def test_video_mapping_allows_existing_direct_path_within_host_root(tmp_path: Path):
    host_videos_dir = tmp_path / "videos"
    host_videos_dir.mkdir()
    clip_path = host_videos_dir / "clip.mp4"
    clip_path.write_bytes(b"clip")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        host_videos_dir=host_videos_dir,
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, clip_path)

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == b"clip"


def test_video_mapping_allows_existing_path_within_trusted_media_root(
    tmp_path: Path,
):
    host_videos_dir = tmp_path / "host-videos"
    trusted_media_dir = tmp_path / "demo-video"
    host_videos_dir.mkdir()
    trusted_media_dir.mkdir()
    clip_path = trusted_media_dir / "clip.mp4"
    clip_path.write_bytes(b"demo clip")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        host_videos_dir=host_videos_dir,
        trusted_media_dirs=[trusted_media_dir],
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, clip_path)

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == b"demo clip"


def test_video_mapping_rejects_existing_direct_path_outside_host_root(
    tmp_path: Path,
):
    host_videos_dir = tmp_path / "videos"
    host_videos_dir.mkdir()
    clip_path = tmp_path / "outside.mp4"
    clip_path.write_bytes(b"outside")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        host_videos_dir=host_videos_dir,
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, clip_path)

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 404


def test_video_mapping_preserves_existing_direct_path_without_host_root(
    tmp_path: Path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        host_videos_dir=None,
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, clip_path)

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == b"clip"


def test_video_mapping_rejects_windows_path_traversal(tmp_path: Path):
    host_videos_dir = tmp_path / "videos"
    host_videos_dir.mkdir()
    (tmp_path / "secret.mp4").write_bytes(b"secret")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        windows_videos_prefix=r"C:\Videos",
        host_videos_dir=host_videos_dir,
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, Path(r"C:\Videos\..\secret.mp4"))

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 404


def test_video_mapping_rejects_symlink_escape(tmp_path: Path):
    host_videos_dir = tmp_path / "videos"
    host_videos_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.mp4").write_bytes(b"secret")
    try:
        (host_videos_dir / "escape").symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks are not supported: {exc}")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=False,
        static_dir=tmp_path / "missing-static",
        windows_videos_prefix=r"C:\Videos",
        host_videos_dir=host_videos_dir,
    )
    app = create_app(settings)
    with connect(settings.resolved_database_path()) as conn:
        init_db(conn)
        clip_id = index_known_clip_path(conn, Path(r"C:\Videos\escape\secret.mp4"))

    response = TestClient(app).get(f"/clips/{clip_id}/video")

    assert response.status_code == 404


def test_demo_reset_preserves_local_feedback(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "local.mp4").write_bytes(b"fake video")
    settings = Settings(
        app_data_dir=tmp_path / "data",
        demo_mode=True,
        static_dir=tmp_path / "missing-static",
    )
    client = TestClient(create_app(settings))
    demo_clip = client.get("/clips").json()[0]
    clips = client.post(
        "/folders/scan", json={"path": str(clips_dir), "enrich": False}
    ).json()["clips"]
    local_clip = next(clip for clip in clips if clip["filename"] == "local.mp4")
    demo_feedback = client.post(
        "/feedback",
        json={"clip_id": demo_clip["id"], "action": "favorite", "label": None},
    )
    local_feedback = client.post(
        "/feedback",
        json={"clip_id": local_clip["id"], "action": "keep", "label": None},
    )
    assert demo_feedback.status_code == 200
    assert local_feedback.status_code == 200
    with connect(settings.resolved_database_path()) as conn:
        before_reset = conn.execute(
            "select clip_id, action from feedback_events order by id"
        ).fetchall()
    assert [(row["clip_id"], row["action"]) for row in before_reset] == [
        (demo_clip["id"], "favorite"),
        (local_clip["id"], "keep"),
    ]

    response = client.post("/demo/reset")

    assert response.status_code == 200
    with connect(settings.resolved_database_path()) as conn:
        rows = conn.execute(
            "select clip_id, action from feedback_events order by id"
        ).fetchall()
    assert [(row["clip_id"], row["action"]) for row in rows] == [
        (local_clip["id"], "keep")
    ]
