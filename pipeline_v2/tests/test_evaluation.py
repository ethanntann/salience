import json
import sqlite3

import pytest

from pipeline_v2 import embed, evaluation, feedback
from pipeline_v2.jsonio import read_json
from pipeline_v2.evaluation_review import validate_labels
from pipeline_v2.evaluation_metrics import event_metrics


def test_historical_labels_survive_missing_original_media(tmp_path):
    db = tmp_path / "legacy.db"
    c = sqlite3.connect(db)
    c.executescript(
        "create table clips(id integer,path text); create table feedback_events(id integer,clip_id integer,action text);"
    )
    missing = tmp_path / "old-user" / "clip.mp4"
    c.execute("insert into clips values(1,?)", (str(missing),))
    c.execute("insert into feedback_events values(1,1,'keep')")
    c.commit()
    c.close()
    assert feedback.load(db) == []
    imported = feedback.load(db, require_media=False)
    assert len(imported) == 1 and imported[0].label == 1
    assert imported[0].path == missing


def test_unseen_words_do_not_enter_training_vocabulary():
    encoder = embed.TfidfEncoder()
    encoder.fit_transform(
        ["rifle elimination", "shotgun headshot", "sniper kill", "quiet travel"]
    )
    before = dict(encoder._vectorizer.vocabulary_)
    result = encoder.transform(["unseenxyz hoverboard"])
    assert encoder._vectorizer.vocabulary_ == before
    assert all(abs(v) < 1e-12 for v in result[0])


def test_frozen_sweep_is_deterministic_contiguous_and_separate(tmp_path):
    run = tmp_path / "run-001"
    run.mkdir()
    for i in range(12):
        (run / f"part-{i:05d}.capture.json").write_text(
            json.dumps(
                {
                    "video_file": f"part-{i:05d}.mkv",
                    "duration_sec": 30,
                    "analysis_start_sec": 11,
                    "analysis_end_sec": 15,
                    "segment_start_sec": i * 30,
                    "segment_end_sec": (i + 1) * 30,
                }
            )
        )
    windows = evaluation.select_windows(tmp_path)
    assert windows == evaluation.select_windows(tmp_path)
    sweep = [w for w in windows if w["cohort"] == "continuous"]
    assert len(sweep) == 48
    assert sweep[0]["timeline_start"] == 0 and sweep[-1]["timeline_end"] == 180
    assert all(
        a["timeline_end"] == b["timeline_start"] for a, b in zip(sweep, sweep[1:])
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"labels": {"unknown": {"keep": "keep"}}},
        {"labels": {"one": {"keep": "automatic"}}},
        {"events": [{"start": 3, "end": 2}]},
        {"events": [{"start": 0, "end": float("nan")}]},
        {"events": [{"start": 0, "end": 181}]},
    ],
)
def test_invalid_review_labels_are_rejected(payload):
    with pytest.raises(ValueError):
        validate_labels(payload, {"one"}, 180)


def test_review_labels_keep_user_provenance():
    saved = validate_labels(
        {"labels": {"one": {"keep": "skip"}}, "events": [{"start": 3, "end": 5}]},
        {"one"},
        180,
    )
    assert saved["annotator"] == "user"
    assert saved["labels"]["one"]["keep"] == "skip"


def test_event_recall_does_not_count_duplicate_windows_twice():
    m = event_metrics([(0, 4), (4, 8), (12, 16)], [{"start": 3, "end": 5}])
    assert m["events_found"] == 1
    assert m["recall"] == 1
    assert m["precision"] == 0.5


def test_no_positive_reference_events_is_not_perfect_recall():
    assert event_metrics([(0, 4)], [])["recall"] is None


def test_browser_cancelling_video_request_is_not_a_server_error(tmp_path):
    from types import SimpleNamespace
    from pipeline_v2.http_media import serve_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")

    def reset(_):
        raise ConnectionResetError("browser changed its seek position")

    handler = SimpleNamespace(
        headers={},
        send_response=lambda _: None,
        send_header=lambda *args: None,
        end_headers=lambda: None,
        wfile=SimpleNamespace(write=reset),
    )
    serve_video(handler, path)


def test_local_review_post_persists_only_valid_user_judgments(tmp_path):
    from functools import partial
    from http.server import ThreadingHTTPServer
    import threading
    import urllib.request
    from pipeline_v2.evaluation_review import Handler

    (tmp_path / "review-data.json").write_text(
        json.dumps({"duration": 180, "windows": [{"id": "one"}]})
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(tmp_path))
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(
            origin + "/labels",
            data=json.dumps(
                {"labels": {"one": {"keep": "skip"}}, "timeline_review_complete": True}
            ).encode(),
            headers={"Origin": origin, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204
        saved = json.loads((tmp_path / "user-labels.json").read_text())
        assert saved["timeline_review_complete"] is True
        assert saved["labels"]["one"]["keep"] == "skip"
        assert not (tmp_path / "v2.sqlite").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_curated_review_retains_labels_and_does_not_invent_session_recall(tmp_path):
    from pipeline_v2.evaluation_review import build
    from pipeline_v2.evaluation_metrics import report

    windows = [
        {
            "id": "action",
            "cohort": "action",
            "start": 4,
            "end": 8,
            "playback_start": 0,
            "playback_end": 12,
            "preview": "clip.mp4",
            "images": [],
        },
        {
            "id": "background",
            "cohort": "uniform",
            "start": 0,
            "end": 4,
            "preview": "clip.mp4",
            "images": [],
        },
    ]
    plan = {
        "study_kind": "curated_action",
        "session_id": "test",
        "windows": windows,
        "reused_results": ["background"],
        "review_note": "Manually selected",
    }
    labels = {
        "annotator": "user",
        "labels": {
            "background": {
                "keep": "skip",
                "accuracy": "accurate",
                "source_study": "original",
            }
        },
        "events": [],
        "timeline_review_complete": True,
    }
    (tmp_path / "results").mkdir()
    records = {
        "plan.json": plan,
        "prepared.json": {"windows": windows},
        "ranking.json": {
            "scores": [
                {"id": "action", "score": 0.2},
                {"id": "background", "score": 0.8},
            ]
        },
        "user-labels.json": labels,
    }
    for name, record in records.items():
        (tmp_path / name).write_text(json.dumps(record))
    for w in windows:
        (tmp_path / "results" / f"{w['id']}.json").write_text(
            json.dumps(
                {
                    "description": {"summary": "Test"},
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cost_usd": 0.01,
                }
            )
        )
    before = (tmp_path / "user-labels.json").read_bytes()
    build(tmp_path)
    data = read_json(tmp_path / "review-data.json")
    assert data["has_timeline"] is False and data["duration"] == 0
    assert data["windows"][0]["playback_start"] == 0
    assert data["windows"][0]["start"] == 4
    assert (tmp_path / "user-labels.json").read_bytes() == before
    metrics = report(tmp_path)
    assert metrics["user_highlight_metrics"] is None
    assert metrics["new_estimated_cost_usd"] == 0.01
    assert metrics["new_responses"] == metrics["reused_responses"] == 1
    assert metrics["human_review_by_cohort"]["uniform"]["preferences"] == {"skip": 1}
    assert metrics["human_review_by_cohort"]["action"]["reviewed"] == 0
