from pathlib import Path
import json

from salience_api.student.reports import load_student_reports


def test_load_student_reports_reads_speed_and_agreement(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    root = tmp_path
    (root / "speed-bench.json").write_text(
        json.dumps(
            {
                "model": "student-v6-test",
                "summary": {
                    "clips": 2,
                    "median_sec": 10.0,
                    "mean_sec": 11.0,
                    "p90_sec": 12.0,
                    "max_sec": 13.0,
                    "phases": {
                        "total_sec": {
                            "mean": 11.0,
                            "median": 10.0,
                            "p90": 12.0,
                            "max": 13.0,
                        }
                    },
                },
                "clips": [
                    {
                        "clip_id": 7,
                        "filename": "demo.mp4",
                        "duration_sec": 12.0,
                        "total_sec": 9.5,
                        "locator_timestamps": [4.2],
                        "locator_events": 1,
                        "attribution_status": "attributed",
                        "yes_labels": ["elimination_or_knock", "shotgun_kill"],
                        "labels": {
                            "elimination_or_knock": "yes",
                            "shotgun_kill": "yes",
                            "sniper_kill": "no",
                        },
                        "events": [
                            {
                                "event_index": 0,
                                "status": "attributed",
                                "event_kind": "elimination",
                                "finish_timestamp": 4.2,
                                "resolved_weapon": "shotgun",
                                "target_was_active": True,
                                "target_was_downed": False,
                            }
                        ],
                        "video_url": "/clips/7/video",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "agreement-v6.json").write_text(
        json.dumps(
            {
                "model": "student-v6-test",
                "labels": {
                    "elimination_or_knock": {
                        "agree": 8,
                        "teacher_yes": 9,
                        "student_yes": 8,
                        "uncertain_rate": 0.1,
                        "true_positive": 8,
                        "false_positive": 0,
                        "false_negative": 1,
                        "precision": 1.0,
                        "recall": 0.89,
                        "clips": 10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = load_student_reports(artifacts)

    assert report["available"] is True
    assert report["model"] == "student-v6-test"
    assert report["speed"]["median_sec"] == 10.0
    assert report["agreement_labels"][0]["label_key"] == "elimination_or_knock"
    assert report["agreement_labels"][0]["agree"] == 0.8
    assert report["example_clips"][0]["clip_id"] == 7
    assert "elimination_or_knock" in report["example_clips"][0]["yes_labels"]


def test_load_student_reports_missing_files(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = load_student_reports(artifacts)
    assert report["available"] is False
    assert report["message"]
