import json
import sqlite3
from pathlib import Path

from salience_api.student.export_dataset import export_live_teacher_dataset
from salience_api.clips.indexer import clip_ids_for_reenrichment


def _seed_minimal_live(conn: sqlite3.Connection, tmp_path: Path) -> None:
    conn.executescript(
        """
        create table clips (
          id integer primary key, path text, filename text,
          duration_sec real, width integer, height integer, fps real
        );
        create table live_teacher_assignments (
          clip_id integer primary key, label_row_id integer, event_row_id integer
        );
        create table teacher_labels (
          id integer primary key, clip_id integer, label_json text, confidence real
        );
        create table teacher_events (
          id integer primary key, clip_id integer, event_json text, status text
        );
        """
    )
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    conn.execute(
        "insert into clips values (1, ?, 'a.mp4', 20.0, 1920, 1080, 60.0)",
        (str(video),),
    )
    labels = {"sniper_kill": "yes", "elimination_or_knock": "yes", "combat_visible": "yes"}
    events = {
        "locator_timestamps": [5.0, 12.0],
        "events": [
            {
                "event_index": 0,
                "status": "attributed",
                "event_kind": "knock",
                "finish_timestamp": 5.0,
                "resolved_weapon": "sniper_or_hunting",
                "target_was_active": True,
                "target_was_downed": False,
                "damage_aim_state": "hipfire",
                "pov_shot_visible": True,
                "new_damage_visible": True,
                "finish_onset_supported": True,
                "visible_defeat_supported": False,
                "damage_hit_count": 2,
            }
        ],
    }
    conn.execute(
        "insert into teacher_labels values (10, 1, ?, 0.9)",
        (json.dumps(labels),),
    )
    conn.execute(
        "insert into teacher_events values (20, 1, ?, 'promoted')",
        (json.dumps(events),),
    )
    conn.execute("insert into live_teacher_assignments values (1, 10, 20)")
    conn.commit()


def test_export_writes_manifest_and_split(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    _seed_minimal_live(conn, tmp_path)
    out = tmp_path / "dataset"
    manifest = export_live_teacher_dataset(conn, output_dir=out, eval_fraction=0.0, seed=7)
    assert manifest.version.startswith("student-dataset-")
    assert (out / "manifest.json").exists()
    assert len(manifest.train_clips) == 1
    assert manifest.train_clips[0].locator_timestamps == [5.0, 12.0]
    assert manifest.train_clips[0].events[0]["resolved_weapon"] == "sniper_or_hunting"
    event = manifest.train_clips[0].events[0]
    assert event["target_state"] == "active"
    assert event["selected_weapon_before_finish"] == "sniper_or_hunting"
    assert event["aim_state_at_shot"] == "hipfire"
    assert event["finish_ui_newly_appeared"] is True
    assert event["target_defeat_visible"] is False
    assert event["damaging_shot_count"] == 2
    assert manifest.target_coverage["finish_ui_newly_appeared"]["true"] == 1
    assert manifest.train_clips[0].teacher_confidence == 1.0
    assert manifest.train_clips[0].label_confidences["sniper_kill"] == 1.0


def test_export_split_puts_most_clips_in_train(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        create table clips (
          id integer primary key, path text, filename text,
          duration_sec real, width integer, height integer, fps real
        );
        create table live_teacher_assignments (
          clip_id integer primary key, label_row_id integer, event_row_id integer
        );
        create table teacher_labels (
          id integer primary key, clip_id integer, label_json text, confidence real
        );
        create table teacher_events (
          id integer primary key, clip_id integer, event_json text, status text
        );
        """
    )
    for clip_id in range(1, 101):
        conn.execute(
            "insert into clips values (?, ?, ?, 10.0, 1920, 1080, 60.0)",
            (clip_id, str(tmp_path / f"{clip_id}.mp4"), f"{clip_id}.mp4"),
        )
        conn.execute(
            "insert into teacher_labels values (?, ?, '{}', 0.9)",
            (clip_id, clip_id),
        )
        conn.execute(
            "insert into teacher_events values (?, ?, '{}', 'promoted')",
            (clip_id * 10, clip_id),
        )
        conn.execute(
            "insert into live_teacher_assignments values (?, ?, ?)",
            (clip_id, clip_id, clip_id * 10),
        )
    conn.commit()
    manifest = export_live_teacher_dataset(
        conn, output_dir=tmp_path / "dataset", eval_fraction=0.15, seed=7
    )
    assert len(manifest.train_clips) + len(manifest.eval_clips) == 100
    assert 5 <= len(manifest.eval_clips) <= 30
    assert len(manifest.train_clips) >= 70


def test_reenrichment_selector_finds_uncertain_assigned_clips(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    _seed_minimal_live(conn, tmp_path)
    conn.execute(
        "update teacher_labels set label_json = ? where id = 10",
        (json.dumps({"sniper_kill": "uncertain", "confidence": 0.95}),),
    )
    conn.commit()

    assert clip_ids_for_reenrichment(conn, limit=10) == [1]
