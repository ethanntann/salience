import json
import sqlite3
from pathlib import Path

from salience_api.db import init_db
from salience_api.training.labels import import_labeled_jsonl, iter_labeled_clip_rows


def test_iter_labeled_clip_rows_ignores_features(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    labels = tmp_path / "clips.jsonl"
    labels.write_text(
        json.dumps(
            {
                "path": str(clip),
                "label": 1,
                "features": {"motion_mean": 999999, "do_not_use": True},
                "dest": "keepers",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(iter_labeled_clip_rows(labels))

    assert rows == [rows[0]]
    assert rows[0].path == clip
    assert rows[0].label == 1


def test_import_labeled_jsonl_records_supervised_feedback(tmp_path: Path):
    keep = tmp_path / "keep.mp4"
    skip = tmp_path / "skip.mp4"
    labels = tmp_path / "clips.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"path": str(keep), "label": 1, "features": {"ignored": 1}}),
                json.dumps(
                    {"path": str(skip), "label": -1, "features": {"ignored": -1}}
                ),
            ]
        ),
        encoding="utf-8",
    )
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    result = import_labeled_jsonl(conn, labels)

    assert result.imported == 2
    assert result.keepers == 1
    assert result.skips == 1
    events = conn.execute(
        "select action, weight from feedback_events order by action"
    ).fetchall()
    assert [(row["action"], row["weight"]) for row in events] == [
        ("supervised_keep", 1.0),
        ("supervised_skip", -1.0),
    ]


def test_import_labeled_jsonl_is_idempotent_for_same_file(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    labels = tmp_path / "clips.jsonl"
    labels.write_text(
        json.dumps({"path": str(clip), "label": 1}) + "\n", encoding="utf-8"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    equivalent_path = nested / ".." / labels.name
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    import_labeled_jsonl(conn, labels)
    import_labeled_jsonl(conn, equivalent_path)

    count = conn.execute("select count(*) from feedback_events").fetchone()[0]
    assert count == 1


def test_import_labeled_jsonl_isolates_same_basename_in_different_directories(
    tmp_path: Path,
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_labels = first_dir / "clips.jsonl"
    second_labels = second_dir / "clips.jsonl"
    first_labels.write_text(
        json.dumps({"path": str(tmp_path / "first.mp4"), "label": 1}) + "\n",
        encoding="utf-8",
    )
    second_labels.write_text(
        json.dumps({"path": str(tmp_path / "second.mp4"), "label": -1}) + "\n",
        encoding="utf-8",
    )
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    import_labeled_jsonl(conn, first_labels)
    import_labeled_jsonl(conn, second_labels)

    events = conn.execute(
        "select action, label from feedback_events order by action"
    ).fetchall()
    assert [(row["action"], row["label"]) for row in events] == [
        ("supervised_keep", f"supervised:{first_labels.resolve()}"),
        ("supervised_skip", f"supervised:{second_labels.resolve()}"),
    ]
