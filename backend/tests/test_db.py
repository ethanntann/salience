import sqlite3
from pathlib import Path

from salience_api.db import connect, init_db


def test_init_db_creates_core_tables(tmp_path: Path):
    db_path = tmp_path / "salience.db"

    with connect(db_path) as conn:
        init_db(conn)
        table_names = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }

    assert {
        "clips",
        "clip_features",
        "clip_scores",
        "feedback_events",
        "teacher_labels",
        "teacher_events",
        "teacher_event_candidates",
        "teacher_label_reviews",
        "user_ranker_metadata",
    }.issubset(table_names)


def test_connect_enables_foreign_keys():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    assert conn.execute("pragma foreign_keys").fetchone()[0] == 0
