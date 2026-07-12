import sqlite3

from salience_api.db import init_db
from salience_api.feedback.service import feedback_weight, record_feedback


def test_feedback_weight_maps_actions():
    assert feedback_weight("favorite") == 1.0
    assert feedback_weight("keep") == 0.6
    assert feedback_weight("skip") == -0.2
    assert feedback_weight("boring") == -0.8
    assert feedback_weight("delete") == -1.0


def test_record_feedback_inserts_event():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute(
        """
        insert into clips(path, filename, indexed_at)
        values('/clips/a.mp4', 'a.mp4', '2026-07-09T00:00:00Z')
        """
    )
    clip_id = conn.execute("select id from clips").fetchone()["id"]

    record_feedback(conn, clip_id=clip_id, action="favorite", label=None)

    row = conn.execute("select action, weight from feedback_events").fetchone()
    assert row["action"] == "favorite"
    assert row["weight"] == 1.0
