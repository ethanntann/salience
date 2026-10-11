from pipeline_v2 import config, review, store
from conftest import add_clip


def test_reindex_preserves_preferences_other_versions_and_embeddings(conn, tmp_path):
    add_clip(conn, tmp_path / "a.mp4", variant="v3.0")
    conn.execute("insert into preferences values ('one', 1, 123)")
    store.put_embedding(conn, "one", "test", [1, 2], "v3.0")
    add_clip(conn, tmp_path / "renamed.mp4")
    assert conn.execute("select label from preferences").fetchone()[0] == 1
    assert len(store.all_clips(conn, "v3.0")) == 1
    assert len(store.all_clips(conn, config.DEFAULT_PROMPT_VERSION)) == 1
    assert store.embeddings(conn, "v3.0")["one"] == [1, 2]


def test_review_counts_current_clips_once(conn, tmp_path):
    add_clip(conn, tmp_path / "a.mp4", variant="v3.0")
    add_clip(conn, tmp_path / "a.mp4")
    add_clip(conn, tmp_path / "old.mp4", sha="old", variant="v3.0")
    next_clip = review._next_clip(conn)
    assert next_clip["remaining"] == 1
    assert next_clip["sha"] == "one"


def test_optional_research_tables_are_not_required(conn):
    assert store.optional_rows(conn, "audio_features") == []
    assert store.optional_rows(conn, "peak_kind") == []
