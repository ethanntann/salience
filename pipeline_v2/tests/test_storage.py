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


def test_windows_are_offered_newest_session_first_in_the_order_played(conn, tmp_path, monkeypatch):
    """Newest-described first ran each session backwards, a finish before its knock (2026-11-04)."""
    for session, name, sha in (("20261101-120000", "002-coverage.mkv", "old"),
                               ("20261103-120000", "005-coverage.mkv", "late"),
                               ("20261103-120000", "001-coverage.mkv", "early")):
        folder = tmp_path / session / "windows"
        folder.mkdir(parents=True, exist_ok=True)
        add_clip(conn, folder / name, sha=sha)
    assert review._next_clip(conn)["sha"] == "early"
    monkeypatch.setattr(review, "_FIRST", lambda: ["late"])  # unsure: marked, not moved ahead
    assert review._next_clip(conn)["sha"] == "early" and not review._next_clip(conn)["unsure"]
    conn.execute("insert into preferences values ('early', 0, 1)")
    pending = review._next_clip(conn)
    assert pending["sha"] == "late" and pending["unsure"]


def test_only_windows_that_could_be_highlights_are_offered(conn, tmp_path, monkeypatch):
    """1 of 1,343 windows that were not kills was ever kept (2026-11-04)."""
    add_clip(conn, tmp_path / "a.mp4", sha="a")
    add_clip(conn, tmp_path / "b.mp4", sha="b")
    monkeypatch.setattr(review, "_ONLY_SHAS", lambda: {"b"})
    pending = review._next_clip(conn)
    assert pending["sha"] == "b" and pending["remaining"] == 1
    monkeypatch.setattr(review, "_ONLY_SHAS", lambda: set())
    assert review._next_clip(conn)["done"]


def test_a_rating_taken_back_is_gone_and_offered_again(conn, tmp_path):
    add_clip(conn, tmp_path / "a.mp4", sha="a")
    conn.execute("insert into preferences values ('a', 1, 1)")
    conn.execute("insert into rating_notes (clip_sha, love, reason) values ('a', 1, null)")
    assert review._next_clip(conn)["done"]
    assert review.unrate(conn, "a") == 1
    assert not conn.execute("select 1 from preferences").fetchone()
    assert not conn.execute("select 1 from rating_notes").fetchone()
    assert review.clip_again(conn, "a")["sha"] == "a" and review._next_clip(conn)["sha"] == "a"
    assert review.unrate(conn, "a") is None  # nothing left to take back


def test_the_rating_page_says_who_laughed(conn, tmp_path, monkeypatch):
    add_clip(conn, tmp_path / "a.mp4", sha="a")
    monkeypatch.setattr(review, "_LAUGHED", lambda path: ["game", "mic", "game"])
    assert review._next_clip(conn)["laughed"] == ["game", "mic"]
    monkeypatch.setattr(review, "_LAUGHED", lambda path: [])
    assert review._next_clip(conn)["laughed"] == []
