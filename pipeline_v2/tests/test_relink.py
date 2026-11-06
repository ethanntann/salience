"""Relinking moved footage must confirm identity, not trust a filename.

The store is keyed by content hash. Matching on name alone could attach a clip's
descriptions to a different video, and nothing would surface that until someone
watched one.
"""

from __future__ import annotations

from pathlib import Path

from pipeline_v2 import cache, cli, store


def _clip(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return cache.clip_sha(path)


def _add(conn, sha: str, path: Path) -> None:
    store.upsert_clip(
        conn,
        clip_sha=sha,
        path=path,
        engagement=None,
        duration_sec=20.0,
        event_sec=15.0,
        event_source="fraction",
        facts={},
        summary="a summary",
        payload={},
        prompt_version="v3.3",
    )


def test_relink_matches_moved_files_and_confirms_their_hash(tmp_path, conn):
    new_root = tmp_path / "moved"
    sha = _clip(new_root / "clip-a.mp4", b"real contents")
    _add(conn, sha, Path("C:/gone/clip-a.mp4"))

    plan = cli.relink_plan(conn, new_root)

    assert plan["missing"] == 1
    assert plan["relinked"] == [(sha, new_root / "clip-a.mp4")]
    assert not plan["mismatched"]


def test_a_same_named_but_different_file_is_refused(tmp_path, conn):
    new_root = tmp_path / "moved"
    _clip(new_root / "clip-a.mp4", b"a completely different video")
    _add(conn, "sha-of-the-original", Path("C:/gone/clip-a.mp4"))

    plan = cli.relink_plan(conn, new_root)

    assert plan["relinked"] == []
    assert plan["mismatched"] == ["clip-a.mp4"]


def test_trusting_names_skips_verification(tmp_path, conn):
    new_root = tmp_path / "moved"
    _clip(new_root / "clip-a.mp4", b"a completely different video")
    _add(conn, "sha-of-the-original", Path("C:/gone/clip-a.mp4"))

    plan = cli.relink_plan(conn, new_root, verify=False)

    assert len(plan["relinked"]) == 1
    assert plan["mismatched"] == []


def test_duplicate_names_are_reported_never_guessed(tmp_path, conn):
    new_root = tmp_path / "moved"
    sha = _clip(new_root / "one" / "clip-a.mp4", b"real contents")
    _clip(new_root / "two" / "clip-a.mp4", b"real contents")
    _add(conn, sha, Path("C:/gone/clip-a.mp4"))

    plan = cli.relink_plan(conn, new_root)

    assert plan["relinked"] == []
    assert plan["ambiguous"] == ["clip-a.mp4"]


def test_clips_that_already_resolve_are_left_alone(tmp_path, conn):
    here = tmp_path / "here"
    sha = _clip(here / "clip-a.mp4", b"real contents")
    _add(conn, sha, here / "clip-a.mp4")

    plan = cli.relink_plan(conn, tmp_path / "elsewhere")

    assert plan["present"] == 1
    assert plan["missing"] == 0
    assert plan["relinked"] == []


def test_missing_footage_is_reported_as_absent(tmp_path, conn):
    _add(conn, "sha-x", Path("C:/gone/clip-x.mp4"))
    (tmp_path / "empty").mkdir()

    plan = cli.relink_plan(conn, tmp_path / "empty")

    assert plan["absent"] == ["clip-x.mp4"]
    assert plan["relinked"] == []
