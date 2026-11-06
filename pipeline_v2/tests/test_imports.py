"""Clips and montages the player brings in count as liked, and only while they help."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from pipeline_v2 import facts, ranking
from pipeline_v2 import taste as taste_mod
from pipeline_v2.capture import imports


def test_windows_never_cross_an_edit_and_skip_flashes():
    # Shots: 0-12 (two windows of 6), 12-13.5 (a flash), 13.5-17.5 (one), 17.5-20 (too short).
    found = imports.windows(20.0, [12.0, 13.5, 17.5])

    assert [(w.start, w.end) for w in found] == [(0.0, 6.0), (6.0, 12.0), (13.5, 17.5)]


def test_an_uncut_clip_is_windows_of_the_recorders_length():
    found = imports.windows(32.0, [])

    assert len(found) == 4 and all(w.end - w.start == imports.WINDOW_SECONDS for w in found)


def test_sidecar_describes_an_import_like_a_recorded_window(tmp_path):
    video = tmp_path / "000-coverage.mkv"
    video.write_bytes(b"")
    window = imports.Window(13.5, 17.5)
    video.with_suffix(".capture.json").write_text(
        __import__("json").dumps(imports.sidecar(tmp_path / "montage.mp4", video, window)),
        encoding="utf-8")

    got = facts.gather(video, 4.0)

    assert got.event_source == "capture_window"  # the window prompt, the player's corpus
    assert (got.analysis_start_sec, got.analysis_end_sec) == (0.0, 4.0)
    assert got.player_name is None


def test_cut_makes_windows_once(tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "IMPORT_DIR", tmp_path / "Imports")
    source = tmp_path / "montage.mp4"
    source.write_bytes(b"x")
    encoded = []

    def runner(cmd, **kw):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="20.0\n", returncode=0)
        if "-f" in cmd and "null" in cmd:
            return SimpleNamespace(stderr="n:1 pts:1 pts_time:12.0\n", returncode=0)
        Path(cmd[-1]).write_bytes(b"v")
        encoded.append(cmd[-1])
        return SimpleNamespace(returncode=0)

    first = imports.cut(source, runner)
    again = imports.cut(source, runner)

    # 0-12 is two windows, 12-20 one.
    assert [p.name for p in first] == ["000-coverage.mkv", "001-coverage.mkv", "002-coverage.mkv"]
    assert again == first and len(encoded) == 3
    assert all(p.with_suffix(".capture.json").is_file() for p in first)


def _ratings(n_sessions: int = 4, per: int = 12, seed: int = 0):
    """Own ratings whose keeps follow the first number of the description."""
    rng = random.Random(seed)
    own, sessions = [], {}
    for s in range(n_sessions):
        for i in range(per):
            keep = i % 3 == 0
            sha = f"s{s}c{i}"
            own.append(taste_mod.Example(
                clip_sha=sha, engagement=sha, label=int(keep), weight=1.0, fields=[],
                embedding=[1.0 if keep else -1.0, *[rng.gauss(0, 0.5) for _ in range(4)]]))
            sessions[sha] = f"session{s}"
    return own, sessions


def _liked(sign: float, n: int = 20, seed: int = 1):
    rng = random.Random(seed)
    return [taste_mod.Example(
        clip_sha=f"i{i}", engagement=f"i{i}", label=1, weight=ranking.IMPORT_WEIGHT, fields=[],
        embedding=[sign, *[rng.gauss(0, 0.5) for _ in range(4)]]) for i in range(n)]


def test_liked_windows_like_the_keeps_are_used():
    own, sessions = _ratings()

    use, tried = ranking.imports_help(own, _liked(1.0), sessions, "embedding")

    assert use and tried["with"] >= tried["without"] - ranking.MIN_SPEECH_GAIN


def test_liked_windows_unlike_the_keeps_are_dropped():
    """Imports that pull against the player's own ratings make it worse: not used."""
    own, sessions = _ratings()

    use, tried = ranking.imports_help(own, _liked(-3.0, n=60), sessions, "embedding")

    assert not use and tried["with"] < tried["without"]


def test_too_few_own_ratings_to_judge_uses_the_imports():
    own, sessions = _ratings(n_sessions=2, per=6)

    assert ranking.imports_help(own, _liked(-3.0), sessions, "embedding") == (True, {})


def _fake_runner(cmd, **kw):
    if cmd[0] == "ffprobe":
        return SimpleNamespace(stdout="16.0\n", returncode=0)
    if "-f" in cmd and "null" in cmd:
        return SimpleNamespace(stderr="", returncode=0)
    Path(cmd[-1]).write_bytes(cmd[-1].encode())
    return SimpleNamespace(returncode=0)


def test_job_cuts_then_describes_only_when_asked(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setattr(imports, "IMPORT_DIR", tmp_path / "Imports")
    source = tmp_path / "saved.mp4"
    source.write_bytes(b"x")
    db = tmp_path / "db.sqlite"
    sqlite3.connect(db).executescript(
        "create table clips (clip_sha text primary key);"
        "create table imported (clip_sha text primary key, source text, added_at real);")
    described = []

    def describe(folder, max_calls):
        from pipeline_v2 import cache

        described.append(folder)
        conn = sqlite3.connect(db)
        with conn:
            for video in folder.glob("*.mkv"):
                conn.execute("insert into clips values (?)", (cache.clip_sha(video),))

    job = imports.Job(runner=_fake_runner, describe=describe, connect=lambda: sqlite3.connect(db))

    assert job.cut([source], wait=True)
    assert job.view()["stage"] == "ready" and job.view()["windows"] == 2
    assert described == []  # nothing paid for until the player says yes

    assert job.describe(max_calls=10, wait=True)
    assert job.view()["stage"] == "done" and job.view()["liked"] == 2
    assert imports.liked_count(lambda: sqlite3.connect(db)) == 2


def test_job_waits_for_the_game_to_close(tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "IMPORT_DIR", tmp_path / "Imports")
    source = tmp_path / "saved.mp4"
    source.write_bytes(b"x")
    games = iter([True, True, False])
    seen = []
    job = imports.Job(playing=lambda: next(games), runner=_fake_runner, wait_seconds=0)
    original = job._set

    def spy(**changes):
        seen.append(changes.get("stage"))
        original(**changes)

    job._set = spy
    job.cut([source], wait=True)

    assert "cutting-waiting" in seen and job.view()["stage"] == "ready"


def test_describe_before_cutting_does_nothing():
    assert imports.Job().describe(max_calls=10, wait=True) is False
