"""Export my stats: counts a tester can send back, and nothing else."""

from __future__ import annotations

import json
import sqlite3

from pipeline_v2.capture import stats


def _session(root, name="20261106-023800"):
    session = root / name
    (session / "highlights").mkdir(parents=True)
    (session / "segments.csv").write_text("part-00000.mkv,0,30\npart-00001.mkv,30,600\n", encoding="utf-8")
    (session / "highlights" / "highlights.json").write_text(json.dumps({
        "gameplay": {"matches": 2}, "captured_fps": 60.0,
        "highlights": [{"clip": "010-coverage.mkv", "summary": "a secret description", "said": "words"},
                       {"clip": "011-coverage.mkv", "summary": "x", "said": ""}]}), encoding="utf-8")
    rows = ["Application,MsBetweenPresents"] + ["FortniteClient-Win64-Shipping.exe,7"] * 1000
    rows += ["FortniteClient-Win64-Shipping.exe,500", "FortniteClient-Win64-Shipping.exe,1200", "Discord.exe,900"]
    (session / "frames.csv").write_text("\n".join(rows), encoding="utf-8")
    return session


def _conn(session):
    conn = sqlite3.connect(":memory:")
    conn.executescript("create table clips (clip_sha text, path text); create table preferences (clip_sha text, label int);")
    conn.execute("insert into clips values ('a', ?)", (str(session / "windows" / "010-coverage.mkv"),))
    conn.execute("insert into clips values ('b', ?)", (str(session / "windows" / "011-coverage.mkv"),))
    conn.executemany("insert into preferences values (?, ?)", [("a", 1), ("b", 0)])
    return conn


def test_counts_per_session_and_nothing_private(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(stats, "_match_spans", lambda s: [(0.0, 600.0)])
    log = tmp_path / "salience.log"
    log.write_text(f"recording -> {session}\n[02:51:24] Making highlights: x\n[02:53:17] Highlights ready: y\n",
                   encoding="utf-8")

    data = stats.build(tmp_path, _conn(session), log, version="t")

    row = data["sessions"][0]
    assert (row["minutes"], row["matches"], row["highlights"]) == (10.0, 2, 2)
    assert (row["highlights_kept"], row["highlights_thrown_out"]) == (1, 1)
    assert row["seconds_to_highlights"] == 113
    assert row["game_stalls_in_matches"] == 2  # Fortnite's two slow frames; Discord's is not the game's
    assert data["ratings"] == {"kept": 1, "skipped": 1}
    text = json.dumps(data)
    for private in ("secret description", "words", str(tmp_path), "coverage.mkv"):
        assert private not in text


def test_stalls_outside_matches_do_not_count(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(stats, "_match_spans", lambda s: [(100.0, 200.0)])  # the slow frames come at ~7 s

    assert stats._frames_stalls(session) == 0
    assert json.loads((session / stats.CACHE).read_text())["stalls"] == 0  # counted once, kept
