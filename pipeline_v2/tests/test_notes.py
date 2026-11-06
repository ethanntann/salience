"""What the v3.9 prompt is told beyond the frames."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_v2 import notes, store

V = "v3.9-said-window@deepseek-v4p1-flash"


def _window(folder: Path, index: int, start: float) -> Path:
    clip = folder / f"{index:03d}-coverage.mkv"
    clip.write_bytes(b"x")
    clip.with_suffix(".capture.json").write_text(
        json.dumps({"segment_start_sec": start, "segment_end_sec": start + 8}), encoding="utf-8")
    return clip


def _describe(conn, clip: Path, sha: str, summary: str) -> None:
    store.upsert_clip(conn, clip_sha=sha, path=clip, engagement=None, duration_sec=8.0,
                      event_sec=4.0, event_source="capture_window", facts={}, summary=summary,
                      payload={}, prompt_version=V)


def test_a_window_is_told_what_came_just_before_and_never_what_was_said(tmp_path):
    """What was said made every window wait on speech, and taste was better without it (2026-11-02)."""
    folder = tmp_path / "windows"
    folder.mkdir()
    first, second = _window(folder, 41, 328.0), _window(folder, 42, 336.0)
    conn = store.connect()
    _describe(conn, first, "a", "The recording player builds a ramp toward an opponent.")
    store.put_transcript(conn, "b", "why are you hiding in the building bro", "mic")

    told = notes.window_notes(conn, second, "b", V)

    assert told == (
        "In the 8 seconds just before these frames: The recording player builds a ramp "
        "toward an opponent.",
    )


def test_nothing_is_claimed_across_a_skipped_lobby(tmp_path):
    """Window 42 follows 41 by name, but 41 ended long before: a lobby between."""
    folder = tmp_path / "windows"
    folder.mkdir()
    first, second = _window(folder, 41, 100.0), _window(folder, 42, 400.0)
    conn = store.connect()
    _describe(conn, first, "a", "The recording player loots a chest.")

    assert notes.window_notes(conn, second, "b", V) == ()


def test_the_first_window_and_a_silent_one_get_no_notes(tmp_path):
    folder = tmp_path / "windows"
    folder.mkdir()
    conn = store.connect()

    assert notes.window_notes(conn, _window(folder, 0, 0.0), "z", V) == ()



def test_older_summaries_are_read_without_the_players_words():
    assert notes.without_said(
        "The recording player knocks X with a shotgun; the player says 'What the f.'") == \
        "The recording player knocks X with a shotgun."
    assert notes.without_said('The player loots, and says "next store"; no combat occurs.') == "The player loots."
    assert notes.without_said("The recording player eliminates X with a sniper.") == \
        "The recording player eliminates X with a sniper."


def test_a_window_is_told_of_the_players_banners_in_the_30_seconds_before(tmp_path, monkeypatch):
    """An ELIMINATION! for an opponent knocked earlier and bleeding out read as a new kill (2026-11-04)."""
    from pipeline_v2 import banner

    session = tmp_path / "20261104-120000"
    (session / "windows").mkdir(parents=True)
    (session / "segments.csv").write_text("part-00000.mkv,0.0,60.0\n", encoding="utf-8")
    clip = session / "windows" / "005-coverage.mkv"
    clip.with_suffix(".capture.json").write_text('{"segment_start_sec": 40.0, "segment_end_sec": 48.0}',
                                                 encoding="utf-8")
    asked = []
    monkeypatch.setattr(banner, "session_found",
                        lambda s, a, b: asked.append((a, b)) or [(4.0, "knock"), (18.0, "kill")])

    note = notes.recent_banners(clip)

    assert asked == [(10.0, 40.0)]
    assert "KNOCKED 26s before, ELIMINATION 12s before" in note
    monkeypatch.setattr(banner, "session_found", lambda s, a, b: [])
    assert notes.recent_banners(clip) is None


def test_a_window_is_told_whether_the_players_banner_appears_in_it(tmp_path, monkeypatch):
    from pipeline_v2 import banner

    session = tmp_path / "20261104-120000"
    (session / "windows").mkdir(parents=True)
    (session / "segments.csv").write_text("part-00000.mkv,0.0,60.0\n", encoding="utf-8")
    clip = session / "windows" / "005-coverage.mkv"
    clip.with_suffix(".capture.json").write_text('{"segment_start_sec": 40.0, "segment_end_sec": 48.0}',
                                                 encoding="utf-8")
    monkeypatch.setattr(banner, "session_found", lambda s, a, b: [])
    assert "no knock or elimination here is theirs" in notes.window_banners(clip)
    monkeypatch.setattr(banner, "session_found", lambda s, a, b: [(2.5, "knock")])
    assert notes.window_banners(clip) == "The player's own banners in these frames: KNOCKED at 2.5s."
