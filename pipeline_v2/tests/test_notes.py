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


def test_a_window_is_told_what_was_said_and_what_came_just_before(tmp_path):
    folder = tmp_path / "windows"
    folder.mkdir()
    first, second = _window(folder, 41, 328.0), _window(folder, 42, 336.0)
    conn = store.connect()
    _describe(conn, first, "a", "The recording player builds a ramp toward an opponent.")
    store.put_transcript(conn, "b", "why are you hiding in the building bro", "mic")

    told = notes.window_notes(conn, second, "b", V)

    assert told == (
        'Said on the player\'s own microphone during these frames: '
        '"why are you hiding in the building bro"',
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
