"""Watching a window whose copy is gone: re-cut once, from the raw footage."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_v2.capture import rating


def _window(tmp_path: Path, released: bool) -> Path:
    windows = tmp_path / "20261022-184935" / "windows"
    windows.mkdir(parents=True)
    clip = windows / "079-coverage.mkv"
    record = {"segment_start_sec": 632.0, "segment_end_sec": 640.0}
    if released:
        record["released_sha"] = "abc"
    else:
        clip.write_bytes(b"copy")
    clip.with_suffix(".capture.json").write_text(json.dumps(record), encoding="utf-8")
    return clip


def test_a_copy_on_disk_is_played_as_it_is(tmp_path):
    clip = _window(tmp_path, released=False)
    assert rating.playable(clip, recut=lambda *a: None) == clip


def test_a_released_window_is_recut_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(rating, "RECUT_DIR", tmp_path / "recuts")
    clip = _window(tmp_path, released=True)
    cuts = []

    def recut(source, destination):
        cuts.append(source)
        destination.write_bytes(b"recut")
        return destination

    first = rating.playable(clip, recut=recut)
    again = rating.playable(clip, recut=recut)

    assert first == again and first.read_bytes() == b"recut"
    assert first.name == "20261022-184935-079-coverage.mkv"  # sessions never collide
    assert cuts == [clip]


def test_a_missing_window_that_was_never_released_is_not_found(tmp_path):
    clip = _window(tmp_path, released=False)
    clip.unlink()
    assert rating.playable(clip, recut=lambda *a: None) is None
