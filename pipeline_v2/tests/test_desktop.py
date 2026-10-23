"""The desktop app's server: names, paths it will serve, and the player's edits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_v2.capture import highlights
from pipeline_v2.capture.desktop import server


@pytest.mark.parametrize("payload, summary, name", [
    ({"outcome": "elimination", "kill_weapon_class": "shotgun"}, "", "Shotgun elimination"),
    ({"outcome": "elimination", "kill_weapon_class": "sniper", "kill_headshot": "yes"}, "", "Sniper headshot"),
    ({"outcome": "elimination", "kill_weapon_class": "sniper"}, "no-scopes a hunting rifle", "No-scope sniper"),
    ({"outcome": "knock", "kill_weapon_class": "smg"}, "", "SMG knock"),
    ({"outcome": "no_event"}, "the Victory Royale banner appears", "Victory Royale"),
    ({"outcome": "elimination"}, "", "Elimination"),
    ({}, "", "Moment"),
])
def test_a_highlight_is_named_from_what_the_describer_said(payload, summary, name):
    assert server.title(payload, summary) == name


def test_only_a_recording_folder_by_its_own_name_is_served(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "recordings_root", lambda: tmp_path)
    session = tmp_path / "20261022-184935"
    session.mkdir()
    (session / "segments.csv").write_text("part-00000.mkv,0,30\n", encoding="utf-8")

    assert server.session_dir("20261022-184935") == session
    for bad in ("..", "../20261022-184935", "20261022-184935/..", "C:\Windows", ""):
        assert server.session_dir(bad) is None
    assert server.session_dir("20261022-184936") is None  # no such recording


def _session(tmp_path: Path, manifest: dict) -> Path:
    session = tmp_path / "20261022-184935"
    out = session / "highlights"
    (out / highlights.REEL_DIR).mkdir(parents=True)
    (session / "segments.csv").write_text("part-00000.mkv,0.0,1800.0\n", encoding="utf-8")
    (session / "part-00000.mkv").write_bytes(b"raw")
    for h in manifest["highlights"]:
        (out / (h.get("file") or f"{h['rank']:02d}-{h['clip']}")).write_bytes(b"clip")
        if h.get("reel"):
            (out / highlights.REEL_DIR / h["reel"]).write_bytes(b"reel")
    (out / "highlights.json").write_text(json.dumps(manifest), encoding="utf-8")
    return session


def test_old_manifests_without_file_names_still_list_their_highlights(tmp_path, isolated_data, monkeypatch):
    """Sessions before 2026-10-19 named only the window; 20 highlights went missing."""
    monkeypatch.setattr(server, "_lengths", lambda out, files: {})
    session = _session(tmp_path, {"highlights": [
        {"rank": 1, "clip": "314-coverage.mkv", "summary": "a kill", "outcome": "elimination"}]})
    from pipeline_v2 import store

    conn = store.connect()
    view = server.session_view(session, conn)
    conn.close()

    assert [h["file"] for h in view["highlights"]] == ["01-314-coverage.mkv"]
    assert view["minutes"] == 30


def test_deleting_a_highlight_takes_its_reel_and_its_entry(tmp_path):
    session = _session(tmp_path, {"highlights": [
        {"rank": 1, "clip": "a.mkv", "file": "01-a.mkv", "reel": "01-a.mp4"},
        {"rank": 2, "clip": "b.mkv", "file": "02-b.mkv", "reel": ""}]})
    out = session / "highlights"

    assert server.delete_highlight(session, "01-a.mkv")
    assert not server.delete_highlight(session, "01-a.mkv")  # once

    assert not (out / "01-a.mkv").exists() and not (out / highlights.REEL_DIR / "01-a.mp4").exists()
    manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    assert [h["file"] for h in manifest["highlights"]] == ["02-b.mkv"]


def test_custom_quality_records_at_the_apps_fps_and_bitrate(tmp_path, monkeypatch):
    from pipeline_v2.capture import app

    for key in (app.QUALITY_KEY, app.FPS_KEY, app.BITRATE_KEY):
        monkeypatch.delenv(key, raising=False)
    app.set_custom_quality(30, 12, tmp_path / "settings.env")
    args = app._recording_args()
    assert (args.fps, args.bitrate) == (30, "12M")
    with pytest.raises(ValueError):
        app.set_custom_quality(144, 12, tmp_path / "settings.env")
    monkeypatch.setenv(app.BITRATE_KEY, "999")  # a hand-edited settings file
    assert app.custom_quality() == (60, "25M")
