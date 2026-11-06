"""Highlights from the player's OBS recordings: found, copied out, lined up."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from pipeline_v2.capture import app, obs_recordings as obs


def _local(h, m, s=0):
    return datetime(2026, 11, 6, h, m, s).astimezone(timezone.utc)


def test_obs_folder_is_read_from_the_profile_in_use(tmp_path):
    base = tmp_path / "obs-studio"
    (base / "basic" / "profiles" / "Gaming").mkdir(parents=True)
    (base / "global.ini").write_text("[Basic]\nProfileDir=Gaming\n", encoding="utf-8")
    recordings = tmp_path / "Clips"
    recordings.mkdir()
    (base / "basic" / "profiles" / "Gaming" / "basic.ini").write_text(
        f"[Output]\nMode=Advanced\n[AdvOut]\nRecFilePath={recordings}\n", encoding="utf-8")

    assert obs.obs_folder(tmp_path) == recordings


def test_start_comes_from_obs_file_name(tmp_path):
    path = tmp_path / "2026-11-06 02-40-00.mkv"
    path.write_bytes(b"")

    assert obs.started_at(path) == _local(2, 40)


def _runner(durations: dict[str, float], keyframes=(0.0, 18.0, 20.0, 22.0)):
    def run(cmd, **kw):
        if cmd[0] == "ffprobe" and "format=duration" in cmd:
            return SimpleNamespace(stdout=f"{durations.get(Path(cmd[-1]).name, 0)}\n", returncode=0)
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="\n".join(f"{k}," for k in keyframes), returncode=0)
        listing = Path(cmd[cmd.index("-segment_list") + 1])
        listing.write_text("part-00000.mkv,0.0,30.0\n", encoding="utf-8")
        (listing.parent / "part-00000.mkv").write_bytes(b"v")
        run.copied = cmd
        return SimpleNamespace(returncode=0)
    return run


def test_the_recording_covering_most_of_the_game_is_used(tmp_path):
    for name in ("2026-11-06 01-00-00.mkv", "2026-11-06 02-40-00.mkv", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    runner = _runner({"2026-11-06 01-00-00.mkv": 600.0, "2026-11-06 02-40-00.mkv": 3600.0})

    found = obs.covering(tmp_path, _local(2, 40, 25), _local(2, 52), runner)

    assert found.path.name == "2026-11-06 02-40-00.mkv"
    assert (found.offset, found.end) == (25.0, 720.0)


def test_adopt_copies_the_game_from_a_keyframe_and_lines_up_the_input(tmp_path):
    recordings = tmp_path / "obs"
    recordings.mkdir()
    (recordings / "2026-11-06 02-40-00.mkv").write_bytes(b"x")
    session = tmp_path / "20261106-024025"
    session.mkdir()
    (session / obs.PENDING).write_text(json.dumps({
        "folder": str(recordings), "began_utc": _local(2, 40, 26).isoformat(),
        "ended_utc": _local(2, 45).isoformat(), "input_utc": _local(2, 40, 25).isoformat()}))
    runner = _runner({"2026-11-06 02-40-00.mkv": 3600.0})

    assert obs.pending(session)
    assert obs.adopt(session, runner, wait=0) is None

    copied = runner.copied
    assert copied[copied.index("-ss") + 1] == "20.000"  # the keyframe before 21 s (26 - 5)
    assert "-c" in copied and copied[copied.index("-c") + 1] == "copy"  # never re-encoded
    stats = json.loads((session / "capture_stats.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(stats["first_frame_utc"]) == _local(2, 40, 20)
    assert stats["input_offset_sec"] == -5.0  # input began 5 s into the copied video
    assert not obs.pending(session)


def test_no_covering_recording_says_why(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    (session / obs.PENDING).write_text(json.dumps({
        "folder": str(tmp_path), "began_utc": _local(2, 40).isoformat(),
        "ended_utc": _local(2, 45).isoformat(), "input_utc": _local(2, 40).isoformat()}))

    assert obs.adopt(session, _runner({}), wait=0) == "no OBS recording covers this game"


def test_a_game_obs_recorded_is_made_up_like_any_unfinished_session(tmp_path):
    session = tmp_path / "20261106-024025"
    session.mkdir()
    (session / obs.PENDING).write_text("{}")

    assert app.unfinished(tmp_path) == [session]


def test_obs_setting_round_trips(tmp_path, monkeypatch):
    monkeypatch.delenv(app.RECORDER_KEY, raising=False)
    monkeypatch.delenv(app.OBS_FOLDER_KEY, raising=False)
    settings = tmp_path / "settings.env"

    assert not app.uses_obs()
    app.set_obs(True, str(tmp_path), settings)

    assert app.uses_obs() and app.obs_folder() == tmp_path
    assert app.read_settings(settings)[app.RECORDER_KEY] == "obs"
    app.set_obs(False, path=settings)
    assert not app.uses_obs() and os.environ[app.RECORDER_KEY] == "salience"
