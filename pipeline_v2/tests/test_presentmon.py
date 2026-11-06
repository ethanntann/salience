"""Frame times by what Salience was doing: matches with speech on and off."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from pipeline_v2.capture import live, presentmon


def write_frames(path, start, rows):
    lines = ["Application,ProcessID,CPUStartDateTime,MsBetweenPresents"]  # PresentMon 2.6
    for offset, ms in rows:
        when = (start + timedelta(seconds=offset)).isoformat(sep=" ", timespec="microseconds") + "500"  # 9 decimals, as it writes
        lines.append(f"FortniteClient-Win64-Shipping.exe,1,{when},{ms}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_frames_are_grouped_by_match_and_speech(tmp_path):
    start = datetime(2026, 10, 25, 20, 0, 0)
    t0 = start.timestamp()
    # Looks every 30s: match 0 with speech (its first 10s), then match 1 without.
    log = [{"start": t0 + 30 * i, "wall": t0 + 30 * i + (10.0 if i < 3 else 0.5), "idle": False,
            "match": 0 if i < 3 else 1, "speech_live": i < 3, "speech_s": 10.0 if i < 3 else 0.5}
           for i in range(6)]
    (tmp_path / live.EXPERIMENT_LOG).write_text("".join(json.dumps(e) + "\n" for e in log), encoding="utf-8")
    rows = []
    for second in range(150):  # 150s at 100 fps; slower while speech runs
        slow = second < 90 and (second % 30) < 10
        rows += [(second + k / 100, 14.0 if slow else 10.0) for k in range(100)]
    write_frames(tmp_path / presentmon.OUTPUT, start, rows)

    got = presentmon.report(tmp_path)

    assert got["match, speech off"]["fps"] == 100.0
    assert got["while speech ran"]["fps"] < 75
    assert got["match, speech on"]["fps"] < got["match, speech off"]["fps"]


def test_presentmons_clock_is_moved_to_the_sessions(tmp_path):
    # 2.6.0 wrote 8:55 for a session begun 12:55 (2026-10-27), hour unpadded.
    session = tmp_path / "20261027-125550"
    session.mkdir()
    path = session / presentmon.OUTPUT
    path.write_text("﻿Application,ProcessID,CPUStartDateTime,MsBetweenPresents\n"
                    "FortniteClient-Win64-Shipping.exe,1,2026-10-27 8:55:52.860068600,5.0\n"
                    "FortniteClient-Win64-Shipping.exe,1,2026-10-27 8:55:52.865000000,6.0\n", encoding="utf-8")
    got = presentmon._frames(path, presentmon._session_start(session))
    assert got[0][0] == datetime(2026, 10, 27, 12, 55, 52, 860068).timestamp() and got[1][1] == 6.0


def test_no_presentmon_without_the_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv(live.MATCH_SPEECH_KEY, "off")
    monkeypatch.delenv("SALIENCE_SKIP", raising=False)
    started = []
    assert presentmon.start(tmp_path, runner=lambda *a, **k: started.append(a)) is None and not started
    assert presentmon.report(tmp_path) == {"why": "no frames recorded"}


def test_parts_left_off_are_skipped_and_their_frames_recorded(tmp_path, monkeypatch):
    from pipeline_v2.capture import record

    monkeypatch.setenv(live.MATCH_SPEECH_KEY, "off")
    monkeypatch.setenv(record.SKIP_KEY, "live, Mouse,nonsense")
    assert record.skipped() == {"live", "mouse"}
    monkeypatch.setattr(presentmon, "exe", lambda: tmp_path / "PresentMon.exe")
    (tmp_path / "PresentMon.exe").write_bytes(b"")
    started = []
    presentmon.start(tmp_path, runner=lambda *a, **k: started.append(a) or "process")
    assert started  # frames recorded while parts are left off, even with speech off
    monkeypatch.setenv(record.SKIP_KEY, "")
    assert record.skipped() == set()
