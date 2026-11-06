"""Fortnite's frame times while Salience works, from Intel's PresentMon.

Schedule D (speech and laughter during matches, `live.MATCH_SPEECH_KEY`) is
what makes highlights a minute after a session instead of ten, and the last
time speech ran mid-match it cost the player frames. Frame times decide it,
not CPU percentages: PresentMon reads the game's presents from Windows' own
event tracing, nothing loaded into the game. It records only while the
experiment is on, beside each session (`frames.csv`), and `report` compares
the matches with D on against those with it off (`experiment.jsonl`).

PresentMon needs administrator rights, or the account in Windows' Performance
Log Users group (2026-10-25: added once, then a sign-out).
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from pipeline_v2.capture.windows import GAME_EXE

OUTPUT = "frames.csv"
SESSION_NAME = "SalienceFrames"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def exe() -> Path:
    from pipeline_v2.capture import app

    return app.APP_DIR / "tools" / "PresentMon.exe"


def start(session: Path, runner=subprocess.Popen):
    """PresentMon recording the game's frames beside `session`, or None: while
    schedule D is tried, or parts are left off to find what costs frames."""
    import os

    from pipeline_v2.capture import live, record

    program = exe()
    testing = live.match_speech_mode() != "off" or record.SKIP_KEY in os.environ
    if not testing or not program.is_file():
        return None
    try:
        return runner([str(program), "--process_name", GAME_EXE, "--output_file", str(session / OUTPUT),
                       "--date_time", "--no_console_stats", "--session_name", SESSION_NAME,
                       "--stop_existing_session", "--terminate_on_proc_exit"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except OSError:
        return None


def stop(process) -> None:
    """End the trace: its own way first (the session is the kernel's), then the process."""
    if process is None:
        return
    try:
        subprocess.run([str(exe()), "--terminate_existing_session", "--session_name", SESSION_NAME],
                       capture_output=True, timeout=15, creationflags=_NO_WINDOW)
        process.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _stamp(text: str) -> float:
    """PresentMon's date and time ("2026-10-27 8:55:52.860068600": the hour
    unpadded, nine decimals) as seconds, read as local time."""
    day, _, clock = text.strip().partition(" ")
    whole, _, fraction = clock.partition(".")
    hour, minute, second = (int(part) for part in whole.split(":"))
    year, month, date = (int(part) for part in day.split("-"))
    return datetime(year, month, date, hour, minute, second, int((fraction + "000000")[:6])).timestamp()


def _session_start(session: Path) -> float | None:
    try:
        return datetime.strptime(session.name, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


def _frames(path: Path, start: float | None = None) -> list[tuple[float, float]]:
    """(wall-clock seconds, frame time in ms) for each of the game's frames.

    PresentMon's clock is whole hours off this one (2.6.0, 2026-10-27: 8:55
    for 12:55 Eastern, neither local nor UTC), so with the session's `start`
    the frames are moved by the whole half-hours that put the first at it:
    recording starts with the game.
    """
    found = []
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                # PresentMon 2.6's names first (2026-10-26), then older ones.
                ms = row.get("MsBetweenPresents") or row.get("FrameTime") or row.get("msBetweenPresents")
                when = row.get("CPUStartDateTime") or row.get("TimeInDateTime") or row.get("CPUStartTime")
                try:
                    found.append((_stamp(when), float(ms)))
                except (AttributeError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    if found and start is not None:
        shift = round((start - found[0][0]) / 1800.0) * 1800.0
        found = [(stamp + shift, ms) for stamp, ms in found]
    return found


def _stats(frame_ms: list[float]) -> dict:
    if len(frame_ms) < 30:
        return {"frames": len(frame_ms)}
    ordered = sorted(frame_ms)
    median = statistics.median(ordered)
    worst = ordered[int(len(ordered) * 0.99):]  # the slowest 1%
    return {
        "frames": len(frame_ms),
        "fps": round(1000.0 * len(frame_ms) / sum(frame_ms), 1),
        "low_1pct_fps": round(1000.0 / statistics.mean(worst), 1),
        "p99_ms": round(ordered[int(len(ordered) * 0.99) - 1], 2),
        "stutters_per_min": round(sum(ms > 2 * median for ms in frame_ms) / (sum(frame_ms) / 60000.0), 1),
    }


def report(session: Path) -> dict:
    """Frame-time statistics in matches with speech on, in matches with it off,
    and inside the moments speech was actually running."""
    frames = _frames(session / OUTPUT, _session_start(session))
    try:
        log = [json.loads(line) for line in (session / "experiment.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    except (OSError, ValueError):
        log = []
    if not frames or not log:
        return {"why": "no frames recorded" if not frames else "no experiment log"}
    # Each look: when it began, when its speech ended, and what it found. A
    # frame belongs to the last look begun before it.
    log = [e for e in log if "start" in e]
    log.sort(key=lambda e: e["start"])
    groups: dict[str, list[float]] = {"match, speech on": [], "match, speech off": [],
                                      "while speech ran": [], "lobby": []}
    at = -1
    for stamp, ms in frames:
        while at + 1 < len(log) and log[at + 1]["start"] <= stamp:
            at += 1
        if at < 0:
            continue
        entry = log[at]
        if entry["idle"] or entry["match"] is None:
            groups["lobby"].append(ms)
            continue
        groups["match, speech on" if entry["speech_live"] else "match, speech off"].append(ms)
        if entry["speech_live"] and stamp <= entry["wall"]:
            groups["while speech ran"].append(ms)
    return {name: _stats(values) for name, values in groups.items()}
