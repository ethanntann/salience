"""Highlights from the player's own OBS recordings, instead of Salience's.

For a player who already records (or streams and records) with OBS: two
recorders on one game is twice the encoding. With this on, Salience records
nothing while Fortnite runs -- only the key-press and mouse logs, which cost
nothing -- and once the game closes it finds the OBS recording that covers it,
copies out just that game's stretch as a session (`adopt`), and makes the
highlights as if it had recorded them.

Never real time: highlights are made after the game anyway, and work beside
a match froze it (2026-11-06). Never required: OBS is the player's choice,
and Salience's own recorder stays the default.

What is lost against Salience's own recording: the mic and friends as their
own tracks. OBS's first audio track is taken as the game's -- by default it
is everything OBS heard, mic included -- so speech and laughter read only it.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PENDING = "obs.json"
SEGMENT_SECONDS = 30
VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".flv", ".ts", ".m4v"}
# OBS's default file name: the local time the recording started.
OBS_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _](\d{2})-(\d{2})-(\d{2})")
# How long a file OBS is still writing is waited for before giving up on it.
FINISH_WAIT = 30 * 60
FINISH_STEP = 10.0


def obs_folder(appdata: Path | None = None) -> Path | None:
    """Where OBS saves recordings, from its own settings: the profile in use."""
    base = (appdata or Path(os.environ.get("APPDATA", Path.home()))) / "obs-studio"
    try:
        glob = configparser.ConfigParser(strict=False, interpolation=None)
        glob.read(base / "global.ini", encoding="utf-8-sig")
        profile = glob.get("Basic", "ProfileDir", fallback="")
        profiles = [base / "basic" / "profiles" / profile] if profile else []
        profiles += sorted((base / "basic" / "profiles").glob("*"))
        for folder in profiles:
            ini = configparser.ConfigParser(strict=False, interpolation=None)
            ini.read(folder / "basic.ini", encoding="utf-8-sig")
            mode = ini.get("Output", "Mode", fallback="Simple")
            path = (ini.get("AdvOut", "RecFilePath", fallback="") if mode == "Advanced"
                    else ini.get("SimpleOutput", "FilePath", fallback=""))
            if path and Path(path).is_dir():
                return Path(path)
    except (OSError, configparser.Error):
        pass
    videos = Path.home() / "Videos"
    return videos if videos.is_dir() else None


def started_at(path: Path) -> datetime:
    """When a recording began: OBS's file name, else when the file was made."""
    match = OBS_NAME.search(path.stem)
    if match:
        local = datetime(*map(int, match.groups()))
        return local.astimezone(timezone.utc)
    return datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)


def duration(path: Path, runner=subprocess.run) -> float | None:
    out = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float((out.stdout or "").strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class Found:
    path: Path
    start: datetime  # the recording's t=0
    offset: float  # where the game begins in it, seconds
    end: float  # where it ends in it


def covering(folder: Path, began: datetime, ended: datetime, runner=subprocess.run) -> Found | None:
    """The recording in `folder` that covers most of the game."""
    best, overlap = None, 0.0
    try:
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    except OSError:
        return None
    for path in candidates:
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < began:
                continue  # finished before the game began
            start = started_at(path)
        except OSError:
            continue
        if start > ended:
            continue
        length = duration(path, runner)
        if not length:
            continue
        a = max(0.0, (began - start).total_seconds())
        b = min(length, (ended - start).total_seconds())
        if b - a > overlap:
            best, overlap = Found(path, start, a, b), b - a
    return best


def keyframe_before(path: Path, at: float, runner=subprocess.run) -> float:
    """The last keyframe at or before `at`: where a stream copy really starts."""
    out = runner(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                  "-read_intervals", f"{max(0.0, at - 20):.3f}%{at + 0.001:.3f}",
                  "-show_entries", "frame=pts_time,best_effort_timestamp_time", "-of", "csv=p=0",
                  str(path)], capture_output=True, text=True)
    times = []
    for line in (out.stdout or "").splitlines():
        for field in line.split(","):
            try:
                times.append(float(field))
                break
            except ValueError:
                continue
    before = [t for t in times if t <= at + 0.001]
    return max(before) if before else 0.0


class Session:
    """Stands in for `record.Session` while OBS records: only the input logs."""

    announce = ("Watching", "Salience is noting this game; it uses your OBS recording when you finish.")
    status_label = "OBS is recording; Salience takes this game when it ends"

    def __init__(self, args, hwnd=None, root: Path | None = None, folder: Path | None = None):
        self.args = args
        self.path = (root or args.out) / time.strftime("%Y%m%d-%H%M%S")
        self.folder = folder
        self.error: str | None = None
        self.warning: str | None = None

    def start(self) -> "Session":
        from pipeline_v2.capture.input_activity import ActivitySampler
        from pipeline_v2.capture.mouse import JOURNAL_NAME as MOUSE_JOURNAL, SHOTS_NAME, MouseMotion
        from pipeline_v2.capture.record import skipped

        self.path.mkdir(parents=True, exist_ok=True)
        self.began = time.time()
        self.activity = ActivitySampler(journal=self.path / "input.tsv")
        skip = skipped()
        if "keys" not in skip:
            self.activity.start()
        self.mouse = MouseMotion(journal=self.path / MOUSE_JOURNAL, shots=self.path / SHOTS_NAME,
                                 started=self.activity.started)
        if "mouse" not in skip:
            self.mouse.start()
        # The input logs count from a monotonic start; kept as wall time here.
        self.input_wall = time.time() - (time.monotonic() - self.activity.started)
        print(f"watching -> {self.path}\n  video : your OBS recording in {self.folder}")
        return self

    def tick(self) -> None:
        return None

    def stop(self):
        print("\nstopping -- the game is over; the OBS recording is used for highlights")
        self.activity.stop()
        self.mouse.stop()
        (self.path / PENDING).write_text(json.dumps({
            "folder": str(self.folder) if self.folder else "",
            "began_utc": datetime.fromtimestamp(self.began, tz=timezone.utc).isoformat(),
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "input_utc": datetime.fromtimestamp(self.input_wall, tz=timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        return None


def pending(session: Path) -> bool:
    return (session / PENDING).is_file() and not (session / "segments.csv").is_file()


def adopt(session: Path, runner=subprocess.run, *, wait: float = FINISH_WAIT,
          step: float = FINISH_STEP, sleep=time.sleep) -> str | None:
    """Copy the game's stretch of its OBS recording into `session`; None, or why not.

    A recording OBS is still writing is tried as it is (Matroska and OBS's
    hybrid MP4 read fine mid-write) and waited for while it cannot be read.
    """
    note = json.loads((session / PENDING).read_text(encoding="utf-8"))
    folder = Path(note.get("folder") or "")
    began = datetime.fromisoformat(note["began_utc"])
    ended = datetime.fromisoformat(note["ended_utc"])
    if not folder.is_dir():
        return f"the OBS folder {folder} is not there"
    deadline = time.time() + wait
    while True:
        found = covering(folder, began - timedelta(seconds=5), ended, runner)
        if found is not None:
            break
        if time.time() >= deadline:
            return "no OBS recording covers this game"
        sleep(step)  # OBS may not have closed the file it is writing
    start = keyframe_before(found.path, found.offset, runner)
    result = runner(["ffmpeg", "-hide_banner", "-nostdin", "-y",
                     "-ss", f"{start:.3f}", "-i", str(found.path),
                     "-t", f"{found.end - start:.3f}",
                     "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                     "-f", "segment", "-segment_time", str(SEGMENT_SECONDS), "-reset_timestamps", "1",
                     "-segment_list", str(session / "segments.csv"), "-segment_list_type", "csv",
                     str(session / "part-%05d.mkv")], capture_output=True, text=True)
    if getattr(result, "returncode", 1) != 0 or not (session / "segments.csv").is_file():
        (session / "segments.csv").unlink(missing_ok=True)
        return f"copying from {found.path.name} failed"
    origin = found.start + timedelta(seconds=start)
    input_utc = datetime.fromisoformat(note["input_utc"])
    (session / "capture_stats.json").write_text(json.dumps({
        "source": "obs", "obs_file": str(found.path),
        "first_frame_utc": origin.isoformat(),
        "input_offset_sec": round((origin - input_utc).total_seconds(), 3),
        "error": None,
    }, indent=2), encoding="utf-8")
    print(f"  from OBS: {found.path.name}, {start:.1f}-{found.end:.1f}s")
    return None
