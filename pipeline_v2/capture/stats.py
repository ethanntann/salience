"""What a beta tester can send back: counts only, made when they ask.

Nothing here is sent anywhere. The player presses Export my stats, reads the
file if they like, and sends it to whoever invited them. It holds no video,
audio, words said, descriptions, names or paths: per session, how long it
was, how many matches and highlights, how many of those the player kept or
threw out, how long the highlights took after the game, and whether the
recording or the game stuttered.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import time
from pathlib import Path

# A frame the game took longer than this to show is a stall a player feels.
STALL_MS = 250.0


CACHE = "stalls.json"


def _frames_stalls(session: Path, spans=None) -> int | None:
    """Fortnite frames slower than `STALL_MS` inside a match, or None.

    Only in matches: loading screens and the lobby stall by design, and every
    session showed dozens of them (31 in a clean one, 2026-11-06). Kept
    beside the session once counted: a frame log is tens of megabytes and
    never changes after the session.
    """
    path = session / "frames.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        kept = json.loads((session / CACHE).read_text(encoding="utf-8"))
        if kept.get("size") == path.stat().st_size and kept.get("in") == "matches":
            return kept["stalls"]
    except (OSError, ValueError, KeyError):
        pass
    if spans is None:
        spans = _match_spans(session)
    if not spans:
        return None
    stalls = _count_stalls(path, spans)
    if stalls is not None:
        try:
            (session / CACHE).write_text(json.dumps({"size": path.stat().st_size, "in": "matches",
                                                     "stalls": stalls}),
                                         encoding="utf-8")
        except OSError:
            pass
    return stalls


def _match_spans(session: Path) -> list | None:
    from pipeline_v2.capture import clips, gameplay

    segments = clips.read_segments(session)
    if not segments:
        return None
    return gameplay.spans(session, max(seg.end for seg in segments)).get("spans")


def _count_stalls(path: Path, spans: list) -> int | None:
    """Stalls inside `spans` (seconds of the video). The frame log starts with
    the recording, so its running total of frame times is the video's clock."""
    stalls, clock = 0, 0.0
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if "fortnite" not in (row.get("Application") or "").lower():
                    continue
                try:
                    frame = float(row.get("MsBetweenPresents") or 0)
                except ValueError:
                    continue
                clock += frame / 1000
                if frame > STALL_MS and any(a <= clock <= b for a, b in spans):
                    stalls += 1
    except OSError:
        return None
    return stalls


def _waits(log: Path) -> dict[str, float]:
    """Seconds from the game closing to highlights ready, per session, from the log."""
    waits: dict[str, float] = {}
    session = started = None
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return waits
    for line in lines:
        moved = re.search(r"(?:recording|watching) -> .*[\\/](\d{8}-\d{6})\s*$", line)
        if moved:
            session, started = moved.group(1), None
            continue
        stamp = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\] (Making highlights|Highlights ready)", line)
        if not stamp or session is None:
            continue
        seconds = int(stamp.group(1)) * 3600 + int(stamp.group(2)) * 60 + int(stamp.group(3))
        if stamp.group(4) == "Making highlights":
            started = seconds
        elif started is not None:
            waits[session] = (seconds - started) % 86400
            started = None
    return waits


def _ratings_by_window(conn) -> dict[tuple[str, str], int]:
    """(session folder, window file) -> the player's keep (1) or skip (0)."""
    found = {}
    for row in conn.execute("select c.path, p.label from preferences p join clips c using (clip_sha)"):
        path = Path(row[0])
        found[(path.parent.parent.name, path.name)] = int(row[1])
    return found


def build(root: Path, conn, log: Path, *, version: str = "", relay: dict | None = None,
          imports: int = 0, now: float | None = None) -> dict:
    rated = _ratings_by_window(conn)
    waits = _waits(log)
    sessions = []
    for session in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        manifest_path = session / "highlights" / "highlights.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        picks = manifest.get("highlights") or []
        verdicts = [rated.get((session.name, h.get("clip", ""))) for h in picks]
        try:
            stats = json.loads((session / "capture_stats.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats = {}
        try:
            segments = (session / "segments.csv").read_text(encoding="utf-8").splitlines()
            minutes = round(float(segments[-1].split(",")[2]) / 60, 1)
        except (OSError, ValueError, IndexError):
            minutes = None
        sessions.append({
            "date": session.name[:8],
            "minutes": minutes,
            "matches": (manifest.get("gameplay") or {}).get("matches"),
            "highlights": len(picks),
            "highlights_kept": sum(v == 1 for v in verdicts),
            "highlights_thrown_out": sum(v == 0 for v in verdicts),
            "seconds_to_highlights": waits.get(session.name),
            "recorded_fps": manifest.get("captured_fps"),
            "game_stalls_in_matches": _frames_stalls(session),
            "recorder": "obs" if stats.get("source") == "obs" else "salience",
            "capture_error": bool(stats.get("error")),
        })
    labels = [int(r[0]) for r in conn.execute("select label from preferences")]
    return {
        "salience_stats": 1,
        "exported": time.strftime("%Y-%m-%d", time.localtime(now)),
        "version": version,
        "windows": platform.release(),
        "sessions": sessions,
        "ratings": {"kept": sum(labels), "skipped": len(labels) - sum(labels)},
        "clips_imported": imports,
        "beta_match_time": ({k: relay.get(k) for k in ("used_seconds", "allowance_seconds")}
                            if relay else None),
    }


def export(folder: Path | None = None) -> Path:
    """Write the stats file (to Documents) and return where."""
    from pipeline_v2 import store
    from pipeline_v2.capture import app, imports

    conn = store.connect()
    try:
        data = build(app._recording_args().out, conn, app.LOG, version=_version(),
                     relay=app.relay_usage(), imports=imports.liked_count())
    finally:
        conn.close()
    folder = folder or Path.home() / "Documents"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"Salience stats {data['exported']}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("salience")
    except Exception:  # noqa: BLE001 - a bundled copy has no package metadata
        return "bundled"
