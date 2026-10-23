"""The desktop app's local server: the library, a session, rating, settings.

Rating is the rating page's own (`review.Handler`: the next window, a rating,
a window's video), so a keep here does exactly what a keep there does,
including becoming a highlight (`rating.rated`). What is added is the player's
view of their sessions: every session's highlights, played, with a still from
each, and the few recording settings that are the player's to choose.

Bound to 127.0.0.1 on a free port: nothing off this machine can reach it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pipeline_v2 import review, store
from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import highlights, rating
from pipeline_v2.http_media import serve_video

STATIC = Path(__file__).parent / "static"
SESSION_ID = re.compile(r"^\d{8}-\d{6}$")
FILE_NAME = re.compile(r"^[\w.-]+$")
THUMBS = ".thumbs"  # inside highlights/, so a re-run's rebuild takes them too
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
         ".jpg": "image/jpeg", ".png": "image/png"}

WEAPONS = {"sniper": "Sniper", "shotgun": "Shotgun", "smg": "SMG", "assault_rifle": "Rifle",
           "pistol": "Pistol", "explosive": "Explosive", "melee": "Pickaxe"}


def recordings_root() -> Path:
    from pipeline_v2.capture import app

    return app._recording_args().out


def session_dir(session_id: str) -> Path | None:
    if not SESSION_ID.match(session_id or ""):
        return None
    path = recordings_root() / session_id
    return path if (path / "segments.csv").is_file() else None


def title(payload: dict, summary: str) -> str:
    """A highlight's name from what the describer filled in, never invented."""
    text = (summary or "").lower()
    if "victory royale" in text:
        return "Victory Royale"
    weapon = WEAPONS.get(str(payload.get("kill_weapon_class") or ""), "")
    outcome = payload.get("outcome")
    if "no-scope" in text or "no scope" in text:
        return f"No-scope {weapon.lower() or 'kill'}".strip()
    if outcome == "elimination":
        what = "headshot" if payload.get("kill_headshot") == "yes" else "elimination"
        return f"{weapon} {what}".strip().capitalize() if weapon else what.capitalize()
    if outcome == "knock":
        return f"{weapon} knock".strip() if weapon else "Knock"
    return "Fight" if outcome == "fight" else "Moment"


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _probe_seconds(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                             timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _lengths(out: Path, files: list[str]) -> dict[str, float | None]:
    """Each highlight's length, probed once and kept beside the thumbnails."""
    cache_path = out / THUMBS / "lengths.json"
    try:
        known = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    missing = [f for f in files if f not in known and (out / f).is_file()]
    if missing:
        with ThreadPoolExecutor(8) as pool:
            for name, seconds in zip(missing, pool.map(lambda f: _probe_seconds(out / f), missing)):
                known[name] = seconds
        try:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(known), encoding="utf-8")
        except OSError:
            pass
    return known


def session_view(session: Path, conn) -> dict:
    """One session as the app shows it: when, how long, its highlights, what is left to rate."""
    out = session / "highlights"
    try:
        manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    entries = [h for h in manifest.get("highlights", []) if (out / h.get("file", "")).is_file()]
    lengths = _lengths(out, [h["file"] for h in entries])
    windows = session / "windows"
    _, inside = highlights._session_clips(conn, windows) if windows.is_dir() else (None, [])
    by_name = {Path(c.path).name: c for c in inside}
    rated = {row[0] for row in conn.execute("select clip_sha from preferences")}
    loved = {row[0] for row in conn.execute("select clip_sha from rating_notes where love = 1")}
    shown = []
    for h in entries:
        clip = by_name.get(h.get("clip", ""))
        payload = (clip.payload if clip else None) or {}
        side = clips_mod._read_sidecar(windows / h.get("clip", "")) if clip else {}
        at = side.get("segment_start_sec")
        shown.append({
            "rank": h.get("rank"), "file": h["file"], "reel": h.get("reel") or "",
            "title": "You asked for this" if h.get("asked") and not clip else title(payload, h.get("summary", "")),
            "summary": h.get("summary", ""), "said": h.get("said", ""),
            "weapon": WEAPONS.get(str(payload.get("kill_weapon_class") or ""), ""),
            "at": _clock(at + side.get("event_sec", 4.0) if at is not None else None),
            "length": _clock(lengths.get(h["file"])),
            "loved": bool(clip and clip.clip_sha in loved),
            "asked": bool(h.get("asked")), "added": h.get("trigger") == highlights.RATED_TRIGGER,
        })
    segments = clips_mod.read_segments(session)
    started = datetime.strptime(session.name, "%Y%m%d-%H%M%S")
    gameplay = manifest.get("gameplay") or {}
    unrated = sum(c.clip_sha not in rated for c in inside)
    return {
        "id": session.name, "started": started.isoformat(),
        "minutes": round(segments[-1].end / 60) if segments else None,
        "matches": gameplay.get("matches"), "highlights": shown,
        "windows": len(inside), "unrated": unrated, "basis": manifest.get("basis", ""),
    }


def library() -> dict:
    root = recordings_root()
    try:
        found = sorted((p for p in root.iterdir() if p.is_dir() and SESSION_ID.match(p.name)
                        and (p / "highlights" / "highlights.json").is_file()),
                       key=lambda p: p.name, reverse=True)
    except OSError:
        found = []
    conn = store.connect()
    try:
        sessions = [session_view(p, conn) for p in found]
        labels = conn.execute("select count(*) from preferences").fetchone()[0]
    finally:
        conn.close()
    return {"sessions": sessions, "ratings": labels}


def thumbnail(out: Path, name: str) -> Path | None:
    """A still from a highlight, made once: where its kill is, about 12s in."""
    source = out / name
    thumb = out / THUMBS / (Path(name).stem + ".jpg")
    if thumb.is_file():
        return thumb
    if not source.is_file():
        return None
    thumb.parent.mkdir(exist_ok=True)
    length = _probe_seconds(source) or 0.0
    at = min(12.0, max(0.0, length / 2))
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{at:.2f}", "-i", str(source),
         "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(thumb)],
        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return thumb if done.returncode == 0 and thumb.is_file() else None


def settings() -> dict:
    from pipeline_v2.capture import app

    from pipeline_v2.capture import record

    args = app._recording_args()
    return {
        "quality": app.quality(),
        "presets": {name: {"fps": fps, "mbps": int(rate.rstrip("M"))}
                    for name, (fps, rate) in record.QUALITY.items()},
        "fps": args.fps, "mbps": int(str(args.bitrate).rstrip("M")),
        "frame_rates": list(app.FRAME_RATES),
        "bitrate_range": [app.BITRATES.start, app.BITRATES.stop - 1],
        "resolution": app.resolution(),
        "resolutions": {k: label for k, (_, label) in app.RESOLUTIONS.items()},
        "starts_with_windows": bool(app.starts_with_windows()),
        "has_key": bool(os.environ.get("FIREWORKS_API_KEY")),
    }


def delete_highlight(session: Path, name: str) -> bool:
    """The player removing one of their highlights: its file, reel and entry."""
    from pipeline_v2.jsonio import write_json

    out = session / "highlights"
    with highlights._HIGHLIGHTS_LOCK:
        manifest = highlights._manifest(out)
        if manifest is None:
            return False
        entries = manifest.get("highlights", [])
        gone = [h for h in entries if h.get("file") == name]
        if not gone:
            return False
        for h in gone:
            (out / h["file"]).unlink(missing_ok=True)
            if h.get("reel"):
                (out / highlights.REEL_DIR / h["reel"]).unlink(missing_ok=True)
            (out / THUMBS / (Path(h["file"]).stem + ".jpg")).unlink(missing_ok=True)
        manifest["highlights"] = [h for h in entries if h not in gone]
        write_json(out / "highlights.json", manifest)
    return True


class Handler(review.Handler):
    """The rating page's routes, under the app's own."""

    show = None  # set by the window: bring it forward, at a session

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, kind: str, session_id: str, name: str) -> Path | None:
        session = session_dir(session_id)
        if session is None or not FILE_NAME.match(name):
            return None
        out = session / "highlights"
        if kind == "thumb":
            return thumbnail(out, name)
        path = out / (highlights.REEL_DIR if kind == "reel" else "") / name
        return path if path.is_file() else None

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if url.path == "/":
            return self._static("index.html")
        if parts[:1] == ["static"] and len(parts) == 2:
            return self._static(parts[1])
        if url.path == "/api/library":
            return self._json(library())
        if url.path == "/api/settings":
            return self._json(settings())
        if url.path == "/api/rate/next":
            session = session_dir(parse_qs(url.query).get("session", [""])[0])
            review._ONLY = str((session / "windows").resolve()) if session else None
            conn = store.connect()
            try:
                return self._json(review._next_clip(conn))
            finally:
                conn.close()
        if parts[:1] in (["video"], ["reel"], ["thumb"]) and len(parts) == 3:
            path = self._file(parts[0], parts[1], parts[2])
            if path is None:
                return self.send_error(404)
            if parts[0] == "thumb":
                return self._send(path.read_bytes(), "image/jpeg")
            return serve_video(self, path)
        return super().do_GET()  # /api/next, /media/<sha>

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 4096:
                return {}
            data = json.loads(self.rfile.read(length))
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/rate":
            return super().do_POST()
        data = self._payload()
        session = session_dir(str(data.get("session", "")))
        if self.path == "/api/open-folder" and session is not None:
            os.startfile(session / "highlights")  # noqa: S606 -- the player's own folder
            return self._json({"ok": True})
        if self.path == "/api/delete" and session is not None:
            name = str(data.get("file", ""))
            ok = FILE_NAME.match(name) is not None and delete_highlight(session, name)
            return self._json({"ok": ok}, 200 if ok else 404)
        if self.path == "/api/settings":
            from pipeline_v2.capture import app

            if data.get("quality") in app.QUALITY_LABELS:
                app.set_quality(data["quality"])
            custom = data.get("custom")
            if isinstance(custom, dict):
                try:
                    app.set_custom_quality(int(custom.get("fps")), int(custom.get("mbps")))
                except (TypeError, ValueError):
                    return self._json({"error": "not a frame rate and bitrate"}, 400)
            if data.get("resolution") in app.RESOLUTIONS:
                app.set_resolution(data["resolution"])
            return self._json(settings())
        if self.path == "/api/show":
            if Handler.show is not None:
                Handler.show(str(data.get("session", "")))
            return self._json({"ok": True})
        return self.send_error(404)


def start(port: int = 0) -> ThreadingHTTPServer:
    """Serve on 127.0.0.1 (a free port unless given), on a background thread."""
    from pipeline_v2.capture import app

    app.load_environment()
    review._VARIANT = highlights._configured_variant()
    review._PLAYABLE = rating.playable
    review._ON_RATED = rating.rated
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, name="desktop-server", daemon=True).start()
    return server
