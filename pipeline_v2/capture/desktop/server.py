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
import shutil
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


KEYFRAME_GRID = 2.0  # seconds: where a stream-copied cut can start (`clips.cut_span`)


def kill_at(clip, inside: list, length: float | None) -> float | None:
    """Where the window's moment sits in its exported highlight, in seconds.

    The export grew the window over the rest of its fight and started the cut
    on a keyframe; the same steps, from the same descriptions, find it again.
    None when that cannot be known (no timing, or the answer is off the clip).
    """
    side = clips_mod._read_sidecar(Path(clip.path))
    try:
        moment = float(side["segment_start_sec"]) + float(side.get("event_sec", 4.0))
    except (KeyError, TypeError, ValueError):
        return None
    span = highlights.fight_extent(clip, highlights.window_index([(0.0, c) for c in inside]),
                                   (highlights.HIGHLIGHT_LEAD, highlights.HIGHLIGHT_LAG))
    if span is None:
        return None
    start = max(0.0, span[0] // KEYFRAME_GRID * KEYFRAME_GRID)
    at = moment - start
    return round(at, 2) if length and 0 <= at <= length else None


def session_view(session: Path, conn) -> dict:
    """One session as the app shows it: when, how long, its highlights, what is left to rate."""
    out = session / "highlights"
    try:
        manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    entries = []
    for h in manifest.get("highlights", []):
        # Manifests before 2026-10-19 name only the window; the file is `NN-<window>`.
        name = h.get("file") or (f"{int(h['rank']):02d}-{h['clip']}" if h.get("rank") and h.get("clip") else "")
        if name and (out / name).is_file():
            entries.append({**h, "file": name})
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
            "trimmed": (trimmed := (out / ORIGINALS / h["file"]).is_file()),
            "kill": (kill_at(clip, inside, lengths.get(h["file"]))
                     if clip and not trimmed and not (h.get("asked") and not clip) else None),
            "seconds": lengths.get(h["file"]),
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
        "autoclip": app.autoclip_on(),
        "has_key": bool(os.environ.get("FIREWORKS_API_KEY")),
    }


# How old the tray's last word can be before it counts as gone (it writes on
# every change of state, and the app checks it is still running anyway).
def recorder_status() -> dict:
    """What the background recorder is doing, for the sidebar."""
    from pipeline_v2.capture import app

    running = app.tray_running()
    status = app.read_status() if running else {}
    state = status.get("state") if running else "off"
    if running and state not in ("waiting", "recording", "highlights"):
        state = "waiting"
    return {"autoclip": app.autoclip_on(), "running": running, "state": state,
            "started": status.get("started"), "session": status.get("session"),
            "has_key": bool(os.environ.get("FIREWORKS_API_KEY"))}


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


ORIGINALS = ".originals"  # a trimmed highlight's untrimmed file, for "Restore original"
MIN_TRIM = 1.0  # seconds: shorter is not a clip
_NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(command: list[str]) -> bool:
    return subprocess.run(command, capture_output=True, creationflags=_NO_CONSOLE).returncode == 0


def _refresh(out: Path, name: str) -> None:
    """After a highlight's file changed: its still, length and reel are made again."""
    (out / THUMBS / (Path(name).stem + ".jpg")).unlink(missing_ok=True)
    cache = out / THUMBS / "lengths.json"
    try:
        known = json.loads(cache.read_text(encoding="utf-8"))
        known.pop(name, None)
        cache.write_text(json.dumps(known), encoding="utf-8")
    except (OSError, ValueError):
        pass
    manifest = highlights._manifest(out) or {}
    reel = next((h.get("reel") for h in manifest.get("highlights", []) if h.get("file") == name), "")
    if reel:
        threading.Thread(target=highlights.make_reel, args=(out / name, out / highlights.REEL_DIR / reel),
                         name="reel", daemon=True).start()


def trim_highlight(session: Path, name: str, start: float, end: float) -> bool:
    """Keep [start, end) of a highlight, re-encoded so the cut is exact.

    Stream-copied, a cut can only start on a keyframe (every 2s), which is the
    difference between a clip that opens on the shot and one that opens late.
    NVENC re-encodes it in about a second. Every audio track is kept with its
    title, the mix still first. The first trim keeps the untrimmed file aside.
    """
    out = session / "highlights"
    source = out / name
    length = _probe_seconds(source) if source.is_file() else None
    if length is None or not (0 <= start < end <= length + 0.05) or end - start < MIN_TRIM:
        return False
    original = out / ORIGINALS / name
    with highlights._HIGHLIGHTS_LOCK:
        if not original.is_file():
            original.parent.mkdir(exist_ok=True)
            shutil.copy2(source, original)
        partial = out / f"trimming-{name}"
        base = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{end - start:.3f}", "-map", "0", "-map_metadata", "0"]
        tail = ["-c:a", "aac", "-b:a", "192k", str(partial)]
        if not (_run(base + ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"] + tail)
                or _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"] + tail)):
            partial.unlink(missing_ok=True)
            return False
        partial.replace(source)
    _refresh(out, name)
    return True


def restore_highlight(session: Path, name: str) -> bool:
    out = session / "highlights"
    original = out / ORIGINALS / name
    if not original.is_file():
        return False
    with highlights._HIGHLIGHTS_LOCK:
        original.replace(out / name)
    _refresh(out, name)
    return True


def export_mp4(source: Path, destination: Path) -> bool:
    """An MP4 anything plays: the video copied, and the one mixed track a phone hears."""
    if destination.suffix.lower() != ".mp4":
        destination = destination.with_suffix(".mp4")
    return _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:v:0",
                 "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(destination)])


class Handler(review.Handler):
    """The rating page's routes, under the app's own."""

    show = None  # set by the window: bring it forward
    goto = None  # where the page is to go next, collected with /api/status
    save_as = None  # set by the window: a Save dialog, (suggested name) -> path or None

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
        if url.path == "/api/status":
            goto, Handler.goto = Handler.goto, None
            return self._json({**recorder_status(), "goto": goto})
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
        name = str(data.get("file", ""))
        if self.path in ("/api/trim", "/api/restore", "/api/export"):
            if session is None or not FILE_NAME.match(name) or not (session / "highlights" / name).is_file():
                return self._json({"ok": False}, 404)
        if self.path == "/api/trim":
            try:
                start, end = float(data.get("start")), float(data.get("end"))
            except (TypeError, ValueError):
                return self._json({"ok": False}, 400)
            ok = trim_highlight(session, name, start, end)
            return self._json({"ok": ok}, 200 if ok else 400)
        if self.path == "/api/restore":
            return self._json({"ok": restore_highlight(session, name)})
        if self.path == "/api/export":
            suggested = f"Salience {session.name} {Path(name).stem}.mp4"
            chosen = Handler.save_as(suggested) if Handler.save_as is not None else None
            if not chosen:
                return self._json({"ok": False, "cancelled": True})
            target = Path(chosen).with_suffix(".mp4")
            ok = export_mp4(session / "highlights" / name, target)
            return self._json({"ok": ok, "path": str(target) if ok else ""})
        if self.path == "/api/reveal":
            path = Path(str(data.get("path", "")))
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)], creationflags=_NO_CONSOLE)
            return self._json({"ok": path.is_file()})
        if self.path == "/api/autoclip":
            from pipeline_v2.capture import app

            on = bool(data.get("on"))
            if on and not os.environ.get("FIREWORKS_API_KEY"):
                return self._json({**recorder_status(), "need_key": True}, 409)
            app.set_autoclip(on)
            return self._json(recorder_status())
        if self.path == "/api/startup":
            from pipeline_v2.capture import app

            app.set_starts_with_windows(bool(data.get("on")))
            return self._json(settings())
        if self.path == "/api/key":
            from pipeline_v2.capture import app

            key = str(data.get("key", "")).strip()
            problem = app.check_key(key) if key else "Paste your Fireworks API key first."
            if problem is not None:
                return self._json({"ok": False, "error": problem}, 400)
            app.write_setting("FIREWORKS_API_KEY", key)
            os.environ["FIREWORKS_API_KEY"] = key
            return self._json({"ok": True, **settings()})
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
            from pipeline_v2.capture import desktop

            # Answered at once: raising the window on this thread once waited
            # forever on pywebview (2026-10-24), the tray took the app for gone
            # and opened a second one. The page goes where it is sent itself.
            session = str(data.get("session", ""))
            Handler.goto = desktop.target(session if SESSION_ID.match(session) else "",
                                          str(data.get("view", "session")))
            if Handler.show is not None:
                threading.Thread(target=Handler.show, name="show", daemon=True).start()
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
