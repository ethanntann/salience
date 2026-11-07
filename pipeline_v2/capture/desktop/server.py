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
from pipeline_v2.capture import cleanup
from pipeline_v2.capture import clips as clips_mod
from pipeline_v2 import banner
from pipeline_v2.capture import highlights, rating
from pipeline_v2.http_media import serve_video

STATIC = Path(__file__).parent / "static"
SESSION_ID = re.compile(r"^\d{8}-\d{6}$")
FILE_NAME = re.compile(r"^[\w.-]+$")
THUMBS = ".thumbs"  # inside highlights/, so a re-run's rebuild takes them too
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
         ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json"}

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
BEFORE_WINDOW = 4.0  # seconds: how early a window's kill can be (a squads knock 3.4s before, 2026-10-25)


# The look the player chose, kept on disk: the page's own storage belongs to its
# address, and the port is new each launch, so it was gone after every restart.
APPEARANCE = {"salience.theme": ("auto", "light", "dark"), "salience.rail": ("open", "collapsed"),
              "salience.palette": None, "salience.pattern": None}  # None: any short name


def appearance_path() -> Path:
    from pipeline_v2.capture import app

    return app.APP_DIR / "appearance.json"


def appearance() -> dict[str, str]:
    try:
        kept = json.loads(appearance_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in kept.items() if _look_ok(k, v)} if isinstance(kept, dict) else {}


def _look_ok(key, value) -> bool:
    allowed = APPEARANCE.get(key, ())
    if allowed is None:
        return isinstance(value, str) and re.fullmatch(r"[a-z0-9-]{1,24}", value) is not None
    return value in allowed


def set_appearance(key: str, value: str) -> bool:
    if not _look_ok(key, value):
        return False
    kept = {**appearance(), key: value}
    path = appearance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept), encoding="utf-8")
    return True


def event_name(clip) -> str:
    """What the marked moment is, in the player's words."""
    if highlights.is_victory(clip):
        return "win"
    return {"elimination": "kill", "knock": "knock"}.get((clip.payload or {}).get("outcome"), "")


def window_in(clip, index: dict, length: float | None) -> tuple[float, float] | None:
    """Where the window sits in its exported highlight, in seconds.

    The export grew the window over the rest of its fight and started the cut
    on a keyframe; the same steps, from the same descriptions, find it again.
    None when that cannot be known (no timing, or the window is off the clip).
    """
    side = clips_mod._read_sidecar(Path(clip.path))
    try:
        a, b = float(side["segment_start_sec"]), float(side["segment_end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    span = highlights.fight_extent(clip, index, (highlights.HIGHLIGHT_LEAD, highlights.HIGHLIGHT_LAG))
    if span is None or not length:
        return None
    start = max(0.0, span[0] // KEYFRAME_GRID * KEYFRAME_GRID)
    a, b = max(0.0, a - start), min(length, b - start)
    return (round(a, 2), round(b, 2)) if a < b else None


def _kills(out: Path, files: dict[str, float | None], finder=banner.found,
           compute: bool = True) -> dict[str, list[tuple[float, str]]]:
    """Every kill and knock in each highlight, as (seconds, "kill" or "knock"),
    found once on screen and kept beside the thumbnails.

    Every one in the clip, not only the window's (2026-10-25): a highlight
    grown over its fight holds the knock, the finish and the next kill. The
    window's centre was the marker until 2026-10-24, up to 4s off; no banner
    found is no marker, not a guess.
    """
    cache_path = out / THUMBS / "kills.json"
    try:
        known = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}

    def key(name: str) -> str:  # a trim or a restore is a new file
        try:
            # The words looked for are in it: a new one is looked for in old clips too.
            return f"all:{(out / name).stat().st_mtime_ns}:{'+'.join(banner.WORDS)}"
        except OSError:
            return ""

    missing = [n for n in files if (known.get(n) or {}).get("key") != key(n)]
    if not compute:  # what is already found, nothing looked for
        known = {n: e for n, e in known.items() if n not in missing}
        missing = []
    if missing:
        def find(name: str) -> list:
            length = files[name] or 0.0
            return finder(out / name, 0.0, length) if length else []

        with ThreadPoolExecutor(4) as pool:
            for name, hits in zip(missing, pool.map(find, missing)):
                known[name] = {"key": key(name), "hits": [[t, kind] for t, kind in hits]}
        try:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(known), encoding="utf-8")
        except OSError:
            pass
    # Not yet looked for (`compute` off) is left out, not []: [] is "no banner".
    return {n: [(float(t), str(kind)) for t, kind in known[n].get("hits") or []] for n in files if n in known}


def main_kill(hits: list[tuple[float, str]], window: tuple[float, float] | None) -> tuple[float, str] | None:
    """The kill the highlight was picked for: the first in its window, or just
    before it (a kill a few seconds earlier leaves its banner in the window,
    and that is what the describer saw); the first in the clip otherwise."""
    if not hits:
        return None
    if window is None:
        return hits[0]
    a, b = window
    inside = [h for h in hits if a - banner.LEAD <= h[0] <= b]
    before = [h for h in hits if a - BEFORE_WINDOW <= h[0] < a - banner.LEAD]
    return inside[0] if inside else (before[-1] if before else None)


def session_view(session: Path, conn, find_markers: bool = False) -> dict:
    """One session as the app shows it: when, how long, its highlights, what is left to rate.

    Kill markers come from what is already found unless `find_markers`: the
    library builds every session, and finding them all at once (after the
    words looked for changed) held the app on its skeleton for minutes
    (2026-10-25). The session page asks for its own (`/api/kills`).
    """
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
    labels = dict(conn.execute("select clip_sha, label from preferences").fetchall())
    rated = set(labels)
    loved = {row[0] for row in conn.execute("select clip_sha from rating_notes where love = 1")}
    built: list[dict] = []

    def index() -> dict:  # once a session, and only if a highlight needs it
        if not built:
            built.append(highlights.window_index([(0.0, c) for c in inside]))
        return built[0]

    # Each kill's window, then the kill in it, all at once: one decode each, side by side.
    spans = {}
    for h in entries:
        clip = by_name.get(h.get("clip", ""))
        trimmed = (out / ORIGINALS / h["file"]).is_file()
        if clip and not trimmed and highlights.event_tier(clip) > 0 \
                and (span := window_in(clip, index(), lengths.get(h["file"]))):
            spans[h["file"]] = span
    found = _kills(out, {h["file"]: lengths.get(h["file"]) for h in entries}, compute=find_markers)

    shown = []
    for h in entries:
        clip = by_name.get(h.get("clip", ""))
        payload = (clip.payload if clip else None) or {}
        side = clips_mod._read_sidecar(windows / h.get("clip", "")) if clip else {}
        at = side.get("segment_start_sec")
        main = main_kill(found.get(h["file"]) or [], spans.get(h["file"]))
        shown.append({
            "rank": h.get("rank"), "file": h["file"], "reel": h.get("reel") or "",
            "title": "You asked for this" if h.get("asked") and not clip else title(payload, h.get("summary", "")),
            "summary": h.get("summary", ""), "said": h.get("said", ""),
            "weapon": WEAPONS.get(str(payload.get("kill_weapon_class") or ""), ""),
            "at": _clock(at + side.get("event_sec", 4.0) if at is not None else None),
            "length": _clock(lengths.get(h["file"])),
            "loved": bool(clip and clip.clip_sha in loved),
            # Rated from the highlight itself: a highlight the player thinks
            # is bad could not be marked a skip (2026-11-05).
            "sha": clip.clip_sha if clip else None,
            "rated": labels.get(clip.clip_sha) if clip else None,
            "asked": bool(h.get("asked")), "added": h.get("trigger") == highlights.RATED_TRIGGER,
            "laughed": bool(h.get("laughed")),
            "trimmed": (trimmed := (out / ORIGINALS / h["file"]).is_file()),
            # Marked only where there is one: a lobby window was marked "kill" (2026-10-24).
            "kill": main[0] if main else None,
            # What the banner said, over what the describer said: in squads
            # the marked moment is a knock.
            "event": main[1] if main else (event_name(clip) if clip else ""),
            "kills": [{"at": t, "event": kind} for t, kind in found.get(h["file"]) or []],
            "seconds": lengths.get(h["file"]),
        })
    segments = clips_mod.read_segments(session)
    started = datetime.strptime(session.name, "%Y%m%d-%H%M%S")
    gameplay = manifest.get("gameplay") or {}
    offered = highlights.rateable(session, inside)
    unrated = sum(c.clip_sha not in rated for c in inside if c.clip_sha in offered)
    return {
        "id": session.name, "started": started.isoformat(),
        "minutes": round(segments[-1].end / 60) if segments else (
            round(m) if (m := cleanup.recorded_minutes(session)) is not None else None),
        "footage": not cleanup.footage_deleted(session),
        "matches": gameplay.get("matches"), "highlights": shown,
        "windows": len(offered), "unrated": unrated, "basis": manifest.get("basis", ""),
        "markers_pending": any(h["file"] not in found for h in entries),
    }


# The kills a session's rating page offers first: those whose keep chance is
# nearest the line (`ranking.KEEP_CHANCE`), where a rating teaches the most.
# Remembered a while: the chances take a second to work out.
_UNSURE: dict[str, tuple[float, list[str]]] = {}
UNSURE_SECONDS = 300.0


_WORKING: set[str] = set()
_UNSURE_LOCK = threading.Lock()


def _rated_sessions(session: Path | None) -> list[Path]:
    """`session`, or every session with highlights (None)."""
    if session is not None:
        return [session]
    try:
        return sorted(p for p in recordings_root().iterdir() if p.is_dir() and SESSION_ID.match(p.name)
                      and (p / "highlights" / "highlights.json").is_file())
    except OSError:
        return []


def playing() -> bool:
    """Whether the game is running: the window's background work waits for
    it to close. Opened on a session as a match began, the window worked out
    unsure kills (27 CPU-seconds) and kill markers (video decoding) during
    it, and the game froze for half a second at a time (2026-11-06)."""
    from pipeline_v2.capture import windows

    try:
        return windows.game_running()
    except Exception:  # noqa: BLE001 -- unknown: as if playing, the cautious side
        return True


def unsure_kills(session: Path | None, wait: bool = False) -> list[str]:
    """The order, once worked out; until then none, and it is worked out
    behind the page: the first time took 27s (2026-11-04), a page frozen."""
    import time

    key = str(session)
    known = _UNSURE.get(key)
    if known is not None and time.monotonic() - known[0] < UNSURE_SECONDS:
        return known[1]
    if wait:
        return _work_out_unsure(session)
    if playing():
        return known[1] if known else []  # the rating page's own order until the game closes
    with _UNSURE_LOCK:
        if key in _WORKING:
            return known[1] if known else []
        _WORKING.add(key)

    def work() -> None:
        try:
            _work_out_unsure(session)
        finally:
            _WORKING.discard(key)

    threading.Thread(target=work, name="unsure-kills", daemon=True).start()
    return known[1] if known else []


def _work_out_unsure(session: Path) -> list[str]:
    import time

    from pipeline_v2.ranking import KEEP_CHANCE

    key = str(session)
    conn = store.connect()
    try:
        rated = {row[0] for row in conn.execute("select clip_sha from preferences")}
        inside = [c for s in _rated_sessions(session) for c in highlights._session_clips(conn, s / "windows")[1]]
        kills = [c for c in inside if c.clip_sha not in rated and highlights.event_tier(c) > 0]
        chances = highlights.keep_chances(conn, kills) if kills else {}
    except Exception:  # noqa: BLE001 -- the rating page in its own order rather than none
        chances = {}
    finally:
        conn.close()
    order = sorted(chances, key=lambda sha: abs(chances[sha] - KEEP_CHANCE))
    _UNSURE[key] = (time.monotonic(), order)
    return order


_RATEABLE: dict[str, tuple[float, set[str]]] = {}


def rateable_in(session: Path | None) -> set[str]:
    """What could be a highlight in `session`, or in every session (None),
    remembered a while: which windows those are does not change with ratings."""
    import time

    key = str(session)
    known = _RATEABLE.get(key)
    if known is not None and time.monotonic() - known[0] < UNSURE_SECONDS:
        return known[1]
    found: set[str] = set()
    conn = store.connect()
    try:
        for s in _rated_sessions(session):
            found |= highlights.rateable(s, highlights._session_clips(conn, s / "windows")[1])
    finally:
        conn.close()
    _RATEABLE[key] = (time.monotonic(), found)
    return found


def laughed_in_window(clip: Path) -> list[str]:
    """Who laughed in the window or just after it: "mic" (the player), "game"
    (the PC's audio: friends on a call)."""
    from pipeline_v2.capture import reactions

    span = highlights._span(type("Window", (), {"path": str(clip)})())
    if span is None:
        return []
    return [laugh.track for laugh in reactions.laughed_in(reactions.laughs(clip.parent.parent), *span)]


def _sessions_without_footage() -> list[Path]:
    try:
        return [p for p in recordings_root().iterdir() if p.is_dir() and cleanup.footage_deleted(p)]
    except OSError:
        return []


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


def _voice():
    from pipeline_v2.capture import voice

    return voice


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
        "transcribe_pc": _voice().pc_enabled(),
        "has_key": bool(os.environ.get("FIREWORKS_API_KEY")),
        "relay": app.relay_usage(),  # match time left, for a beta tester on an invite
        "invites_open": bool(app.relay_url()),
        "uses_obs": app.uses_obs(),
        "obs_folder": str(app.obs_folder() or ""),
    }


# How old the tray's last word can be before it counts as gone (it writes on
# every change of state, and the app checks it is still running anyway).
def recorder_status() -> dict:
    """What the background recorder is doing, for the sidebar."""
    from pipeline_v2.capture import app

    running = app.tray_running()
    last = app.read_status()
    status = last if running else {}
    # Quit from its menu, the recorder stays closed until asked for again.
    state = status.get("state") if running else ("closed" if last.get("state") == "closed" else "off")
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


# Clips and montages the player likes, one import at a time (`capture.imports`).
# Most model calls one import may pay for: about a ten-minute montage.
IMPORT_MAX_CALLS = 120
_IMPORT = None


def import_job():
    global _IMPORT
    if _IMPORT is None:
        from pipeline_v2.capture import imports

        _IMPORT = imports.Job(playing=playing)
    return _IMPORT


def import_view() -> dict:
    from pipeline_v2.capture import imports

    return {**import_job().view(), "total": imports.liked_count(),
            "has_key": bool(os.environ.get("FIREWORKS_API_KEY"))}


class Handler(review.Handler):
    """The rating page's routes, under the app's own."""

    show = None  # set by the window: bring it forward
    goto = None  # where the page is to go next, collected with /api/status
    save_as = None  # set by the window: a Save dialog, (suggested name) -> path or None
    choose_videos = None  # set by the window: an Open dialog for videos, () -> [paths]
    choose_folder = None  # set by the window: a folder dialog, (start) -> path or None
    quit = None  # set by the window: close it (Quit Salience in the tray)

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        if name == "index.html":  # the kept look, before anything paints
            body = body.replace(b"<script>", b"<script>window.SALIENCE_LOOK = "
                                + json.dumps(appearance()).encode() + b";", 1)
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
        if url.path == "/api/kills":
            # The open session's markers, found now if they are not yet.
            session = session_dir(parse_qs(url.query).get("session", [""])[0])
            if session is None:
                return self._json({}, 404)
            unsure_kills(None)  # worked out now, ready by the rating page
            conn = store.connect()
            try:
                view = session_view(session, conn, find_markers=not playing())
            finally:
                conn.close()
            return self._json({h["file"]: {"kill": h["kill"], "event": h["event"], "kills": h["kills"]}
                               for h in view["highlights"]})
        if url.path == "/api/settings":
            return self._json(settings())
        if url.path == "/api/import":
            return self._json(import_view())
        if url.path == "/api/status":
            goto, Handler.goto = Handler.goto, None
            return self._json({**recorder_status(), "goto": goto})
        if url.path == "/api/rate/next":
            # Every session's, one queue: a session at a time said "all rated"
            # with more to rate in others (2026-11-04).
            review._ONLY = None
            review._FIRST = lambda: unsure_kills(None)
            review._ONLY_SHAS = lambda: rateable_in(None)
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
        if self.path in ("/api/rate", "/api/unrate"):
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
        if self.path == "/api/import/choose":
            chosen = Handler.choose_videos() if Handler.choose_videos is not None else []
            if chosen:
                import_job().cut([Path(p) for p in chosen])
            return self._json({**import_view(), "cancelled": not chosen})
        if self.path == "/api/import/describe":
            if not os.environ.get("FIREWORKS_API_KEY"):
                return self._json({**import_view(), "need_key": True}, 409)
            import_job().describe(max_calls=IMPORT_MAX_CALLS)
            return self._json(import_view())
        if self.path == "/api/import/discard":
            import_job().discard()
            return self._json(import_view())
        if self.path == "/api/appearance":
            ok = set_appearance(str(data.get("key", "")), data.get("value"))
            return self._json({"ok": ok}, 200 if ok else 400)
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
        if self.path == "/api/transcribe-pc":
            from pipeline_v2.capture import app

            value = "on" if data.get("on") else "off"
            app.write_setting(_voice().PC_KEY, value)
            os.environ[_voice().PC_KEY] = value
            return self._json(settings())
        if self.path == "/api/obs":
            from pipeline_v2.capture import app

            folder = None
            if data.get("choose"):
                folder = Handler.choose_folder(str(app.obs_folder() or "")) if Handler.choose_folder else None
                if not folder:
                    return self._json({**settings(), "cancelled": True})
            app.set_obs(bool(data.get("on", app.uses_obs())), folder)
            return self._json(settings())
        if self.path == "/api/startup":
            from pipeline_v2.capture import app

            app.set_starts_with_windows(bool(data.get("on")))
            return self._json(settings())
        if self.path == "/api/stats-export":
            from pipeline_v2.capture import stats

            path = stats.export()
            subprocess.Popen(["explorer", "/select,", str(path)], creationflags=_NO_CONSOLE)
            return self._json({"ok": True, "path": str(path)})
        if self.path == "/api/invite":
            from pipeline_v2.capture import app

            problem = app.redeem_invite(str(data.get("code", "")).strip())
            if problem is not None:
                return self._json({"ok": False, "error": problem}, 400)
            return self._json({"ok": True, **settings()})
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
        if self.path == "/api/quit" and Handler.quit is not None:
            self._json({"ok": True})
            threading.Thread(target=Handler.quit, daemon=True).start()
            return None
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
    review._ON_UNRATED = rating.unrated
    review._LAUGHED = laughed_in_window
    review._GONE = lambda: [str(s / "windows") for s in _sessions_without_footage()]
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, name="desktop-server", daemon=True).start()
    return server
