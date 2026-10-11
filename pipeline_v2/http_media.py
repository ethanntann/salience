"""Seekable local video responses shared by both review interfaces."""

import hashlib
import os
import re
import subprocess
import threading
from pathlib import Path

# Containers a browser `<video>` element plays natively. Everything else -- the
# native recorder writes Matroska -- is remuxed once into an MP4 and cached.
_PLAYABLE = {".mp4", ".m4v", ".webm"}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _audio_tracks(path: Path) -> int:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([line for line in out.splitlines() if line.strip()])


def playable(path: Path) -> Path:
    """A version of `path` a browser can play, made once and cached.

    Two problems, one fix. Matroska was served under a `video/mp4` header, which
    Firefox refuses and Chromium plays only sometimes -- so the review page could
    show a black box where the clip should be. And the recorder writes game and
    microphone as separate tracks, of which a browser plays only the first, so
    even a clip that did play hid the player's own reaction -- the thing most
    worth hearing when judging it. Video is copied untouched; the two audio
    tracks are mixed into one, which is the only part re-encoded.
    """
    if path.suffix.lower() in _PLAYABLE:
        return path
    from pipeline_v2 import config

    stat = path.stat()
    key = hashlib.sha256(
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()[:24]
    target = config.DATA_DIR / "playable" / f"{key}.mp4"
    if target.is_file():
        return target
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:  # a browser fires several range requests at once
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tracks = _audio_tracks(path)
        if tracks >= 2:
            audio = ["-filter_complex",
                     "[0:a:0][0:a:1]amix=inputs=2:normalize=0[mixed]",
                     "-map", "0:v:0", "-map", "[mixed]", "-c:a", "aac", "-b:a", "160k"]
        elif tracks == 1:
            audio = ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "copy"]
        else:
            audio = ["-map", "0:v:0"]
        partial = target.with_suffix(".partial.mp4")
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
             *audio, "-c:v", "copy", "-movflags", "+faststart", str(partial)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            return path  # serve the original rather than nothing
        os.replace(partial, target)
    return target


def serve_video(handler, path: Path) -> None:
    path = playable(path)
    size = path.stat().st_size
    # Range support: without it the browser cannot seek, and seeking to the
    # event is the whole point of the layout.
    start, end = 0, size - 1
    requested = handler.headers.get("Range", "") or ""
    partial = bool(requested)
    if partial:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if match and any(match.groups()) and size:
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
        else:
            start = size
        if start >= size or end < start:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
    length = max(0, end - start + 1)

    handler.send_response(206 if partial else 200)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except ConnectionError:
                return  # browser seeked away; not an error
            remaining -= len(chunk)
