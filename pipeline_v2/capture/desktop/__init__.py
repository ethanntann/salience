"""The Salience desktop app: the player's highlights, rating and settings in one window.

A window of its own process (pywebview, on Windows' built-in WebView2) around
a local server (`server`). Its own process because the tray and the window
each need a process's main thread. The tray starts it (`launch`); a second
start brings the open one forward instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _port_file() -> Path:
    from pipeline_v2.capture import app

    return app.APP_DIR / "desktop.port"


def _running() -> tuple[int, str] | None:
    """The open window's port and launch secret (`server.SECRET_HEADER`)."""
    try:
        port, secret = (_port_file().read_text(encoding="utf-8").split() + [""])[:2]
        return int(port), secret
    except (OSError, ValueError):
        return None


def write_running(port: int, secret: str) -> None:
    path = _port_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{port} {secret}", encoding="utf-8")


def _ask(path: str, body: dict, timeout: float) -> bool:
    """POST to the open window's server; False if none answers."""
    running = _running()
    if running is None:
        return False
    port, secret = running
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Salience": secret}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except OSError:
        return False


VIEWS = ("session", "rate")


def target(session: str = "", view: str = "session") -> str:
    """The app's address for a view of a session: `#/session/<id>`, `#/rate/<id>`."""
    view = view if view in VIEWS else "session"
    return f"#/{view}/{session}" if session else ("#/rate" if view == "rate" else "#/")


def show_running(session: str = "", timeout: float = 2.0, view: str = "session") -> bool:
    """Ask an open window to come forward (at `session`); False if none answers."""
    if _running() is None:
        return False
    try:
        # Windows lets a process raise another's window only with its leave.
        import ctypes

        ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
    except (AttributeError, OSError):
        pass
    return _ask("/api/show", {"session": session, "view": view}, timeout)


def close_running(timeout: float = 2.0) -> bool:
    """Ask an open window to close: Quit Salience in the tray closes everything."""
    return _ask("/api/quit", {}, timeout)


def launch(session: Path | None = None, view: str = "session") -> None:
    """Open the app at a view of `session`; an open one is brought forward."""
    name = session.name if session is not None else ""
    if show_running(name, view=view):
        return
    executable = Path(sys.executable)
    windowless = executable.with_name("pythonw.exe")
    command = [str(windowless if windowless.is_file() else executable),
               "-m", "pipeline_v2.capture.desktop", *([f"{view}/{name}"] if name else [])]
    subprocess.Popen(command, cwd=os.getcwd(),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
