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


def _running_port() -> int | None:
    try:
        return int(_port_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


VIEWS = ("session", "rate")


def target(session: str = "", view: str = "session") -> str:
    """The app's address for a view of a session: `#/session/<id>`, `#/rate/<id>`."""
    view = view if view in VIEWS else "session"
    return f"#/{view}/{session}" if session else ("#/rate" if view == "rate" else "#/")


def show_running(session: str = "", timeout: float = 2.0, view: str = "session") -> bool:
    """Ask an open window to come forward (at `session`); False if none answers."""
    port = _running_port()
    if port is None:
        return False
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/show", data=json.dumps({"session": session, "view": view}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except OSError:
        return False


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
