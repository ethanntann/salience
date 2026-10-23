"""`python -m pipeline_v2.capture.desktop [session]`: the app window."""

from __future__ import annotations

import re
import sys

from pipeline_v2.capture import desktop
from pipeline_v2.capture.desktop import server

SESSION_ID = re.compile(r"^\d{8}-\d{6}$")


def main(argv: list[str]) -> int:
    # `[<view>/]<session id>`: which screen to open at.
    view, _, session = (argv[0] if argv else "").rpartition("/")
    session = session if SESSION_ID.match(session) else ""
    view = view if view in desktop.VIEWS else "session"
    if desktop.show_running(session, view=view):
        return 0  # one window; the open one has come forward
    import webview

    from pipeline_v2.capture import app

    # Every ffmpeg the app starts would otherwise open, and flash, a console.
    app.hide_child_consoles()
    running = server.start()
    port = running.server_address[1]
    port_file = desktop._port_file()
    port_file.parent.mkdir(parents=True, exist_ok=True)
    port_file.write_text(str(port), encoding="utf-8")
    window = webview.create_window(
        "Salience", f"http://127.0.0.1:{port}/" + desktop.target(session, view),
        width=1360, height=880, min_size=(980, 640), background_color="#0D0F13")

    def show(at: str, at_view: str = "session") -> None:
        window.restore()
        window.show()
        # Windows only lets a window that is already in front take focus;
        # on-top for a moment puts it there.
        window.on_top = True
        window.on_top = False
        if SESSION_ID.match(at or "") or at_view == "rate":
            hash_ = desktop.target(at if SESSION_ID.match(at or "") else "", at_view)
            window.evaluate_js(f"location.hash = '{hash_}'")

    def save_as(suggested: str) -> str | None:
        from pathlib import Path

        chosen = window.create_file_dialog(
            webview.FileDialog.SAVE, directory=str(Path.home() / "Videos"),
            save_filename=suggested, file_types=("MP4 video (*.mp4)",))
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0] if chosen else None
        return chosen or None

    server.Handler.show = staticmethod(show)
    server.Handler.save_as = staticmethod(save_as)
    try:
        webview.start(private_mode=False)
    finally:
        running.shutdown()
        if desktop._running_port() == port:
            port_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
