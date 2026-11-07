"""`python -m pipeline_v2.capture.desktop [session]`: the app window."""

from __future__ import annotations

import re
import sys

from pipeline_v2.capture import desktop
from pipeline_v2.capture.desktop import server

SESSION_ID = re.compile(r"^\d{8}-\d{6}$")


def _windows(title: str) -> list:
    """This process's top-level windows called `title`."""
    import ctypes
    import os
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        text = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, text, 256)
        if pid.value == os.getpid() and text.value == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found


def bring_forward(title: str) -> bool:
    """Restore and raise this process's window called `title`."""
    import ctypes

    found = _windows(title)
    for hwnd in found:
        ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    return bool(found)


def close(title: str) -> None:
    """Close this process's window as its X does; gone regardless after a few seconds."""
    import ctypes
    import os
    import threading

    for hwnd in _windows(title):
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    threading.Timer(5.0, lambda: os._exit(0)).start()  # a dialog open, say


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
    import os

    # The app is what the player opens; the recorder comes with it while
    # auto-clip is on (after a restart with Start with Windows off, say).
    if app.autoclip_on() and os.environ.get("FIREWORKS_API_KEY"):
        # Behind the window: one quitting may take seconds to let go.
        import threading

        threading.Thread(target=app.start_tray_when_free, name="start-recorder", daemon=True).start()
    port = running.server_address[1]
    desktop.write_running(port, server.Handler.secret)
    window = webview.create_window(
        "Salience", f"http://127.0.0.1:{port}/" + desktop.target(session, view),
        width=1360, height=880, min_size=(980, 640), background_color="#0D0F13")

    def show() -> None:
        # Plain Win32, never pywebview's own calls: from a thread other than
        # the window's, `on_top` waited forever and froze the window.
        bring_forward("Salience")

    def save_as(suggested: str) -> str | None:
        from pathlib import Path

        chosen = window.create_file_dialog(
            webview.FileDialog.SAVE, directory=str(Path.home() / "Videos"),
            save_filename=suggested, file_types=("MP4 video (*.mp4)",))
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0] if chosen else None
        return chosen or None

    def choose_videos() -> list[str]:
        from pathlib import Path

        chosen = window.create_file_dialog(
            webview.FileDialog.OPEN, directory=str(Path.home() / "Videos"), allow_multiple=True,
            file_types=("Videos (*.mp4;*.mkv;*.mov;*.webm;*.avi;*.m4v)",))
        return [str(c) for c in chosen or []]

    server.Handler.show = staticmethod(show)
    def choose_folder(start: str) -> str | None:
        from pathlib import Path

        chosen = window.create_file_dialog(webview.FileDialog.FOLDER, directory=start or str(Path.home()))
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0] if chosen else None
        return chosen or None

    server.Handler.choose_videos = staticmethod(choose_videos)
    server.Handler.choose_folder = staticmethod(choose_folder)
    server.Handler.quit = staticmethod(lambda: close("Salience"))
    server.Handler.save_as = staticmethod(save_as)
    try:
        webview.start(private_mode=False)
    finally:
        running.shutdown()
        if (desktop._running() or (None,))[0] == port:
            desktop._port_file().unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
