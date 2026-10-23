"""`python -m pipeline_v2.capture.desktop [session]`: the app window."""

from __future__ import annotations

import re
import sys

from pipeline_v2.capture import desktop
from pipeline_v2.capture.desktop import server

SESSION_ID = re.compile(r"^\d{8}-\d{6}$")


def main(argv: list[str]) -> int:
    session = argv[0] if argv and SESSION_ID.match(argv[0]) else ""
    if desktop.show_running(session):
        return 0  # one window; the open one has come forward
    import webview

    running = server.start()
    port = running.server_address[1]
    port_file = desktop._port_file()
    port_file.parent.mkdir(parents=True, exist_ok=True)
    port_file.write_text(str(port), encoding="utf-8")
    window = webview.create_window(
        "Salience", f"http://127.0.0.1:{port}/" + (f"#/session/{session}" if session else ""),
        width=1360, height=880, min_size=(980, 640), background_color="#0D0F13")

    def show(at: str) -> None:
        window.restore()
        window.show()
        # Windows only lets a window that is already in front take focus;
        # on-top for a moment puts it there.
        window.on_top = True
        window.on_top = False
        if SESSION_ID.match(at or ""):
            window.evaluate_js(f"location.hash = '#/session/{at}'")

    server.Handler.show = staticmethod(show)
    try:
        webview.start(private_mode=False)
    finally:
        running.shutdown()
        if desktop._running_port() == port:
            port_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
