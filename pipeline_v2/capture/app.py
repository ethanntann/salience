"""The tray app: records Fortnite on its own and hands back highlights.

    pythonw -m pipeline_v2.capture.app        (no console; `python -m` shows the log)

It waits in the tray. When the Fortnite window appears it starts recording,
when the game has quit (or its window has been gone `GRACE` seconds) it stops, makes the
highlights and says so. The player never opens a terminal or presses Ctrl+C,
which is what made session 3 lose its highlights step: a second Ctrl+C
during a silent stop cancelled it.

Settings live in %APPDATA%\\Salience\\settings.env, written by the first-run
dialog: the player's own Fireworks key (bring your own; a key shipped inside
the app could be pulled out and spent by anyone). A developer `.env` in the
repo still applies, and wins over the product defaults below.

`Controller` decides when to record and holds no UI, so it is tested without
a tray or a game. `main` is the thin pystray shell around it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / "Salience"
SETTINGS = APP_DIR / "settings.env"
LOG = APP_DIR / "salience.log"
REVIEW_PORT = 8777

# What a player gets without a developer .env: the describer, prompt and prices
# measured best (README, "Measured"). The budget is a runaway guard on the
# spend this machine has recorded, not a monthly cap.
PRODUCT_DEFAULTS = {
    "FIREWORKS_MODEL": "accounts/fireworks/models/deepseek-v4p1-flash",
    "V2_PRICE_MODEL": "deepseek-v4p1-flash",
    "V2_INPUT_PRICE_PER_MTOK": "0.30",
    "V2_OUTPUT_PRICE_PER_MTOK": "1.20",
    "V2_CACHED_INPUT_PRICE_PER_MTOK": "0.30",
    "V2_ABLATE": "react",
    "V2_BUDGET_USD": "100",
}

# Seconds the game window may be missing before the session ends. Fortnite
# recreates its window on some resolution changes, which should not split a
# session in two.
GRACE = 15.0
TICK = 0.5


def read_settings(path: Path = SETTINGS) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            found[key.strip()] = value.strip()
    return found


def write_setting(key: str, value: str, path: Path = SETTINGS) -> None:
    settings = read_settings(path)
    settings[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in settings.items()), encoding="utf-8")


def load_environment(path: Path = SETTINGS) -> None:
    """Player settings, then the repo .env (via `config`), then product defaults.

    Each only fills what is still unset, so the order is the precedence.
    """
    for key, value in read_settings(path).items():
        os.environ.setdefault(key, value)
    from pipeline_v2 import config  # noqa: F401 -- loads the repo .env on import

    for key, value in PRODUCT_DEFAULTS.items():
        os.environ.setdefault(key, value)


def check_key(key: str, base_url: str = "https://api.fireworks.ai/inference/v1") -> str | None:
    """None if Fireworks accepts the key, else why not."""
    import urllib.error
    import urllib.request

    from pipeline_v2.vlm import USER_AGENT

    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20):
            return None
    except urllib.error.HTTPError as exc:
        return "Fireworks rejected that key" if exc.code in (401, 403) else f"HTTP {exc.code}"
    except OSError as exc:
        return f"could not reach Fireworks ({exc})"


class Controller:
    """When to record, when to stop, and making the highlights afterwards.

    `tick` is called every `TICK` seconds from one worker thread; everything
    else only sets flags, so the tray's menu callbacks never race it.
    """

    def __init__(self, *, find_game, start_session, make_highlights, notify,
                 clock=time.monotonic, grace: float = GRACE, game_running=lambda: True):
        self.find_game = find_game
        self.game_running = game_running
        self.start_session = start_session
        self.make_highlights = make_highlights
        self.notify = notify
        self.clock = clock
        self.grace = grace
        self.state = "waiting"  # waiting | recording | highlights
        self.paused = False
        self.session = None
        self.last_session: Path | None = None
        self._gone_since: float | None = None
        self._stop_requested = False
        self._warned: str | None = None

    @property
    def status(self) -> str:
        if self.paused and self.state == "waiting":
            return "Paused (not recording)"
        return {
            "waiting": "Waiting for Fortnite",
            "recording": "Recording",
            "highlights": "Making highlights...",
        }[self.state]

    def request_stop(self) -> None:
        self._stop_requested = True

    def set_paused(self, paused: bool) -> None:
        """Paused: no new session starts. Pausing mid-session ends it."""
        self.paused = paused
        if paused and self.state == "recording":
            self.request_stop()

    def tick(self) -> None:
        if self.state == "waiting":
            if self.paused:
                return
            hwnd = self.find_game()
            if hwnd is not None:
                self.session = self.start_session(hwnd)
                self.last_session = self.session.path
                self._gone_since, self._stop_requested = None, False
                self.state = "recording"
                self.notify("Recording", "Salience is recording this session.")
            return
        if self.state != "recording":
            return
        self.session.tick()
        warning = getattr(self.session, "warning", None)
        if warning and warning != self._warned:
            self._warned = warning
            self.notify("Check your audio", warning)
        if self.session.error:
            self._finish(failed=True)
            return
        if self.find_game() is None:
            self._gone_since = self._gone_since or self.clock()
        else:
            self._gone_since = None
        # The grace is for a window the game recreates; once the game itself
        # has exited there is nothing to wait for, and the player is waiting.
        gone = self._gone_since is not None and (
            self.clock() - self._gone_since >= self.grace or not self.game_running())
        if gone or self._stop_requested:
            self._finish()

    def _finish(self, failed: bool = False) -> None:
        session = self.session
        session.stop()
        self.session = None
        if failed:
            self.notify("Recording stopped", f"Capture failed: {session.error}")
            self.state = "waiting"
            return
        self.state = "highlights"
        # At once, so quitting the game is answered before the work is done.
        self.notify("Making highlights", "Your highlights will open in a few seconds.")
        try:
            manifest = self.make_highlights(session.path)
        except Exception as exc:  # noqa: BLE001 -- report it, keep the app alive
            self.notify("Highlights failed", f"{type(exc).__name__}: {exc}")
        else:
            count = len(manifest.get("highlights", []))
            self.notify("Highlights ready",
                        f"{count} highlights from this session." if count
                        else manifest.get("why", "No highlights this time."))
        self.state = "waiting"


def latest_session(root: Path) -> Path | None:
    """The newest recording folder, so the menu works before this run records."""
    try:
        found = [p for p in root.iterdir() if p.is_dir() and (p / "segments.csv").exists()]
    except OSError:
        return None
    return max(found, key=lambda p: p.name, default=None)


def _recording_args():
    from pipeline_v2.capture import record

    args = record._parser().parse_args([])
    args.mic = os.environ.get("SALIENCE_MIC", "")  # "" = Windows' default mic
    return args


def _single_instance() -> bool:
    """False if another copy is already running (a named mutex per user)."""
    if os.name != "nt":
        return True
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\SalienceTray")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def ask_for_key(message: str = "") -> str | None:
    """A small dialog for the player's Fireworks key; None if cancelled."""
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        prompt = (message + "\n\n" if message else "") + (
            "Paste your Fireworks API key (fireworks.ai -> API Keys).\n"
            "It stays on this PC and is only sent to Fireworks.")
        key = simpledialog.askstring("Salience", prompt, parent=root, show="*")
    finally:
        root.destroy()
    return key.strip() if key else None


def setup_key(ask=ask_for_key, check=check_key) -> bool:
    """Ask until Fireworks accepts a key or the player cancels."""
    message = ""
    while True:
        key = ask(message)
        if not key:
            return False
        problem = check(key)
        if problem is None:
            write_setting("FIREWORKS_API_KEY", key)
            os.environ["FIREWORKS_API_KEY"] = key
            return True
        message = problem


_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def starts_with_windows() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, "Salience")
            return True
    except OSError:
        return False


def set_starts_with_windows(enabled: bool) -> None:
    import winreg

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    command = f'"{pythonw if pythonw.exists() else sys.executable}" -m pipeline_v2.capture.app'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, "Salience", 0, winreg.REG_SZ, command)
        else:
            try:
                winreg.DeleteValue(key, "Salience")
            except OSError:
                pass


def _icon_image(recording: bool):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill=(24, 28, 36, 255))
    draw.polygon([(24, 18), (24, 46), (46, 32)], fill=(230, 232, 235, 255))
    if recording:
        draw.ellipse((40, 4, 60, 24), fill=(232, 64, 64, 255))
    return image


class _Review:
    """The rating page, served once and pointed at the latest session."""

    def __init__(self):
        self._thread: threading.Thread | None = None

    def open(self, session: Path | None) -> None:
        from pipeline_v2 import review
        from pipeline_v2.capture import highlights

        variant = highlights._configured_variant()
        if self._thread is None:
            self._thread = threading.Thread(
                target=review.serve, kwargs=dict(port=REVIEW_PORT, variant=variant, only=session),
                name="review", daemon=True)
            self._thread.start()
            time.sleep(0.5)
        else:
            review._VARIANT = variant
            review._ONLY = str(session.resolve()) if session else None
        webbrowser.open(f"http://localhost:{REVIEW_PORT}/")


def hide_child_consoles() -> None:
    """Start every child process without a console window.

    The tray app has no console, so each ffmpeg it starts got a new one: a
    window flashing up every few seconds, taking focus from the game and
    freezing it (session 4's first try). Run from a terminal, the children
    shared that terminal and nothing showed, which is why no earlier session
    hit it. Done once for the whole process, so no call site can miss it.
    """
    import subprocess

    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not flag or getattr(subprocess.Popen, "_salience_hidden", False):
        return
    original = subprocess.Popen.__init__

    def __init__(self, *args, creationflags=0, **kwargs):
        original(self, *args, creationflags=creationflags | flag, **kwargs)

    subprocess.Popen.__init__ = __init__
    subprocess.Popen._salience_hidden = True


def main() -> int:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:  # pythonw: no console, so keep the log on disk
        sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
    hide_child_consoles()
    if not _single_instance():
        return 0
    load_environment()
    if not os.environ.get("FIREWORKS_API_KEY") and not setup_key():
        return 1

    import pystray

    from pipeline_v2.capture import highlights, record
    from pipeline_v2.capture.windows import game_running, game_window

    icon = pystray.Icon("salience", _icon_image(False), "Salience")

    def notify(title: str, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {title}: {message}", flush=True)
        try:
            icon.notify(message, title)
        except Exception:  # noqa: BLE001 -- a missed toast is not worth a crash
            pass

    def start_session(hwnd):
        return record.Session(_recording_args(), hwnd).start()

    def make_highlights(session: Path) -> dict:
        manifest = highlights.run(session, top=10)
        highlights.report(manifest)
        out = session / "highlights"
        if out.is_dir() and hasattr(os, "startfile"):
            os.startfile(out)  # noqa: S606 -- the player's own folder
        return manifest

    controller = Controller(find_game=game_window, start_session=start_session,
                            make_highlights=make_highlights, notify=notify,
                            game_running=game_running)
    controller.last_session = latest_session(_recording_args().out)
    review = _Review()
    quitting = threading.Event()

    def worker() -> None:
        shown = None
        while not quitting.is_set():
            try:
                controller.tick()
            except Exception as exc:  # noqa: BLE001 -- the tray must outlive a bad tick
                print(f"tick failed: {type(exc).__name__}: {exc}", flush=True)
            if (controller.state, controller.paused) != shown:
                shown = (controller.state, controller.paused)
                icon.icon = _icon_image(controller.state == "recording")
                icon.title = f"Salience: {controller.status}"
                icon.update_menu()
            quitting.wait(TICK)
        if controller.session is not None:
            controller.session.stop()  # quitting mid-session keeps the footage

    def open_folder(path: Path | None) -> None:
        if path is not None and path.exists():
            os.startfile(path)  # noqa: S606

    def change_key(*_):
        threading.Thread(target=setup_key, daemon=True).start()

    def quit_app(*_):
        quitting.set()
        icon.stop()

    def toggle_start(*_):
        set_starts_with_windows(not starts_with_windows())

    def toggle_pause(*_):
        controller.set_paused(not controller.paused)

    menu = pystray.Menu(
        pystray.MenuItem(lambda _: controller.status, None, enabled=False),
        pystray.MenuItem("Stop recording now", lambda *_: controller.request_stop(),
                         visible=lambda _: controller.state == "recording"),
        pystray.MenuItem("Pause auto-recording", toggle_pause,
                         checked=lambda _: controller.paused),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open latest highlights",
                         lambda *_: open_folder(
                             controller.last_session and controller.last_session / "highlights"),
                         default=True),
        pystray.MenuItem("Rate latest session", lambda *_: review.open(controller.last_session)),
        pystray.MenuItem("Open recordings folder",
                         lambda *_: open_folder(_recording_args().out)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Change Fireworks key...", change_key),
        pystray.MenuItem("Start with Windows", toggle_start,
                         checked=lambda _: starts_with_windows()),
        pystray.MenuItem("Quit", quit_app),
    )
    icon.menu = menu
    threading.Thread(target=worker, name="salience-worker", daemon=True).start()
    icon.run()
    quitting.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
