"""The tray app's decisions: when to record, when to stop, the player's key."""

from __future__ import annotations

from pathlib import Path

from pipeline_v2.capture import app


class _Session:
    def __init__(self, path: Path):
        self.path = path
        self.error = None
        self.ticks = 0
        self.stopped = False

    def tick(self):
        self.ticks += 1

    def stop(self):
        self.stopped = True


class _World:
    """A game window that comes and goes, a clock, and what was shown."""

    def __init__(self, tmp_path: Path):
        self.window = None
        self.now = 0.0
        self.sessions: list[_Session] = []
        self.made: list[Path] = []
        self.toasts: list[str] = []
        self.tmp = tmp_path

    def controller(self, **kwargs) -> app.Controller:
        def start(hwnd):
            session = _Session(self.tmp / f"s{len(self.sessions)}")
            self.sessions.append(session)
            return session

        def highlights(path):
            self.made.append(path)
            return {"highlights": [1, 2, 3]}

        return app.Controller(
            find_game=lambda: self.window, start_session=start,
            make_highlights=highlights, notify=lambda title, _: self.toasts.append(title),
            clock=lambda: self.now, **kwargs)


def _run(controller, world, seconds: float, step: float = 0.5):
    end = world.now + seconds
    while world.now < end:
        controller.tick()
        world.now += step


def test_records_while_the_game_is_open_and_makes_highlights_after(tmp_path):
    world = _World(tmp_path)
    controller = world.controller(grace=10)

    _run(controller, world, 5)
    assert world.sessions == []  # no game yet

    world.window = 1234
    _run(controller, world, 60)
    assert controller.state == "recording" and world.sessions[0].ticks > 100

    world.window = None
    _run(controller, world, 20)
    assert world.sessions[0].stopped and world.made == [world.sessions[0].path]
    assert controller.state == "waiting"
    assert world.toasts == ["Recording", "Making highlights", "Highlights ready"]


def test_a_window_that_blinks_out_does_not_split_the_session(tmp_path):
    """Fortnite recreates its window on some resolution changes."""
    world = _World(tmp_path)
    controller = world.controller(grace=10)
    world.window = 1
    _run(controller, world, 30)
    world.window = None
    _run(controller, world, 5)
    world.window = 1
    _run(controller, world, 30)

    assert len(world.sessions) == 1 and not world.sessions[0].stopped


def test_once_the_game_has_quit_highlights_start_without_the_grace(tmp_path):
    """The grace is for a recreated window; a closed game has nothing to wait for."""
    world = _World(tmp_path)
    running = [True]
    controller = world.controller(grace=15, game_running=lambda: running[0])
    world.window = 1
    _run(controller, world, 30)

    world.window, running[0] = None, False
    controller.tick()

    assert world.sessions[0].stopped and world.made == [world.sessions[0].path]


def test_stop_now_ends_the_session_while_the_game_is_still_open(tmp_path):
    world = _World(tmp_path)
    controller = world.controller()
    world.window = 1
    _run(controller, world, 5)

    controller.request_stop()
    controller.tick()
    assert world.sessions[0].stopped and world.made

    # The game is still open, so the next look starts a new session: "stop"
    # ends this recording, pausing is what stops recording altogether.
    controller.tick()
    assert len(world.sessions) == 2


def test_paused_never_starts_and_pausing_mid_session_ends_it(tmp_path):
    world = _World(tmp_path)
    controller = world.controller()
    world.window = 1
    _run(controller, world, 5)

    controller.set_paused(True)
    _run(controller, world, 30)

    assert len(world.sessions) == 1 and world.sessions[0].stopped
    assert controller.status == "Paused (not recording)"
    controller.set_paused(False)
    controller.tick()
    assert len(world.sessions) == 2


def test_a_capture_failure_is_reported_and_makes_no_highlights(tmp_path):
    world = _World(tmp_path)
    controller = world.controller()
    world.window = 1
    controller.tick()
    world.sessions[0].error = "encoder died"

    controller.tick()

    assert world.made == [] and world.toasts[-1] == "Recording stopped"
    assert controller.state == "waiting"


def test_a_highlights_crash_keeps_the_app_alive(tmp_path):
    world = _World(tmp_path)
    controller = world.controller()

    def broken(path):
        raise RuntimeError("disk full")

    controller.make_highlights = broken
    world.window = 1
    controller.tick()
    controller.request_stop()
    controller.tick()

    assert world.toasts[-1] == "Highlights failed" and controller.state == "waiting"


def test_settings_round_trip_and_keep_other_keys(tmp_path):
    path = tmp_path / "settings.env"
    app.write_setting("SALIENCE_MIC", "HyperX", path)
    app.write_setting("FIREWORKS_API_KEY", "fw_one", path)
    app.write_setting("FIREWORKS_API_KEY", "fw_two", path)

    assert app.read_settings(path) == {"SALIENCE_MIC": "HyperX", "FIREWORKS_API_KEY": "fw_two"}


def test_key_setup_retries_until_fireworks_accepts(monkeypatch):
    saved = {}
    monkeypatch.setattr(app, "write_setting", lambda key, value: saved.update({key: value}))
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    asked, messages = iter(["fw_bad", "fw_good"]), []

    def ask(message):
        messages.append(message)
        return next(asked)

    ok = app.setup_key(ask=ask, check=lambda key: None if key == "fw_good" else "rejected")

    assert ok and messages == ["", "rejected"]  # told why, then asked again
    assert saved == {"FIREWORKS_API_KEY": "fw_good"}
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


def test_cancelling_key_setup_gives_up(monkeypatch):
    assert app.setup_key(ask=lambda message: None, check=lambda key: None) is False


def test_latest_session_is_the_newest_recording(tmp_path):
    for name in ("20261014-175600", "20261018-172215", "notes"):
        (tmp_path / name).mkdir()
    for name in ("20261014-175600", "20261018-172215"):
        (tmp_path / name / "segments.csv").write_text("", encoding="utf-8")

    assert app.latest_session(tmp_path).name == "20261018-172215"


def test_a_recording_warning_is_shown_once(tmp_path):
    world = _World(tmp_path)
    controller = world.controller()
    world.window = 1
    controller.tick()
    world.sessions[0].warning = "Game audio is silent."
    controller.tick()
    controller.tick()

    assert world.toasts.count("Check your audio") == 1


def test_child_processes_get_no_console_window(monkeypatch):
    """Under pythonw every ffmpeg opened a console that stole focus from the game."""
    import subprocess

    seen = []

    class Popen:
        def __init__(self, *args, creationflags=0, **kwargs):
            seen.append(creationflags)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    app.hide_child_consoles()

    subprocess.Popen(["ffmpeg"])
    subprocess.Popen(["ffmpeg"], creationflags=0x4000)  # below normal priority, kept

    assert seen == [0x08000000, 0x08000000 | 0x4000]


def test_recording_quality_defaults_to_60fps_and_follows_the_menu(tmp_path, monkeypatch):
    monkeypatch.delenv(app.QUALITY_KEY, raising=False)
    args = app._recording_args()
    assert (args.fps, args.bitrate) == (60, "25M")
    app.set_quality("standard", tmp_path / "settings.env")
    assert app.read_settings(tmp_path / "settings.env")[app.QUALITY_KEY] == "standard"
    args = app._recording_args()
    assert (args.fps, args.bitrate) == (30, "8M")
    monkeypatch.setenv(app.QUALITY_KEY, "nonsense")  # a hand-edited settings file
    assert app.quality() == "high"
    monkeypatch.delenv(app.QUALITY_KEY)


def test_recording_resolution_defaults_to_full_size_and_follows_the_menu(tmp_path, monkeypatch):
    monkeypatch.delenv(app.RESOLUTION_KEY, raising=False)
    assert app._recording_args().height is None
    app.set_resolution("720", tmp_path / "settings.env")
    assert app.read_settings(tmp_path / "settings.env")[app.RESOLUTION_KEY] == "720"
    assert app._recording_args().height == 720
    monkeypatch.setenv(app.RESOLUTION_KEY, "4k")  # a hand-edited settings file
    assert app.resolution() == "full"
    monkeypatch.delenv(app.RESOLUTION_KEY)


def test_autoclip_is_on_until_turned_off_and_off_asks_the_recorder_to_finish(tmp_path, monkeypatch):
    monkeypatch.delenv(app.AUTOCLIP_KEY, raising=False)
    monkeypatch.setattr(app, "STOP_REQUEST", tmp_path / "stop.request")
    started = []
    monkeypatch.setattr(app, "start_tray", lambda: started.append(1))
    monkeypatch.setattr(app, "tray_running", lambda: True)
    assert app.autoclip_on()  # the tray ran before the switch existed

    app.set_autoclip(False, tmp_path / "settings.env")
    assert not app.autoclip_on() and (tmp_path / "stop.request").is_file()
    assert app.read_settings(tmp_path / "settings.env")[app.AUTOCLIP_KEY] == "off"

    monkeypatch.setattr(app, "tray_running", lambda: False)
    app.set_autoclip(True, tmp_path / "settings.env")
    assert app.autoclip_on() and started == [1]
    assert not (tmp_path / "stop.request").exists()  # a stale request must not stop the new one
    monkeypatch.delenv(app.AUTOCLIP_KEY)


def test_a_running_tray_records_with_what_the_app_set_since(tmp_path, monkeypatch):
    """`load_environment` only fills gaps; settings changed by the app must win."""
    monkeypatch.setenv(app.QUALITY_KEY, "high")
    app.write_setting(app.QUALITY_KEY, "max", tmp_path / "settings.env")
    app.reload_settings(tmp_path / "settings.env")
    assert app.quality() == "max"
    monkeypatch.delenv(app.QUALITY_KEY)


def test_the_recorders_status_reaches_the_app(tmp_path):
    app.write_status("recording", "Recording", 1000.0, "20261022-184935", tmp_path / "status.json")
    status = app.read_status(tmp_path / "status.json")
    assert (status["state"], status["started"], status["session"]) == ("recording", 1000.0, "20261022-184935")
    assert app.read_status(tmp_path / "missing.json") == {}
