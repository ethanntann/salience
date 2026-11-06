"""Describing during play: only what the final run would describe, the same way."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from pipeline_v2 import cli, vlm
from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import live


def _segments(session: Path, ends: list[float]) -> None:
    session.mkdir(parents=True, exist_ok=True)
    rows, start = [], 0.0
    for index, end in enumerate(ends):
        part = session / f"part-{index:05d}.mkv"
        part.write_bytes(b"x")
        rows.append(f"{part.name},{start},{end}")
        start = end
    (session / "segments.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _gameplay(monkeypatch, spans):
    monkeypatch.setattr(live.gameplay_mod, "spans",
                        lambda session, duration: {"spans": spans})


def test_only_fully_recorded_windows_inside_a_match_are_ready(tmp_path, monkeypatch):
    _segments(tmp_path, [30.0, 60.0])
    _gameplay(monkeypatch, [(20.0, 1e9)])

    ready, _ = live.ready_windows(tmp_path)

    # Windows tile [0,8), [8,16) ... [56,64). Before 20s is the lobby; the last
    # would run past the 60s of footage on disk, so it waits for the next segment.
    assert [r.at for r in ready] == [20.0, 28.0, 36.0, 44.0, 52.0]


def test_nothing_is_ready_until_the_log_places_a_match(tmp_path, monkeypatch):
    """No log yet, or no match in it: left to the final run, never guessed."""
    _segments(tmp_path, [30.0])
    monkeypatch.setattr(live.gameplay_mod, "spans",
                        lambda session, duration: {"spans": None, "why": "no match"})

    assert live.ready_windows(tmp_path)[0] == []


class _Conn:
    def execute(self, sql):
        return []

    def close(self):
        pass


def test_a_step_cuts_ready_windows_and_describes_each_once(tmp_path, monkeypatch):
    _segments(tmp_path, [30.0])
    _gameplay(monkeypatch, [(0.0, 1e9)])
    monkeypatch.setattr("pipeline_v2.cache.clip_sha", lambda path: path.name)
    asked: list[set] = []

    def harvest(session, clip_dir, *, skip, **_):
        requests = clips_mod.coverage_requests(clips_mod.read_segments(session))
        wanted = [(i, r) for i, r in enumerate(requests) if not skip(r)]
        asked.append({r.at for _, r in wanted})
        clip_dir.mkdir(exist_ok=True)
        for i, _ in wanted:
            (clip_dir / f"{i:03d}-coverage.mkv").write_bytes(b"w")
        return [{"video_file": f"{i:03d}-coverage.mkv"} for i, _ in wanted]

    def describe(cfg, paths, max_calls, workers):
        for path in paths:
            yield path, (8.0, None, None)

    stored: list[str] = []
    seen: set = set()
    kwargs = dict(
        seen=seen, cfg=SimpleNamespace(), harvest_fn=harvest, describe_fn=describe,
        store_fn=lambda conn, cfg, clip, *rest: stored.append(clip.name), connect=_Conn)

    first = live.step(tmp_path, **kwargs)
    again = live.step(tmp_path, **kwargs)

    # Windows at 4, 12 and 20s; the one at 28s runs past the 30s on disk.
    assert (first, again) == (3, 0)
    assert stored == ["000-coverage.mkv", "001-coverage.mkv", "002-coverage.mkv"]
    assert asked == [{4.0, 12.0, 20.0}]  # the second step had nothing new to cut


def test_a_window_waits_until_all_of_it_is_transcribed(tmp_path, monkeypatch):
    """Mid-match nothing is transcribed, so a window not yet heard is held, not
    described without its speech, and comes up again once it is heard."""
    _segments(tmp_path, [30.0, 60.0])
    _gameplay(monkeypatch, [(0.0, 1e9)])
    monkeypatch.setattr("pipeline_v2.cache.clip_sha", lambda path: path.name)
    cut: list[set] = []

    def harvest(session, clip_dir, *, skip, **_):
        requests = clips_mod.coverage_requests(clips_mod.read_segments(session))
        wanted = [(i, r) for i, r in enumerate(requests) if not skip(r)]
        cut.append({r.at for _, r in wanted})
        clip_dir.mkdir(exist_ok=True)
        for i, _ in wanted:
            (clip_dir / f"{i:03d}-coverage.mkv").write_bytes(b"w")
        return [{"video_file": f"{i:03d}-coverage.mkv"} for i, _ in wanted]

    def describe(cfg, paths, max_calls, workers):
        for path in paths:
            yield path, (8.0, None, None)

    heard = [(0.0, 30.0)]
    attached: list[list[str]] = []
    seen: set = set()
    kwargs = dict(
        seen=seen, cfg=SimpleNamespace(), harvest_fn=harvest, describe_fn=describe,
        store_fn=lambda *a: None, connect=_Conn, heard=lambda s: heard,
        listen=lambda s, windows: attached.append([w.name for w in windows]))

    first = live.step(tmp_path, **kwargs)
    heard.append((30.0, 60.0))  # the lobby transcribed the second segment
    second = live.step(tmp_path, **kwargs)

    # 4, 12, 20 lie in the first segment; 28 straddles both; 36, 44, 52 the second.
    assert cut == [{4.0, 12.0, 20.0}, {28.0, 36.0, 44.0, 52.0}]
    assert (first, second) == (3, 4)
    assert len(attached) == 2


def test_speech_never_holds_up_describing(tmp_path):
    """Its own thread: one look of speech mid-match took ten minutes, and
    nothing was described during play (2026-10-27)."""
    stepped, release = threading.Event(), threading.Event()
    passed: list[dict] = []

    def step(session, seen, **kw):
        passed.append(kw)
        stepped.set()
        return 0

    describer = live.LiveDescriber(
        tmp_path, interval=0.01, step_fn=step, idle_fn=lambda s: True,
        laugh_fn=lambda s, holder: 0, listen_fn=lambda s, holder: release.wait(20) and 0).start()
    assert stepped.wait(10.0)  # the loop imports sklearn first (~2s, more on a busy PC)
    release.set()
    describer.stop()

    assert "heard" not in passed[0] and passed[0]["describe_now"]  # nothing waits to be heard


def test_without_speech_windows_are_not_held(tmp_path):
    passed: list[dict] = []
    done = threading.Event()

    def step(session, seen, **kw):
        passed.append(kw)
        done.set()
        return 0

    describer = live.LiveDescriber(tmp_path, interval=0.01, step_fn=step, speech=False,
                                   idle_fn=lambda s: False).start()
    assert done.wait(10.0)  # the loop imports sklearn first (~2s, more on a busy PC)
    describer.stop()

    assert "heard" not in passed[0] and "listen" not in passed[0]


def test_a_failing_step_never_stops_the_recording(tmp_path):
    calls = threading.Event()

    def broken(session, seen, **kw):
        calls.set()
        raise RuntimeError("network down")

    describer = live.LiveDescriber(tmp_path, interval=0.01, step_fn=broken).start()
    assert calls.wait(2.0)
    describer.stop()

    assert describer.error == "RuntimeError: network down"


def test_priority_is_lowered_only_while_live(tmp_path):
    from salience_media import ffmpeg

    describer = live.LiveDescriber(tmp_path, interval=60, step_fn=lambda s, seen, **kw: 0).start()
    during = ffmpeg.LOW_PRIORITY
    describer.stop()

    assert during is True and ffmpeg.LOW_PRIORITY is False


# describe_many: the parallel describer both paths use.

def _fake_describe(monkeypatch, cached: set[str], paid: list[str]):
    lock = threading.Lock()
    monkeypatch.setattr(cli, "_prepare",
                        lambda clip: (8.0, None, [SimpleNamespace(path=clip)]))

    def describe(cfg, clip, facts, frames, *, allow_call=True, **_):
        if clip.name in cached:
            return SimpleNamespace(cached=True, name=clip.name)
        if not allow_call:
            raise vlm.BudgetExceeded("call limit reached before this clip")
        with lock:
            paid.append(clip.name)
        return SimpleNamespace(cached=False, name=clip.name)

    monkeypatch.setattr(cli.vlm, "describe", describe)


def test_describe_many_keeps_order_and_never_pays_past_the_limit(monkeypatch):
    paid: list[str] = []
    _fake_describe(monkeypatch, cached={"c1", "c3"}, paid=paid)
    clips = [Path(f"c{i}") for i in range(6)]

    out = list(cli.describe_many(None, clips, max_calls=2, workers=6))

    assert [c.name for c, _ in out] == [c.name for c in clips]
    assert len(paid) == 2
    budgeted = [c.name for c, o in out if isinstance(o, vlm.BudgetExceeded)]
    # Cached answers are free and never count against the limit.
    assert {"c1", "c3"}.isdisjoint(budgeted) and len(budgeted) == 2


def test_a_windows_frames_are_deleted_once_it_is_described(monkeypatch):
    """They exist only to be sent: kept, 2.8 GB of JPEGs."""
    from pipeline_v2 import config

    frames = config.FRAME_DIR / "somesha" / "window"
    frames.mkdir(parents=True)
    (frames / "window-000.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(cli, "_prepare", lambda clip: (
        8.0, None, [SimpleNamespace(path=frames / "window-000.jpg")]))
    monkeypatch.setattr(cli.vlm, "describe", lambda *a, **k: SimpleNamespace(cached=True))

    list(cli.describe_many(None, [Path("c0")], max_calls=1, workers=1))

    assert not (config.FRAME_DIR / "somesha").exists()
    assert config.FRAME_DIR.is_dir()


def test_speech_waits_for_the_game_to_close_unless_asked_for_during_it(tmp_path, monkeypatch):
    """Transcribing the lobby cost the 1% lows (2026-11-05); nothing waits on it any more."""
    monkeypatch.setenv(live.MATCH_SPEECH_KEY, "off")
    listened: list = []
    describer = live.LiveDescriber(
        tmp_path, interval=0.01, step_fn=lambda s, seen, **kw: 0, laugh_fn=lambda s, holder: 0,
        idle_fn=lambda s: True, listen_fn=lambda s, holder: listened.append(1) or 0)
    describer._thread = threading.Thread(target=lambda: None)
    describer.start()
    time.sleep(0.2)
    describer.stop()
    assert listened == [] and not describer._listener.is_alive()


def test_speech_is_allowed_between_matches_and_in_them_by_schedule_d():
    allowed = live.speech_allowed
    assert allowed("off", True, None) and not allowed("off", False, 0) and not allowed("off", False, None)
    assert allowed("on", False, 3)
    assert allowed("alternate", False, 0) and not allowed("alternate", False, 1)
    assert allowed("alternate", False, None)  # before the log knows of a match: may be one


def test_threads_a_model_starts_are_put_at_the_lowest_priority_on_the_efficiency_cores(monkeypatch):
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    ready, release, seen = threading.Event(), threading.Event(), {}

    def worker():
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentThread.restype = ctypes.c_void_p
        kernel.GetThreadPriority.argtypes = [ctypes.c_void_p]
        ready.set()
        release.wait(10)
        seen["priority"] = kernel.GetThreadPriority(kernel.GetCurrentThread())
        kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
        kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        seen["cores"] = kernel.SetThreadAffinityMask(kernel.GetCurrentThread(), 0b11)  # returns the mask it had

    def load():  # as onnxruntime makes its pool
        thread = threading.Thread(target=worker)
        thread.start()
        ready.wait(10)
        return thread

    monkeypatch.setattr(live, "_efficiency_cores", lambda: 0b10)  # the second core "efficient"
    thread = live.load_lowest(load)
    release.set()
    thread.join(10)
    assert seen["priority"] == live.THREAD_PRIORITY_LOWEST
    assert seen["cores"] == 0b10  # and kept on the efficiency cores


def test_newest_footage_in_the_lobby_counts_as_between_matches(tmp_path, monkeypatch):
    _segments(tmp_path, [30.0, 60.0, 90.0])
    _gameplay(monkeypatch, [(0.0, 40.0)])  # died at 40s; now spectating

    assert live.between_matches(tmp_path) is True


def test_mid_match_and_unknown_are_not_between_matches(tmp_path, monkeypatch):
    _segments(tmp_path, [30.0, 60.0])
    _gameplay(monkeypatch, [(10.0, 1e9)])
    assert live.between_matches(tmp_path) is False

    monkeypatch.setattr(live.gameplay_mod, "spans",
                        lambda session, duration: {"spans": None})
    assert live.between_matches(tmp_path) is False  # no log looks like this too


def test_without_the_speech_model_live_stops_asking(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def no_whisper(name, *args, **kwargs):
        if name == "pipeline_v2.audio":
            raise ImportError("onnx_asr")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_whisper)

    assert live.listen_step(tmp_path, {}) is None

def test_stopping_early_never_runs_the_queue(monkeypatch):
    paid: list[str] = []
    _fake_describe(monkeypatch, cached=set(), paid=paid)
    clips = [Path(f"c{i}") for i in range(20)]

    for _ in cli.describe_many(None, clips, max_calls=20, workers=1):
        break

    assert len(paid) <= 2  # the one in flight (and at most one picked up next)


def test_a_slow_look_is_not_followed_by_a_whole_interval_of_waiting(tmp_path):
    import time as time_mod

    starts = []

    def slow(session, seen, **kw):
        starts.append(time_mod.monotonic())
        time_mod.sleep(0.15)
        return 0

    describer = live.LiveDescriber(tmp_path, interval=0.2, step_fn=slow,
                                   idle_fn=lambda s: False).start()
    time_mod.sleep(1.0)
    describer.stop()

    gaps = [b - a for a, b in zip(starts, starts[1:])]
    # Looks start every 0.2s; waiting the full interval after each was 0.35s.
    assert gaps and max(gaps) < 0.3


def test_mid_match_every_window_is_cut_and_prepared_and_described_once_heard(tmp_path, monkeypatch):
    """Cutting and the cheap work happen while playing; the wait after was 2.5
    minutes of cutting alone (2026-10-25). Describing still waits for speech."""
    import json

    _segments(tmp_path, [30.0, 60.0])
    _gameplay(monkeypatch, [(0.0, 1e9)])
    monkeypatch.setattr("pipeline_v2.cache.clip_sha", lambda path: path.name)
    harvested: list[set] = []

    def harvest(session, clip_dir, *, skip, **_):
        requests = clips_mod.coverage_requests(clips_mod.read_segments(session))
        wanted = [(i, r) for i, r in enumerate(requests) if not skip(r)]
        harvested.append({r.at for _, r in wanted})
        clip_dir.mkdir(exist_ok=True)
        out = []
        for i, r in wanted:
            clip = clip_dir / f"{i:03d}-coverage.mkv"
            clip.write_bytes(b"w")
            side = {"video_file": clip.name, "segment_start_sec": r.at - 4.0, "segment_end_sec": r.at + 4.0}
            clip.with_suffix(".capture.json").write_text(json.dumps(side), encoding="utf-8")
            out.append(side)
        return out

    described: list[str] = []

    def describe(cfg, paths, max_calls, workers):
        for path in paths:
            described.append(path.name)
            yield path, (8.0, None, None)

    prepared_paths: list[str] = []
    heard: list[tuple] = []
    prepared: set = set()
    kwargs = dict(seen=set(), cfg=SimpleNamespace(), harvest_fn=harvest, describe_fn=describe,
                  store_fn=lambda *a: None, connect=_Conn, heard=lambda s: heard,
                  listen=lambda s, windows: None, prepared=prepared,
                  prepare_fn=lambda paths: prepared_paths.extend(p.name for p in paths))

    assert live.step(tmp_path, **kwargs) == 0  # mid-match: nothing heard yet
    assert len(prepared_paths) == len(harvested[0]) > 0 and described == []
    heard.append((0.0, 60.0))  # the lobby transcribes it all
    assert live.step(tmp_path, **kwargs) == len(prepared_paths)
    assert sorted(described) == sorted(prepared_paths)
    assert live.step(tmp_path, **kwargs) == 0  # each once


def test_every_window_is_described_mid_match_unheard_where_efficiency_cores_take_it(tmp_path, monkeypatch):
    """Describing waited on every window's transcript, and a 20-minute session
    left all 124 windows to the wait (2026-11-01). What was said is no longer
    in the prompt, so nothing waits on it."""
    import json

    _segments(tmp_path, [30.0, 60.0])
    _gameplay(monkeypatch, [(0.0, 1e9)])
    monkeypatch.setattr("pipeline_v2.cache.clip_sha", lambda path: path.name)

    def harvest(session, clip_dir, *, skip, **_):
        requests = clips_mod.coverage_requests(clips_mod.read_segments(session))
        clip_dir.mkdir(exist_ok=True)
        out = []
        for i, r in enumerate(requests):
            if skip(r):
                continue
            clip = clip_dir / f"{i:03d}-coverage.mkv"
            side_path = clip.with_suffix(".capture.json")
            if not clip.is_file():  # already cut: as harvest, its sidecar stands
                clip.write_bytes(b"w")
                side_path.write_text(json.dumps({"video_file": clip.name, "segment_start_sec": r.at - 4.0,
                                                 "segment_end_sec": r.at + 4.0}), encoding="utf-8")
            out.append(json.loads(side_path.read_text(encoding="utf-8")))
        return out

    def prepare(paths):
        pass

    described: list[str] = []

    def describe(cfg, paths, max_calls, workers):
        for path in paths:
            described.append(path.name)
            yield path, (8.0, None, None)

    kwargs = dict(seen=set(), cfg=SimpleNamespace(), harvest_fn=harvest, describe_fn=describe,
                  store_fn=lambda *a: None, connect=_Conn,
                  prepared=set(), prepare_fn=prepare)

    live.step(tmp_path, **kwargs, describe_now=False)  # no efficiency cores: cut, not described
    windows = sorted(p.name for p in (tmp_path / "windows").glob("*.mkv"))
    assert windows and described == []
    live.step(tmp_path, **kwargs)  # with them, or between matches: all, nothing transcribed
    assert sorted(described) == windows
    live.step(tmp_path, **kwargs)
    assert sorted(described) == windows  # none twice


def test_match_speech_alternates_by_match_and_is_logged(tmp_path, monkeypatch):
    """Schedule D for every other match, so one session compares frame times (2026-10-25).
    Mid-match only where efficiency cores take it; without them, never."""
    import json

    for mode, expected, cores in (("alternate", [0, 2], 0xFF000), ("off", [], 0xFF000), ("on", [], None)):
        monkeypatch.setattr(live, "_efficiency_cores", lambda cores=cores: cores)
        monkeypatch.setenv(live.MATCH_SPEECH_KEY, mode)
        session = tmp_path / mode
        session.mkdir()
        calls: list[int] = []
        matches = iter([0, 1, 2])
        state = {"match": 0}
        done = threading.Event()

        def match(s):
            state["match"] = next(matches, None)
            if state["match"] is None:
                done.set()
                state["match"] = 1
            return state["match"]

        describer = live.LiveDescriber(
            session, interval=0.01, step_fn=lambda s, seen, **kw: 0, idle_fn=lambda s: False,
            listen_fn=lambda s, holder: calls.append(state["match"]) or 0,
            laugh_fn=lambda s, holder: 0)
        describer.match_fn = match
        describer.start()
        if mode == "off":  # nothing listens during the session at all
            describer.stop()
            assert calls == [] and not (session / live.EXPERIMENT_LOG).exists()
            continue
        assert done.wait(10.0)
        describer.stop()
        assert calls[:2] == expected, mode
        log = session / live.EXPERIMENT_LOG
        if mode == "off":
            assert not log.exists()
        elif cores is None:
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            assert not any(e["speech_live"] for e in entries)
        else:
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            assert [e["speech_live"] for e in entries[:3]] == [True, False, True]


def test_banked_frames_line_up_with_where_a_window_really_starts(tmp_path, monkeypatch):
    """A stream-copied window begins on the next keyframe: 272.0 began at 272.021 (2026-11-03)."""
    from pipeline_v2 import cli, frames

    _segments(tmp_path, [30.021, 60.021])
    clip = tmp_path / "windows" / "000-coverage.mkv"
    clip.parent.mkdir()
    clip.write_bytes(b"w")
    clip.with_suffix(".capture.json").write_text('{"segment_start_sec": 32.0}', encoding="utf-8")
    monkeypatch.setattr(clips_mod, "keyframe_interval", lambda path: 2.0)
    monkeypatch.setattr(cli, "frame_plan", lambda c: (tmp_path / "f", [0.5], [3.0]))
    asked: dict = {}
    monkeypatch.setattr(frames, "bank", lambda segment, wanted, gpu=True: asked.setdefault(segment.name, wanted) and len(wanted))

    live.bank_frames(tmp_path, [clip])

    (only,) = asked.values()
    assert [(tag, round(in_segment, 3)) for _, tag, _, _, in_segment in only] == [("window", 2.5), ("kill", 5.0)]


def test_nothing_runs_during_play_unless_asked_for(monkeypatch):
    """Cutting and scanning as each segment closed cost 7fps (2026-11-05): all after the game."""
    monkeypatch.setenv(live.MATCH_SPEECH_KEY, "off")
    monkeypatch.setenv(live.DESCRIBE_KEY, "off")
    assert not live.work_during_play()
    monkeypatch.setenv(live.DESCRIBE_KEY, "on")
    assert live.work_during_play()
    monkeypatch.setenv(live.DESCRIBE_KEY, "off")
    monkeypatch.setenv(live.MATCH_SPEECH_KEY, "alternate")
    assert live.work_during_play()
