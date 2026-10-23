"""Describing during play: only what the final run would describe, the same way."""

from __future__ import annotations

import threading
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


def test_between_matches_speech_is_transcribed_before_the_step(tmp_path):
    order: list[str] = []
    done = threading.Event()

    def step(session, seen, **kw):
        order.append("step:" + ",".join(sorted(k for k in kw if k in ("listen", "heard"))))
        done.set()
        return 0

    describer = live.LiveDescriber(
        tmp_path, interval=0.01, step_fn=step, idle_fn=lambda s: True,
        listen_fn=lambda s, holder: order.append("listen") or True).start()
    assert done.wait(2.0)
    describer.stop()

    assert order[:2] == ["listen", "step:heard,listen"]


def test_without_speech_windows_are_not_held(tmp_path):
    passed: list[dict] = []
    done = threading.Event()

    def step(session, seen, **kw):
        passed.append(kw)
        done.set()
        return 0

    describer = live.LiveDescriber(tmp_path, interval=0.01, step_fn=step, speech=False,
                                   idle_fn=lambda s: False).start()
    assert done.wait(2.0)
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

    def describe(cfg, clip, facts, frames, *, allow_call=True):
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


def test_speech_runs_only_between_matches(tmp_path):
    states = iter([False, True, False, True])
    looked, listened = threading.Event(), []

    def idle(session):
        state = next(states, None)
        if state is None:
            looked.set()
            return False
        return state

    describer = live.LiveDescriber(
        tmp_path, interval=0.01, step_fn=lambda s, seen, **kw: 0, idle_fn=idle,
        listen_fn=lambda s, holder: listened.append(1) or True).start()
    assert looked.wait(2.0)
    describer.stop()

    assert len(listened) == 2


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

    assert live.listen_step(tmp_path, {}) is False

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
