"""Finding a kill by the ELIMINATION! banner, and the app's marker from it."""

from __future__ import annotations

import json
import os

import numpy as np

from pipeline_v2 import banner
from pipeline_v2.capture.desktop import server


def region_shape(size=(1920, 1200)):
    width = round(size[0] * banner.HEIGHT / size[1] / 2) * 2
    x0, x1, y0, y1 = banner.REGION
    return int(banner.HEIGHT * (y1 - y0)), int(width * (x1 - x0))


def frame_with_word(at=(36, 20), shape=None, rows=banner.STENCIL) -> np.ndarray:
    frame = np.full(shape or region_shape(), 90, np.uint8)  # a grey wall
    word = np.array([[c == "#" for c in row] for row in rows])
    y, x = at
    frame[y:y + word.shape[0], x:x + word.shape[1]][word] = 240
    return frame


def test_the_word_matches_wherever_it_sits_and_a_wall_does_not():
    wall = np.full(region_shape(), 90, np.uint8)
    white = np.full(region_shape(), 250, np.uint8)
    got = banner.scores(np.stack([frame_with_word((36, 20)), frame_with_word((60, 40)), wall, white]))
    assert got[0] > 0.95 and got[1] > 0.95
    assert got[2] < banner.MATCH and got[3] < banner.MATCH


def test_one_banner_is_one_kill_and_a_gap_starts_the_next():
    series = np.zeros(80)
    series[13:35] = 0.8  # 1.3s to 3.4s
    series[37:40] = 0.4  # sliding out: not a new one
    series[60:70] = 0.7  # 6.0s
    assert banner.onsets(series) == [1.3, 6.0]


def fake_ffmpeg(frames: np.ndarray, size=(1920, 1200)):
    calls = []

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(args, **_):
        calls.append(args)
        if args[0] == "ffprobe":
            return Done(f"{size[0]},{size[1]}\n")
        return Done(frames.tobytes())
    return run, calls


def test_a_kill_is_the_banner_less_its_lead_and_one_already_up_is_not_one():
    stack = [np.full(region_shape(), 90, np.uint8)] * 30
    late = stack[:]
    for i in range(14, 30):
        late[i] = frame_with_word()
    run, calls = fake_ffmpeg(np.stack(late))
    assert banner.kills("clip.mkv", 8.0, 16.0, runner=run) == [round(8.0 + 1.4 - banner.LEAD, 2)]
    assert calls[1][calls[1].index("-ss") + 1] == "8.00"

    already = [frame_with_word()] * 30
    run, _ = fake_ffmpeg(np.stack(already))
    assert banner.kills("clip.mkv", 8.0, 16.0, runner=run) == []


def test_the_region_is_cut_to_the_size_ffmpeg_is_asked_for():
    # 16:9 and 16:10 alike: the frame is scaled to the stencil's height first.
    for size in ((1920, 1080), (2560, 1440), (1920, 1200)):
        width = round(size[0] * banner.HEIGHT / size[1] / 2) * 2
        x0, x1, y0, y1 = banner.REGION
        w, h = int(width * (x1 - x0)), int(banner.HEIGHT * (y1 - y0))
        run, calls = fake_ffmpeg(np.zeros((5, h, w), np.uint8), size)
        assert banner.frames("clip.mkv", 0, 1, runner=run).shape == (5, h, w)
        assert f"crop={w}:{h}:" in calls[1][calls[1].index("-vf") + 1]


def test_every_kill_in_a_clip_is_found_once_and_the_main_one_is_the_windows(tmp_path):
    out = tmp_path / "highlights"
    out.mkdir()
    for name in ("01-a.mkv", "02-b.mkv", "03-c.mkv"):
        (out / name).write_bytes(b"x")
    found = {"01-a.mkv": [(3.0, "kill"), (13.4, "knock"), (20.0, "kill")], "02-b.mkv": [(23.0, "kill")], "03-c.mkv": []}
    asked = []

    def finder(path, start, end):
        asked.append((path.name, start, end))
        return found[path.name]

    lengths = {"01-a.mkv": 30.0, "02-b.mkv": 34.0, "03-c.mkv": 18.0}
    got = server._kills(out, lengths, finder)
    assert got == found  # every one, not only the window's
    assert ("01-a.mkv", 0.0, 30.0) in asked  # the whole clip

    # The one it was picked for: the window's; one just before when that is
    # all (its banner is what the describer saw); nothing near, none.
    assert server.main_kill(got["01-a.mkv"], (8.0, 16.0)) == (13.4, "knock")
    assert server.main_kill(got["02-b.mkv"], (24.0, 32.0)) == (23.0, "kill")
    assert server.main_kill(got["02-b.mkv"], (40.0, 48.0)) is None
    assert server.main_kill(got["01-a.mkv"], None) == (3.0, "kill")

    asked.clear()
    assert server._kills(out, lengths, finder) == got and asked == []
    assert server._kills(out, {"04-new.mkv": 10.0}, finder, compute=False) == {}  # not looked for: left out
    # A trim is a new file: found again.
    stat = (out / "02-b.mkv").stat()
    os.utime(out / "02-b.mkv", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    server._kills(out, lengths, finder)
    assert [a[0] for a in asked] == ["02-b.mkv"]
    assert json.loads((out / server.THUMBS / "kills.json").read_text())["02-b.mkv"]["hits"] == [[23.0, "kill"]]


def test_a_squads_kill_is_found_by_its_knocked_banner():
    # Half of a squads session's highlights had no marker: KNOCKED!, not ELIMINATION! (2026-10-25).
    rows, cut = banner.WORDS["knock"]
    stack = [np.full(region_shape(), 90, np.uint8)] * 10 + [frame_with_word((40, 30), rows=rows)] * 20
    run, _ = fake_ffmpeg(np.stack(stack))
    assert banner.found("clip.mkv", 8.0, 16.0, runner=run) == [(round(8.0 + 1.0 - banner.LEAD, 2), "knock")]


def test_a_squads_last_elimination_is_found_by_its_elim_banner_and_counted_once():
    # DOUBLE ELIM! under ENEMY TEAM WIPED!: the finish had no marker (2026-10-25).
    rows, _ = banner.WORDS["elim"]
    stack = [np.full(region_shape(), 90, np.uint8)] * 10 + [frame_with_word((50, 40), rows=rows)] * 20
    run, _ = fake_ffmpeg(np.stack(stack))
    assert banner.found("clip.mkv", 8.0, 16.0, runner=run) == [(round(8.0 + 1.0 - banner.LEAD, 2), "kill")]
    # ELIMINATION! starts with ELIM: one banner, one kill.
    both = [np.full(region_shape(), 90, np.uint8)] * 10 + [frame_with_word()] * 20
    run, _ = fake_ffmpeg(np.stack(both))
    assert [kind for _, kind in banner.found("clip.mkv", 8.0, 16.0, runner=run)] == ["kill"]


def test_a_session_is_scanned_a_segment_at_a_time_and_a_banner_across_two_is_one(tmp_path, monkeypatch):
    """Per window, the GPU decoder started every 8s: 2.2 CPU-s each, against 1.2 a 30s segment."""
    import numpy as np

    for i in range(2):
        (tmp_path / f"part-{i:05d}.mkv").write_bytes(b"x")
    (tmp_path / "segments.csv").write_text("part-00000.mkv,0.0,30.0\npart-00001.mkv,30.0,60.0\n", encoding="utf-8")
    decoded = []

    def frames(path, start, end, runner=None, fps=banner.FPS, gpu=True):
        decoded.append(path.name)
        return np.zeros((300, 4, 4), np.uint8) + (0 if path.name == "part-00000.mkv" else 1)

    def all_scores(stack, words=tuple(banner.WORDS)):
        lit = np.zeros(len(stack), np.float32)
        if stack[0, 0, 0] == 0:
            lit[-10:] = 0.9  # a kill banner up for the last second of the first segment...
        else:
            lit[:10] = 0.9  # ...and the first of the next
        return {w: (lit if w == "kill" else np.zeros(len(stack), np.float32)) for w in words}

    monkeypatch.setattr(banner, "frames", frames)
    monkeypatch.setattr(banner, "all_scores", all_scores)

    assert banner.session_found(tmp_path, 24.0, 32.0) == [(round(29.0 - banner.LEAD - 24.0, 2), "kill")]
    assert banner.session_found(tmp_path, 32.0, 40.0) == []  # still the same banner
    assert sorted(decoded) == ["part-00000.mkv", "part-00001.mkv"]  # each segment once, side by side
    banner.session_found(tmp_path, 0.0, 8.0)
    assert len(decoded) == 2  # then from the cache


def test_a_banner_flickering_under_the_match_is_one_kill():
    """ELIMINATION! at 0.4-0.5 for two seconds came back as kills at 0:17 and 0:18 (2026-11-03)."""
    import numpy as np

    flicker = np.array([0.0] * 5 + [0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.55] + [0.0] * 20)
    assert len(banner.onsets(flicker)) == 1
    gone = np.array([0.0] * 5 + [0.6] + [0.0] * 12 + [0.6] + [0.0] * 5)  # gone 1.2s, then a new one
    assert len(banner.onsets(gone)) == 2
