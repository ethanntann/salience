"""Reading the player's kills and knocks off their own HUD."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_v2.capture import hud, highlights

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _frame_with(template, at=(40, 20), size=(160, 420), seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 140, size=size, dtype=np.uint8)
    if template is not None:
        y, x = at
        frame[y:y + template.shape[0], x:x + template.shape[1]] = template
    return frame


def test_the_word_matches_and_a_scene_does_not():
    templates = hud._templates(hud.REFERENCE_HEIGHT, hud.KINDS["knock"])
    banner, scene = _frame_with(templates[0][0]), _frame_with(None, seed=1)

    (hit, where), (miss, _) = hud.match_scores([banner, scene], templates)

    assert hit > 0.95 and miss < hud.THRESHOLD
    assert where == (20, 40, 1.0)  # where and at what size: the plate is read below it


def test_a_banner_held_for_two_seconds_is_one_kill():
    times = [i / 10 for i in range(60)]
    scores = [0.0] * 60
    for i in range(10, 30):
        scores[i] = 0.7 + i / 1000  # one banner, peaking at its last frame
    for i in range(45, 50):
        scores[i] = 0.8  # a second kill 1.5s after the first faded

    found = hud.peaks(times, scores)

    assert [round(at, 1) for at, _, _ in found] == [2.9, 4.5]


def test_scanning_again_adds_only_new_segments_and_never_a_duplicate(tmp_path):
    session = tmp_path / "run"
    session.mkdir()
    rows = []
    for i in range(3):
        (session / f"part-{i:05d}.mkv").write_bytes(b"x")
        rows.append(f"part-{i:05d}.mkv,{i * 30.0},{i * 30.0 + 30}")
    (session / "segments.csv").write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
    calls = []

    def scan(session, runner, segments):
        calls.append([s.path.name for s in segments])
        # A banner at the 30s boundary is seen from both sides of it.
        return [hud.Elimination(at=29.9 if len(calls) == 1 else 30.2, score=0.8, name="Nanochunks")]

    assert hud.update(session, scan_fn=scan) == 1
    (session / "segments.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert hud.update(session, scan_fn=scan) == 0

    assert calls == [["part-00000.mkv", "part-00001.mkv"], ["part-00002.mkv"]]
    assert [e.name for e in hud.read_events(session)] == ["Nanochunks"]


def test_no_banner_at_all_leaves_kills_to_the_describer():
    """It may be a HUD layout the reader does not know."""
    assert hud.tiers([]) is None and hud.tiers(None) is None


def _clip(tmp_path, start, outcome="no_event", summary=""):
    path = tmp_path / f"{int(start):03d}-coverage.mkv"
    path.write_bytes(b"x")
    path.with_suffix(".capture.json").write_text(
        json.dumps({"segment_start_sec": start, "segment_end_sec": start + 8}), encoding="utf-8")
    return SimpleNamespace(path=str(path), payload={"outcome": outcome}, summary=summary)


def test_the_hud_decides_kills_when_it_was_read(tmp_path):
    banners = hud.tiers([hud.Elimination(at=12.3, score=0.8),
                         hud.Elimination(at=20.1, score=1.0, kind="knock")])
    feed_line = _clip(tmp_path, 40.0, outcome="elimination")  # no banner near: the describer's mistake
    real = _clip(tmp_path, 8.0)                               # a kill it did not see
    knock = _clip(tmp_path, 16.0)

    assert [highlights.event_tier(c, banners) for c in (feed_line, real, knock)] == [0, 2, 1]
    assert highlights.event_tier(feed_line) == 2  # without the HUD, as before


def test_a_banner_at_a_window_edge_counts_for_both_windows(tmp_path):
    """The banner lands a moment after the shot, which may be in the window
    before; the kill itself plays out in the window it is drawn in."""
    banners = hud.tiers([hud.Elimination(at=8.3, score=0.8)])

    assert highlights.event_tier(_clip(tmp_path, 0.0), banners) == 2
    assert highlights.event_tier(_clip(tmp_path, 8.0), banners) == 2
    assert highlights.event_tier(_clip(tmp_path, 16.0), banners) == 0


def test_templates_are_in_the_repo():
    for names in hud.KINDS.values():
        for name in names:
            assert (Path(hud.ASSETS) / name).is_file()


def test_a_banner_seen_twice_is_one_kill_but_a_double_is_two():
    events = [hud.Elimination(at=100.0, score=0.88, name="Nanochunks"),
              hud.Elimination(at=101.6, score=0.68, name="Nanochunks7"),   # the same banner
              hud.Elimination(at=200.0, score=0.82, name="IckyWombat"),
              hud.Elimination(at=201.5, score=0.80, name="AngryRoommate10")]  # a double

    merged = hud.merge(events)

    assert [e.name for e in merged] == ["Nanochunks", "IckyWombat", "AngryRoommate10"]


def test_the_shot_before_a_banner_is_part_of_the_kill(tmp_path):
    """The describer saw the kill; the banner landed in the next window."""
    banners = hud.tiers([hud.Elimination(at=12.3, score=0.8)])

    assert highlights.event_tier(_clip(tmp_path, 0.0, outcome="elimination"), banners) == 2
    assert highlights.event_tier(_clip(tmp_path, 0.0), banners) == 0  # unless it saw one
