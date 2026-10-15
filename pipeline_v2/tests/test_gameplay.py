"""Own gameplay from Fortnite's log, placed on the video timeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pipeline_v2.capture import gameplay
from pipeline_v2.capture.game_log import GameEvent

ORIGIN = datetime(2026, 10, 14, 21, 56, 0, tzinfo=timezone.utc)


def _e(kind, seconds):
    return GameEvent(kind=kind, at=ORIGIN + timedelta(seconds=seconds))


def test_a_match_runs_from_start_to_death_plus_the_death_itself():
    spans = gameplay.play_spans([_e("match_start", 118), _e("death", 752)], ORIGIN, 2558)

    assert spans == [(118, 752 + gameplay.DEATH_TAIL)]


def test_a_win_keeps_the_victory_banner_after_match_end():
    """Session 2's Victory Royale was drawn 5s after the log's match end."""
    spans = gameplay.play_spans([_e("match_start", 1408), _e("match_end", 2523)], ORIGIN, 2558)

    assert spans == [(1408, 2523 + gameplay.WIN_TAIL)]


def test_a_match_still_running_when_recording_stopped_runs_to_the_end():
    assert gameplay.play_spans([_e("match_start", 100)], ORIGIN, 500) == [(100, 500)]


def test_recording_started_mid_match_starts_at_zero():
    assert gameplay.play_spans([_e("match_start", -60), _e("death", 40)], ORIGIN, 500) == [
        (0.0, 40 + gameplay.DEATH_TAIL)]


def test_requeue_without_death_closes_at_the_next_start():
    spans = gameplay.play_spans([_e("match_start", 10), _e("match_start", 300)], ORIGIN, 500)

    assert spans == [(10, 300), (300, 500)]


def test_recorded_origin_beats_the_folder_name(tmp_path):
    session = tmp_path / "20261014-175600"
    session.mkdir()
    (session / gameplay.STATS_NAME).write_text(
        json.dumps({"first_frame_utc": "2026-10-14T21:56:01.500000+00:00"}), encoding="utf-8")

    origin, source = gameplay.video_origin(session)

    assert (origin.second, origin.microsecond, source) == (1, 500000, "recorded")


def test_no_log_means_no_gating(tmp_path):
    session = tmp_path / "20261014-175600"
    session.mkdir()

    result = gameplay.spans(session, 100.0, log_dir=tmp_path / "missing")

    assert result["spans"] is None


def test_spans_are_read_from_the_log_files(tmp_path):
    session = tmp_path / "20261014-175600"
    session.mkdir()
    (session / gameplay.STATS_NAME).write_text(
        json.dumps({"first_frame_utc": ORIGIN.isoformat()}), encoding="utf-8")
    logs = tmp_path / "Logs"
    logs.mkdir()
    (logs / "FortniteGame.log").write_text(
        "[2026.10.14-21.57.58:396][  1]LogFort: MatchState changed "
        "previous=WaitingToStart current=InProgress\n"
        "[2026.10.14-22.08.35:850][  2]LogFortDeathCameraMode: Going to use death report\n",
        encoding="utf-8")

    result = gameplay.spans(session, 2558.0, log_dir=logs)

    assert [(round(a), round(b)) for a, b in result["spans"]] == [(118, 760)]
    assert result["matches"] == 1


def _phase(name, seconds):
    return GameEvent(kind="phase", at=ORIGIN + timedelta(seconds=seconds), detail=name)


def test_the_pre_game_island_is_not_gameplay():
    """Warmup fights do not count; play starts when the Battle Bus does."""
    events = [_e("match_start", 118), _phase("Warmup", 118), _phase("Aircraft", 161),
              _phase("SafeZones", 193), _e("death", 752)]

    assert gameplay.play_spans(events, ORIGIN, 2558) == [(161, 752 + gameplay.DEATH_TAIL)]


def test_the_phase_line_is_parsed():
    from pipeline_v2.capture import game_log

    event = game_log.parse_line(
        "[2026.10.14-21.58.41:231][700]LogBattleRoyaleGamePhaseLogic: HandleGamePhaseChanged. "
        "OldPhase = EAthenaGamePhase::Warmup. NewPhase = EAthenaGamePhase::Aircraft")

    assert (event.kind, event.detail) == ("phase", "Aircraft")
