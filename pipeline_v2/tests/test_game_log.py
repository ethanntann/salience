"""Gameplay events read from Fortnite's own log.

Exact and free, where every other trigger estimates. Validated against three
real session logs: 5 match starts, 3 match ends, 5 deaths.
"""

from __future__ import annotations

from datetime import timezone

from pipeline_v2.capture import game_log


def test_a_match_start_is_recognised():
    line = (
        "[2026.09.23-20.18.22:135][236]LogFort: MatchState changed "
        "previous=EnteringMap current=InProgress"
    )

    event = game_log.parse_line(line)

    assert event.kind == "match_start"
    assert event.at.tzinfo is timezone.utc
    assert (event.at.hour, event.at.minute, event.at.second) == (20, 18, 22)
    assert event.at.microsecond == 135000


def test_a_match_end_is_recognised():
    line = (
        "[2026.09.23-20.34.40:616][789]LogFort: MatchState changed "
        "previous=InProgress current=WaitingPostMatch"
    )

    assert game_log.parse_line(line).kind == "match_end"


def test_lobby_state_changes_are_not_events():
    """EnteringMap -> WaitingToStart is queue churn, not a match boundary."""
    line = (
        "[2026.09.23-20.38.33:777][478]LogFort: MatchState changed "
        "previous=EnteringMap current=WaitingToStart"
    )

    assert game_log.parse_line(line) is None


def test_the_local_players_death_is_recognised():
    line = (
        "[2026.09.23-20.42.49:796][617]LogFortDeathCameraMode: Going to use "
        "death report (loc=X=-31755.000 Y=-38865.000 Z=638.000) to replace"
    )

    event = game_log.parse_line(line)

    assert event.kind == "death"


def test_ordinary_engine_noise_is_ignored():
    """The log is ~8MB per session and almost all of it is irrelevant."""
    for line in (
        "[2026.09.23-20.14.16:174][  0]LogConfig: Set CVar [[gc.Enabled:0]]",
        "[2026.09.23-20.14.25:429][751]LogGameFeatures: Loading 4135 plugins.",
        "not a log line at all",
        "",
    ):
        assert game_log.parse_line(line) is None


def test_timestamps_are_utc_not_local():
    """They look local but are not; treating them as local shifts every event."""
    moment = game_log.parse_timestamp("2026.09.23-20.18.22", "135")

    assert moment.tzinfo is timezone.utc
    assert moment.hour == 20


def test_a_watcher_reports_only_what_arrives_after_it_starts(tmp_path):
    log = tmp_path / "FortniteGame.log"
    log.write_text(
        "[2026.09.23-20.18.22:135][236]LogFort: MatchState changed "
        "previous=EnteringMap current=InProgress\n",
        encoding="utf-8",
    )
    watcher = game_log.LogWatcher(log).seek_to_end()

    assert watcher.poll() == []

    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "[2026.09.23-20.42.49:796][617]LogFortDeathCameraMode: Going to "
            "use death report (loc=X=1 Y=2 Z=3) to replace\n"
        )

    fresh = watcher.poll()

    assert [e.kind for e in fresh] == ["death"]
    assert watcher.poll() == []  # not repeated


def test_a_rotated_log_is_reread_rather_than_seeked_into(tmp_path):
    """A new match can rotate the log; a stale offset would skip real events."""
    log = tmp_path / "FortniteGame.log"
    log.write_text("x" * 5000, encoding="utf-8")
    watcher = game_log.LogWatcher(log).seek_to_end()

    log.unlink()
    log.write_text(
        "[2026.09.23-21.00.00:000][1]LogFort: MatchState changed "
        "previous=EnteringMap current=InProgress\n",
        encoding="utf-8",
    )

    assert [e.kind for e in watcher.poll()] == ["match_start"]


def test_a_missing_log_is_not_an_error(tmp_path):
    assert game_log.read_events(tmp_path / "nope.log") == []
    assert game_log.LogWatcher(tmp_path / "nope.log").poll() == []


def _at(stamp, body):
    return game_log.parse_line(f"[{stamp}:000][1]{body}")


def test_log_player_kills_is_a_view_change_not_a_kill():
    """The category name is misleading and cost real investigation.

    Timing settles it: a death at 19.22.58 is followed four seconds later by
    this line carrying a *different* player's name -- the spectate target.
    """
    event = _at(
        "2026.09.22-19.23.02",
        "LogPlayerKills: HandlePlayerStateChanged : charliefattybomb",
    )

    assert event.kind == "viewing"
    assert event.player == "charliefattybomb"


def test_the_local_player_is_inferred_from_the_first_view_of_a_match():
    """Hardcoding a name would be wrong for every other user of the product."""
    events = [
        _at("2026.09.22-19.07.42",
            "LogFort: MatchState changed previous=WaitingToStart current=InProgress"),
        _at("2026.09.22-19.07.44",
            "LogPlayerKills: HandlePlayerStateChanged : BabyRico.YT"),
        _at("2026.09.22-19.23.02",
            "LogPlayerKills: HandlePlayerStateChanged : charliefattybomb"),
    ]

    assert game_log.local_player(events) == "BabyRico.YT"


def test_spectating_is_watching_anyone_who_is_not_the_owner():
    """Exactly the viewpoint signal the VLM got wrong on 14 of 18 windows."""
    events = [
        _at("2026.09.22-19.07.42",
            "LogFort: MatchState changed previous=WaitingToStart current=InProgress"),
        _at("2026.09.22-19.07.44",
            "LogPlayerKills: HandlePlayerStateChanged : BabyRico.YT"),
        _at("2026.09.22-19.22.58",
            "LogFortDeathCameraMode: Going to use death report (loc=X=1 Y=2 Z=3)"),
        _at("2026.09.22-19.23.02",
            "LogPlayerKills: HandlePlayerStateChanged : charliefattybomb"),
    ]

    spans = game_log.spectating_spans(events)

    assert [who for _, who in spans] == ["charliefattybomb"]


def test_non_ascii_player_names_survive():
    """Real logs contain them; a console encoding error must not lose data."""
    event = _at(
        "2026.09.23-20.30.00",
        "LogPlayerKills: HandlePlayerStateChanged : 流苏坨子",
    )

    assert event.player == "流苏坨子"


def test_an_unknown_owner_reports_no_spectating():
    """Without a known owner every name would look like someone else."""
    events = [
        _at("2026.09.22-19.23.02",
            "LogPlayerKills: HandlePlayerStateChanged : somebody"),
    ]

    assert game_log.spectating_spans(events) == []
