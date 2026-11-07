"""The output device the game is playing through, from its audio sessions."""

from pipeline_v2.capture import game_audio

GAME = game_audio.GAME_EXE


def names(table):
    return lambda pid: table.get(pid, "")


def test_the_device_the_game_plays_on_beats_the_default_and_others():
    # Headphones picked in Fortnite's settings: its session plays there, and
    # only sits open on the speakers (2026-10-24: "game audio not detected").
    def listing():
        return [("speakers", 4212, 0), ("broadcast", 4212, 0), ("headphones", 4212, 1),
                ("headphones", 6608, 1)]
    assert game_audio.game_device(GAME, listing, names({4212: GAME, 6608: "discord.exe"})) == "headphones"


def test_no_session_or_only_open_ones_keep_the_default():
    def pick(rows):
        return game_audio.game_device(GAME, lambda: rows, names({1: GAME, 2: "discord.exe"}))
    assert pick([]) is None
    assert pick([("headphones", 2, 1)]) is None  # someone else's playing
    assert pick([("a", 1, 0), ("b", 1, 0)]) is None  # open on two, playing on neither
    assert pick([("a", 1, 0)]) == "a"  # its one device, quiet for now


def test_a_failing_lookup_keeps_the_default():
    def broken():
        raise OSError("no audio service")
    assert game_audio.game_device(GAME, broken, names({})) is None
