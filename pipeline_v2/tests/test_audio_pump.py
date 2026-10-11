"""Recorder audio: two tracks, kept separable, never blocking capture.

Every chunk of the September 7 session is silent, so the A/V drift gate has
never been evaluable and saved highlights have no sound. These cover the shape
of the output and the shutdown ordering that decides whether the tracks end up
the same length.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pipeline_v2.capture import audio_pump


def test_each_track_gets_its_own_output_stream_not_a_mix():
    """Summing them makes the voice-versus-effects question unanswerable."""
    command = audio_pump.mux_command(
        "1920x1200", 30, "8M", "h264_nvenc", 30,
        Path("part-%05d.mkv"), Path("segments.csv"),
        [r"\\.\pipe\a", r"\\.\pipe\b"],
    )

    assert command.count("-map") == 3          # one video, two audio
    assert "0:v:0" in command
    assert "1:a:0" in command and "2:a:0" in command
    assert "amix" not in " ".join(command)
    assert "amerge" not in " ".join(command)


def test_audio_inputs_declare_the_raw_pcm_format():
    command = audio_pump.mux_command(
        "1920x1200", 30, "8M", "h264_nvenc", 30,
        Path("p.mkv"), Path("s.csv"), [r"\\.\pipe\a"],
    )
    joined = " ".join(command)

    assert "-f f32le" in joined
    assert f"-ar {audio_pump.SAMPLE_RATE}" in joined
    assert f"-ac {audio_pump.CHANNELS}" in joined


def test_a_dropped_track_must_not_truncate_the_video():
    """`-shortest` would end the recording when any one input stops."""
    command = audio_pump.mux_command(
        "1920x1200", 30, "8M", "h264_nvenc", 30,
        Path("p.mkv"), Path("s.csv"), [r"\\.\pipe\a"],
    )

    assert "-shortest" not in command


def test_no_audio_means_no_audio_inputs_at_all():
    command = audio_pump.mux_command(
        "1920x1200", 30, "8M", "h264_nvenc", 30,
        Path("p.mkv"), Path("s.csv"), [],
    )

    assert command.count("-map") == 1
    # No audio codec flags at all when there is no audio to encode.
    assert "-c:a" not in command
    assert "-f f32le" not in " ".join(command)


def test_pipe_names_are_valid_and_unique():
    first, second = audio_pump.pipe_name("game"), audio_pump.pipe_name("game")

    assert first.startswith("\\\\.\\pipe\\")
    assert first != second


def test_track_seconds_come_from_block_count():
    stats = audio_pump.TrackStats(name="game")
    stats.blocks = audio_pump.SAMPLE_RATE // audio_pump.BLOCK_FRAMES

    assert stats.seconds == pytest.approx(1.0, abs=0.05)


def test_silence_is_reported_rather_than_treated_as_an_error():
    """A muted game yields a valid but useless track; the report must say so."""
    stats = audio_pump.TrackStats(name="mic", blocks=100, silent_blocks=100)

    assert stats.silent_fraction == 1.0
    assert stats.error is None


def test_a_missing_device_is_recorded_not_raised():
    stats = audio_pump.AudioStats(
        tracks=[audio_pump.TrackStats(name="mic", error="no such device")]
    )

    assert stats.summary()["mic"]["error"] == "no such device"


def test_signal_stop_does_not_wait():
    """Every producer must be signalled before any is joined.

    Joining one first lets the others keep recording for the duration of that
    join, which left the audio track 191ms longer than the video.
    """
    pump = audio_pump.TrackPump("game", device=None, sink=None)

    pump.signal_stop()

    assert pump._stop.is_set()
    assert isinstance(pump._stop, threading.Event)


def test_stop_is_safe_when_nothing_ever_started():
    pump = audio_pump.TrackPump("game", device=None, sink=None)

    assert pump.stop().blocks == 0


def test_a_wasapi_discontinuity_is_counted_not_just_warned():
    """soundcard warns and carries on; the gap is real and must be visible.

    A discontinuity means WASAPI lost samples, so everything after it no longer
    lines up with the video. A warning printed to stderr during a 2-hour
    session is not something anyone will see.
    """
    stats = audio_pump.TrackStats(name="game", blocks=100, discontinuities=3)

    summary = audio_pump.AudioStats(tracks=[stats]).summary()

    assert summary["game"]["discontinuities"] == 3
    assert summary["game"]["error"] is None  # not fatal, but recorded


def test_a_clean_track_reports_zero_discontinuities():
    summary = audio_pump.AudioStats(
        tracks=[audio_pump.TrackStats(name="mic", blocks=50)]
    ).summary()

    assert summary["mic"]["discontinuities"] == 0
