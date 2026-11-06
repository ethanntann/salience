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


class FakeDevice:
    """Records blocks filled with `level`; `fail` makes recording raise."""

    def __init__(self, id, level=0.5, fail=False):
        self.id, self.level, self.fail = id, level, fail

    def recorder(self, **kwargs):
        device = self

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def record(self, numframes):
                import numpy as np

                if device.fail:
                    raise RuntimeError("device invalidated")
                return np.full((numframes, audio_pump.CHANNELS), device.level, np.float32)

        return Recorder()


class Sink:
    def __init__(self, pump_ref, limit):
        self.blocks, self.pump_ref, self.limit = [], pump_ref, limit

    def write(self, data):
        import numpy as np

        self.blocks.append(float(np.frombuffer(data, np.float32)[0]))
        if len(self.blocks) >= self.limit:
            self.pump_ref[0].signal_stop()


def run_pump(device, choose, limit):
    ref = []
    sink = Sink(ref, limit)
    clock = iter(range(10**6))  # every look at the clock is a second later
    pump = audio_pump.TrackPump("mic", device, sink, choose=choose, clock=lambda: next(clock))
    ref.append(pump)
    pump._loop()
    return pump, sink.blocks


def test_a_mic_plugged_in_mid_session_is_recorded_from_then_on():
    """Session 4 started before the mic was plugged in and recorded the old
    default for all 38 minutes."""
    laptop, headset = FakeDevice("laptop", 0.1), FakeDevice("headset", 0.9)
    plugged = []

    def choose():
        plugged.append(1)
        return headset if len(plugged) > 1 else laptop

    pump, blocks = run_pump(laptop, choose, limit=6)

    assert blocks[0] == pytest.approx(0.1)
    assert blocks[-1] == pytest.approx(0.9)
    assert pump.stats.switches == 1


def test_no_mic_at_the_start_is_silence_until_one_appears():
    headset = FakeDevice("headset", 0.9)
    asked = []

    def choose():
        asked.append(1)
        return headset if len(asked) > 2 else None

    pump, blocks = run_pump(None, choose, limit=12)

    assert blocks[0] == 0.0
    assert blocks[-1] == pytest.approx(0.9)


def test_an_unplugged_mic_leaves_silence_not_an_ended_track():
    pump, blocks = run_pump(FakeDevice("headset", fail=True), lambda: None, limit=3)

    assert pump.stats.error is None
    assert blocks == [0.0, 0.0, 0.0]


def test_without_choosing_a_failing_device_ends_the_track_with_the_error():
    pump, blocks = run_pump(FakeDevice("headset", fail=True), None, limit=3)

    assert "device invalidated" in pump.stats.error
    assert blocks == []


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
