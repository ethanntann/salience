"""Capture inside FFmpeg: the command, and the clock the session hangs on it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pipeline_v2.capture import gfx


def test_the_window_is_captured_on_the_nvidia_gpu_and_never_piped_through_python():
    command = gfx.command(gfx.source(1234, None, 60), 30, "8M", 30,
                          Path("part-%05d.mkv"), Path("segments.csv"), [r"\\.\pipe\game"])

    graph = command[command.index("-filter_complex") + 1]
    assert graph.startswith("gfxcapture=hwnd=1234:max_framerate=60")
    assert graph.endswith("showinfo,fps=30[video]")
    assert "d3d11va=gpu:,vendor_id=0x10de" in command
    assert "pipe:0" not in command and "rawvideo" not in command
    # Video from the graph, audio from input 0: there is no video input.
    assert command[command.index("-map") + 1] == "[video]"
    assert "0:a:0" in command
    assert "-pix_fmt" not in command  # GPU frames go to NVENC as they are


def test_without_a_window_the_monitor_is_captured():
    assert gfx.source(None, 1, 60).startswith("gfxcapture=monitor_idx=0:")


def test_frame_zero_is_when_ffmpeg_logs_it_not_when_it_is_muxed():
    now = [100.0]
    progress = gfx.Progress(30, clock=lambda: now[0])
    assert progress.timeline() is None  # audio flows until video exists

    progress.feed("frame=0")
    now[0] = 100.7
    progress.logged("[Parsed_showinfo_1 @ 0000] n:   0 pts:      0 pts_time:0")
    now[0] = 101.0
    progress.feed("frame=3")  # muxed late: it must not move frame 0
    now[0] = 102.7

    assert progress.origin == 100.7
    assert progress.frames == 3
    assert abs(progress.timeline() - 2.0) < 1e-9


def test_only_ffmpeg_8_and_later_can_capture():
    def runner(output):
        return lambda *a, **k: SimpleNamespace(stdout=output)

    assert gfx.available(runner("Filter gfxcapture\ngfxcapture AVOptions:\n"))
    assert not gfx.available(runner("Unknown filter 'gfxcapture'.\n"))


def test_a_smaller_resolution_keeps_the_screens_shape_with_even_sides():
    assert gfx.output_size((1920, 1200), 720) == (1152, 720)  # 16:10
    assert gfx.output_size((2560, 1440), 1080) == (1920, 1080)
    assert gfx.output_size((1366, 768), 720) == (1280, 720)


def test_a_resolution_is_never_scaled_up():
    """A 1080p screen asked for 1080p records as it is, not resampled."""
    assert gfx.output_size((1920, 1080), 1080) is None
    assert gfx.output_size((1280, 720), 1080) is None
    assert gfx.output_size((1920, 1200), None) is None
    assert gfx.output_size(None, 720) is None


def test_the_capture_scales_itself_only_when_asked():
    assert ":width=" not in gfx.source(42, None, 120)
    sized = gfx.source(42, None, 120, (1152, 720))
    assert sized.endswith(":width=1152:height=720:scale_mode=bicubic")
    assert "hwnd=42" in sized and "resize_mode=scale_aspect" in sized
