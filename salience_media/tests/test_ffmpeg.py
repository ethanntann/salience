from salience_media.ffmpeg import ffmpeg_worker_count, parallel_map


def test_parallel_map_preserves_input_order(monkeypatch):
    monkeypatch.setenv("SALIENCE_FFMPEG_WORKERS", "2")

    assert parallel_map([3, 1, 2], lambda value: value * 2) == [6, 2, 4]


def test_parallel_map_runs_sequentially_when_pinned_to_one_worker(monkeypatch):
    monkeypatch.setenv("SALIENCE_FFMPEG_WORKERS", "1")

    assert parallel_map([1, 2, 3], lambda value: value + 1) == [2, 3, 4]


def test_worker_count_is_bounded_and_ignores_nonsense(monkeypatch):
    monkeypatch.setenv("SALIENCE_FFMPEG_WORKERS", "99")
    assert ffmpeg_worker_count() == 8

    monkeypatch.setenv("SALIENCE_FFMPEG_WORKERS", "not-a-number")
    assert 1 <= ffmpeg_worker_count() <= 4


def test_the_efficiency_cores_are_the_lowest_class_and_none_when_all_alike():
    from salience_media.ffmpeg import efficient_mask

    # i5-14500HX: 12 performance threads (class 1), then 8 efficiency cores (0).
    hybrid = [(0, i, 1) for i in range(12)] + [(0, i, 0) for i in range(12, 20)]
    assert efficient_mask(hybrid) == 0xFF000
    assert efficient_mask([(0, i, 0) for i in range(8)]) is None
    assert efficient_mask([(1, 3, 0), (0, 0, 1)]) is None  # only group 0 can be masked


def test_while_a_game_runs_a_helper_is_moved_off_its_cores_on_two_threads(monkeypatch):
    import os

    from salience_media import ffmpeg

    if os.name != "nt":
        return
    import shutil

    if shutil.which("ffmpeg") is None:
        return
    moved = []
    monkeypatch.setattr(ffmpeg, "LOW_PRIORITY", True)
    done = ffmpeg.run(["ffmpeg", "-v", "error", "-version"], capture_output=True, text=True, timeout=30,
                      confine=moved.append)
    assert moved and done.returncode == 0 and "ffmpeg version" in done.stdout
    assert done.args[:5] == ["ffmpeg", "-threads", "2", "-filter_threads", "2"]


def test_without_a_game_a_helper_is_plain_subprocess_run(monkeypatch):
    import sys

    from salience_media import ffmpeg

    monkeypatch.setattr(ffmpeg, "LOW_PRIORITY", False)
    moved = []
    done = ffmpeg.run([sys.executable, "-c", "print('hi')"], capture_output=True, text=True, confine=moved.append)
    assert done.stdout.strip() == "hi" and not moved
