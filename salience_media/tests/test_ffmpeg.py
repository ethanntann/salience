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
