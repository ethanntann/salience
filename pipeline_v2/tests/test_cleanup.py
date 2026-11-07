"""Raw footage deleted once it has done its job; never what is still needed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline_v2.capture import cleanup

DAY = 86400.0


def _session(root, name, *, made=True, parts=2):
    s = root / name
    (s / "highlights").mkdir(parents=True)
    if made:
        (s / "highlights" / "highlights.json").write_text("{}", encoding="utf-8")
        (s / "highlights" / "01-000-coverage.mkv").write_bytes(b"h")
    for i in range(parts):
        (s / f"part-{i:05d}.mkv").write_bytes(b"x" * 1000)
    (s / "segments.csv").write_text("".join(f"part-{i:05d}.mkv,{i*30}.0,{(i+1)*30}.0\n" for i in range(parts)),
                                    encoding="utf-8")
    return s


def _now(name):
    return datetime.strptime(name, "%Y%m%d-%H%M%S").timestamp()


def test_footage_goes_after_the_days_and_highlights_stay(tmp_path):
    old, recent = _session(tmp_path, "20261020-120000"), _session(tmp_path, "20261101-120000")
    done = cleanup.clean(tmp_path, now=_now("20261102-120000"), days=7, free_bytes=lambda p: 1e12)

    assert [d[0] for d in done] == [old.name]
    assert not list(old.glob("part-*.mkv")) and (old / "highlights" / "01-000-coverage.mkv").is_file()
    assert cleanup.footage_deleted(old) and cleanup.recorded_minutes(old) == 1.0
    assert list(recent.glob("part-*.mkv"))


def test_never_the_session_recording_nor_one_not_made_yet(tmp_path):
    live = _session(tmp_path, "20261001-120000")
    waiting = _session(tmp_path, "20261002-120000", made=False)
    assert cleanup.clean(tmp_path, active={live.name}, now=_now("20261102-120000"), days=7,
                         free_bytes=lambda p: 0) == []
    assert list(live.glob("part-*.mkv")) and list(waiting.glob("part-*.mkv"))


def test_a_full_drive_frees_the_oldest_first_even_when_kept_forever(tmp_path):
    a, b = _session(tmp_path, "20261030-120000"), _session(tmp_path, "20261031-120000")
    answers = iter([5e9, 30e9])  # short of room, then enough once one session went

    def room(path):
        return next(answers)

    done = cleanup.clean(tmp_path, now=_now("20261101-120000"), days=None, free_bytes=room)
    assert [d[0] for d in done] == [a.name] and list(b.glob("part-*.mkv"))


def test_the_setting(monkeypatch):
    monkeypatch.setenv(cleanup.KEEP_KEY, "forever")
    assert cleanup.keep_days() is None
    monkeypatch.setenv(cleanup.KEEP_KEY, "3")
    assert cleanup.keep_days() == 3.0
    monkeypatch.delenv(cleanup.KEEP_KEY)
    assert cleanup.keep_days() == cleanup.KEEP_DAYS


def test_a_session_rated_through_goes_before_its_days(tmp_path):
    rated, open_ = _session(tmp_path, "20261105-120000"), _session(tmp_path, "20261106-120000")
    done = cleanup.clean(tmp_path, now=_now("20261107-120000"), days=7, free_bytes=lambda p: 1e12,
                         rated=lambda s: s == rated)
    assert done == [(rated.name, "every moment rated", 2000)]
    assert (rated / "highlights" / "01-000-coverage.mkv").is_file() and list(open_.glob("part-*.mkv"))
    # Kept forever is kept, rated or not.
    assert cleanup.clean(tmp_path, now=_now("20261107-120000"), days=None, free_bytes=lambda p: 1e12,
                         rated=lambda s: True) == []


def test_nothing_described_is_not_rated_through(tmp_path):
    assert not cleanup.fully_rated(_session(tmp_path, "20261105-120000"))


def _segments(n):
    from pipeline_v2.capture.clips import Segment

    return [Segment(Path(f"part-{i:05d}.mkv"), i * 30.0, (i + 1) * 30.0) for i in range(n)]


def test_the_lobby_is_what_lies_outside_every_match_before_the_newest():
    segments = _segments(40)  # 20 minutes
    # A match 5:00-11:00, then lobby, then a match from 16:00 still on.
    lobby = cleanup.lobby_segments(segments, [(300.0, 660.0), (960.0, 1200.0)])
    # A minute clear of each match: to 4:00, and 12:00 to 15:00.
    assert [s.start for s in lobby] == [i * 30.0 for i in range(8)] + [i * 30.0 for i in range(24, 30)]
    assert cleanup.lobby_segments(segments, None) == []
    # On the island, after the newest match began in the log: nothing after it goes.
    assert all(s.end <= 300 - 60 for s in cleanup.lobby_segments(segments, [(300.0, 660.0)]))


def test_drop_lobby_deletes_only_lobby_and_remembers_it(tmp_path, monkeypatch):
    from pipeline_v2.capture import gameplay

    monkeypatch.delenv(cleanup.KEEP_KEY, raising=False)
    s = _session(tmp_path, "20261107-200000", parts=20)
    monkeypatch.setattr(gameplay, "spans", lambda session, duration, log_dir=None: {"spans": [(400.0, 600.0)]})
    assert cleanup.drop_lobby(s) == 11 * 1000  # to 5:30, a minute before the match
    assert sorted(p.name for p in s.glob("part-*.mkv"))[0] == "part-00011.mkv"
    assert cleanup.recorded_minutes(s) == 10.0  # the listing still says how long it was
    monkeypatch.setenv(cleanup.KEEP_KEY, "forever")
    monkeypatch.setattr(gameplay, "spans", lambda session, duration, log_dir=None: {"spans": [(560.0, 600.0)]})
    assert cleanup.drop_lobby(s) == 0


def test_rated_through_waits_days_after_the_last_rating(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pipeline_v2 import store
    from pipeline_v2.capture import highlights

    s = _session(tmp_path, "20261105-120000")
    (s / "windows").mkdir()
    ratings = {"a": 1000.0, "b": 1000.0 + 2 * DAY}

    class Conn:
        def execute(self, sql):
            return SimpleNamespace(fetchall=lambda: list(ratings.items()))

        def close(self):
            pass

    monkeypatch.setattr(store, "connect", Conn)
    monkeypatch.setattr(highlights, "_session_clips", lambda conn, w: (None, ["x"]))
    monkeypatch.setattr(highlights, "rateable", lambda session, inside: {"a", "b"})
    assert not cleanup.fully_rated(s, now=1000.0 + 4 * DAY)  # "b" rated 2 days ago
    assert cleanup.fully_rated(s, now=1000.0 + 5 * DAY)
    ratings.pop("b")
    assert not cleanup.fully_rated(s, now=1000.0 + 30 * DAY)  # one left to rate
