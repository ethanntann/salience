"""Raw footage deleted once it has done its job; never what is still needed."""

from __future__ import annotations

from datetime import datetime

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
