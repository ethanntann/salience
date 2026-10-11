"""Hotkey marks, aligned to the recording they annotate.

Marks are what every detector is scored against, and the project's one audio
result was tuned on a session with three marks that were also its test set.
Cheap marking is the fix.
"""

from __future__ import annotations

from pathlib import Path

from pipeline_v2.capture import marks as marks_mod
from pipeline_v2.jsonio import read_json


class FakeUser:
    """Stands in for user32, so key state is scriptable."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.step = -1

    def advance(self):
        self.step += 1

    def GetAsyncKeyState(self, key):  # noqa: N802 - matches the win32 name
        held = self.sequence[self.step] if self.step < len(self.sequence) else ()
        return 0x8000 if key in held else 0


def _recorder(monkeypatch, sequence):
    fake = FakeUser(sequence)
    monkeypatch.setattr(marks_mod.ctypes, "WinDLL", lambda *a, **k: fake)
    recorder = marks_mod.MarkRecorder(started=0.0)
    return recorder, fake


def test_a_press_is_timestamped_against_the_recording_start(monkeypatch):
    recorder, fake = _recorder(monkeypatch, [(), (marks_mod.VK_MARK,)])

    fake.advance()
    assert recorder._poll_once(now=1.0) is None
    fake.advance()
    mark = recorder._poll_once(now=12.5)

    assert mark.kind == "keep"
    assert mark.at == 12.5


def test_a_held_key_marks_once_not_continuously(monkeypatch):
    """Sixty marks a second would drown the real signal."""
    held = (marks_mod.VK_MARK,)
    recorder, fake = _recorder(monkeypatch, [held, held, held, held])

    found = []
    for index in range(4):
        fake.advance()
        result = recorder._poll_once(now=float(index))
        if result is not None:
            found.append(result)

    assert len(found) == 1


def test_releasing_and_pressing_again_marks_twice(monkeypatch):
    held = (marks_mod.VK_MARK,)
    recorder, fake = _recorder(monkeypatch, [held, (), held])

    found = []
    for index in range(3):
        fake.advance()
        result = recorder._poll_once(now=float(index))
        if result is not None:
            found.append(result)

    assert [m.at for m in found] == [0.0, 2.0]


def test_the_dull_key_records_a_negative_example(monkeypatch):
    """Explicit "nothing happened" is a label too, and rarer than it should be."""
    recorder, fake = _recorder(monkeypatch, [(marks_mod.VK_DULL,)])

    fake.advance()
    mark = recorder._poll_once(now=5.0)

    assert mark.kind == "dull"


def test_saved_marks_are_readable_by_the_tuner(tmp_path, monkeypatch):
    recorder, fake = _recorder(monkeypatch, [(marks_mod.VK_MARK,), (), (marks_mod.VK_DULL,)])
    for index in range(3):
        fake.advance()
        recorder._poll_once(now=float(index) * 10)

    path = recorder.save(tmp_path / "marks.json")
    saved = read_json(Path(path))

    assert saved["keep_seconds"] == [0.0]
    assert saved["dull_seconds"] == [20.0]
    assert len(saved["marks"]) == 2


def test_no_presses_saves_an_empty_but_valid_file(tmp_path, monkeypatch):
    recorder, _ = _recorder(monkeypatch, [()])

    saved = read_json(Path(recorder.save(tmp_path / "marks.json")))

    assert saved["keep_seconds"] == []


def test_marks_are_journalled_as_they_happen(tmp_path, monkeypatch):
    """On Windows `terminate()` is TerminateProcess: no cleanup ever runs.

    An end-of-session write therefore loses the entire session, so each press
    is appended immediately.
    """
    recorder, fake = _recorder(monkeypatch, [(marks_mod.VK_MARK,)])
    recorder.journal = tmp_path / "marks.tsv"

    fake.advance()
    recorder._poll_once(now=7.5)

    assert (tmp_path / "marks.tsv").read_text(encoding="utf-8") == "7.5\tkeep\n"


def test_a_killed_session_can_be_recovered_from_its_journal(tmp_path):
    journal = tmp_path / "marks.tsv"
    journal.write_text("12.5\tkeep\n30.0\tdull\n", encoding="utf-8")

    found = marks_mod.MarkRecorder.read_journal(journal)

    assert [(m.at, m.kind) for m in found] == [(12.5, "keep"), (30.0, "dull")]


def test_a_corrupt_journal_line_is_skipped_not_fatal(tmp_path):
    journal = tmp_path / "marks.tsv"
    journal.write_text("12.5\tkeep\ngarbage\nnotanumber\tkeep\n8.0\tdull\n",
                       encoding="utf-8")

    found = marks_mod.MarkRecorder.read_journal(journal)

    assert [m.at for m in found] == [12.5, 8.0]


def test_a_journal_failure_never_stops_the_recording(tmp_path, monkeypatch):
    recorder, fake = _recorder(monkeypatch, [(marks_mod.VK_MARK,)])
    recorder.journal = tmp_path / "nodir" / "marks.tsv"
    monkeypatch.setattr(
        marks_mod.Path, "mkdir",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    fake.advance()
    mark = recorder._poll_once(now=1.0)

    assert mark is not None          # the mark still counted in memory
    assert recorder.marks == [mark]
