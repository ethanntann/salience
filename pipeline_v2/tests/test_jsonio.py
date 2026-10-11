"""Atomic JSON writes, and what happens when Windows refuses one.

The September 7 recording session died in the control window on a status-file
write: an antivirus or indexer held the temp file, `replace` raised WinError 5,
and the exception escaped into the Tkinter callback. The caller is guarded now
(`launcher.publish_status`), and these cover the writer's own behaviour.
"""

from __future__ import annotations

import pytest

from pipeline_v2.jsonio import read_json, write_json


def test_a_written_file_reads_back(tmp_path):
    target = tmp_path / "s.json"

    write_json(target, {"clips": 2, "nested": {"a": [1, 2]}})

    assert read_json(target) == {"clips": 2, "nested": {"a": [1, 2]}}


def test_a_write_replaces_previous_content_whole(tmp_path):
    target = tmp_path / "s.json"
    write_json(target, {"old": True, "extra": 1})

    write_json(target, {"new": True})

    assert read_json(target) == {"new": True}


def test_no_temp_files_are_left_behind_on_success(tmp_path):
    write_json(tmp_path / "s.json", {"a": 1})

    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_a_locked_target_surfaces_the_real_error_not_a_cleanup_one(
    tmp_path, monkeypatch
):
    """The failure must name the write that failed, not the temp tidy-up."""
    target = tmp_path / "s.json"

    def refuse(self, other):
        raise PermissionError("WinError 5: target is held open")

    def also_refuse(self, missing_ok=False):
        raise PermissionError("WinError 5: temp is held open")

    monkeypatch.setattr("pathlib.Path.replace", refuse)
    monkeypatch.setattr("pathlib.Path.unlink", also_refuse)

    with pytest.raises(PermissionError, match="target is held open"):
        write_json(target, {"a": 1})


def test_a_transient_lock_is_retried_and_succeeds(tmp_path):
    import pathlib

    target = tmp_path / "s.json"
    real = pathlib.Path.replace
    attempts = {"n": 0}

    def flaky(self, other):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError("briefly held")
        return real(self, other)

    pathlib.Path.replace = flaky
    try:
        write_json(target, {"a": 1})
    finally:
        pathlib.Path.replace = real

    assert read_json(target) == {"a": 1}
    assert attempts["n"] == 3
