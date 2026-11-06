"""The stop path of the recorder: it must reach the highlights step."""

from __future__ import annotations

import json

from pipeline_v2.capture import record
from pipeline_v2.capture.wgc import WgcStats


def test_summary_reads_only_fields_the_stats_have():
    """A missing field here crashed `finally` and silently skipped highlights."""
    line = record._summary(WgcStats(delivered=900, written=1000, superseded=9))

    assert "1000 frames from 900 captured (1.0% superseded)" in line


def test_stats_are_saved_beside_the_footage(tmp_path):
    record._save_stats(tmp_path, WgcStats(delivered=10, written=12, superseded=1), None)

    saved = json.loads((tmp_path / "capture_stats.json").read_text(encoding="utf-8"))
    assert saved["written"] == 12
    assert saved["superseded_rate"] == 0.1
    assert saved["error"] is None
