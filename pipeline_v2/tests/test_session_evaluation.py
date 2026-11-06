import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline_v2 import config, vlm
from pipeline_v2.evaluation_session import select_stretch, report


def test_shared_budget_counts_uncached_attempts_and_not_cache_hits(
    isolated_data, monkeypatch
):
    clip = isolated_data / "clip.mkv"
    clip.write_bytes(b"video")
    cfg = replace(config.load_config(), api_key="test")
    calls = []

    def post(*args):
        calls.append(1)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(vlm, "_post", post)
    budget = vlm.CallBudget(1)
    vlm.request_raw(cfg, clip, [{"type": "text", "text": "one"}], call_budget=budget)
    vlm.request_raw(cfg, clip, [{"type": "text", "text": "one"}], call_budget=budget)
    with pytest.raises(vlm.BudgetExceeded, match="shared"):
        vlm.request_raw(
            cfg, clip, [{"type": "text", "text": "two"}], call_budget=budget
        )
    assert len(calls) == budget.attempted == 1


def test_ambiguous_network_attempt_reserves_budget(isolated_data, monkeypatch):
    clip = isolated_data / "clip.mkv"
    clip.write_bytes(b"video")
    cfg = replace(config.load_config(), api_key="test")

    def fail(*args):
        raise TimeoutError("unknown provider outcome")

    monkeypatch.setattr(vlm, "_post", fail)
    budget = vlm.CallBudget(1)
    with pytest.raises(TimeoutError):
        vlm.request_raw(cfg, clip, [], call_budget=budget)
    assert budget.attempted == 1


def test_selection_is_deterministic_and_excludes_used_or_discontinuous_chunks(tmp_path):
    run = tmp_path / "run-001"
    run.mkdir()
    for i in range(8):
        (run / f"part-{i}.capture.json").write_text(
            json.dumps(
                {
                    "video_file": f"part-{i}.mkv",
                    "segment_start_sec": i * 30,
                    "segment_end_sec": (i + 1) * 30,
                    "duration_sec": 30,
                }
            )
        )
    excluded = {(run / "part-3.mkv").resolve()}
    result = select_stretch(tmp_path, excluded, 3, 0)
    assert result == select_stretch(tmp_path, excluded, 3, 0)
    group, candidates = result
    assert candidates == 3
    assert all(Path(w["clip"]) not in excluded for w in group)
    for a, b in zip(group, group[1:]):
        assert a["segment_end"] == b["segment_start"]


def test_no_session_recall_until_human_review_and_all_chunks_complete(tmp_path):
    plan = {"duration": 30, "windows": [{"id": "chunk-00", "offset": 0}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    labels = {"events": [], "timeline_review_complete": True}
    (tmp_path / "user-labels.json").write_text(json.dumps(labels))
    assert report(tmp_path)["user_highlight_metrics"] is None
    directory = tmp_path / "chunk-00"
    directory.mkdir()
    (directory / "proposals.json").write_text(
        json.dumps(
            {
                "refined": [
                    {
                        "start": 1,
                        "end": 2,
                        "actor": "recording_player",
                        "kind": "elimination",
                    }
                ]
            }
        )
    )
    m = report(tmp_path)["user_highlight_metrics"]
    assert m["recall"] is None and m["precision"] == 0
    assert m["unwanted_proposals_per_hour"] == 120
    labels["timeline_review_complete"] = False
    (tmp_path / "user-labels.json").write_text(json.dumps(labels))
    assert report(tmp_path)["user_highlight_metrics"] is None
