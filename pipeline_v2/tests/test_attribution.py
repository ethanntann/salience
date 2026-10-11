import hashlib
import json

import pytest

from pipeline_v2 import config, prompt
from pipeline_v2.evaluation_attribution import file_sha, prompt_sha, run, variant
from pipeline_v2.facts import ClipFacts
from pipeline_v2.schema import ClipDescription


def test_baseline_frozen_prompt_is_unchanged(monkeypatch):
    monkeypatch.delenv("V2_ABLATE", raising=False)
    assert (
        prompt_sha()
        == "c6451c56c64972dda01e4c9c1f52b64a55980b2e275e5d9872a9ae200d9aeb67"
    )
    assert config.load_config().prompt_version == "v3.3"


def test_capture_attribution_is_opt_in_and_restores_environment(monkeypatch):
    monkeypatch.setenv("V2_ABLATE", "track")
    with variant("attribution"):
        text = prompt.build_prompt(ClipFacts(event_source="capture_window"))
        assert '"viewpoint"' in text and '"elimination_weapon_name"' in text
        assert config.load_config().prompt_version == "v3.4-attribution-dev"
    assert config.load_config().prompt_version == "v3.2b-track"


def test_legacy_descriptions_do_not_acquire_own_player_attribution():
    old = ClipDescription(summary="An opponent is eliminated.", outcome="elimination")
    assert old.viewpoint == old.event_actor == old.elimination_weapon_name == "unknown"
    assert old.weapon_evidence_frames == []
    with pytest.raises(ValueError):
        ClipDescription(
            summary="An opponent is eliminated.", weapon_evidence_frames=[0]
        )


def test_evidence_audit_does_not_certify_visual_correctness():
    from pipeline_v2.evaluation_attribution import contract_issues

    d = {
        "outcome": "death",
        "elimination_weapon_name": "Pump Shotgun",
        "weapon_evidence_frames": [6, 7],
        "viewpoint": "spectating",
        "event_actor": "recording_player",
    }
    assert len(contract_issues(d, 7)) == 2
    d.update(outcome="elimination", event_actor="spectated_player")
    assert contract_issues(d, 7) == []
    assert contract_issues(d, 5)
    d["weapon_evidence_frames"] = []
    assert contract_issues(d, 7)


def test_changed_comparison_image_stops_before_paid_inference(tmp_path, monkeypatch):
    from pipeline_v2 import evaluation_attribution as experiment

    monkeypatch.delenv("V2_ABLATE", raising=False)
    cfg = config.load_config()
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"video")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"changed")
    (tmp_path / "results").mkdir()
    plan = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "frame_width": experiment.frames.FRAME_WIDTH,
        "prompts": {
            "baseline": {
                "prompt_version": cfg.prompt_version,
                "prompt_sha": prompt_sha(),
            }
        },
        "arms": [
            {
                "id": "one",
                "variant": "baseline",
                "clip": str(clip),
                "clip_sha": file_sha(clip),
                "images": [
                    {
                        "path": str(frame),
                        "sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ],
            }
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))

    def fail(*args, **kwargs):
        pytest.fail("Paid inference must not run on altered inputs")

    monkeypatch.setattr(experiment.vlm, "describe", fail)
    with pytest.raises(ValueError, match="input changed"):
        run(tmp_path, 1)


def test_cross_chunk_input_checks_additional_source(tmp_path):
    from pipeline_v2.evaluation_attribution import validate_inputs

    current, previous = tmp_path / "current.mkv", tmp_path / "previous.mkv"
    current.write_bytes(b"current")
    previous.write_bytes(b"previous")
    arm = {
        "clip": str(current),
        "clip_sha": file_sha(current),
        "images": [],
        "source_segments": [{"clip": str(previous), "sha256": file_sha(previous)}],
    }
    validate_inputs(arm)
    previous.write_bytes(b"altered")
    with pytest.raises(ValueError, match="Additional source chunk"):
        validate_inputs(arm)
