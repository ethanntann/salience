import json
from dataclasses import replace

import pytest

from pipeline_v2 import cache, config, frames, localization, vlm


@pytest.mark.parametrize(
    "event",
    [
        {"start": 1, "end": float("nan")},
        {"start": 0, "end": 40},
        {"start": 3, "end": 2},
        {"start": True, "end": 3},
        {
            "start": 1,
            "end": 2,
            "kind": "gunfire",
            "actor": "recording_player",
            "evidence": "shot",
        },
    ],
)
def test_invalid_localization_is_rejected(event):
    with pytest.raises(ValueError):
        localization.validate_events({"events": [event]}, 0.1, 29.9)


def test_no_event_is_valid_and_spectator_ownership_survives():
    assert localization.validate_events({"events": []}, 0.1, 29.9) == []
    event = {
        "start": 12,
        "end": 14,
        "kind": "elimination",
        "actor": "spectated_player",
        "evidence": "opponent dissolves",
    }
    assert (
        localization.validate_events({"events": [event]}, 0.1, 29.9)[0]["actor"]
        == "spectated_player"
    )


def test_point_confirmation_brackets_samples_without_changing_raw_payload():
    payload = {
        "events": [
            {
                "start": 29.9,
                "end": 29.9,
                "kind": "death",
                "actor": "recording_player",
                "evidence": "placement screen",
            }
        ]
    }
    result = localization.bracket_point_events(payload, [27.9, 28.9, 29.9])
    assert result["events"][0]["start"] == 28.9
    assert payload["events"][0]["start"] == 29.9
    assert localization.validate_events(result, 27.9, 29.9)


def test_point_at_first_frame_is_not_a_located_new_shot():
    payload = {
        "events": [
            {
                "start": 0.1,
                "end": 0.1,
                "kind": "elimination",
                "actor": "recording_player",
                "evidence": "persistent banner",
            }
        ]
    }
    event = localization.bracket_point_events(payload, [0.1, 1.1])["events"][0]
    assert event["timing_adjustment"] == "confirmation_already_visible_at_first_frame"
    assert localization.shot_window(event, 30) is None


def test_off_grid_point_is_not_given_invented_timing():
    payload = {
        "events": [
            {
                "start": 13.456,
                "end": 13.456,
                "kind": "death",
                "actor": "recording_player",
                "evidence": "screen",
            }
        ]
    }
    result = localization.bracket_point_events(payload, [12, 14, 16])
    with pytest.raises(ValueError):
        localization.validate_events(result, 12, 16)


def test_shot_window_is_only_for_narrow_own_eliminations():
    event = {
        "start": 13.3,
        "end": 13.6,
        "kind": "elimination",
        "actor": "recording_player",
    }
    assert localization.shot_window(event, 30) == (12.3, 13.75)
    assert localization.shot_window({**event, "actor": "spectated_player"}, 30) is None
    assert localization.shot_window({**event, "end": 15}, 30) is None
    assert localization.shot_window({**event, "kind": "victory"}, 30) is None
    assert localization.shot_window({**event, "start": 0.1, "end": 0.3}, 30)[0] == 0.1


def test_empty_scan_does_not_refine_and_changed_images_fail(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"video")
    out = tmp_path / "study"
    monkeypatch.setattr(frames, "probe_duration", lambda _: 30)
    calls = []

    def extract(clip, duration, midpoint, *, pre_sec, post_sec, samples, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for i in range(samples):
            path = out_dir / f"{i}.jpg"
            path.write_bytes(b"image")
            result.append(
                frames.Frame(
                    path,
                    round(
                        midpoint - pre_sec + (pre_sec + post_sec) * i / (samples - 1), 3
                    ),
                )
            )
        return result

    monkeypatch.setattr(frames, "window_frames", extract)

    def request(*args, **kwargs):
        calls.append(1)
        return cache.CachedResponse('{"events":[]}', 100, 10, 0), True

    monkeypatch.setattr(vlm, "request_raw", request)
    assert localization.locate(clip, out, 0) == []
    assert len(calls) == 1 and not (out / "refine-0.json").exists()
    assert localization.locate(clip, out, 0) == [] and len(calls) == 1
    (out / "coarse/0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frames changed"):
        localization.locate(clip, out, 0)


def test_shared_transport_preserves_invalid_localization_without_rebilling(
    isolated_data, monkeypatch
):
    clip = isolated_data / "clip.mkv"
    clip.write_bytes(b"video")
    cfg = replace(
        config.load_config(), api_key="test", prompt_version="localization-test"
    )
    calls = []

    def post(*args):
        calls.append(1)
        return {
            "choices": [{"message": {"content": json.dumps({"events": "invalid"})}}]
        }

    monkeypatch.setattr(vlm, "_post", post)
    for _ in range(2):
        response, _ = vlm.request_raw(cfg, clip, [{"type": "text", "text": "locate"}])
        with pytest.raises(ValueError):
            localization.validate_events(vlm._extract_json(response.raw), 0, 30)
    assert len(calls) == 1
