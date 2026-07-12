import numpy as np
import pytest

from salience_api.student.event_heads import EventHeadPrediction
from salience_api.student.backbone import CONTEXT_HEADS
from salience_api.student.onnx_runtime import (
    IMAGE_SIZE,
    StudentOnnxModels,
    preprocess_frame_nchw,
)
from salience_api.student.schema import (
    AIM_STATES,
    EVENT_KINDS,
    TARGET_STATES,
    WEAPON_CLASSES,
)

_EVIDENCE_STATES = ("yes", "no", "unknown")


def _one_hot_logits(num_classes: int, index: int, *, batch: int = 1) -> np.ndarray:
    logits = np.full((batch, num_classes), -8.0, dtype=np.float32)
    logits[:, index] = 8.0
    return logits


class _MockOnnxSession:
    def __init__(self, *, input_name: str, outputs: dict[str, np.ndarray]) -> None:
        self._input_name = input_name
        self._outputs = outputs
        self.input_names = [input_name]
        self.output_names = list(outputs.keys())

    def run(self, output_names: list[str], feed_dict: dict) -> list[np.ndarray]:
        batch = next(iter(feed_dict.values())).shape[0]
        result = []
        for name in output_names:
            value = self._outputs[name]
            if value.shape[0] == 1 and batch > 1:
                result.append(np.repeat(value, batch, axis=0))
            else:
                result.append(value)
        return result


def _fake_nchw(count: int = 1) -> list[np.ndarray]:
    return [np.zeros((3, 224, 224), dtype=np.float32) for _ in range(count)]


def _event_head_outputs(*, batch: int = 1) -> dict[str, np.ndarray]:
    evidence = {
        "pov_shot_visible_logits": _one_hot_logits(3, 0, batch=batch),
        "target_reaction_visible_logits": _one_hot_logits(3, 2, batch=batch),
        "new_damage_visible_logits": _one_hot_logits(3, 0, batch=batch),
        "target_defeat_visible_logits": _one_hot_logits(3, 0, batch=batch),
        "finish_ui_newly_appeared_logits": _one_hot_logits(3, 0, batch=batch),
        "kill_feed_corroborates_pov_logits": _one_hot_logits(3, 2, batch=batch),
        "visual_action_supported_logits": _one_hot_logits(3, 2, batch=batch),
        "stationary_target_logits": _one_hot_logits(3, 2, batch=batch),
        "stationary_duration_supported_logits": _one_hot_logits(3, 2, batch=batch),
        "flick_shot_logits": _one_hot_logits(3, 2, batch=batch),
        "trickshot_logits": _one_hot_logits(3, 2, batch=batch),
        "cleanup_kill_logits": _one_hot_logits(3, 2, batch=batch),
        "opponent_likely_bot_logits": _one_hot_logits(3, 2, batch=batch),
        "damage_display_is_cumulative_logits": _one_hot_logits(3, 1, batch=batch),
        "single_shot_damage_known_logits": _one_hot_logits(3, 0, batch=batch),
    }
    return {
        "event_kind_logits": _one_hot_logits(len(EVENT_KINDS), EVENT_KINDS.index("elimination"), batch=batch),
        "target_state_logits": _one_hot_logits(len(TARGET_STATES), TARGET_STATES.index("active"), batch=batch),
        "weapon_logits": _one_hot_logits(len(WEAPON_CLASSES), WEAPON_CLASSES.index("shotgun"), batch=batch),
        "aim_state_logits": _one_hot_logits(len(AIM_STATES), AIM_STATES.index("hipfire"), batch=batch),
        "damaging_shot_count": np.array([[2.0]] * batch, dtype=np.float32),
        "single_shot_damage": np.array([[1.24]] * batch, dtype=np.float32),
        **{
            f"context_{name}_logits": _one_hot_logits(3, 1, batch=batch)
            for name in CONTEXT_HEADS
        },
        **evidence,
    }


class _OrtStyleInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _OrtStyleOnnxSession:
    """Mock real onnxruntime.InferenceSession (get_inputs, no input_names)."""

    def __init__(self, *, input_name: str, outputs: dict[str, np.ndarray]) -> None:
        self._input_name = input_name
        self._outputs = outputs
        self.output_names = list(outputs.keys())

    def get_inputs(self) -> list[_OrtStyleInput]:
        return [_OrtStyleInput(self._input_name)]

    def run(self, output_names: list[str], feed_dict: dict) -> list[np.ndarray]:
        batch = next(iter(feed_dict.values())).shape[0]
        result = []
        for name in output_names:
            value = self._outputs[name]
            if value.shape[0] == 1 and batch > 1:
                result.append(np.repeat(value, batch, axis=0))
            else:
                result.append(value)
        return result


def test_session_input_name_from_get_inputs():
    locator = _OrtStyleOnnxSession(
        input_name="images",
        outputs={"logits": np.array([[0.0]], dtype=np.float32)},
    )
    event_heads = _OrtStyleOnnxSession(
        input_name="images",
        outputs=_event_head_outputs(),
    )
    models = StudentOnnxModels.from_sessions(
        locator_session=locator,
        event_heads_session=event_heads,
    )

    scores = models.score_frames(_fake_nchw(1))
    assert scores == [pytest.approx(0.5)]


def test_score_frames_applies_sigmoid_to_locator_logits():
    locator = _MockOnnxSession(
        input_name="images",
        outputs={"logits": np.array([[0.0], [2.0]], dtype=np.float32)},
    )
    models = StudentOnnxModels.from_sessions(
        locator_session=locator,
        event_heads_session=_MockOnnxSession(input_name="images", outputs=_event_head_outputs()),
    )

    scores = models.score_frames(_fake_nchw(2))

    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.880797, rel=1e-4)


def test_predict_event_decodes_mock_head_logits():
    event_heads = _MockOnnxSession(
        input_name="images",
        outputs=_event_head_outputs(batch=2),
    )
    models = StudentOnnxModels.from_sessions(
        locator_session=_MockOnnxSession(
            input_name="images",
            outputs={"logits": np.array([[1.0]], dtype=np.float32)},
        ),
        event_heads_session=event_heads,
    )

    pred = models.predict_event(_fake_nchw(2))

    assert isinstance(pred, EventHeadPrediction)
    assert pred.event_kind == "elimination"
    assert pred.target_state == "active"
    assert pred.weapon == "shotgun"
    assert pred.aim_state == "hipfire"
    assert pred.weapon_confidence == pytest.approx(1.0, abs=1e-4)
    assert pred.pov_shot_visible == "yes"
    assert pred.new_damage_visible == "yes"
    assert pred.target_defeat_visible == "yes"
    assert pred.finish_ui_newly_appeared == "yes"
    assert pred.damaging_shot_count == 2
    assert pred.single_shot_damage == 124


def test_decode_event_logits_damage_cutoff_drops_regressed_claim():
    from salience_api.student.onnx_runtime import decode_event_logits

    outputs = _event_head_outputs()

    # Regression says 124 damage and the "known" head says yes.
    default_pred = decode_event_logits(outputs)
    assert default_pred.single_shot_damage_known == "yes"
    assert default_pred.single_shot_damage == 124

    # A calibrated cutoff above the regressed value drops the claim entirely.
    gated_pred = decode_event_logits(
        outputs, thresholds={"single_shot_damage_cutoff": 140.0}
    )
    assert gated_pred.single_shot_damage_known == "no"


def test_preprocess_frame_nchw_normalizes_hwc_uint8():
    frame = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    nchw = preprocess_frame_nchw(frame)
    assert nchw.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert nchw.dtype == np.float32
    expected = (1.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    np.testing.assert_allclose(nchw[:, 0, 0], expected, rtol=1e-5)


def test_predict_context_evenly_subsamples_long_timelines():
    captured: dict[str, object] = {}

    class _CaptureSession(_MockOnnxSession):
        def __init__(self) -> None:
            super().__init__(
                input_name="images",
                outputs={
                    f"context_{name}_logits": _one_hot_logits(3, 0)
                    for name in CONTEXT_HEADS
                },
            )
            self.input_names = ["images", "hud_images"]

        def run(self, output_names: list[str], feed_dict: dict):
            captured["images"] = feed_dict["images"]
            return super().run(output_names, feed_dict)

    models = StudentOnnxModels.from_sessions(
        locator_session=_MockOnnxSession(
            input_name="images",
            outputs={"logits": np.array([[1.0]], dtype=np.float32)},
        ),
        event_heads_session=_CaptureSession(),
    )

    # Distinct frame values so we can verify even subsample indices [0,3,6,10,13,16,19].
    frames = [
        np.full((3, IMAGE_SIZE, IMAGE_SIZE), float(index), dtype=np.float32)
        for index in range(20)
    ]
    labels = models.predict_context(frames)

    assert labels["combat_visible"] == "yes"
    window = captured["images"]
    assert window.shape == (1, 7, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert [float(window[0, index, 0, 0, 0]) for index in range(7)] == [
        0.0,
        3.0,
        6.0,
        10.0,
        13.0,
        16.0,
        19.0,
    ]


def test_from_artifacts_requires_onnx_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="locator.onnx"):
        StudentOnnxModels.from_artifacts(tmp_path)

    (tmp_path / "locator.onnx").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="event_heads.onnx"):
        StudentOnnxModels.from_artifacts(tmp_path)


def test_predict_event_respects_custom_weapon_threshold():
    # Weapon logits give the "shotgun" class a moderate lead over the rest
    # (softmax prob ~0.62 with 6 classes) - above the default 0.55 cutoff but
    # below a stricter 0.65 cutoff, which should fall back to "unknown".
    logits = np.zeros((1, len(WEAPON_CLASSES)), dtype=np.float32)
    shotgun_index = WEAPON_CLASSES.index("shotgun")
    logits[:, shotgun_index] = 2.1

    event_heads = _MockOnnxSession(
        input_name="images",
        outputs={**_event_head_outputs(), "weapon_logits": logits},
    )
    default_models = StudentOnnxModels.from_sessions(
        locator_session=_MockOnnxSession(
            input_name="images", outputs={"logits": np.array([[1.0]], dtype=np.float32)}
        ),
        event_heads_session=event_heads,
    )
    strict_models = StudentOnnxModels.from_sessions(
        locator_session=_MockOnnxSession(
            input_name="images", outputs={"logits": np.array([[1.0]], dtype=np.float32)}
        ),
        event_heads_session=event_heads,
        thresholds={"weapon": 0.65},
    )

    assert default_models.predict_event(_fake_nchw(1)).weapon == "shotgun"
    assert strict_models.predict_event(_fake_nchw(1)).weapon == "unknown"


def test_raw_event_logits_exposes_uncalibrated_outputs():
    event_heads = _MockOnnxSession(input_name="images", outputs=_event_head_outputs())
    models = StudentOnnxModels.from_sessions(
        locator_session=_MockOnnxSession(
            input_name="images", outputs={"logits": np.array([[1.0]], dtype=np.float32)}
        ),
        event_heads_session=event_heads,
    )

    by_name = models.raw_event_logits(_fake_nchw(1))

    assert "weapon_logits" in by_name
    assert by_name["weapon_logits"].shape[-1] == len(WEAPON_CLASSES)


def test_decode_event_logits_is_a_pure_function_of_cached_logits():
    from salience_api.student.onnx_runtime import decode_event_logits

    by_name = _event_head_outputs()
    prediction = decode_event_logits(by_name)
    assert prediction.event_kind == "elimination"
    assert prediction.weapon == "shotgun"


def test_load_thresholds_json_helper(tmp_path):
    import json

    from salience_api.student.onnx_runtime import _load_thresholds

    assert _load_thresholds(tmp_path) == {}
    (tmp_path / "thresholds.json").write_text(
        json.dumps({"weapon": 0.65}), encoding="utf-8"
    )
    assert _load_thresholds(tmp_path) == {"weapon": 0.65}
