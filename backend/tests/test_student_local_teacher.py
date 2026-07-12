from pathlib import Path

import numpy as np
import pytest

from salience_api.features.fireworks_teacher import ClipTeacherInput
from salience_api.student.event_heads import EventHeadPrediction
from salience_api.student.backbone import CONTEXT_HEADS
from salience_api.student.local_teacher import LocalTeacherClient
from salience_api.student.onnx_runtime import StudentOnnxModels
from salience_api.student.schema import AIM_STATES, EVENT_KINDS, TARGET_STATES, WEAPON_CLASSES


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
        "event_kind_logits": _one_hot_logits(
            len(EVENT_KINDS), EVENT_KINDS.index("elimination"), batch=batch
        ),
        "target_state_logits": _one_hot_logits(
            len(TARGET_STATES), TARGET_STATES.index("active"), batch=batch
        ),
        "weapon_logits": _one_hot_logits(
            len(WEAPON_CLASSES), WEAPON_CLASSES.index("shotgun"), batch=batch
        ),
        "aim_state_logits": _one_hot_logits(
            len(AIM_STATES), AIM_STATES.index("hipfire"), batch=batch
        ),
        "damaging_shot_count": np.array([[1.0]] * batch, dtype=np.float32),
        "single_shot_damage": np.array([[1.1]] * batch, dtype=np.float32),
        **{
            f"context_{name}_logits": _one_hot_logits(
                3, 0 if name in {"combat_visible", "enemy_visible"} else 1, batch=batch
            )
            for name in CONTEXT_HEADS
        },
        **evidence,
    }


def _fake_nchw(count: int = 1) -> list[np.ndarray]:
    return [np.zeros((3, 224, 224), dtype=np.float32) for _ in range(count)]


def _shotgun_client(*, locator_scores: list[float] | None = None) -> LocalTeacherClient:
    scores = locator_scores or [0.9]
    locator = _MockOnnxSession(
        input_name="images",
        outputs={
            "logits": np.array([[score] for score in scores], dtype=np.float32)
        },
    )
    models = StudentOnnxModels.from_sessions(
        locator_session=locator,
        event_heads_session=_MockOnnxSession(
            input_name="images", outputs=_event_head_outputs()
        ),
    )
    return LocalTeacherClient.from_sessions(models=models, model="test-student")


def _clip_with_event(*, with_ocr: bool = False) -> ClipTeacherInput:
    paths = [Path(f"frame-{index}.jpg") for index in range(3)]
    ocr_observations = (
        [
            {
                "event_index": 0,
                "timestamp": 4.8,
                "text": "Pump Shotgun",
                "confidence": 0.95,
                "category": "shotgun",
            }
        ]
        if with_ocr
        else []
    )
    return ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=60.0,
        tags=[],
        image_paths=paths,
        image_timestamps=[4.5, 5.0, 5.5],
        image_views=["event composite"] * 3,
        image_event_indices=[0, 0, 0],
        image_event_centers=[5.0, 5.0, 5.0],
        ocr_observations=ocr_observations,
    )


@pytest.fixture
def stub_frames(monkeypatch):
    def _load(_paths: list[Path]) -> list[np.ndarray]:
        return _fake_nchw(len(_paths))

    monkeypatch.setattr(
        "salience_api.student.local_teacher.load_frames_nchw", _load
    )
    monkeypatch.setattr(
        "salience_api.student.local_teacher.load_hud_frames_nchw", _load
    )


def test_label_weapon_event_shotgun_elimination(stub_frames):
    client = _shotgun_client()
    clip = _clip_with_event()

    attribution = client.label_weapon_event(clip)

    assert attribution.labels["shotgun_kill"] == "yes"
    assert attribution.status == "attributed"
    assert len(attribution.events) == 1
    assert attribution.events[0]["resolved_weapon"] == "shotgun"


def test_label_weapon_event_applies_ocr_when_present(stub_frames):
    client = _shotgun_client()
    clip = _clip_with_event(with_ocr=True)

    attribution = client.label_weapon_event(clip)

    assert attribution.labels["shotgun_kill"] == "yes"
    assert attribution.events[0].get("local_ocr", {}).get("applied") is True


def test_locate_event_timestamps_peak_decode_and_cap(stub_frames):
    # Locator mock emits raw logits; runtime applies sigmoid.
    client = _shotgun_client(locator_scores=[-2.0, 3.0, -2.0, 2.5, -3.0])
    clip = ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=60.0,
        tags=[],
        image_paths=[Path(f"f{i}.jpg") for i in range(5)],
        image_timestamps=[1.0, 5.0, 10.0, 15.0, 20.0],
    )

    timestamps = client.locate_event_timestamps(clip)

    assert timestamps == [5.0, 15.0]
    assert len(timestamps) <= 4


def test_locate_event_timestamps_falls_back_to_best_frame_below_threshold(stub_frames):
    # Every raw logit is negative -> sigmoid < 0.5 for every frame, so the
    # default peak-decode would return [] and merge_event_labels would force
    # every event-owned label to "uncertain". The fallback should instead
    # attempt attribution on the single best-scoring frame.
    client = _shotgun_client(locator_scores=[-3.0, -1.0, -2.0])
    clip = ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=60.0,
        tags=[],
        image_paths=[Path(f"f{i}.jpg") for i in range(3)],
        image_timestamps=[1.0, 2.0, 3.0],
    )

    timestamps = client.locate_event_timestamps(clip)

    assert timestamps == [2.0]


def test_label_clip_combat_visible_when_attributed_finish(stub_frames):
    client = _shotgun_client()
    clip = _clip_with_event()

    labels = client.label_clip(clip)

    assert labels.combat_visible == "yes"
    assert labels.enemy_visible == "yes"
    assert labels.shotgun_kill == "uncertain"


def test_provider_and_model_fields():
    client = _shotgun_client()

    assert client.provider == "local"
    assert client.model == "test-student"


def test_label_clip_marks_victory_when_banner_detected(stub_frames, tmp_path, monkeypatch):
    from PIL import Image

    from salience_api.features.hud_ocr import RapidHudOcr

    client = _shotgun_client()
    frame_path = tmp_path / "frame-0.jpg"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(frame_path)
    clip = ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=60.0,
        tags=[],
        image_paths=[frame_path],
        image_timestamps=[29.0],
        image_views=["full gameplay frame"],
    )

    class _StubResult:
        def __init__(self, texts):
            self.txts = texts
            self.scores = [0.9] * len(texts)

    class _StubEngine:
        def __call__(self, image):
            return _StubResult(["#1 VICTORY ROYALE"])

    ocr = RapidHudOcr(enabled=True)
    monkeypatch.setattr(ocr, "_get_engine", lambda: _StubEngine())

    labels = client.label_clip(clip, ocr=ocr)

    assert labels.victory == "yes"


def test_label_clip_marks_victory_from_split_ocr_boxes(stub_frames, tmp_path, monkeypatch):
    from PIL import Image

    from salience_api.features.hud_ocr import RapidHudOcr

    client = _shotgun_client()
    frame_paths = []
    for index in range(6):
        frame_path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(frame_path)
        frame_paths.append(frame_path)
    clip = ClipTeacherInput(
        filename="clip.mp4",
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=60.0,
        tags=[],
        image_paths=frame_paths,
        image_timestamps=[5.0 * i for i in range(6)],
        image_views=["full gameplay frame"] * 6,
    )

    class _StubResult:
        def __init__(self, texts):
            self.txts = texts
            self.scores = [0.9] * len(texts)

    class _SplitBoxEngine:
        # Banner only visible on the final frame, split into two OCR boxes.
        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            if self.calls >= 6:
                return _StubResult(["VICTORY", "ROYALE"])
            return _StubResult(["100 HP"])

    engine = _SplitBoxEngine()
    ocr = RapidHudOcr(enabled=True)
    monkeypatch.setattr(ocr, "_get_engine", lambda: engine)

    labels = client.label_clip(clip, ocr=ocr)

    assert labels.victory == "yes"


def test_label_clip_defaults_to_no_ocr_and_keeps_prior_behavior(stub_frames):
    client = _shotgun_client()
    clip = _clip_with_event()

    labels = client.label_clip(clip)  # no ocr argument at all

    assert labels.victory == "no"
