"""ONNX Runtime wrapper for student locator and event-head models.

Preprocess contract (runtime input to both models):
- Shape: ``(3, 224, 224)`` float32 NCHW
- Color: RGB
- Normalization: ImageNet mean/std per channel
  (mean ``[0.485, 0.456, 0.406]``, std ``[0.229, 0.224, 0.225]``)
- Callers may pass already-preprocessed tensors; ``preprocess_frame_nchw`` is
  provided for uint8 HWC frames that are already resized to 224×224.

Locator ONNX I/O:
- input ``images``: float32 ``[batch, 3, 224, 224]``
- output ``logits``: float32 ``[batch, 1]`` eventness logits (sigmoid at runtime)

Event-head ONNX I/O (multi-head logits, class order matches ``schema.py`` tuples):
- input ``images``: float32 ``[batch, 7, 3, 224, 224]`` ordered scene windows
  (longer clip timelines are evenly subsampled to 7 frames; same policy as training)
- input ``hud_images``: same shape, cropped selected-weapon HUD windows
- outputs: ``event_kind_logits``, ``target_state_logits``, ``weapon_logits``,
  ``aim_state_logits``, evidence ``*_logits`` (yes/no/unknown), and
  ``damaging_shot_count`` (scalar regression per frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from salience_api.student.event_heads import EventHeadPrediction
from salience_api.student.backbone import (
    ARTIFACT_VERSION,
    CONTEXT_HEADS,
    SUPPORTED_ARTIFACT_VERSIONS,
)
from salience_api.student.schema import (
    AIM_STATES,
    EVENT_KINDS,
    TARGET_STATES,
    WEAPON_CLASSES,
    EVENT_WINDOW_FRAMES,
    select_window_indices,
)

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_EVIDENCE_HEADS = (
    "pov_shot_visible",
    "target_reaction_visible",
    "new_damage_visible",
    "target_defeat_visible",
    "finish_ui_newly_appeared",
    "kill_feed_corroborates_pov",
    "visual_action_supported",
    "stationary_target",
    "stationary_duration_supported",
    "flick_shot",
    "trickshot",
    "cleanup_kill",
    "opponent_likely_bot",
    "damage_display_is_cumulative",
    "single_shot_damage_known",
)
_EVIDENCE_STATES = ("yes", "no", "unknown")

_LOCATOR_OUTPUT = "logits"
_EVENT_OUTPUTS = (
    "event_kind_logits",
    "target_state_logits",
    "weapon_logits",
    "aim_state_logits",
    "damaging_shot_count",
    "single_shot_damage",
    *(
        f"{name}_logits"
        for name in _EVIDENCE_HEADS
    ),
)


class _OnnxSession(Protocol):
    input_names: list[str]
    output_names: list[str]

    def run(self, output_names: list[str], feed_dict: dict[str, Any]) -> list[np.ndarray]: ...


def preprocess_frame_nchw(frame: np.ndarray) -> np.ndarray:
    """Convert a 224×224 RGB frame to ImageNet-normalized NCHW float32."""
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE):
        normalized = np.asarray(array, dtype=np.float32)
        if normalized.max() > 1.5:
            normalized = normalized / 255.0
        mean = IMAGENET_MEAN[:, None, None]
        std = IMAGENET_STD[:, None, None]
        return ((normalized - mean) / std).astype(np.float32)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("frame must be HWC RGB or CHW with 3 channels")
    if array.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"frame must be {IMAGE_SIZE}x{IMAGE_SIZE} before preprocessing")

    rgb = np.asarray(array, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    return ((chw - mean) / std).astype(np.float32)


def _stack_batch(images_nchw: Sequence[np.ndarray]) -> np.ndarray:
    if not images_nchw:
        raise ValueError("images_nchw must contain at least one frame")
    batch = [np.asarray(frame, dtype=np.float32) for frame in images_nchw]
    for frame in batch:
        if frame.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"each frame must be (3, {IMAGE_SIZE}, {IMAGE_SIZE}); got {frame.shape}"
            )
    return np.stack(batch, axis=0)


def _stack_event_window(images_nchw: Sequence[np.ndarray]) -> np.ndarray:
    frames = list(images_nchw)
    if not frames:
        raise ValueError("event window must contain at least one frame")
    selected = [
        np.asarray(frames[index])
        for index in select_window_indices(len(frames), EVENT_WINDOW_FRAMES)
    ]
    while len(selected) < EVENT_WINDOW_FRAMES:
        selected.append(np.asarray(selected[-1]).copy())
    return _stack_batch(selected)[None, ...]


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _mean_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return _softmax(logits[None, :])[0]
    return _softmax(logits).mean(axis=0)


DEFAULT_THRESHOLDS: dict[str, float] = {
    "event_kind": 0.5,
    "target_state": 0.55,
    "weapon": 0.55,
    "aim_state": 0.55,
}
DEFAULT_EVIDENCE_THRESHOLD = 0.6
# The damage regression is trained only on events where the teacher read a
# damage number, so it regresses toward ~100+ on every event. Claims below
# this cutoff are dropped; calibration may raise it above 100.
DEFAULT_SINGLE_SHOT_DAMAGE_CUTOFF = 100.0


def _resolve_threshold(thresholds: dict[str, float], field: str, default: float) -> float:
    return float(thresholds.get(field, default))


def _class_label(
    classes: tuple[str, ...],
    logits: np.ndarray,
    *,
    fallback: str | None = None,
    min_confidence: float = 0.0,
) -> str:
    probs = _mean_softmax(logits)
    index = int(np.argmax(probs))
    if fallback is not None and float(probs[index]) < min_confidence:
        return fallback
    return classes[index]


def _evidence_label(logits: np.ndarray, *, threshold: float = DEFAULT_EVIDENCE_THRESHOLD) -> str:
    return _class_label(
        _EVIDENCE_STATES,
        logits,
        fallback="unknown",
        min_confidence=threshold,
    )


def decode_event_logits(
    by_name: dict[str, np.ndarray], *, thresholds: dict[str, float] | None = None
) -> EventHeadPrediction:
    thresholds = thresholds or {}
    weapon_logits = by_name["weapon_logits"]
    weapon_probs = _mean_softmax(weapon_logits)

    damaging = by_name["damaging_shot_count"].reshape(-1)
    damaging_shot_count = int(round(float(np.mean(damaging))))
    damage = by_name["single_shot_damage"].reshape(-1)
    single_shot_damage = int(round(max(0.0, float(np.mean(damage))) * 100.0))

    def _evidence(field: str) -> str:
        return _evidence_label(
            by_name[f"{field}_logits"],
            threshold=_resolve_threshold(thresholds, field, DEFAULT_EVIDENCE_THRESHOLD),
        )

    single_shot_damage_known = _evidence("single_shot_damage_known")
    damage_cutoff = _resolve_threshold(
        thresholds, "single_shot_damage_cutoff", DEFAULT_SINGLE_SHOT_DAMAGE_CUTOFF
    )
    if single_shot_damage < damage_cutoff:
        single_shot_damage_known = "no"

    return EventHeadPrediction(
        event_kind=_class_label(
            EVENT_KINDS,
            by_name["event_kind_logits"],
            fallback="none",
            min_confidence=_resolve_threshold(
                thresholds, "event_kind", DEFAULT_THRESHOLDS["event_kind"]
            ),
        ),
        target_state=_class_label(
            TARGET_STATES,
            by_name["target_state_logits"],
            fallback="unknown",
            min_confidence=_resolve_threshold(
                thresholds, "target_state", DEFAULT_THRESHOLDS["target_state"]
            ),
        ),
        weapon=_class_label(
            WEAPON_CLASSES,
            weapon_logits,
            fallback="unknown",
            min_confidence=_resolve_threshold(thresholds, "weapon", DEFAULT_THRESHOLDS["weapon"]),
        ),
        weapon_confidence=float(np.max(weapon_probs)),
        aim_state=_class_label(
            AIM_STATES,
            by_name["aim_state_logits"],
            fallback="unknown",
            min_confidence=_resolve_threshold(
                thresholds, "aim_state", DEFAULT_THRESHOLDS["aim_state"]
            ),
        ),
        pov_shot_visible=_evidence("pov_shot_visible"),
        target_reaction_visible=_evidence("target_reaction_visible"),
        new_damage_visible=_evidence("new_damage_visible"),
        target_defeat_visible=_evidence("target_defeat_visible"),
        finish_ui_newly_appeared=_evidence("finish_ui_newly_appeared"),
        kill_feed_corroborates_pov=_evidence("kill_feed_corroborates_pov"),
        visual_action_supported=_evidence("visual_action_supported"),
        damaging_shot_count=max(0, damaging_shot_count),
        stationary_target=_evidence("stationary_target"),
        stationary_duration_supported=_evidence("stationary_duration_supported"),
        flick_shot=_evidence("flick_shot"),
        trickshot=_evidence("trickshot"),
        cleanup_kill=_evidence("cleanup_kill"),
        opponent_likely_bot=_evidence("opponent_likely_bot"),
        damage_display_is_cumulative=_evidence("damage_display_is_cumulative"),
        single_shot_damage_known=single_shot_damage_known,
        single_shot_damage=single_shot_damage,
    )


def decode_context_logits(
    by_name: dict[str, np.ndarray], *, thresholds: dict[str, float] | None = None
) -> dict[str, str]:
    thresholds = thresholds or {}
    result: dict[str, str] = {}
    for name in CONTEXT_HEADS:
        logits = by_name[f"context_{name}_logits"]
        result[name] = _evidence_label(
            logits,
            threshold=_resolve_threshold(thresholds, name, DEFAULT_EVIDENCE_THRESHOLD),
        )
    return result


def _load_thresholds(artifacts_dir: Path) -> dict[str, float]:
    path = Path(artifacts_dir) / "thresholds.json"
    if not path.is_file():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def _session_input_name(session: _OnnxSession) -> str:
    names = getattr(session, "input_names", None)
    if names:
        return names[0]
    get_inputs = getattr(session, "get_inputs", None)
    if callable(get_inputs):
        inputs = get_inputs()
        if inputs:
            return inputs[0].name
    raise AttributeError("ONNX session has no input name")


def _session_input_names(session: _OnnxSession) -> list[str]:
    names = getattr(session, "input_names", None)
    if names:
        return list(names)
    get_inputs = getattr(session, "get_inputs", None)
    if callable(get_inputs):
        return [item.name for item in get_inputs()]
    raise AttributeError("ONNX session has no input names")


class StudentOnnxModels:
    """Loads and runs student locator + event-head ONNX sessions."""

    def __init__(
        self,
        *,
        locator_session: _OnnxSession,
        event_heads_session: _OnnxSession,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._locator_session = locator_session
        self._event_heads_session = event_heads_session
        self._locator_input = _session_input_name(locator_session)
        self._event_inputs = _session_input_names(event_heads_session)
        self._event_input = self._event_inputs[0]
        self._thresholds = dict(thresholds or {})

    @classmethod
    def from_sessions(
        cls,
        *,
        locator_session: _OnnxSession,
        event_heads_session: _OnnxSession,
        thresholds: dict[str, float] | None = None,
    ) -> StudentOnnxModels:
        return cls(
            locator_session=locator_session,
            event_heads_session=event_heads_session,
            thresholds=thresholds,
        )

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> StudentOnnxModels:
        import json
        import onnxruntime as ort

        root = Path(artifacts_dir)
        locator_path = root / "locator.onnx"
        event_heads_path = root / "event_heads.onnx"
        if not locator_path.is_file():
            raise FileNotFoundError(f"locator.onnx not found under {root}")
        if not event_heads_path.is_file():
            raise FileNotFoundError(f"event_heads.onnx not found under {root}")
        meta_path = root / "artifact_meta.json"
        if meta_path.is_file():
            version = str(json.loads(meta_path.read_text(encoding="utf-8")).get("version", ""))
            if version not in SUPPORTED_ARTIFACT_VERSIONS:
                raise RuntimeError(
                    f"student artifact version {version!r} is incompatible; "
                    f"expected one of {sorted(SUPPORTED_ARTIFACT_VERSIONS)!r}"
                )

        return cls(
            locator_session=ort.InferenceSession(str(locator_path)),
            event_heads_session=ort.InferenceSession(str(event_heads_path)),
            thresholds=_load_thresholds(root),
        )

    def score_frames(self, images_nchw: list[np.ndarray]) -> list[float]:
        batch = _stack_batch(images_nchw)
        outputs = self._locator_session.run(
            [_LOCATOR_OUTPUT],
            {self._locator_input: batch},
        )
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        return [float(score) for score in _sigmoid(logits)]

    def _event_feed(
        self,
        images_nchw: list[np.ndarray],
        hud_images_nchw: list[np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        batch = _stack_event_window(images_nchw)
        feed = {self._event_inputs[0]: batch}
        if len(self._event_inputs) > 1:
            feed[self._event_inputs[1]] = _stack_event_window(
                hud_images_nchw or images_nchw
            )
        return feed

    def raw_event_logits(
        self,
        images_nchw: list[np.ndarray],
        hud_images_nchw: list[np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        outputs = self._event_heads_session.run(
            list(_EVENT_OUTPUTS),
            self._event_feed(images_nchw, hud_images_nchw),
        )
        return {
            name: np.asarray(tensor, dtype=np.float32)
            for name, tensor in zip(_EVENT_OUTPUTS, outputs, strict=True)
        }

    def raw_context_logits(self, images_nchw: list[np.ndarray]) -> dict[str, np.ndarray]:
        output_names = [f"context_{name}_logits" for name in CONTEXT_HEADS]
        outputs = self._event_heads_session.run(
            output_names,
            self._event_feed(images_nchw),
        )
        return {
            name: np.asarray(tensor, dtype=np.float32)
            for name, tensor in zip(output_names, outputs, strict=True)
        }

    def predict_event(
        self,
        images_nchw: list[np.ndarray],
        hud_images_nchw: list[np.ndarray] | None = None,
    ) -> EventHeadPrediction:
        return decode_event_logits(
            self.raw_event_logits(images_nchw, hud_images_nchw),
            thresholds=self._thresholds,
        )

    def predict_context(self, images_nchw: list[np.ndarray]) -> dict[str, str]:
        return decode_context_logits(
            self.raw_context_logits(images_nchw), thresholds=self._thresholds
        )
