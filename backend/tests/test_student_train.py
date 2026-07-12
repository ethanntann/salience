import json
from pathlib import Path

import numpy as np
import pytest

from salience_api.student.schema import STUDENT_DATASET_VERSION_PREFIX


def _write_synthetic_frame(path: Path, *, color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color).save(path)


def _build_synthetic_dataset(root: Path) -> None:
    clips = []
    for clip_id, locator_ts, event in (
        (
            1,
            [2.0],
            {
                "event_index": 0,
                "event_kind": "elimination",
                "event_timestamp": 2.0,
                "target_state": "active",
                "selected_weapon_before_finish": "shotgun",
                "aim_state_at_shot": "hipfire",
                "pov_shot_visible": "yes",
                "new_damage_visible": "yes",
                "damaging_shot_count": 1,
            },
        ),
        (
            2,
            [8.0],
            {
                "event_index": 0,
                "event_kind": "knock",
                "event_timestamp": 8.0,
                "target_state": "active",
                "selected_weapon_before_finish": "sniper_or_hunting",
                "aim_state_at_shot": "scope_overlay",
                "pov_shot_visible": "yes",
                "damaging_shot_count": 2,
            },
        ),
    ):
        timeline_dir = root / "frames" / str(clip_id) / "timeline"
        timestamps = [0.5, 1.5, 2.0, 3.0, 7.5, 8.0, 9.0]
        for index, timestamp in enumerate(timestamps):
            _write_synthetic_frame(
                timeline_dir / f"{index:04d}.jpg",
                color=(40 + clip_id * 20, 80, 120 + index * 10),
            )
        (timeline_dir / "meta.json").write_text(
            json.dumps({"timestamps": timestamps}, indent=2),
            encoding="utf-8",
        )

        events_dir = root / "frames" / str(clip_id) / "events"
        for frame_index in range(3):
            _write_synthetic_frame(
                events_dir / f"{frame_index:04d}.jpg",
                color=(180, 60 + frame_index * 20, 40 + clip_id * 15),
            )
        (events_dir / "meta.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"filename": "0000.jpg", "event_index": 0},
                        {"filename": "0001.jpg", "event_index": 0},
                        {"filename": "0002.jpg", "event_index": 0},
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        clips.append(
            {
                "clip_id": clip_id,
                "path": str(root / f"missing-{clip_id}.mp4"),
                "filename": f"missing-{clip_id}.mp4",
                "duration_sec": 12.0,
                "locator_timestamps": locator_ts,
                "label_json": {},
                "events": [event],
            }
        )

    manifest = {
        "version": STUDENT_DATASET_VERSION_PREFIX,
        "train_clips": clips,
        "eval_clips": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_context_sample_weights_upweights_rare_positive_labels():
    from salience_api.student.backbone import CONTEXT_HEADS
    from salience_api.student.train_event_heads import ContextSample, context_sample_weights

    def _sample(positive_fields: set[str]) -> ContextSample:
        targets = {field: (0 if field in positive_fields else 1) for field in CONTEXT_HEADS}
        masks = {field: True for field in CONTEXT_HEADS}
        return ContextSample(image_paths=(), targets=targets, masks=masks)

    samples = [_sample({"victory"}), _sample(set()), _sample(set()), _sample(set())]

    weights = context_sample_weights(samples)

    assert weights[0] > weights[1]
    assert weights[1] == weights[2] == weights[3]


def test_event_sample_weights_upweights_rare_known_weapons():
    from salience_api.student.train_event_heads import EventHeadSample, event_sample_weights

    def _sample(weapon: int, known: bool = True) -> EventHeadSample:
        return EventHeadSample(
            image_paths=(),
            targets={"weapon": weapon},
            masks={"weapon": known},
        )

    weights = event_sample_weights([_sample(0), _sample(0), _sample(0), _sample(3)])

    assert weights[3] > weights[0]


def test_event_confidence_weight_preserves_weak_unresolved_examples():
    from salience_api.student.train_event_heads import event_confidence_weight

    assert event_confidence_weight({"status": "attributed", "teacher_confidence": 0.8}) == 0.8
    assert event_confidence_weight({"status": "no_finish"}) == 0.25


def test_multiclass_metrics_reports_macro_f1():
    from salience_api.student.train_event_heads import multiclass_metrics

    metrics = multiclass_metrics([0, 1, 1, 2], [0, 0, 1, 2], 3)

    assert metrics["known"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(((2 / 3) + (2 / 3) + 1.0) / 3)


def test_should_stop_early_triggers_after_patience_epochs_without_improvement():
    from salience_api.student.train_event_heads import should_stop_early

    assert should_stop_early([0.5, 0.6, 0.6, 0.6], patience=2) is True
    assert should_stop_early([0.5, 0.6, 0.7, 0.65], patience=2) is False
    assert should_stop_early([0.5, 0.6], patience=2) is False


@pytest.mark.train
def test_smoke_train_exports_onnx_artifacts(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "artifacts"
    _build_synthetic_dataset(dataset_dir)

    from salience_api.student.train_event_heads import train_event_heads
    from salience_api.student.train_locator import train_locator

    locator_metrics = train_locator(dataset_dir, out_dir, epochs=1, batch_size=4)
    event_metrics = train_event_heads(dataset_dir, out_dir, epochs=1, batch_size=4)

    assert locator_metrics["samples"] >= 2
    assert event_metrics["samples"] >= 2
    assert (out_dir / "locator.onnx").is_file()
    assert (out_dir / "event_heads.onnx").is_file()
    assert (out_dir / "artifact_meta.json").is_file()

    meta = json.loads((out_dir / "artifact_meta.json").read_text(encoding="utf-8"))
    assert meta["version"]
    assert meta["train_date"]
    assert "locator" in meta["metrics"]
    assert "event_heads" in meta["metrics"]

    import onnxruntime as ort

    locator = ort.InferenceSession(str(out_dir / "locator.onnx"))
    event_heads = ort.InferenceSession(str(out_dir / "event_heads.onnx"))
    assert locator.get_inputs()[0].name == "images"
    assert event_heads.get_inputs()[0].name == "images"
    assert event_heads.get_inputs()[1].name == "hud_images"
    frame = np.zeros((1, 3, 224, 224), dtype=np.float32)
    event_window = np.zeros((1, 7, 3, 224, 224), dtype=np.float32)
    logits = locator.run(["logits"], {"images": frame})[0]
    assert logits.shape == (1, 1)
    outputs = event_heads.run(
        None, {"images": event_window, "hud_images": event_window}
    )
    assert len(outputs) == 31
