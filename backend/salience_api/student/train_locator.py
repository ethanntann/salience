"""Train the student locator model and export ONNX artifacts.

Usage::

    python -m salience_api.student.train_locator \\
        --dataset .local-data/student/dataset \\
        --out .local-data/student/artifacts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from salience_api.clips.keyframes import extract_timeline_keyframes, keyframe_timestamps
from salience_api.student.backbone import (
    build_locator_net,
    export_locator_onnx,
    frames_root,
    load_manifest,
    write_artifact_meta,
)
from salience_api.student.schema import ClipStudentRecord

LOCATOR_POSITIVE_WINDOW_SEC = 0.5


@dataclass(frozen=True)
class LocatorSample:
    image_path: Path
    timestamp: float
    label: float


def _timeline_cache_meta(cache_dir: Path) -> dict | None:
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_timeline_cache(cache_dir: Path, *, timestamps: list[float]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(
        json.dumps({"timestamps": timestamps, "policy": "inference-density"}, indent=2),
        encoding="utf-8",
    )


def _collect_timeline_samples(
    dataset_dir: Path,
    clip: ClipStudentRecord,
) -> list[tuple[Path, float]]:
    cache_dir = frames_root(dataset_dir, clip.clip_id, "timeline")
    expected_timestamps = keyframe_timestamps(clip.duration_sec)
    cached = _timeline_cache_meta(cache_dir)
    if (
        cached
        and cached.get("timestamps")
        and cached.get("policy") == "inference-density"
        and len(cached["timestamps"]) == len(expected_timestamps)
    ):
        timestamps = [float(value) for value in cached["timestamps"]]
        frames = []
        for index, timestamp in enumerate(timestamps):
            path = cache_dir / f"{index:04d}.jpg"
            if path.is_file():
                frames.append((path, timestamp))
        if frames:
            return frames

    video_path = Path(clip.path)
    if not video_path.is_file():
        # Fall back to any existing cache so synthetic fixtures still train.
        if cached and cached.get("timestamps"):
            timestamps = [float(value) for value in cached["timestamps"]]
            frames = []
            for index, timestamp in enumerate(timestamps):
                path = cache_dir / f"{index:04d}.jpg"
                if path.is_file():
                    frames.append((path, timestamp))
            if frames:
                return frames
        return []

    keyframes = extract_timeline_keyframes(video_path, clip.duration_sec)
    if not keyframes:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("*.jpg"):
        stale.unlink()
    frames: list[tuple[Path, float]] = []
    timestamps: list[float] = []
    for index, frame in enumerate(keyframes):
        dest = cache_dir / f"{index:04d}.jpg"
        dest.write_bytes(frame.path.read_bytes())
        frames.append((dest, float(frame.timestamp_sec)))
        timestamps.append(float(frame.timestamp_sec))
    _write_timeline_cache(cache_dir, timestamps=timestamps)
    return frames


def _locator_label(timestamp: float, locator_timestamps: list[float]) -> float:
    for center in locator_timestamps:
        if abs(timestamp - float(center)) <= LOCATOR_POSITIVE_WINDOW_SEC:
            return 1.0
    return 0.0


def build_locator_dataset(
    dataset_dir: Path, *, split: str = "train"
) -> list[LocatorSample]:
    manifest = load_manifest(dataset_dir)
    samples: list[LocatorSample] = []
    clips = manifest.train_clips if split == "train" else manifest.eval_clips
    for clip in clips:
        for image_path, timestamp in _collect_timeline_samples(dataset_dir, clip):
            samples.append(
                LocatorSample(
                    image_path=image_path,
                    timestamp=timestamp,
                    label=_locator_label(timestamp, clip.locator_timestamps),
                )
            )
    return samples


def _load_image_tensor(path: Path):
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return transform(rgb)


def train_locator(
    dataset_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    patience: int = 3,
) -> dict[str, Any]:
    import copy

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    samples = build_locator_dataset(dataset_dir, split="train")
    validation_samples = build_locator_dataset(dataset_dir, split="eval")
    if not samples:
        raise RuntimeError(f"no locator training samples found under {dataset_dir}")

    class _LocatorDataset(Dataset):
        def __init__(self, source: list[LocatorSample]) -> None:
            self.source = source

        def __len__(self) -> int:
            return len(self.source)

        def __getitem__(self, index: int):
            sample = self.source[index]
            image = _load_image_tensor(sample.image_path)
            label = torch.tensor([sample.label], dtype=torch.float32)
            return image, label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_locator_net().to(device)
    loader = DataLoader(_LocatorDataset(samples), batch_size=batch_size, shuffle=True)
    validation_loader = (
        DataLoader(_LocatorDataset(validation_samples), batch_size=batch_size, shuffle=False)
        if validation_samples
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    pos_weight = torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _classification_metrics(eval_loader) -> dict[str, float | int]:
        if eval_loader is None:
            return {"samples": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        true_positive = false_positive = false_negative = 0
        model.eval()
        with torch.no_grad():
            for images, labels in eval_loader:
                labels = labels.to(device)
                predictions = (torch.sigmoid(model(images.to(device))) >= 0.5).float()
                true_positive += int(((predictions == 1) & (labels == 1)).sum())
                false_positive += int(((predictions == 1) & (labels == 0)).sum())
                false_negative += int(((predictions == 0) & (labels == 1)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return {
            "samples": len(validation_samples),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }

    model.train()
    last_loss = 0.0
    best_state = None
    best_f1 = float("-inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for images, labels in tqdm(loader, desc="train-locator", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        last_loss = epoch_loss / max(batches, 1)
        epochs_ran += 1
        validation = _classification_metrics(validation_loader)
        if validation["f1"] > best_f1:
            best_f1 = float(validation["f1"])
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        model.train()
        if validation_loader is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    validation = _classification_metrics(validation_loader)

    out_dir = Path(out_dir)
    export_locator_onnx(model.cpu(), out_dir / "locator.onnx")

    metrics = {
        "samples": len(samples),
        "positive_samples": int(positives),
        "validation": validation,
        "epochs": epochs,
        "epochs_ran": epochs_ran,
        "early_stopped": epochs_ran < epochs,
        "patience": patience,
        "loss": last_loss,
    }
    write_artifact_meta(out_dir, task="locator", metrics=metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train student locator ONNX model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args(argv)
    metrics = train_locator(
        args.dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
