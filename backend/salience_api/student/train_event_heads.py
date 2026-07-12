"""Train the student event-head model and export ONNX artifacts.

Usage::

    python -m salience_api.student.train_event_heads \\
        --dataset .local-data/student/dataset \\
        --out .local-data/student/artifacts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from salience_api.clips.keyframes import extract_event_keyframes
from salience_api.student.backbone import (
    CONTEXT_HEADS,
    EVIDENCE_HEADS,
    build_event_heads_net,
    event_target_masks,
    event_targets,
    export_event_heads_onnx,
    frames_root,
    load_manifest,
    write_artifact_meta,
)
from salience_api.student.train_locator import _collect_timeline_samples
from salience_api.student.schema import (
    AIM_STATES,
    EVENT_KINDS,
    EVENT_WINDOW_FRAMES,
    TARGET_STATES,
    WEAPON_CLASSES,
    ClipStudentRecord,
    hud_crop_left,
    select_window_indices,
)


@dataclass(frozen=True)
class EventHeadSample:
    image_paths: tuple[Path, ...]
    targets: dict[str, int | float]
    masks: dict[str, bool]
    confidence_weight: float = 1.0
    confidence_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSample:
    image_paths: tuple[Path, ...]
    targets: dict[str, int]
    masks: dict[str, bool]
    confidence_weight: float = 1.0
    confidence_weights: dict[str, float] = field(default_factory=dict)


def confidence_weight(value: object, *, fallback: float = 0.25) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(0.05, min(1.0, parsed))


def event_confidence_weight(event: dict[str, Any]) -> float:
    """Use event confidence while keeping unresolved events as weak examples."""
    if event.get("teacher_confidence") is not None:
        return confidence_weight(event.get("teacher_confidence"))
    return 1.0 if str(event.get("status", "")).lower() == "attributed" else 0.25


def event_sample_weights(samples: list[EventHeadSample]) -> list[float]:
    """Balance known weapon classes without oversampling unknown labels."""
    weapon_counts: dict[int, int] = {}
    for sample in samples:
        if sample.masks.get("weapon"):
            weapon = int(sample.targets["weapon"])
            weapon_counts[weapon] = weapon_counts.get(weapon, 0) + 1
    known_total = sum(weapon_counts.values())
    weights: list[float] = []
    for sample in samples:
        if not sample.masks.get("weapon"):
            weights.append(0.75)
            continue
        weapon = int(sample.targets["weapon"])
        count = max(weapon_counts.get(weapon, 1), 1)
        weight = (known_total / count) ** 0.5 if known_total else 1.0
        weights.append(min(max(weight, 1.0), 4.0))
    return weights


def multiclass_metrics(
    predictions: list[int], targets: list[int], class_count: int
) -> dict[str, float | int]:
    """Return accuracy and macro-F1 over classes present in the targets."""
    if not targets:
        return {"known": 0, "accuracy": 0.0, "macro_f1": 0.0}
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for prediction, target in zip(predictions, targets, strict=True):
        if 0 <= target < class_count and 0 <= prediction < class_count:
            confusion[target][prediction] += 1
    scores: list[float] = []
    for class_index in range(class_count):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(row[class_index] for row in confusion) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        support = true_positive + false_negative
        if support == 0:
            continue
        scores.append(
            2.0 * true_positive
            / max(2.0 * true_positive + false_positive + false_negative, 1)
        )
    return {
        "known": len(targets),
        "accuracy": sum(
            int(prediction == target)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        / len(targets),
        "macro_f1": sum(scores) / len(scores) if scores else 0.0,
    }


def _event_cache_meta(cache_dir: Path) -> dict | None:
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_event_cache(cache_dir: Path, *, entries: list[dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(json.dumps({"frames": entries}, indent=2), encoding="utf-8")


def _collect_event_samples(
    dataset_dir: Path,
    clip: ClipStudentRecord,
) -> list[tuple[Path, int]]:
    if not clip.events:
        return []

    cache_dir = frames_root(dataset_dir, clip.clip_id, "events")
    cached = _event_cache_meta(cache_dir)
    if cached and cached.get("frames"):
        frames: list[tuple[Path, int]] = []
        for entry in cached["frames"]:
            path = cache_dir / str(entry["filename"])
            if path.is_file():
                frames.append((path, int(entry["event_index"])))
        if frames:
            return frames

    video_path = Path(clip.path)
    if not video_path.is_file():
        return []

    event_timestamps = [
        float(event.get("event_timestamp") or event.get("finish_timestamp") or 0.0)
        for event in clip.events
    ]
    keyframes = extract_event_keyframes(video_path, clip.duration_sec, event_timestamps)
    if not keyframes:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    frames: list[tuple[Path, int]] = []
    for index, frame in enumerate(keyframes):
        if frame.event_index is None:
            continue
        filename = f"{index:04d}.jpg"
        dest = cache_dir / filename
        dest.write_bytes(frame.path.read_bytes())
        entries.append({"filename": filename, "event_index": int(frame.event_index)})
        frames.append((dest, int(frame.event_index)))
    _write_event_cache(cache_dir, entries=entries)
    return frames


def build_event_heads_dataset(
    dataset_dir: Path, *, split: str = "train"
) -> list[EventHeadSample]:
    manifest = load_manifest(dataset_dir)
    samples: list[EventHeadSample] = []
    clips = manifest.train_clips if split == "train" else manifest.eval_clips
    for clip in clips:
        events_by_index = {
            int(event.get("event_index", index)): event
            for index, event in enumerate(clip.events)
        }
        grouped: dict[int, list[Path]] = {}
        for image_path, event_index in _collect_event_samples(dataset_dir, clip):
            grouped.setdefault(event_index, []).append(image_path)
        for event_index, image_paths in grouped.items():
            event = events_by_index.get(event_index)
            if not event:
                continue
            samples.append(
                EventHeadSample(
                    image_paths=tuple(image_paths),
                    targets=event_targets(event),
                    masks=event_target_masks(event),
                    confidence_weight=event_confidence_weight(event),
                    confidence_weights={
                        str(key): confidence_weight(value)
                        for key, value in (event.get("label_confidences") or {}).items()
                    },
                )
            )
    return samples


def context_sample_weights(samples: list[ContextSample]) -> list[float]:
    """Inverse-frequency resampling weight per clip, driven by its rarest positive label.

    ``_context_value`` maps "yes" to index 0, so a masked target of 0 is a positive.
    """
    positive_counts = {field: 0 for field in CONTEXT_HEADS}
    for sample in samples:
        for field in CONTEXT_HEADS:
            if sample.masks[field] and sample.targets[field] == 0:
                positive_counts[field] += 1

    total = len(samples)
    weights: list[float] = []
    for sample in samples:
        weight = 1.0
        for field in CONTEXT_HEADS:
            if sample.masks[field] and sample.targets[field] == 0:
                count = positive_counts[field]
                if count:
                    weight = max(weight, total / count)
        weights.append(weight)
    return weights


def _context_value(value: object) -> tuple[int, bool]:
    normalized = str(value or "uncertain").strip().lower()
    if normalized == "yes":
        return 0, True
    if normalized == "no":
        return 1, True
    return 2, False


def build_context_dataset(
    dataset_dir: Path, *, split: str = "train"
) -> list[ContextSample]:
    manifest = load_manifest(dataset_dir)
    clips = manifest.train_clips if split == "train" else manifest.eval_clips
    samples: list[ContextSample] = []
    for clip in clips:
        frames = _collect_timeline_samples(dataset_dir, clip)
        if not frames:
            continue
        values = {field: _context_value(clip.label_json.get(field)) for field in CONTEXT_HEADS}
        known_confidences = [
            confidence_weight(clip.label_confidences.get(field), fallback=clip.teacher_confidence)
            for field, value in values.items()
            if value[1]
        ]
        label_confidences = {
            field: confidence_weight(
                clip.label_confidences.get(field),
                fallback=clip.teacher_confidence,
            )
            for field in CONTEXT_HEADS
        }
        samples.append(
            ContextSample(
                image_paths=tuple(path for path, _ in frames),
                targets={field: value[0] for field, value in values.items()},
                masks={field: value[1] for field, value in values.items()},
                confidence_weight=(
                    sum(known_confidences) / len(known_confidences)
                    if known_confidences
                    else confidence_weight(clip.teacher_confidence)
                ),
                confidence_weights=label_confidences,
            )
        )
    return samples


def should_stop_early(history: list[float], *, patience: int) -> bool:
    """True once ``patience`` consecutive epochs failed to beat the best score seen before them."""
    if len(history) <= patience:
        return False
    best_before_window = max(history[:-patience])
    recent = history[-patience:]
    return all(score <= best_before_window for score in recent)


def _load_event_tensor(paths: tuple[Path, ...], *, include_hud: bool, augment: bool = False):
    import torch
    from PIL import Image
    from torchvision import transforms

    scene_steps = [transforms.Resize((224, 224))]
    if augment:
        scene_steps.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
        )
    scene_steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    scene_transform = transforms.Compose(scene_steps)
    # HUD crop stays unaugmented: weapon-name pixels are spatially anchored,
    # and color jitter would fight the OCR/weapon-classification signal there.
    hud_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    selected = list(paths)
    if len(selected) > EVENT_WINDOW_FRAMES:
        selected = [
            selected[index]
            for index in select_window_indices(len(selected), EVENT_WINDOW_FRAMES)
        ]
    images = []
    hud_images = []
    for path in selected:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images.append(scene_transform(rgb))
            if include_hud:
                left = hud_crop_left(rgb.width)
                hud_images.append(hud_transform(rgb.crop((left, 0, rgb.width, rgb.height))))
            else:
                hud_images.append(images[-1].clone())
    if not images:
        raise ValueError("event window has no frames")
    while len(images) < EVENT_WINDOW_FRAMES:
        images.append(images[-1].clone())
        hud_images.append(hud_images[-1].clone())
    return torch.stack(images, dim=0), torch.stack(hud_images, dim=0)


def train_event_heads(
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

    samples = build_event_heads_dataset(dataset_dir, split="train")
    validation_samples = build_event_heads_dataset(dataset_dir, split="eval")
    context_samples = build_context_dataset(dataset_dir, split="train")
    validation_context_samples = build_context_dataset(dataset_dir, split="eval")
    if not samples:
        raise RuntimeError(f"no event-head training samples found under {dataset_dir}")

    class _EventDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            sample = samples[index]
            images, hud_images = _load_event_tensor(
                sample.image_paths, include_hud=True, augment=True
            )
            return (
                images,
                hud_images,
                sample.targets,
                sample.masks,
                sample.confidence_weight,
                sample.confidence_weights,
            )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        hud_images = torch.stack([item[1] for item in batch], dim=0)
        keys = batch[0][2].keys()
        targets = {
            key: torch.tensor([item[2][key] for item in batch], dtype=torch.float32)
            for key in keys
        }
        masks = {
            key: torch.tensor([item[3][key] for item in batch], dtype=torch.bool)
            for key in keys
        }
        sample_weights = torch.tensor(
            [item[4] for item in batch], dtype=torch.float32
        )
        confidence_weights = {
            key: torch.tensor(
                [item[5].get(key, item[4]) for item in batch], dtype=torch.float32
            )
            for key in keys
        }
        return images, hud_images, targets, masks, sample_weights, confidence_weights

    class _ValidationDataset(Dataset):
        def __len__(self) -> int:
            return len(validation_samples)

        def __getitem__(self, index: int):
            sample = validation_samples[index]
            images, hud_images = _load_event_tensor(
                sample.image_paths, include_hud=True, augment=False
            )
            return (
                images,
                hud_images,
                sample.targets,
                sample.masks,
                sample.confidence_weight,
                sample.confidence_weights,
            )

    class _ContextDataset(Dataset):
        def __init__(self, source: list[ContextSample], *, augment: bool) -> None:
            self.source = source
            self.augment = augment

        def __len__(self) -> int:
            return len(self.source)

        def __getitem__(self, index: int):
            sample = self.source[index]
            images, hud_images = _load_event_tensor(
                sample.image_paths, include_hud=False, augment=self.augment
            )
            return (
                images,
                hud_images,
                sample.targets,
                sample.masks,
                sample.confidence_weight,
                sample.confidence_weights,
            )

    def _masked_ce(logits, target, mask, *, weight=None, sample_weights=None):
        if not bool(mask.any()):
            return logits.sum() * 0.0
        losses = nn.functional.cross_entropy(
            logits[mask], target[mask].long(), weight=weight, reduction="none"
        )
        if sample_weights is None:
            return losses.mean()
        selected_weights = sample_weights[mask].to(losses.device)
        return (losses * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)

    def _weighted_smooth_l1(prediction, target, mask, sample_weights):
        if not bool(mask.any()):
            return prediction.sum() * 0.0
        losses = nn.functional.smooth_l1_loss(
            prediction[mask], target[mask], reduction="none"
        )
        selected_weights = sample_weights[mask].to(losses.device)
        return (losses * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)

    classification_keys = ("event_kind", "target_state", "weapon", "aim_state", *EVIDENCE_HEADS)
    class_sizes = {
        "event_kind": len(EVENT_KINDS),
        "target_state": len(TARGET_STATES),
        "weapon": len(WEAPON_CLASSES),
        "aim_state": len(AIM_STATES),
        **{field: 3 for field in EVIDENCE_HEADS},
    }
    # Reducer-critical heads get extra loss weight.
    task_loss_weights = {
        "event_kind": 2.0,
        "weapon": 2.5,
        "target_state": 1.25,
        "aim_state": 1.25,
    }

    def _class_weights(key: str):
        values = [
            int(sample.targets[key])
            for sample in samples
            if sample.masks[key]
        ]
        if not values:
            return None
        counts = torch.bincount(
            torch.tensor(values), minlength=class_sizes[key]
        ).float()
        weights = torch.zeros_like(counts)
        present = counts > 0
        weights[present] = torch.sqrt(counts[present].sum() / counts[present]).clamp(0.25, 4.0)
        return weights.to(device)

    def _context_class_weights(key: str):
        values = [
            sample.targets[key]
            for sample in context_samples
            if sample.masks[key]
        ]
        if not values:
            return None
        counts = torch.bincount(torch.tensor(values), minlength=3).float()
        weights = torch.zeros_like(counts)
        present = counts > 0
        weights[present] = torch.sqrt(
            counts[present].sum() / counts[present]
        ).clamp(0.25, 4.0)
        return weights.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_event_heads_net().to(device)
    event_sampler = torch.utils.data.WeightedRandomSampler(
        event_sample_weights(samples),
        num_samples=len(samples),
        replacement=True,
    )
    loader = DataLoader(
        _EventDataset(),
        batch_size=batch_size,
        sampler=event_sampler,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        _ValidationDataset(), batch_size=batch_size, shuffle=False, collate_fn=_collate
    ) if validation_samples else None
    context_loader = None
    if context_samples:
        context_weights_per_sample = context_sample_weights(context_samples)
        context_sampler = torch.utils.data.WeightedRandomSampler(
            context_weights_per_sample,
            num_samples=len(context_weights_per_sample),
            replacement=True,
        )
        context_loader = DataLoader(
            _ContextDataset(context_samples, augment=True),
            batch_size=batch_size,
            sampler=context_sampler,
            collate_fn=_collate,
        )
    validation_context_loader = (
        DataLoader(
            _ContextDataset(validation_context_samples, augment=False),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
        if validation_context_samples
        else None
    )
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_params}
    task_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": task_params, "lr": learning_rate},
        ]
    )
    weights = {key: _class_weights(key) for key in classification_keys}
    context_weights = {key: _context_class_weights(key) for key in CONTEXT_HEADS}

    def _evaluate() -> tuple[float, dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
        nonlocal model
        if validation_loader is None:
            return 0.0, {}, {}
        model = model.to(device).eval()
        predictions_by_key = {key: [] for key in classification_keys}
        targets_by_key = {key: [] for key in classification_keys}
        with torch.no_grad():
            for images, hud_images, targets, masks, _sample_weights, _confidence_weights in validation_loader:
                outputs = model(images.to(device), hud_images.to(device))
                for key in classification_keys:
                    mask = masks[key]
                    predictions_by_key[key].extend(
                        outputs[f"{key}_logits"].argmax(dim=1).cpu()[mask].tolist()
                    )
                    targets_by_key[key].extend(targets[key][mask].long().tolist())
        validation = {
            key: multiclass_metrics(
                predictions_by_key[key],
                targets_by_key[key],
                class_sizes[key],
            )
            for key in classification_keys
        }

        context_validation: dict[str, dict[str, float | int]] = {}
        if validation_context_loader is not None:
            context_predictions = {key: [] for key in CONTEXT_HEADS}
            context_targets = {key: [] for key in CONTEXT_HEADS}
            with torch.no_grad():
                for images, hud_images, targets, masks, _sample_weights, _confidence_weights in validation_context_loader:
                    outputs = model(images.to(device), hud_images.to(device))
                    for key in CONTEXT_HEADS:
                        mask = masks[key]
                        context_predictions[key].extend(
                            outputs[f"context_{key}_logits"].argmax(dim=1).cpu()[mask].tolist()
                        )
                        context_targets[key].extend(targets[key][mask].long().tolist())
            context_validation = {
                key: multiclass_metrics(
                    context_predictions[key],
                    context_targets[key],
                    3,
                )
                for key in CONTEXT_HEADS
            }

        model.train()
        monitored_keys = ("event_kind", "target_state", "weapon", "aim_state")
        scores = [
            validation[key]["macro_f1"]
            for key in monitored_keys
            if validation[key]["known"] > 0
        ]
        monitored = sum(scores) / len(scores) if scores else 0.0
        return monitored, validation, context_validation

    model.train()
    history: list[float] = []
    best_state: dict[str, Any] | None = None
    best_validation: dict[str, dict[str, float | int]] = {}
    best_context_validation: dict[str, dict[str, float | int]] = {}
    best_score = float("-inf")
    last_loss = 0.0
    epochs_ran = 0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for images, hud_images, targets, masks, sample_weights, confidence_weights in tqdm(
            loader, desc="train-event-heads", leave=False
        ):
            images = images.to(device)
            hud_images = hud_images.to(device)
            optimizer.zero_grad()
            outputs = model(images, hud_images)
            loss = outputs["event_kind_logits"].sum() * 0.0
            for key in ("event_kind", "target_state", "weapon", "aim_state"):
                loss = loss + task_loss_weights.get(key, 1.0) * _masked_ce(
                    outputs[f"{key}_logits"],
                    targets[key].to(device),
                    masks[key].to(device),
                    weight=weights[key],
                    sample_weights=confidence_weights[key].to(device),
                )
            damage_mask = masks["damaging_shot_count"].to(device)
            if bool(damage_mask.any()):
                loss = loss + _weighted_smooth_l1(
                    outputs["damaging_shot_count"].squeeze(-1),
                    targets["damaging_shot_count"].to(device),
                    damage_mask,
                    confidence_weights["damaging_shot_count"].to(device),
                )
            single_damage_mask = masks["single_shot_damage"].to(device)
            if bool(single_damage_mask.any()):
                loss = loss + _weighted_smooth_l1(
                    outputs["single_shot_damage"].squeeze(-1),
                    targets["single_shot_damage"].to(device),
                    single_damage_mask,
                    confidence_weights["single_shot_damage"].to(device),
                )
            for field in EVIDENCE_HEADS:
                loss = loss + _masked_ce(
                    outputs[f"{field}_logits"],
                    targets[field].to(device),
                    masks[field].to(device),
                    weight=weights[field],
                    sample_weights=confidence_weights[field].to(device),
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        if context_loader is not None:
            for images, hud_images, targets, masks, sample_weights, confidence_weights in tqdm(
                context_loader, desc="train-context-heads", leave=False
            ):
                optimizer.zero_grad()
                outputs = model(images.to(device), hud_images.to(device))
                loss = outputs["event_kind_logits"].sum() * 0.0
                for field in CONTEXT_HEADS:
                    loss = loss + _masked_ce(
                        outputs[f"context_{field}_logits"],
                        targets[field].to(device),
                        masks[field].to(device),
                        weight=context_weights[field],
                        sample_weights=confidence_weights[field].to(device),
                    )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1
        last_loss = epoch_loss / max(batches, 1)
        epochs_ran += 1

        monitored, validation, context_validation = _evaluate()
        if validation_loader is not None:
            history.append(monitored)
            if best_state is None or monitored > best_score:
                best_score = monitored
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation
                best_context_validation = context_validation
            if should_stop_early(history, patience=patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(out_dir)
    export_event_heads_onnx(model.cpu(), out_dir / "event_heads.onnx")

    metrics = {
        "events": len(samples),
        "samples": len(samples),
        "frames": sum(len(sample.image_paths) for sample in samples),
        "validation_events": len(validation_samples),
        "context_clips": len(context_samples),
        "validation_context_clips": len(validation_context_samples),
        "epochs": epochs,
        "epochs_ran": epochs_ran,
        "early_stopped": epochs_ran < epochs,
        "patience": patience,
        "loss": last_loss,
        "validation": best_validation,
        "context_validation": best_context_validation,
    }
    write_artifact_meta(out_dir, task="event_heads", metrics=metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train student event-head ONNX model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args(argv)
    metrics = train_event_heads(
        args.dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
