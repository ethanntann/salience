"""Shared MobileNetV3-small backbone for student locator and event-head models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from salience_api.student.schema import (
    AIM_STATES,
    EVENT_KINDS,
    TARGET_STATES,
    WEAPON_CLASSES,
    DatasetManifest,
    EVENT_WINDOW_FRAMES,
)

EVIDENCE_HEADS = (
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
EVIDENCE_STATES = ("yes", "no", "unknown")
CONTEXT_HEADS = (
    "combat_visible",
    "enemy_visible",
    "build_fight",
    "clutch",
    "competitive_context",
    "victory",
    "fast_edit",
    "rotation_traversal",
    "looting_or_menu",
    "downtime",
)
ARTIFACT_VERSION = "student-v8"
SUPPORTED_ARTIFACT_VERSIONS = frozenset(
    {
        ARTIFACT_VERSION,
        "student-v7-prefinish-pool",
        "student-v6-mobilenetv3-prefinish-weapon",
    }
)
_MOBILENET_FEATURE_DIM = 576


def build_feature_backbone():
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone.classifier = nn.Identity()
    return backbone


def build_locator_net():
    import torch
    from torch import nn

    class LocatorNet(nn.Module):
        """Maps (B, 3, 224, 224) frame batches to (B, 1) eventness logits."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = build_feature_backbone()
            self.head = nn.Linear(_MOBILENET_FEATURE_DIM, 1)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.backbone(images)
            if features.dim() > 2:
                features = features.flatten(1)
            return self.head(features)

    return LocatorNet()


def build_event_heads_net():
    import torch
    from torch import nn

    class EventHeadsNet(nn.Module):
        """Multi-task classifier over an ordered event frame window."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = build_feature_backbone()
            dim = _MOBILENET_FEATURE_DIM
            temporal_dim = 256
            self.frame_projection = nn.Linear(dim, temporal_dim)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, EVENT_WINDOW_FRAMES, temporal_dim)
            )
            self.weapon_position_embedding = nn.Parameter(
                torch.zeros(1, EVENT_WINDOW_FRAMES, temporal_dim)
            )
            self.temporal = nn.Sequential(
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.weapon_temporal = nn.Sequential(
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.event_kind_head = nn.Linear(temporal_dim, len(EVENT_KINDS))
            self.target_state_head = nn.Linear(temporal_dim, len(TARGET_STATES))
            self.weapon_head = nn.Linear(temporal_dim * 2, len(WEAPON_CLASSES))
            self.aim_state_head = nn.Linear(temporal_dim, len(AIM_STATES))
            self.damaging_shot_count_head = nn.Linear(temporal_dim, 1)
            self.single_shot_damage_head = nn.Linear(temporal_dim, 1)
            self.evidence_heads = nn.ModuleDict(
                {
                    name: nn.Linear(temporal_dim, len(EVIDENCE_STATES))
                    for name in EVIDENCE_HEADS
                }
            )
            self.context_heads = nn.ModuleDict(
                {
                    name: nn.Linear(temporal_dim, len(EVIDENCE_STATES))
                    for name in CONTEXT_HEADS
                }
            )

        def _encode(self, images, position_embedding, temporal):
            if images.dim() != 5:
                raise ValueError("event images must have shape (B, T, C, H, W)")
            batch, frames, channels, height, width = images.shape
            flat = images.reshape(batch * frames, channels, height, width)
            features = self.backbone(flat)
            if features.dim() > 2:
                features = features.flatten(1)
            features = self.frame_projection(features).reshape(batch, frames, -1)
            features = features + position_embedding[:, :frames]
            pooled = temporal(features.transpose(1, 2)).mean(dim=2)
            return pooled, features

        def forward(
            self, images: torch.Tensor, hud_images: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            features, scene_sequence = self._encode(
                images, self.position_embedding, self.temporal
            )
            _, weapon_sequence = self._encode(
                hud_images,
                self.weapon_position_embedding,
                self.weapon_temporal,
            )
            # Weapon attribution should survive a one-frame HUD miss or an
            # immediate post-shot weapon swap. Pool the pre-finish portion of
            # the ordered window instead of trusting one sampled frame.
            pre_finish_frames = min(3, EVENT_WINDOW_FRAMES)
            scene_weapon_features = scene_sequence[:, :pre_finish_frames].mean(dim=1)
            weapon_hud_features = weapon_sequence[:, :pre_finish_frames].mean(dim=1)
            weapon_features = torch.cat(
                (scene_weapon_features, weapon_hud_features), dim=1
            )
            outputs: dict[str, torch.Tensor] = {
                "event_kind_logits": self.event_kind_head(features),
                "target_state_logits": self.target_state_head(features),
                "weapon_logits": self.weapon_head(weapon_features),
                "aim_state_logits": self.aim_state_head(features),
                "damaging_shot_count": self.damaging_shot_count_head(features),
                "single_shot_damage": self.single_shot_damage_head(features),
            }
            for name, head in self.evidence_heads.items():
                outputs[f"{name}_logits"] = head(features)
            for name, head in self.context_heads.items():
                outputs[f"context_{name}_logits"] = head(features)
            return outputs

    return EventHeadsNet()


def build_event_heads_onnx_wrapper(net):
    import torch
    from torch import nn

    output_names = event_head_output_names()

    class EventHeadsOnnxWrapper(nn.Module):
        """Thin wrapper so ONNX export returns tensors in runtime order."""

        def __init__(self, wrapped) -> None:
            super().__init__()
            self.net = wrapped

        def forward(self, images: torch.Tensor, hud_images: torch.Tensor):
            outputs = self.net(images, hud_images)
            return tuple(outputs[name] for name in output_names)

    return EventHeadsOnnxWrapper(net)


def event_head_output_names() -> tuple[str, ...]:
    return (
        "event_kind_logits",
        "target_state_logits",
        "weapon_logits",
        "aim_state_logits",
        "damaging_shot_count",
        "single_shot_damage",
        *(f"{name}_logits" for name in EVIDENCE_HEADS),
        *(f"context_{name}_logits" for name in CONTEXT_HEADS),
    )


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    from salience_api.student.schema import ClipStudentRecord

    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found under {dataset_dir}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetManifest(
        version=str(payload["version"]),
        train_clips=[ClipStudentRecord(**record) for record in payload["train_clips"]],
        eval_clips=[ClipStudentRecord(**record) for record in payload.get("eval_clips", [])],
        target_coverage=dict(payload.get("target_coverage") or {}),
    )


def frames_root(dataset_dir: Path, clip_id: int, kind: str) -> Path:
    return Path(dataset_dir) / "frames" / str(clip_id) / kind


def write_artifact_meta(
    out_dir: Path,
    *,
    task: str,
    metrics: dict[str, Any],
    version: str = ARTIFACT_VERSION,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "artifact_meta.json"
    existing: dict[str, Any] = {}
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    merged_metrics = dict(existing.get("metrics") or {})
    merged_metrics[task] = metrics
    meta = {
        "version": version,
        "train_date": datetime.now(timezone.utc).isoformat(),
        "metrics": merged_metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_locator_onnx(model, out_path: Path) -> None:
    import torch

    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def export_event_heads_onnx(model, out_path: Path) -> None:
    import torch

    wrapper = build_event_heads_onnx_wrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, EVENT_WINDOW_FRAMES, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy, dummy),
        str(out_path),
        input_names=["images", "hud_images"],
        output_names=list(event_head_output_names()),
        dynamic_axes={"images": {0: "batch"}, "hud_images": {0: "batch"}},
        opset_version=17,
    )


def _class_index(classes: tuple[str, ...], value: str, *, default: str) -> int:
    normalized = str(value or default).strip().lower()
    if normalized not in classes:
        normalized = default
    return classes.index(normalized)


def _evidence_label(event: dict[str, Any], field: str) -> str:
    value = event.get(field)
    if isinstance(value, bool):
        return "yes" if value else "no"
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in EVIDENCE_STATES else "unknown"


def event_targets(event: dict[str, Any]) -> dict[str, int | float]:
    target_state = event.get("target_state")
    if not target_state:
        if event.get("target_was_downed"):
            target_state = "already_downed"
        elif event.get("target_was_active"):
            target_state = "active"
        else:
            target_state = "unknown"

    weapon = (
        event.get("selected_weapon_before_finish")
        or event.get("resolved_weapon")
        or "unknown"
    )
    aim_state = event.get("aim_state_at_shot") or event.get("damage_aim_state") or "unknown"

    targets: dict[str, int | float] = {
        "event_kind": _class_index(EVENT_KINDS, str(event.get("event_kind", "none")), default="none"),
        "target_state": _class_index(TARGET_STATES, str(target_state), default="unknown"),
        "weapon": _class_index(WEAPON_CLASSES, str(weapon), default="unknown"),
        "aim_state": _class_index(AIM_STATES, str(aim_state), default="unknown"),
        "damaging_shot_count": float(
            min(
                10,
                max(
                    0,
                    event.get("damaging_shot_count")
                    if event.get("damaging_shot_count") is not None
                    else event.get("damage_hit_count") or 0,
                ),
            )
        ),
        "single_shot_damage": float(event.get("single_shot_damage") or 0) / 100.0,
    }
    for field in EVIDENCE_HEADS:
        if field == "single_shot_damage_known":
            value = "yes" if event.get("single_shot_damage") is not None else "no"
        else:
            value = _evidence_label(event, field)
        targets[field] = _class_index(EVIDENCE_STATES, value, default="unknown")
    return targets


def event_target_masks(event: dict[str, Any]) -> dict[str, bool]:
    """Return which teacher targets are known enough to contribute to loss."""
    targets = event_targets(event)
    masks = {
        "event_kind": True,
        "target_state": TARGET_STATES[int(targets["target_state"])] != "unknown",
        "weapon": WEAPON_CLASSES[int(targets["weapon"])] != "unknown",
        "aim_state": AIM_STATES[int(targets["aim_state"])] != "unknown",
        "damaging_shot_count": (
            event.get("damaging_shot_count") is not None
            or event.get("damage_hit_count") is not None
        ),
        "single_shot_damage": event.get("single_shot_damage") is not None,
    }
    for field in EVIDENCE_HEADS:
        masks[field] = (
            True
            if field == "single_shot_damage_known"
            else EVIDENCE_STATES[int(targets[field])] != "unknown"
        )
    return masks
