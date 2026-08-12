"""Datasets and losses shared by SEC284-L cache, training, and evaluation."""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


EXPECTED_CONDITION_SHAPE = (284, 768)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_refs(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        refs = manifest["splits"][split]["samples"]
    except KeyError as exc:
        raise KeyError(f"manifest has no samples for split={split!r}") from exc
    if not refs:
        raise ValueError(f"manifest split={split!r} has no samples")
    return refs


class TensorShardStore:
    """Read-only random access over same-schema tensor shards."""

    def __init__(
        self,
        root: Path,
        *,
        preload: bool = False,
        keys: tuple[str, ...] | None = None,
    ) -> None:
        self.paths = sorted(root.glob("shard-*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no shard files under {root}")
        self.lengths: list[int] = []
        self.offsets = [0]
        self.preloaded: list[dict[str, torch.Tensor]] | None = [] if preload else None
        self.loaded_index = -1
        self.loaded: dict[str, torch.Tensor] | None = None
        self.keys = keys
        for path in self.paths:
            obj = self._load(path)
            if not obj:
                raise ValueError(f"empty tensor shard {path}")
            length = int(next(iter(obj.values())).shape[0])
            if any(int(value.shape[0]) != length for value in obj.values()):
                raise ValueError(f"inconsistent first dimension in {path}")
            self.lengths.append(length)
            self.offsets.append(self.offsets[-1] + length)
            if self.preloaded is not None:
                self.preloaded.append(obj)

    def _load(self, path: Path) -> dict[str, torch.Tensor]:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if self.keys is None:
            return obj
        missing = [key for key in self.keys if key not in obj]
        if missing:
            raise KeyError(f"missing keys {missing} in {path}")
        return {key: obj[key] for key in self.keys}

    def __len__(self) -> int:
        return self.offsets[-1]

    def get(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.offsets, index) - 1
        local = index - self.offsets[shard]
        if shard != self.loaded_index:
            self.loaded = (
                self.preloaded[shard]
                if self.preloaded is not None
                else self._load(self.paths[shard])
            )
            self.loaded_index = shard
        assert self.loaded is not None
        return {key: value[local] for key, value in self.loaded.items()}


class SEC284Dataset(Dataset[dict[str, torch.Tensor | int | str]]):
    """Strictly aligned DINO three-view and VLM-condition dataset."""

    def __init__(
        self,
        feature_cache: Path,
        wrist_cache: Path,
        teacher_cache: Path,
        split: str,
        *,
        preload: bool = False,
    ) -> None:
        self.split = split
        self.feature = TensorShardStore(
            feature_cache / split, preload=preload, keys=("vision_t",)
        )
        self.wrist = TensorShardStore(
            wrist_cache / split,
            preload=preload,
            keys=("vision_left_t", "vision_right_t"),
        )
        self.teacher = TensorShardStore(
            teacher_cache / split,
            preload=preload,
            keys=("teacher_condition", "teacher_mask", "episode_id", "base_index"),
        )
        self.refs = load_manifest_refs(feature_cache / "manifest.json", split)
        lengths = {
            "feature": len(self.feature),
            "wrist": len(self.wrist),
            "teacher": len(self.teacher),
            "manifest": len(self.refs),
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"SEC284 split={split} alignment mismatch: {lengths}")
        self._validate_teacher_identity()

    def _validate_teacher_identity(self) -> None:
        """Reject any teacher/manifest permutation before a training step runs."""
        for index, ref in enumerate(self.refs):
            teacher = self.teacher.get(index)
            if "episode_id" not in teacher or "base_index" not in teacher:
                raise KeyError("teacher cache must contain episode_id and base_index")
            if int(teacher["episode_id"]) != int(ref["episode"]):
                raise RuntimeError(f"teacher episode mismatch at index={index}")
            if int(teacher["base_index"]) != int(ref["base_index"]):
                raise RuntimeError(f"teacher base_index mismatch at index={index}")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        feature = self.feature.get(index)
        wrist = self.wrist.get(index)
        teacher = self.teacher.get(index)
        visual = torch.stack(
            [feature["vision_t"], wrist["vision_left_t"], wrist["vision_right_t"]], dim=0
        ).float()
        if tuple(visual.shape) != (3, 256, 768):
            raise RuntimeError(f"unexpected DINO shape at index={index}: {tuple(visual.shape)}")
        condition = teacher["teacher_condition"].float()
        if tuple(condition.shape) != EXPECTED_CONDITION_SHAPE:
            raise RuntimeError(
                f"unexpected teacher condition at index={index}: {tuple(condition.shape)}"
            )
        teacher_mask = teacher.get("teacher_mask")
        if teacher_mask is None:
            teacher_mask = torch.ones(EXPECTED_CONDITION_SHAPE[0], dtype=torch.bool)
        teacher_mask = teacher_mask.to(dtype=torch.bool)
        if tuple(teacher_mask.shape) != (EXPECTED_CONDITION_SHAPE[0],):
            raise RuntimeError(f"unexpected teacher mask at index={index}: {tuple(teacher_mask.shape)}")
        ref = self.refs[index]
        if int(teacher["episode_id"]) != int(ref["episode"]) or int(teacher["base_index"]) != int(ref["base_index"]):
            raise RuntimeError(f"teacher/manifest identity mismatch at index={index}")
        return {
            "visual_tokens": visual,
            "view_mask": torch.ones(3, dtype=torch.bool),
            "teacher_condition": condition,
            "teacher_mask": teacher_mask,
            "episode_id": int(ref["episode"]),
            "base_index": int(ref["base_index"]),
            "domain": str(ref.get("domain", "unknown")),
        }


@dataclass(frozen=True)
class SEC284LossWeights:
    whitened_mse: float = 1.0
    raw_mse: float = 0.5
    cosine: float = 0.05


def bounded_variance_weights(position_variance: torch.Tensor) -> torch.Tensor:
    if tuple(position_variance.shape) != EXPECTED_CONDITION_SHAPE:
        raise ValueError(f"position_variance must be {EXPECTED_CONDITION_SHAPE}")
    variance = position_variance.float().clamp_min(0.0)
    base = variance.mean() / (variance + 1e-4)
    # Find one global scale whose clipped weights retain mean 1.  Dividing by
    # the mean *after* clipping can silently violate the advertised bounds.
    low, high = 0.0, 1.0
    while (base * high).clamp(0.25, 4.0).mean() < 1.0:
        high *= 2.0
    for _ in range(40):
        middle = (low + high) * 0.5
        if (base * middle).clamp(0.25, 4.0).mean() < 1.0:
            low = middle
        else:
            high = middle
    return (base * high).clamp(min=0.25, max=4.0)


def sec284_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    teacher_mask: torch.Tensor,
    position_variance: torch.Tensor,
    weights: SEC284LossWeights = SEC284LossWeights(),
) -> dict[str, torch.Tensor]:
    if student.shape != teacher.shape or tuple(student.shape[1:]) != EXPECTED_CONDITION_SHAPE:
        raise ValueError(f"student/teacher must both be [B,{EXPECTED_CONDITION_SHAPE[0]},{EXPECTED_CONDITION_SHAPE[1]}]")
    if tuple(teacher_mask.shape) != tuple(student.shape[:2]):
        raise ValueError("teacher_mask must be [B,284]")
    error2 = (student.float() - teacher.float()).square()
    mask = teacher_mask.to(device=student.device, dtype=torch.float32).unsqueeze(-1)
    raw = (error2 * mask).sum() / (mask.sum() * student.shape[-1]).clamp_min(1.0)
    variance_weights = bounded_variance_weights(position_variance).to(student.device)
    white_denom = (variance_weights.unsqueeze(0) * mask).sum().clamp_min(1.0)
    white = (error2 * variance_weights.unsqueeze(0) * mask).sum() / white_denom
    student_fp32, teacher_fp32 = student.float(), teacher.float()
    dot = (student_fp32 * teacher_fp32).sum(dim=-1)
    norm_product = student_fp32.square().sum(dim=-1).sqrt() * teacher_fp32.square().sum(dim=-1).sqrt()
    similarity = torch.where(norm_product > 1e-8, dot / norm_product.clamp_min(1e-8), torch.ones_like(dot))
    cosine = 1.0 - similarity
    cosine = (cosine * teacher_mask.to(student.device, torch.float32)).sum() / teacher_mask.sum().clamp_min(1)
    total = weights.whitened_mse * white + weights.raw_mse * raw + weights.cosine * cosine
    return {"total": total, "raw_mse": raw, "whitened_mse": white, "cosine": cosine}
