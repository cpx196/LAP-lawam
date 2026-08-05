#!/usr/bin/env python3
"""Single-GPU FP32 Stage-1 training for the task-14 60M LAP.

The script intentionally keeps the conceptual names from the training plan:

    frozen DINO/V-JEPA -> F_t, F_t1
    frozen IDM(F_t, F_t1) -> z_idm (teacher)
    LAP60M(F_t, s_t) -> z_lap, scene_tokens
    LaWM(F_t, z) -> F_hat_t1

The expensive frozen visual/IDM part is cached once.  Training then operates on
feature shards and does not decode the 74-GiB video dataset repeatedly.  All
training computations are FP32; this script does not use autocast or GradScaler.

Examples (use the currently free card 7 on this machine):

    CUDA_VISIBLE_DEVICES=7 python tools/train_lap_stage1.py \
      --mode build_cache --cache-dir cache/lap_stage1_task14 \
      --cache-dtype float32 --cache-batch-size 2

    CUDA_VISIBLE_DEVICES=7 python tools/train_lap_stage1.py \
      --mode train --phase 1 --cache-dir cache/lap_stage1_task14 \
      --steps 10000 --batch-size 1 --grad-accumulation 8
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Make direct execution (`python tools/train_lap_stage1.py`) independent of an
# externally configured PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.data_loader.video_aug import imagenet_normalize_
from starVLA.dataloader.gr00t_lerobot.data_config import RobotwinEEFDataConfig
from starVLA.dataloader.gr00t_lerobot.data_config_lam import build_lam_dataset_transform
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoTransform
from starVLA.model.lap_stage1 import LAP60M, count_parameters


SEED = 42
DEFAULT_DATASET_ROOT = Path("dataset/robotwin_merged")
DEFAULT_LAM_CKPT = Path(
    "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
)
DEFAULT_LAM_YAML = Path("latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml")
TASK14_RAND = list(range(6500, 7000))
TASK14_CLEAN = list(range(25650, 25700))


@dataclass(frozen=True)
class SampleRef:
    episode: int
    base_index: int
    domain: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_manifest(dataset: LeRobotSingleDataset, seed: int = SEED) -> dict[str, Any]:
    """Create the episode-level split required by the training plan."""
    rng = random.Random(seed)
    rand_ids = TASK14_RAND.copy()
    clean_ids = TASK14_CLEAN.copy()
    rng.shuffle(rand_ids)
    rng.shuffle(clean_ids)
    split_eps = {
        "train": {"randomized": rand_ids[:440], "clean": clean_ids[:40]},
        "val": {"randomized": rand_ids[440:470], "clean": clean_ids[40:45]},
        "test": {"randomized": rand_ids[470:], "clean": clean_ids[45:]},
    }
    lengths = {int(e): int(l) for e, l in zip(dataset.trajectory_ids, dataset.trajectory_lengths)}
    manifest: dict[str, Any] = {"seed": seed, "task": "move_pillbottle_pad", "splits": {}}
    for split, domains in split_eps.items():
        samples: list[dict[str, Any]] = []
        for domain, episodes in domains.items():
            for ep in episodes:
                if ep not in lengths:
                    raise KeyError(f"Episode {ep} from task14 is absent from {dataset.dataset_path}")
                length = lengths[ep]
                anchors = list(range(0, length, 3))
                terminal = max(0, length - 35)
                if terminal not in anchors:
                    anchors.append(terminal)
                for base in sorted(set(anchors)):
                    samples.append({"episode": ep, "base_index": int(base), "domain": domain})
        # Keep the split ordering deterministic while mixing clean/randomized
        # at the documented natural ~10:1 ratio.
        rng.shuffle(samples)
        manifest["splits"][split] = {
            "episodes": {k: [int(x) for x in v] for k, v in domains.items()},
            "samples": samples,
        }
    return manifest


def load_or_make_manifest(
    dataset: LeRobotSingleDataset, cache_dir: Path, seed: int = SEED
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    manifest = make_manifest(dataset, seed=seed)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] wrote {path}")
    for split, value in manifest["splits"].items():
        print(f"[manifest] {split}: {len(value['samples'])} frame pairs")
    return manifest


def _unit_quaternion(state: torch.Tensor) -> torch.Tensor:
    state = state.clone()
    for start in (3, 11):
        quat = state[..., start : start + 4]
        state[..., start : start + 4] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return state


class Task14RawReader:
    """Thin adapter around the existing LeRobot reader.

    It returns the two endpoint frames and normalized endpoint EEF states.  No
    language, action sequence, history, or task token is passed to LAP.
    """

    def __init__(self, dataset_root: Path, delta: int = 35) -> None:
        cfg = RobotwinEEFDataConfig()
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=[0, delta], modality_keys=["video.cam_high"]
            ),
            "state": ModalityConfig(
                delta_indices=[0, delta], modality_keys=cfg.state_keys
            ),
        }
        self.state_keys = list(cfg.state_keys)
        self.dataset = LeRobotSingleDataset(
            dataset_root,
            modality_configs,
            EmbodimentTag.AGILEX,
            mode="all",
            video_backend="pyav",
        )
        # Keep exactly the deterministic video preprocessing from the existing
        # config, but do not apply its global state statistics.  State stats
        # are computed below from the train episodes only, as required by the
        # Stage-1 plan.
        base_transform = RobotwinEEFDataConfig().transform(image_hw=(256, 256))
        video_transforms = []
        for transform in base_transform.transforms:
            if not isinstance(transform, VideoTransform):
                continue
            transform_copy = copy.deepcopy(transform)
            transform_copy.apply_to = ["video.cam_high"]
            video_transforms.append(transform_copy)
        self.video_transform = ComposedModalityTransform(transforms=video_transforms)
        self.video_transform.set_metadata(self.dataset.metadata)
        self.video_transform.eval()
        self.state_mean = torch.zeros(16, dtype=torch.float32)
        self.state_std = torch.ones(16, dtype=torch.float32)

    def set_state_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (16,) or std.shape != (16,):
            raise ValueError(f"state statistics must be [16], got {mean.shape}/{std.shape}")
        self.state_mean = mean.float().clone()
        self.state_std = std.float().clamp_min(1e-6).clone()

    def get_raw_states(self, ref: SampleRef) -> torch.Tensor:
        parts = [
            torch.from_numpy(
                np.asarray(
                    self.dataset.get_state_or_action(
                        ref.episode, "state", key, ref.base_index
                    )
                )
            ).float()
            for key in self.state_keys
        ]
        return _unit_quaternion(torch.cat(parts, dim=-1))

    def get(self, ref: SampleRef) -> dict[str, torch.Tensor]:
        ds = self.dataset
        raw: dict[str, Any] = {
            "video.cam_high": ds.get_video(
                ref.episode, "video.cam_high", ref.base_index
            )
        }
        data = self.video_transform(raw)
        frames = data["video.cam_high"]
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(np.asarray(frames))
        frames = frames.to(dtype=torch.float32).div_(255.0)
        imagenet_normalize_(frames)
        states = self.get_raw_states(ref)
        states = (states - self.state_mean) / self.state_std
        if frames.shape != (2, 3, 256, 256):
            raise RuntimeError(f"Unexpected transformed frame shape: {tuple(frames.shape)}")
        if states.shape != (2, 16):
            raise RuntimeError(f"Unexpected transformed EEF state shape: {tuple(states.shape)}")
        return {
            "videos": frames,
            "state_t": states[0],
            "state_t1": states[1],
        }


def build_train_state_stats(reader: Task14RawReader, refs: list[SampleRef]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute state statistics from train references only."""
    count = 0
    mean = torch.zeros(16, dtype=torch.float64)
    m2 = torch.zeros(16, dtype=torch.float64)
    for idx, ref in enumerate(refs):
        values = reader.get_raw_states(ref).to(dtype=torch.float64).reshape(-1, 16)
        for row in values:
            count += 1
            delta = row - mean
            mean += delta / count
            m2 += delta * (row - mean)
        if (idx + 1) % 2000 == 0:
            print(f"[stats] {idx + 1}/{len(refs)} train samples", flush=True)
    if count < 2:
        raise RuntimeError("Not enough train states to compute statistics")
    std = torch.sqrt((m2 / (count - 1)).clamp_min(1e-12))
    return mean.float(), std.float()


def load_or_build_state_stats(
    reader: Task14RawReader, manifest: dict[str, Any], cache_dir: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    path = cache_dir / "state_stats.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        return torch.tensor(obj["mean"], dtype=torch.float32), torch.tensor(obj["std"], dtype=torch.float32)
    refs = refs_from_manifest(manifest, "train")
    print(f"[stats] computing train-only EEF stats from {len(refs)} frame pairs", flush=True)
    mean, std = build_train_state_stats(reader, refs)
    path.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[stats] wrote {path}")
    return mean, std


class RefDataset(Dataset):
    def __init__(self, refs: list[SampleRef], reader: Task14RawReader) -> None:
        self.refs = refs
        self.reader = reader

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.reader.get(self.refs[index])


def refs_from_manifest(manifest: dict[str, Any], split: str) -> list[SampleRef]:
    return [
        SampleRef(int(x["episode"]), int(x["base_index"]), str(x["domain"]))
        for x in manifest["splits"][split]["samples"]
    ]


def _save_cache_shard(path: Path, batch: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in batch.items()}, path)


def build_cache(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[cache] CUDA is unavailable; feature extraction will run on CPU")
    print(f"[cache] device={device}, dtype={args.cache_dtype}")

    reader = Task14RawReader(Path(args.dataset_root), delta=args.frame_delta)
    manifest = load_or_make_manifest(reader.dataset, Path(args.cache_dir), args.seed)
    state_mean, state_std = load_or_build_state_stats(
        reader, manifest, Path(args.cache_dir)
    )
    reader.set_state_stats(state_mean, state_std)
    lam = load_latent_action_model(args.lam_ckpt, args.lam_yaml).to(device).eval()
    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    embodiment_ids = torch.full(
        (args.cache_batch_size,), int(reader.dataset.embodiment_id), dtype=torch.long, device=device
    )

    splits: Iterable[str] = [args.split] if args.split != "all" else ("train", "val", "test")
    for split in splits:
        refs = refs_from_manifest(manifest, split)
        if args.max_samples > 0:
            refs = refs[: args.max_samples]
            print(f"[cache] {split}: limiting smoke run to {len(refs)} samples")
        split_dir = Path(args.cache_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(split_dir.glob("shard-*.pt"))
        if existing and not args.overwrite_cache:
            print(f"[cache] {split}: {len(existing)} shards already exist; skip")
            continue
        if args.overwrite_cache:
            for path in existing:
                path.unlink()
        shard_size = int(args.cache_shard_size)
        for shard_start in range(0, len(refs), shard_size):
            shard_refs = refs[shard_start : shard_start + shard_size]
            shard_path = split_dir / f"shard-{shard_start // shard_size:05d}.pt"
            if shard_path.exists() and not args.overwrite_cache:
                continue
            out: dict[str, list[torch.Tensor]] = {
                "vision_t": [],
                "vision_t1": [],
                "z_idm": [],
                "state_t": [],
                "state_t1": [],
            }
            for batch_start in range(0, len(shard_refs), args.cache_batch_size):
                batch_refs = shard_refs[batch_start : batch_start + args.cache_batch_size]
                raw_batch = [reader.get(ref) for ref in batch_refs]
                videos = torch.stack([x["videos"] for x in raw_batch], dim=0).to(device)
                with torch.no_grad():
                    features = lam.extract_vision_features(videos).float()
                    ids = embodiment_ids[: len(batch_refs)]
                    z_idm = lam.get_latent_action(
                        videos=videos,
                        states=None,
                        dec_videos=videos,
                        predict_future_frame=False,
                        embodiment_ids=ids,
                    )["quantized"].float()
                out["vision_t"].append(features[:, 0].cpu().to(cache_dtype))
                out["vision_t1"].append(features[:, -1].cpu().to(cache_dtype))
                out["z_idm"].append(z_idm.cpu().to(cache_dtype))
                out["state_t"].append(torch.stack([x["state_t"] for x in raw_batch]))
                out["state_t1"].append(torch.stack([x["state_t1"] for x in raw_batch]))
                if (shard_start + batch_start) % max(args.cache_batch_size * 10, 1) == 0:
                    print(
                        f"[cache] {split} {shard_start + batch_start}/{len(refs)}",
                        flush=True,
                    )
                del videos, features, z_idm
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            shard = {key: torch.cat(values, dim=0) for key, values in out.items()}
            _save_cache_shard(shard_path, shard)
            print(f"[cache] wrote {shard_path} ({len(shard_refs)} samples)", flush=True)


class FeatureShardDataset(Dataset):
    """Random-access view over cached features, optionally preloaded into RAM."""

    def __init__(
        self,
        split_dir: Path,
        *,
        auxiliary_split_dir: Path | None = None,
        preload: bool = False,
        verbose: bool = True,
    ) -> None:
        self.paths = sorted(split_dir.glob("shard-*.pt"))
        if not self.paths:
            raise FileNotFoundError(
                f"No feature shards under {split_dir}. Run this script with --mode build_cache first."
            )
        self.lengths: list[int] = []
        self.auxiliary_paths = (
            sorted(auxiliary_split_dir.glob("shard-*.pt"))
            if auxiliary_split_dir is not None
            else []
        )
        if auxiliary_split_dir is not None:
            if len(self.auxiliary_paths) != len(self.paths):
                raise RuntimeError(
                    f"Main/wrist shard count mismatch: {len(self.paths)} vs "
                    f"{len(self.auxiliary_paths)} ({auxiliary_split_dir})"
                )
            if [p.name for p in self.auxiliary_paths] != [p.name for p in self.paths]:
                raise RuntimeError("Main/wrist shard filenames are not aligned")
        self._preloaded: list[dict[str, torch.Tensor]] | None = [] if preload else None
        self._preloaded_auxiliary: list[dict[str, torch.Tensor]] | None = (
            [] if preload and self.auxiliary_paths else None
        )
        started = time.perf_counter()
        for index, path in enumerate(self.paths):
            obj = torch.load(path, map_location="cpu", weights_only=True)
            self.lengths.append(int(obj["vision_t"].shape[0]))
            auxiliary_obj = None
            if self.auxiliary_paths:
                auxiliary_obj = torch.load(
                    self.auxiliary_paths[index], map_location="cpu", weights_only=True
                )
                if int(auxiliary_obj["vision_left_t"].shape[0]) != self.lengths[-1]:
                    raise RuntimeError(f"Main/wrist sample mismatch in {path.name}")
            if self._preloaded is not None:
                self._preloaded.append(obj)
                assert self._preloaded_auxiliary is not None and auxiliary_obj is not None
                self._preloaded_auxiliary.append(auxiliary_obj)
                if verbose and ((index + 1) % 20 == 0 or index + 1 == len(self.paths)):
                    print(
                        f"[data] preloaded {index + 1}/{len(self.paths)} shards "
                        f"from {split_dir}",
                        flush=True,
                    )
        self._loaded_index = -1
        self._loaded: dict[str, torch.Tensor] | None = None
        self._loaded_auxiliary: dict[str, torch.Tensor] | None = None
        self._offsets = np.cumsum([0] + self.lengths)
        if self._preloaded is not None and verbose:
            gib = sum(path.stat().st_size for path in self.paths) / (1024**3)
            print(
                f"[data] {len(self):,} samples ({gib:.1f} GiB) resident in CPU RAM; "
                f"load_time={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _ensure_loaded(self, shard_index: int) -> None:
        if shard_index == self._loaded_index:
            return
        if self._preloaded is not None:
            self._loaded = self._preloaded[shard_index]
            if self._preloaded_auxiliary is not None:
                self._loaded_auxiliary = self._preloaded_auxiliary[shard_index]
        else:
            self._loaded = torch.load(
                self.paths[shard_index], map_location="cpu", weights_only=True
            )
            if self.auxiliary_paths:
                self._loaded_auxiliary = torch.load(
                    self.auxiliary_paths[shard_index], map_location="cpu", weights_only=True
                )
        self._loaded_index = shard_index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        shard = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = int(index - self._offsets[shard])
        self._ensure_loaded(shard)
        assert self._loaded is not None
        result = {key: value[local].float() for key, value in self._loaded.items()}
        if self._loaded_auxiliary is not None:
            result.update(
                {key: value[local].float() for key, value in self._loaded_auxiliary.items()}
            )
        return result


class EEFHead(nn.Module):
    def __init__(self, dim: int = 768, state_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, state_dim))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled.squeeze(1))


def diversity_loss(scene_tokens: torch.Tensor) -> torch.Tensor:
    x = F.normalize(scene_tokens, dim=-1)
    gram = torch.matmul(x, x.transpose(1, 2))
    target = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return F.mse_loss(gram, target.expand_as(gram))


def parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).norm(2).cpu())


def load_lawm_decoder(ckpt_path: str | Path, yaml_path: str | Path) -> LAMDecoder_v2:
    """Load only LaWM from the Stage-1 teacher checkpoint.

    DINO and IDM produced the cached tensors and are not used by cached
    training.  Constructing the decoder directly avoids allocating those
    frozen modules on the GPU without changing any LaWM weights.
    """
    with open(yaml_path, "r", encoding="utf-8") as handle:
        cfg = (yaml.safe_load(handle) or {}).get("model", {})
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    prefix = "lam.decoder."
    decoder_state = {
        key[len(prefix) :]: value for key, value in checkpoint.items() if key.startswith(prefix)
    }
    if not decoder_state:
        raise RuntimeError(f"No {prefix} weights found in {ckpt_path}")
    context_dim = int(cfg["dim"])
    project_input = decoder_state.get("project_input.weight")
    input_dim = int(project_input.shape[1]) if project_input is not None else context_dim
    image_hw = tuple(int(x) for x in cfg["image_hw"])
    patch_size = int(cfg["patch_size"])
    decoder = LAMDecoder_v2(
        context_dim=context_dim,
        input_dim=input_dim,
        num_queries=int(cfg.get("num_queries", 1)),
        num_layers=int(cfg.get("dec_layers", 6)),
        num_heads=int(cfg.get("num_heads", 16)),
        dropout=float(cfg.get("dropout", 0.1)),
        grid_hw=(image_hw[0] // patch_size, image_hw[1] // patch_size),
        train_in_latent=True,
        ffn_expansion_factor=int(cfg.get("ffn_expansion_factor", 2)),
        num_embodiments=int(cfg.get("num_embodiments", 32)),
        code_dim=int(cfg["code_dim"]),
        last_ln=bool(cfg.get("decoder_last_ln", True)),
    )
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def train(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    seed_everything(args.seed + rank)
    if is_main:
        print(
            f"[train] device={device}; world_size={world_size}; "
            f"precision=FP32; phase={args.phase}"
        )
    cache_dir = Path(args.cache_dir)
    train_ds = FeatureShardDataset(
        cache_dir / "train",
        auxiliary_split_dir=(Path(args.wrist_cache_dir) / "train" if args.wrist_cache_dir else None),
        preload=args.preload_cache,
        verbose=is_main,
    )
    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    num_views = 3 if args.wrist_cache_dir else 1
    lap: nn.Module = LAP60M(
        num_views=num_views,
        view_dropout=args.view_dropout if num_views > 1 else 0.0,
    ).to(device=device, dtype=torch.float32)
    eef_head: nn.Module = EEFHead().to(device=device, dtype=torch.float32)
    if is_main:
        print(f"[train] LAP parameters={count_parameters(lap):,}")
    lawm: nn.Module = load_lawm_decoder(args.lam_ckpt, args.lam_yaml).to(
        device=device, dtype=torch.float32
    ).eval()
    if is_main:
        print(f"[train] LaWM parameters={count_parameters(lawm):,}; DINO/IDM not allocated")
    for p in lawm.parameters():
        p.requires_grad_(False)
    if args.phase >= 2:
        for p in lawm.parameters():
            p.requires_grad_(True)
        if is_main:
            print(f"[train] phase {args.phase}: LaWM decoder is trainable")

    if distributed:
        lap = DDP(lap, device_ids=[local_rank], output_device=local_rank)
        eef_head = DDP(eef_head, device_ids=[local_rank], output_device=local_rank)
        if args.phase >= 2:
            lawm = DDP(lawm, device_ids=[local_rank], output_device=local_rank)

    groups = [
        {"params": list(lap.parameters()), "lr": args.lap_lr},
        {"params": list(eef_head.parameters()), "lr": args.lap_lr},
    ]
    if args.phase >= 2:
        groups.append({"params": [p for p in lawm.parameters() if p.requires_grad], "lr": args.lawm_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    step = 0
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    lap.train()
    eef_head.train()
    lawm.train(args.phase >= 2)
    train_started = time.perf_counter()
    last_log_time = train_started
    last_log_step = 0
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        metrics = {
            "loss": 0.0,
            "latent": 0.0,
            "student_world": 0.0,
            "teacher_world": 0.0,
            "eef": 0.0,
            "diversity": 0.0,
        }
        for micro in range(args.grad_accumulation):
            if distributed:
                should_sync = micro == args.grad_accumulation - 1
                lap.require_backward_grad_sync = should_sync
                eef_head.require_backward_grad_sync = should_sync
                if args.phase >= 2:
                    lawm.require_backward_grad_sync = should_sync
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            vision_t = batch["vision_t"].to(device=device, dtype=torch.float32, non_blocking=True)
            vision_t1 = batch["vision_t1"].to(device=device, dtype=torch.float32, non_blocking=True)
            z_idm = batch["z_idm"].to(device=device, dtype=torch.float32, non_blocking=True)
            state_t = batch["state_t"].to(device=device, dtype=torch.float32, non_blocking=True)
            state_t1 = batch["state_t1"].to(device=device, dtype=torch.float32, non_blocking=True)

            lap_vision = vision_t
            if args.wrist_cache_dir:
                vision_left_t = batch["vision_left_t"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                vision_right_t = batch["vision_right_t"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                lap_vision = torch.stack(
                    [vision_t, vision_left_t, vision_right_t], dim=1
                )
            lap_out = lap(lap_vision, state_t)
            z_lap = lap_out["z_lap"]
            # Frozen LaWM parameters still permit gradients from its output to
            # z_lap; only the decoder parameters are omitted from the graph in
            # phase 1.
            if args.phase >= 2:
                # One DDP forward per micro-step.  Student and teacher paths
                # are concatenated so the jointly trainable LaWM is safe under
                # multi-GPU DDP and both losses update the same decoder.
                world_pred = lawm(
                    torch.cat([vision_t, vision_t], dim=0),
                    torch.cat([z_lap, z_idm.detach()], dim=0),
                ).squeeze(1)
                student_pred, teacher_pred = world_pred.chunk(2, dim=0)
            else:
                student_pred = lawm(vision_t, z_lap).squeeze(1)
                with torch.no_grad():
                    teacher_pred = lawm(vision_t, z_idm.detach()).squeeze(1)

            latent = F.mse_loss(z_lap, z_idm.detach())
            student_world = F.mse_loss(student_pred, vision_t1)
            teacher_world = F.mse_loss(teacher_pred, vision_t1)
            eef = F.mse_loss(eef_head(lap_out["pooled"]), state_t1)
            diversity = diversity_loss(lap_out["scene_tokens"])
            total_loss = (
                args.w_latent * latent
                + args.w_student_world * student_world
                + args.w_teacher_world * teacher_world
                + args.w_eef * eef
                + args.w_diversity * diversity
            )
            loss = total_loss / args.grad_accumulation
            loss.backward()
            micro_metrics = {
                "loss": float(total_loss.detach().cpu()),
                "latent": float(latent.detach().cpu()),
                "student_world": float(student_world.detach().cpu()),
                "teacher_world": float(teacher_world.detach().cpu()),
                "eef": float(eef.detach().cpu()),
                "diversity": float(diversity.detach().cpu()),
            }
            for key, value in micro_metrics.items():
                metrics[key] += value / args.grad_accumulation
        lap_grad_norm = parameter_grad_norm(lap.parameters())
        lawm_grad_norm = parameter_grad_norm(lawm.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(lap.parameters()) + list(eef_head.parameters()) + [p for p in lawm.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        step += 1
        if is_main and (step == 1 or step % args.log_every == 0):
            now = time.perf_counter()
            seconds_per_step = (now - last_log_time) / (step - last_log_step)
            eta_hours = seconds_per_step * (args.steps - step) / 3600
            print(
                f"[train] step={step}/{args.steps} "
                + " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                + f" grad={float(grad_norm):.3f} lap_grad={lap_grad_norm:.3f} "
                + f"lawm_grad={lawm_grad_norm:.3f} step_time={seconds_per_step:.2f}s "
                + f"eta={eta_hours:.2f}h",
                flush=True,
            )
            last_log_time = now
            last_log_step = step
        periodic_save = args.save_every > 0 and step % args.save_every == 0
        should_save = periodic_save or (args.save_final and step == args.steps)
        if should_save:
            if is_main:
                out = Path(args.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "phase": args.phase,
                        "lap": unwrap_model(lap).state_dict(),
                        "eef_head": unwrap_model(eef_head).state_dict(),
                        "lawm_decoder": (
                            unwrap_model(lawm).state_dict() if args.phase >= 2 else None
                        ),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    out / f"stage1_phase{args.phase}_step{step:07d}.pt",
                )
                print(f"[train] checkpoint saved at step {step}", flush=True)
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["build_cache", "train"], required=True)
    p.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--cache-dir", default="cache/lap_stage1_task14")
    p.add_argument(
        "--wrist-cache-dir",
        default=None,
        help="Aligned left/right current-frame DINO cache; enables three-view LAP input",
    )
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--frame-delta", type=int, default=35)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--cache-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--cache-batch-size", type=int, default=2)
    p.add_argument("--cache-shard-size", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0, help="Limit each split (smoke test only)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=8)
    p.add_argument(
        "--preload-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all train feature shards in CPU RAM (recommended; about 36 GiB for task 14)",
    )
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lap-lr", type=float, default=2e-4)
    p.add_argument("--lawm-lr", type=float, default=1e-5)
    p.add_argument("--view-dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a checkpoint at the final step (disable for throughput tests)",
    )
    p.add_argument("--output-dir", default="outputs/lap_stage1_task14")
    p.add_argument("--w-latent", type=float, default=1.0)
    p.add_argument("--w-student-world", type=float, default=1.0)
    p.add_argument("--w-teacher-world", type=float, default=0.25)
    p.add_argument("--w-eef", type=float, default=0.1)
    p.add_argument("--w-diversity", type=float, default=0.01)
    args = p.parse_args()
    if args.mode == "train" and not Path(args.cache_dir, "train").exists():
        p.error("training requires feature cache; run --mode build_cache first")
    if args.mode == "train" and args.wrist_cache_dir:
        wrist_train = Path(args.wrist_cache_dir, "train")
        if not wrist_train.exists():
            p.error(f"wrist training cache does not exist: {wrist_train}")
    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    arguments = parse_args()
    if arguments.mode == "build_cache":
        build_cache(arguments)
    else:
        train(arguments)
