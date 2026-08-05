#!/usr/bin/env python3
"""Build and train the VLM-free LAP8 Phase-1 Action-Expert conditioner.

The formal training path allocates only LAP8, LaWM, and the released Action
Expert.  Qwen/VLM modules and language inputs are never constructed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import RobotwinEEFDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.model.framework.vlas.flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
)
from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.lap_stage2 import LAP8
from tools.train_lap_stage1 import FeatureShardDataset, diversity_loss, load_lawm_decoder


DEFAULT_STAGE1 = Path(
    "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
)
DEFAULT_POLICY = Path(
    "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
)
DEFAULT_POLICY_CONFIG = Path(
    "results/Checkpoints/robotwin/lawam_robotwin_sft_release/config.yaml"
)
DEFAULT_POLICY_STATS = Path(
    "results/Checkpoints/robotwin/lawam_robotwin_sft_release/dataset_statistics.json"
)
DEFAULT_EXPERT = Path("cache/lap8_phase1_official_action_expert.pt")
DEFAULT_LAM_CKPT = Path(
    "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
)
DEFAULT_LAM_YAML = Path("latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml")
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary")
    return obj


def extract_action_expert(args: argparse.Namespace) -> None:
    source = Path(args.official_policy)
    state = _load_checkpoint_state(source)
    prefixes = ("policy_backend.flow.", "flow.")
    flow_state: dict[str, torch.Tensor] = {}
    used_prefix = ""
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if candidate:
            flow_state = candidate
            used_prefix = prefix
            break
    if not flow_state:
        raise RuntimeError(f"No Action Expert weights found in {source}")
    with Path(args.official_config).open("r", encoding="utf-8") as handle:
        policy_cfg = yaml.safe_load(handle)
    action_cfg = policy_cfg["framework"]["action_model"]
    output = Path(args.expert_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "flow": flow_state,
            "flow_cfg": action_cfg["flow_cfg"],
            "action_horizon": int(action_cfg["action_horizon"]),
            "source": str(source),
            "source_prefix": used_prefix,
        },
        output,
    )
    size_gib = output.stat().st_size / 1024**3
    print(f"[extract] wrote {output} ({len(flow_state)} tensors, {size_gib:.2f} GiB)")


class ActionReader:
    def __init__(self, dataset_root: Path, stats_path: Path, horizon: int = 36) -> None:
        cfg = RobotwinEEFDataConfig()
        self.action_keys = list(cfg.action_keys)
        self.dataset = LeRobotSingleDataset(
            dataset_root,
            {
                "action": ModalityConfig(
                    delta_indices=list(range(horizon)), modality_keys=self.action_keys
                )
            },
            EmbodimentTag.AGILEX,
            mode="all",
            video_backend="pyav",
        )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))["agilex"]["action"]
        self.minimum = torch.tensor(stats["min"], dtype=torch.float32)
        self.maximum = torch.tensor(stats["max"], dtype=torch.float32)
        if self.minimum.shape != (16,) or self.maximum.shape != (16,):
            raise ValueError("Official RoboTwin EEF action statistics must be 16-D")
        self.horizon = int(horizon)

    def get(self, episode: int, base_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [
            torch.from_numpy(
                np.asarray(
                    self.dataset.get_state_or_action(
                        episode, "action", key, base_index
                    )
                )
            ).float()
            for key in self.action_keys
        ]
        raw = torch.cat(parts, dim=-1)
        if raw.shape != (self.horizon, 16):
            raise RuntimeError(f"Unexpected raw action shape {tuple(raw.shape)}")

        normalized = raw.clone()
        # RobotwinEEFDataConfig normalizes only xyz and grippers. Quaternion
        # values remain in their original unit representation.
        normalized_indices = (0, 1, 2, 7, 8, 9, 10, 15)
        idx = torch.tensor(normalized_indices, dtype=torch.long)
        span = (self.maximum[idx] - self.minimum[idx]).clamp_min(1e-12)
        normalized[:, idx] = 2.0 * (raw[:, idx] - self.minimum[idx]) / span - 1.0
        normalized[:, 7] = -normalized[:, 7]
        normalized[:, 15] = -normalized[:, 15]

        actions = torch.zeros(50, 32, dtype=torch.float32)
        mask = torch.zeros(50, 32, dtype=torch.bool)
        actions[: self.horizon, :16] = normalized
        mask[: self.horizon, :16] = True
        return actions, mask


def build_action_cache(args: argparse.Namespace) -> None:
    manifest_path = Path(args.feature_cache) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reader = ActionReader(
        Path(args.dataset_root), Path(args.official_statistics), horizon=36
    )
    output_root = Path(args.action_cache)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        refs = manifest["splits"][split]["samples"]
        main_paths = sorted((Path(args.feature_cache) / split).glob("shard-*.pt"))
        if not main_paths:
            raise FileNotFoundError(f"No Stage-1 feature shards for split {split}")
        cursor = 0
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, main_path in enumerate(main_paths):
            output_path = out_dir / main_path.name
            main_obj = torch.load(main_path, map_location="cpu", weights_only=True)
            shard_size = int(main_obj["vision_t"].shape[0])
            if output_path.exists() and not args.overwrite_cache:
                existing = torch.load(output_path, map_location="cpu", weights_only=True)
                if int(existing["actions"].shape[0]) != shard_size:
                    raise RuntimeError(f"Misaligned existing action shard {output_path}")
                cursor += shard_size
                continue
            actions, masks = [], []
            for ref in refs[cursor : cursor + shard_size]:
                action, mask = reader.get(int(ref["episode"]), int(ref["base_index"]))
                actions.append(action)
                masks.append(mask)
            if len(actions) != shard_size:
                raise RuntimeError(f"Manifest ended while building {main_path.name}")
            torch.save(
                {"actions": torch.stack(actions), "actions_mask": torch.stack(masks)},
                output_path,
            )
            cursor += shard_size
            if shard_index == 0 or (shard_index + 1) % 20 == 0 or cursor == len(refs):
                print(
                    f"[action-cache] {split}: {cursor}/{len(refs)} -> {output_path.name}",
                    flush=True,
                )
        if cursor != len(refs):
            raise RuntimeError(
                f"Feature/manifest length mismatch for {split}: shards={cursor}, refs={len(refs)}"
            )


class TensorShardDataset(Dataset):
    def __init__(self, split_dir: Path, *, preload: bool, verbose: bool) -> None:
        self.paths = sorted(split_dir.glob("shard-*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No tensor shards under {split_dir}")
        self.lengths: list[int] = []
        self.objects: list[dict[str, torch.Tensor]] | None = [] if preload else None
        for index, path in enumerate(self.paths):
            obj = torch.load(path, map_location="cpu", weights_only=True)
            self.lengths.append(int(next(iter(obj.values())).shape[0]))
            if self.objects is not None:
                self.objects.append(obj)
            if verbose and preload and ((index + 1) % 50 == 0 or index + 1 == len(self.paths)):
                print(f"[data] preloaded actions {index + 1}/{len(self.paths)} shards", flush=True)
        self.offsets = np.cumsum([0] + self.lengths)
        self.loaded_index = -1
        self.loaded: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = int(index - self.offsets[shard])
        if shard != self.loaded_index:
            self.loaded = (
                self.objects[shard]
                if self.objects is not None
                else torch.load(self.paths[shard], map_location="cpu", weights_only=True)
            )
            self.loaded_index = shard
        assert self.loaded is not None
        return {key: value[local] for key, value in self.loaded.items()}


class Phase1Dataset(Dataset):
    def __init__(
        self,
        feature_dir: Path,
        wrist_dir: Path,
        action_dir: Path,
        *,
        preload: bool,
        verbose: bool,
    ) -> None:
        self.features = FeatureShardDataset(
            feature_dir, auxiliary_split_dir=wrist_dir, preload=preload, verbose=verbose
        )
        self.actions = TensorShardDataset(action_dir, preload=preload, verbose=verbose)
        if len(self.features) != len(self.actions):
            raise RuntimeError(
                f"Feature/action sample mismatch: {len(self.features)} vs {len(self.actions)}"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = self.features[index]
        result.update(self.actions[index])
        return result


def load_action_expert(path: Path) -> ConditionalFlowMatchingHead:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    config = ConditionalFlowMatchingConfig(**obj["flow_cfg"])
    expert = ConditionalFlowMatchingHead(config)
    expert.action_horizon = int(obj["action_horizon"])
    expert.load_state_dict(obj["flow"], strict=True)
    return expert


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    norms = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(norms).norm().cpu()) if norms else 0.0


def off_diagonal_cosine(tokens: torch.Tensor) -> float:
    x = F.normalize(tokens.detach(), dim=-1)
    gram = x @ x.transpose(1, 2)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=torch.bool).unsqueeze(0)
    return float(gram.masked_select(~eye).mean().cpu())


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def train(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-1 training requires CUDA")
    if distributed:
        torch.cuda.set_device(local_rank)
        # Some project imports construct an Accelerate PartialState when
        # torchrun variables are present.  In that case the default process
        # group already exists before this entry point is reached.
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank if distributed else 0)
    is_main = rank == 0
    seed_everything(args.seed + rank)

    dataset = Phase1Dataset(
        Path(args.feature_cache) / "train",
        Path(args.wrist_cache) / "train",
        Path(args.action_cache) / "train",
        preload=args.preload_cache,
        verbose=is_main,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.2)
    lap6.load_state_dict(stage1["lap"], strict=True)
    lap8: nn.Module = LAP8(lap6, view_dropout=args.view_dropout).to(device, torch.float32)

    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml)
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    lawm.to(device, torch.float32).eval()
    for parameter in lawm.parameters():
        parameter.requires_grad_(False)

    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32).eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)

    if distributed:
        lap8 = DDP(lap8, device_ids=[local_rank], output_device=local_rank)
    trainable = [parameter for parameter in lap8.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )

    if is_main:
        total_lap = count_parameters(unwrap(lap8))
        print(
            f"[train] device={device} world_size={world_size} precision=FP32 "
            f"samples={len(dataset):,} effective_batch={world_size * args.batch_size * args.grad_accumulation}",
            flush=True,
        )
        print(
            f"[model] LAP8={total_lap:,} trainable={sum(x.numel() for x in trainable):,} "
            f"LaWM={count_parameters(lawm):,} Expert={count_parameters(expert):,} VLM=not-loaded",
            flush=True,
        )

    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    lap8.train()
    started = time.perf_counter()
    last_log_time = started
    last_log_step = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": 0.0, "flow": 0.0, "div": 0.0, "cond_rms": 0.0, "cond_cos": 0.0}
        for micro in range(args.grad_accumulation):
            if distributed:
                lap8.require_backward_grad_sync = micro == args.grad_accumulation - 1
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            vision_t = batch["vision_t"].to(device, torch.float32, non_blocking=True)
            lap_visual = torch.stack(
                [
                    vision_t,
                    batch["vision_left_t"].to(device, torch.float32, non_blocking=True),
                    batch["vision_right_t"].to(device, torch.float32, non_blocking=True),
                ],
                dim=1,
            )
            state_t = batch["state_t"].to(device, torch.float32, non_blocking=True)
            actions = batch["actions"].to(device, torch.float32, non_blocking=True)
            actions_mask = batch["actions_mask"].to(device, torch.bool, non_blocking=True)

            lap_out = lap8(lap_visual, state_t)
            with torch.no_grad():
                h_t1 = lawm(vision_t, lap_out["z_lap"]).squeeze(1)
            cond_lap = lap_out["cond_lap"]
            if args.condition_dropout > 0.0:
                keep = (
                    torch.rand(cond_lap.shape[0], 1, 1, device=device)
                    >= args.condition_dropout
                ).to(cond_lap.dtype)
                expert_condition = cond_lap * keep
            else:
                expert_condition = cond_lap
            batch_size = vision_t.shape[0]
            flow_loss = expert(
                h_t=vision_t,
                h_t1_star=h_t1,
                h_vlm=None,
                h_lap=expert_condition,
                state=torch.zeros(batch_size, 32, device=device, dtype=torch.float32),
                actions=actions,
                action_hz=torch.full((batch_size,), 30.0, device=device),
                embodiment_id=torch.ones(batch_size, device=device, dtype=torch.long),
                state_mask=torch.zeros(batch_size, 32, device=device, dtype=torch.bool),
                actions_mask=actions_mask,
                attention_mask=torch.ones(batch_size, cond_lap.shape[1], device=device, dtype=torch.bool),
            )
            div_loss = diversity_loss(lap_out["scene_lap8"])
            total = flow_loss + args.diversity_weight * div_loss
            (total / args.grad_accumulation).backward()
            values = {
                "loss": float(total.detach().cpu()),
                "flow": float(flow_loss.detach().cpu()),
                "div": float(div_loss.detach().cpu()),
                "cond_rms": float(cond_lap.detach().square().mean().sqrt().cpu()),
                "cond_cos": off_diagonal_cosine(cond_lap),
            }
            for key, value in values.items():
                metrics[key] += value / args.grad_accumulation

        raw_grad = grad_norm(trainable)
        clip_input_grad = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        grad_after_clip = min(raw_grad, float(args.grad_clip))
        optimizer.step()
        scheduler.step()

        if is_main and (step == 1 or step % args.log_every == 0):
            now = time.perf_counter()
            seconds_per_step = (now - last_log_time) / (step - last_log_step)
            eta_hours = seconds_per_step * (args.steps - step) / 3600.0
            print(
                f"[train] step={step}/{args.steps} "
                + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                + f" grad_before_clip={raw_grad:.4f} grad_after_clip={grad_after_clip:.4f} "
                + f"lr={scheduler.get_last_lr()[0]:.3e} step_time={seconds_per_step:.2f}s eta={eta_hours:.2f}h",
                flush=True,
            )
            last_log_time, last_log_step = now, step

        should_save = (args.save_every > 0 and step % args.save_every == 0) or step == args.steps
        if should_save:
            if is_main:
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "lap8": unwrap(lap8).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                        "stage1_checkpoint": str(args.stage1_checkpoint),
                        "expert_checkpoint": str(args.expert_checkpoint),
                    },
                    output / f"lap8_phase1_step{step:07d}.pt",
                )
                print(f"[train] checkpoint saved at step {step}", flush=True)
            if distributed:
                dist.barrier()

    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract_expert", "build_action_cache", "train"), required=True)
    parser.add_argument("--dataset-root", default="dataset/robotwin_merged")
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    parser.add_argument("--official-policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--official-config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--official-statistics", default=str(DEFAULT_POLICY_STATS))
    parser.add_argument("--expert-checkpoint", default=str(DEFAULT_EXPERT))
    parser.add_argument("--stage1-checkpoint", default=str(DEFAULT_STAGE1))
    parser.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    parser.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--view-dropout", type=float, default=0.2)
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default="outputs/lap8_phase1_task14_1000step")
    args = parser.parse_args()
    if args.mode == "train":
        required = (
            Path(args.feature_cache) / "train",
            Path(args.wrist_cache) / "train",
            Path(args.action_cache) / "train",
            Path(args.stage1_checkpoint),
            Path(args.expert_checkpoint),
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            parser.error("Missing training inputs: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parsed = parse_args()
    if parsed.mode == "extract_expert":
        extract_action_expert(parsed)
    elif parsed.mode == "build_action_cache":
        build_action_cache(parsed)
    else:
        train(parsed)
