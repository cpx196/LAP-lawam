#!/usr/bin/env python3
"""Aligned-Results (AR) training for LAP10V3 + RoboTwin Action Expert.

This entry point deliberately has no VLM teacher, teacher cache, or token
alignment objective.  It trains only against the action chunks collected from
RoboTwin.  AR-A trains the Expert with LAP10V3 detached; AR-B additionally
unfreezes LAP9--10 and the residual head.
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
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.lap_stage2 import LAP10V3
from tools.train_lap8_phase1 import Phase1Dataset, grad_norm, load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, load_lawm_decoder


DEFAULT_JOINT = Path("outputs/lap10v3_expert_joint_task14_2000step/lap10v3_expert_step0002000.pt")
DEFAULT_STAGE1 = Path("outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt")
DEFAULT_EXPERT = Path("cache/lap8_phase1_official_action_expert.pt")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_lap_from_joint(path: Path) -> LAP10V3:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("lap10v3")
    if state is None:
        raise KeyError(f"{path} does not contain lap10v3")
    lap6_state = {
        k.removeprefix("lap6."): v for k, v in state.items() if k.startswith("lap6.")
    }
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap6.load_state_dict(lap6_state, strict=True)
    position_mean = state["teacher_position_mean"]
    if position_mean.ndim == 3:
        position_mean = position_mean.squeeze(0)
    lap = LAP10V3(lap6, position_mean, output_tokens=284, view_dropout=0.0)
    lap.load_state_dict(state, strict=True)
    return lap


class ARModel(nn.Module):
    def __init__(self, lap: LAP10V3, expert: nn.Module) -> None:
        super().__init__()
        self.lap = lap
        self.expert = expert

    def forward(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        actions: torch.Tensor,
        actions_mask: torch.Tensor,
        element_weights: Optional[torch.Tensor],
        lap_grad: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if lap_grad:
            lap_out = self.lap(visual, state)
        else:
            with torch.no_grad():
                lap_out = self.lap(visual, state)
            lap_out = {k: v.detach() for k, v in lap_out.items()}
        bsz = visual.shape[0]
        cond = lap_out["cond_lap"]
        details = self.expert(
            h_t=h_t,
            h_t1_star=h_t1,
            h_vlm=None,
            h_lap=cond,
            state=torch.zeros(bsz, 32, device=visual.device, dtype=visual.dtype),
            actions=actions,
            action_hz=torch.full((bsz,), 30.0, device=visual.device, dtype=visual.dtype),
            embodiment_id=torch.ones(bsz, device=visual.device, dtype=torch.long),
            state_mask=torch.zeros(bsz, 32, device=visual.device, dtype=torch.bool),
            actions_mask=actions_mask,
            attention_mask=torch.ones(
                bsz, cond.shape[1], device=visual.device, dtype=torch.bool
            ),
            return_training_details=True,
            element_weights=element_weights,
        )
        return lap_out, details


def build_action_weights(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-element flow weights and per-time event weights.

    The cache stores grippers as -1=open, +1=closed.  Gripper transitions and
    the short neighborhood around them receive extra weight; this does not
    alter the samples or the valid action mask.
    """
    b, t, d = actions.shape
    if d < 16:
        raise ValueError(f"expected at least 16 action dims, got {d}")
    grip = actions[..., 7]
    event = torch.zeros((b, t), dtype=torch.bool, device=actions.device)
    changes = grip[:, 1:].sub(grip[:, :-1]).abs() > 0.1
    event[:, 1:] |= changes
    event[:, :-1] |= changes
    vicinity = torch.zeros_like(event)
    for off in range(-4, 5):
        if off < 0:
            vicinity[:, -off:] |= event[:, : t + off]
        elif off > 0:
            vicinity[:, : t - off] |= event[:, off:]
        else:
            vicinity |= event
    time_weight = torch.ones((b, t), device=actions.device, dtype=actions.dtype)
    time_weight = torch.where(vicinity, torch.full_like(time_weight, 4.0), time_weight)
    time_weight = torch.where(grip > 0.8, torch.maximum(time_weight, torch.full_like(time_weight, 3.0)), time_weight)
    dim_weight = torch.ones((b, t, d), device=actions.device, dtype=actions.dtype)
    dim_weight[..., 0:3] = 2.0
    dim_weight[..., 7] = 4.0
    return dim_weight * time_weight.unsqueeze(-1), time_weight


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = F.smooth_l1_loss(pred, target, reduction="none")
    valid = mask.to(dtype=err.dtype) * weight
    return (err * valid).sum() / valid.sum().clamp_min(1.0)


def action_aux_losses(
    details: dict[str, torch.Tensor],
    actions: torch.Tensor,
    actions_mask: torch.Tensor,
    time_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = details["action_pred"]
    valid_xyz = actions_mask[..., 0:3]
    valid_grip = actions_mask[..., 7]
    rec_xyz = masked_smooth_l1(pred[..., 0:3], actions[..., 0:3], valid_xyz, time_weight.unsqueeze(-1))
    rec_grip = masked_smooth_l1(pred[..., 7], actions[..., 7], valid_grip, time_weight)
    rec = 0.75 * rec_xyz + 0.25 * rec_grip
    pair_mask = valid_xyz[:, 1:] & valid_xyz[:, :-1]
    pred_delta = pred[:, 1:, 0:3] - pred[:, :-1, 0:3]
    true_delta = actions[:, 1:, 0:3] - actions[:, :-1, 0:3]
    pair_weight = 0.5 * (time_weight[:, 1:] + time_weight[:, :-1])
    delta = masked_smooth_l1(pred_delta, true_delta, pair_mask, pair_weight.unsqueeze(-1))
    xyz_mae = ((pred[..., 0:3] - actions[..., 0:3]).abs() * valid_xyz).sum() / valid_xyz.sum().clamp_min(1)
    grip_mae = ((pred[..., 7] - actions[..., 7]).abs() * valid_grip).sum() / valid_grip.sum().clamp_min(1)
    return rec, delta, xyz_mae, grip_mae


def cosine_ratio(index: int, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    if index < warmup:
        return max(1, index + 1) / max(1, warmup)
    progress = min(1.0, (index - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def phase_lrs(step: int, args: argparse.Namespace) -> tuple[float, float, str]:
    if args.freeze_lap:
        ratio = cosine_ratio(step - 1, args.steps, args.warmup_steps)
        return 0.0, args.expert_lr_a * ratio, "FLOW-ONLY"
    if step < args.phase_b_start:
        r = cosine_ratio(step - 1, args.phase_b_start - 1, args.warmup_steps)
        return 0.0, args.expert_lr_a * r, "AR-A"
    index = step - args.phase_b_start
    r = cosine_ratio(index, args.steps - args.phase_b_start + 1, args.phase_b_warmup_steps)
    return args.lap_lr_b * r, args.expert_lr_b * r, "AR-B"


def reduce_values(values: dict[str, float], device: torch.device, world: int) -> dict[str, float]:
    if world == 1:
        return values
    keys = list(values)
    tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}


def save_checkpoint(
    out: Path, step: int, model: DDP, optimizer: torch.optim.Optimizer,
    args: argparse.Namespace, phase: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "phase": phase,
            "lap10v3": model.module.lap.state_dict(),
            "expert": model.module.expert.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": {"step": step, "phase": phase},
            "args": vars(args),
            "source_joint_checkpoint": str(args.joint_checkpoint),
            "trainable_parameter_names": [n for n, p in model.module.named_parameters() if p.requires_grad],
        },
        out / f"lap10v3_ar_step{step:07d}.pt",
    )


def main(args: argparse.Namespace) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("AR training requires CUDA")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed + rank)

    dataset = Phase1Dataset(
        Path(args.feature_cache) / "train", Path(args.wrist_cache) / "train",
        Path(args.action_cache) / "train", preload=args.preload_cache, verbose=rank == 0,
    )
    if rank == 0 and len(dataset) != 24140:
        raise RuntimeError(f"AR expects full 24140-sample train cache, got {len(dataset)}")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)

    joint_obj = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=True)
    lap = load_lap_from_joint(Path(args.joint_checkpoint))
    expert = load_action_expert(Path(args.expert_checkpoint))
    expert.load_state_dict(joint_obj["expert"], strict=True)
    for p in expert.parameters():
        p.requires_grad_(True)
    if hasattr(expert, "enc_vlm"):
        for p in expert.enc_vlm.parameters():
            p.requires_grad_(False)
    for p in lap.parameters():
        p.requires_grad_(False)
    # Register AR-B parameters with DDP from the beginning.  AR-A uses a
    # detached LAP forward and zero LAP LR; AR-B enables their gradients.
    lap_b_params: list[nn.Parameter] = []
    if not args.freeze_lap:
        for name, p in lap.named_parameters():
            if name.startswith("blocks.2") or name.startswith("blocks.3") or name in {"residual_norm.weight", "residual_norm.bias", "residual_head.weight", "residual_head.bias", "residual_scale"}:
                p.requires_grad_(True)
                lap_b_params.append(p)
    lap = lap.to(device, torch.float32)
    expert = expert.to(device, torch.float32)
    model = DDP(ARModel(lap, expert), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    lap_b_params = [p for p in model.module.lap.parameters() if p.requires_grad]
    expert_params = [p for p in model.module.expert.parameters() if p.requires_grad]
    optimizer_groups = [{"params": expert_params, "lr": args.expert_lr_a}]
    if lap_b_params:
        optimizer_groups.insert(0, {"params": lap_b_params, "lr": 0.0})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml).to(device, torch.float32).eval()
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    for p in lawm.parameters():
        p.requires_grad_(False)

    if rank == 0:
        print(f"[AR] no VLM/teacher cache; samples={len(dataset):,} world={world} FP32 flow_only={args.flow_only} freeze_lap={args.freeze_lap}", flush=True)
        print(f"[model] LAP10V3={count_parameters(model.module.lap):,} AR-B trainable={sum(p.numel() for p in lap_b_params):,}; Expert trainable={sum(p.numel() for p in expert_params):,}", flush=True)

    iterator = iter(loader)
    epoch = 0
    sampler.set_epoch(epoch)
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        lap_lr, expert_lr, phase = phase_lrs(step, args)
        if lap_b_params:
            optimizer.param_groups[0]["lr"] = lap_lr
            optimizer.param_groups[1]["lr"] = expert_lr
        else:
            optimizer.param_groups[0]["lr"] = expert_lr
        lap_grad_enabled = (not args.freeze_lap) and step >= args.phase_b_start
        model.module.lap.train(lap_grad_enabled)
        model.module.expert.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {k: 0.0 for k in ("loss", "flow", "recon", "delta", "xyz_mae", "grip_mae")}
        for micro in range(args.grad_accumulation):
            model.require_backward_grad_sync = micro == args.grad_accumulation - 1
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            main = batch["vision_t"].to(device, torch.float32, non_blocking=True)
            visual = torch.stack([main, batch["vision_left_t"].to(device, torch.float32, non_blocking=True), batch["vision_right_t"].to(device, torch.float32, non_blocking=True)], dim=1)
            state = batch["state_t"].to(device, torch.float32, non_blocking=True)
            actions = batch["actions"].to(device, torch.float32, non_blocking=True)
            mask = batch["actions_mask"].to(device, torch.bool, non_blocking=True)
            if args.flow_only:
                weights = time_weight = None
            else:
                weights, time_weight = build_action_weights(actions)
            with torch.no_grad():
                z_lap = model.module.lap.lap6(visual, state)["z_lap"]
                h_t1 = lawm(main, z_lap).squeeze(1)
            _, details = model(
                visual, state, main, h_t1, actions, mask,
                weights, lap_grad_enabled,
            )
            if args.flow_only:
                recon = delta = xyz_mae = grip_mae = torch.zeros((), device=device)
            else:
                assert time_weight is not None
                recon, delta, xyz_mae, grip_mae = action_aux_losses(details, actions, mask, time_weight)
            flow = details["loss"]
            loss = flow if args.flow_only else flow + args.recon_weight * recon + args.delta_weight * delta
            (loss / args.grad_accumulation).backward()
            for k, v in (("loss", loss), ("flow", flow), ("recon", recon), ("delta", delta), ("xyz_mae", xyz_mae), ("grip_mae", grip_mae)):
                sums[k] += float(v.detach()) / args.grad_accumulation
        total_grad = grad_norm(list(model.module.lap.parameters()) + list(model.module.expert.parameters()))
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.module.lap.parameters()) + list(model.module.expert.parameters()), args.grad_clip)
        optimizer.step()
        lap_g = grad_norm(lap_b_params)
        expert_g = grad_norm(expert_params)
        if step % args.log_every == 0 or step == 1:
            metrics = reduce_values({**sums, "lap_grad": lap_g, "expert_grad": expert_g, "total_grad": total_grad}, device, world)
            if rank == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[AR][{phase}] step={step:04d}/{args.steps} loss={metrics['loss']:.6f} flow={metrics['flow']:.6f} recon={metrics['recon']:.6f} delta={metrics['delta']:.6f} xyz_mae={metrics['xyz_mae']:.6f} grip_mae={metrics['grip_mae']:.6f} lap_grad={metrics['lap_grad']:.3e} expert_grad={metrics['expert_grad']:.3e} lr_lap={lap_lr:.2e} lr_exp={expert_lr:.2e} samples={step*args.grad_accumulation*world*args.batch_size} elapsed={elapsed/60:.1f}m", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                save_checkpoint(Path(args.output_dir), step, model, optimizer, args, phase)
                print(f"[AR] checkpoint saved step {step}", flush=True)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-checkpoint", default=str(DEFAULT_JOINT))
    p.add_argument("--stage1-checkpoint", default=str(DEFAULT_STAGE1))
    p.add_argument("--expert-checkpoint", default=str(DEFAULT_EXPERT))
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    p.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    p.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--phase-b-start", type=int, default=1001)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--expert-lr-a", type=float, default=1e-6)
    p.add_argument("--expert-lr-b", type=float, default=5e-7)
    p.add_argument("--lap-lr-b", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--phase-b-warmup-steps", type=int, default=50)
    p.add_argument("--flow-only", action="store_true", help="Use only the unweighted official flow loss.")
    p.add_argument("--freeze-lap", action="store_true", help="Freeze all LAP parameters for the entire run.")
    p.add_argument("--recon-weight", type=float, default=0.5)
    p.add_argument("--delta-weight", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="outputs/lap10v3_ar_expert_task14_2000step")
    return p.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main(parse_args())
