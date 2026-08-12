#!/usr/bin/env python3
"""Full-data SEC284 fine-tuning with Frozen-Expert inference-grid KD."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.lap_stage1 import LAP60M
from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.sec284_data import SEC284LossWeights, sec284_distillation_loss
from tools.train_lap10_alignment import LAP10Dataset
from tools.train_lap8_phase1 import grad_norm, load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, load_lawm_decoder


class FullGridDataset(Dataset[dict[str, torch.Tensor]]):
    """Join grid target shards to the original aligned full training cache."""

    def __init__(self, args: argparse.Namespace, *, verbose: bool) -> None:
        self.base = LAP10Dataset(args, verbose=verbose)
        self.paths = sorted(Path(args.grid_cache).glob("rank-*-shard-*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no grid shards under {args.grid_cache}")
        self.lengths: list[int] = []
        self.offsets = [0]
        self.loaded_index = -1
        self.loaded: dict[str, torch.Tensor] | None = None
        for path in self.paths:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            n = int(obj["sample_index"].shape[0])
            self.lengths.append(n)
            self.offsets.append(self.offsets[-1] + n)
        if self.offsets[-1] != len(self.base):
            raise RuntimeError(
                f"grid/base sample mismatch: grid={self.offsets[-1]} base={len(self.base)}"
            )

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = int(index - self.offsets[shard])
        if shard != self.loaded_index:
            self.loaded = torch.load(self.paths[shard], map_location="cpu", weights_only=True)
            self.loaded_index = shard
        assert self.loaded is not None
        global_index = int(self.loaded["sample_index"][local])
        base = self.base[global_index]
        visual = torch.stack([
            base["vision_t"], base["vision_left_t"], base["vision_right_t"]
        ], dim=0).float()
        return {
            "visual_tokens": visual,
            "teacher_condition": base["teacher_condition"].float(),
            "teacher_x_inputs": self.loaded["teacher_x_inputs"][local].float(),
            "teacher_velocities": self.loaded["teacher_velocities"][local].float(),
            "state_t": base["state_t"].float(),
        }


def setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local_rank, torch.device("cuda", local_rank)


def unwrap(module: nn.Module) -> SEC284L:
    return module.module if isinstance(module, DDP) else module  # type: ignore[return-value]


def cosine_schedule(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def reduced(value: torch.Tensor, world: int) -> float:
    value = value.detach().float().clone()
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return float(value.cpu())


def grad_stats(a: torch.Tensor, b: torch.Tensor, world: int) -> tuple[float, float, float]:
    a, b = a.detach().float(), b.detach().float()
    values = torch.stack([a.square().sum(), b.square().sum(), (a * b).sum()])
    if world > 1:
        dist.all_reduce(values)
    na, nb = values[0].sqrt(), values[1].sqrt()
    return float(na), float(nb), float(values[2] / (na * nb).clamp_min(1e-12))


def dynamic_loss(
    student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Light condition anchor that preserves cross-sample directions and scale."""
    valid = mask.to(device=student.device, dtype=torch.float32).unsqueeze(-1)
    student_fp32 = student.float() * valid
    teacher_fp32 = teacher.float() * valid
    student_centered = student_fp32 - student_fp32.mean(dim=0, keepdim=True)
    teacher_centered = teacher_fp32 - teacher_fp32.mean(dim=0, keepdim=True)
    direction = 1.0 - F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=-1, eps=1e-8
    ).mean()
    student_std = torch.sqrt(
        student_centered.square().sum() / valid.sum().clamp_min(1.0) + 1e-8
    )
    teacher_std = torch.sqrt(
        teacher_centered.square().sum() / valid.sum().clamp_min(1.0) + 1e-8
    )
    scale = torch.log((student_std + 1e-6) / (teacher_std + 1e-6)).square()
    return direction + scale, student_std / teacher_std.clamp_min(1e-8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-cache", default="cache/sec284_task14_inference_grid/train")
    p.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    p.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    p.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    p.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    p.add_argument("--stage1-checkpoint", default="outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt")
    p.add_argument("--sec284-checkpoint", default="outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt")
    p.add_argument("--expert-checkpoint", default="cache/lap8_phase1_official_action_expert.pt")
    p.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--output-dir", default="outputs/sec284_inference_grid_kd_1000step")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32, help="local/per-GPU batch size")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--min-lr", type=float, default=3e-7)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--loss-mode", choices=("repr-primary", "output-primary"), default="repr-primary"
    )
    p.add_argument("--grid-gradient-ratio", type=float, default=0.25)
    p.add_argument("--grid-lambda-ema", type=float, default=0.9)
    p.add_argument("--grid-lambda-min", type=float, default=1e-3)
    p.add_argument("--grid-lambda-max", type=float, default=10.0)
    p.add_argument("--condition-gradient-ratio", type=float, default=0.1)
    p.add_argument("--condition-lambda-ema", type=float, default=0.9)
    p.add_argument("--condition-lambda-min", type=float, default=1e-4)
    p.add_argument("--condition-lambda-max", type=float, default=10.0)
    p.add_argument("--dynamic-weight", type=float, default=0.1)
    p.add_argument(
        "--uniform-action-weights", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--gripper-weight", type=float, default=4.0)
    p.add_argument("--xyz-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=284)
    p.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup()
    main_rank = rank == 0
    random.seed(args.seed + rank); np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank); torch.cuda.manual_seed_all(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    dataset = FullGridDataset(args, verbose=main_rank)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    iterator = iter(loader)

    sec_obj = torch.load(args.sec284_checkpoint, map_location="cpu", weights_only=False)
    config = SEC284Config(**sec_obj["config"])
    sec: nn.Module = SEC284L(config).to(device, torch.float32)
    sec.load_state_dict(sec_obj["sec284"], strict=True)
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap6.load_state_dict(stage1["lap"], strict=True)
    lap6 = lap6.to(device, torch.float32).eval()
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml)
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    lawm = lawm.to(device, torch.float32).eval()
    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32).eval()
    for module in (lap6, lawm, expert):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if world > 1:
        sec = DDP(sec, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(sec.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, args.warmup_steps, args.steps, args.min_lr / args.lr)
    )
    variance = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)["position_variance"].to(device, torch.float32)
    repr_weights = SEC284LossWeights()
    auxiliary_lambda = 0.0
    output = Path(args.output_dir)
    if main_rank:
        output.mkdir(parents=True, exist_ok=True)
        print(f"[data] full_samples={len(dataset):,} world={world} local_batch={args.batch_size} grid_cache={args.grid_cache}", flush=True)
        print(f"[model] trainable=SEC284({unwrap(sec).parameter_count:,}) frozen=LAP6,LaWM,ActionExpert", flush=True)
        print(
            f"[objective] mode={args.loss_mode} uniform_action_weights={args.uniform_action_weights} "
            f"steps_per_epoch={len(loader)} planned_epochs={args.steps / len(loader):.3f}",
            flush=True,
        )

    sec.train()
    running = {key: 0.0 for key in ("total", "repr", "grid", "lambda", "cos", "grad", "std_ratio")}
    count = 0
    epoch = 0
    last_log = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); batch = next(iterator)
        visual = batch["visual_tokens"].to(device, torch.float32, non_blocking=True)
        state_t = batch["state_t"].to(device, torch.float32, non_blocking=True)
        teacher = batch["teacher_condition"].to(device, torch.float32, non_blocking=True)
        forced_x_all = batch["teacher_x_inputs"].to(device, torch.float32, non_blocking=True).permute(1, 0, 2, 3).contiguous()
        teacher_velocity_all = batch["teacher_velocities"].to(device, torch.float32, non_blocking=True)
        flow_steps = int(forced_x_all.shape[0])
        grid_step = random.randrange(flow_steps)
        forced_x = forced_x_all[grid_step : grid_step + 1]
        teacher_velocity = teacher_velocity_all[:, grid_step]
        batch_size = visual.shape[0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_lap = lap6(visual, state_t)["z_lap"]
            h_t1 = lawm(visual[:, 0], z_lap).squeeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student = sec(visual, torch.ones(visual.shape[:2], device=device, dtype=torch.bool))
        mask = torch.ones(student.shape[:2], device=device, dtype=torch.bool)
        repr_parts = sec284_distillation_loss(student, teacher, mask, variance, repr_weights)
        dynamic, std_ratio = dynamic_loss(student, teacher, mask)
        repr_loss = repr_parts["total"] + args.dynamic_weight * dynamic

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, grid_trace = expert.sample_actions_cfg_train(
                h_t=visual[:, 0], h_t1_star=h_t1, h_vlm=None, h_lap=student,
                state=torch.zeros(batch_size, 32, device=device),
                state_mask=torch.zeros(batch_size, 32, device=device, dtype=torch.bool),
                action_hz=torch.full((batch_size,), 30.0, device=device),
                embodiment_id=torch.ones(batch_size, device=device, dtype=torch.long),
                attention_mask=mask, cfg_scale=float(expert.config.cfg_guidance_scale),
                num_inference_steps=1, flow_total_steps=flow_steps,
                flow_step_offset=grid_step, return_padded=False, return_trace=True,
                forced_x_inputs=forced_x, gradient_trace_step=0,
            )
        predicted_velocity = grid_trace["velocities"][0].float()
        valid = grid_trace["time_valid"].float().unsqueeze(-1)
        dim_weight = torch.ones(predicted_velocity.shape[-1], device=device)
        if not args.uniform_action_weights:
            dim_weight[[0, 1, 2, 8, 9, 10]] = args.xyz_weight
            dim_weight[[7, 15]] = args.gripper_weight
        weights = valid * dim_weight.view(1, 1, -1)
        grid_loss = ((predicted_velocity - teacher_velocity).square() * weights).sum() / weights.sum().clamp_min(1.0)
        repr_grad = torch.autograd.grad(repr_loss, student, retain_graph=True)[0]
        grid_grad = torch.autograd.grad(grid_loss, student, retain_graph=True)[0]
        repr_norm, grid_norm, grad_cos = grad_stats(repr_grad, grid_grad, world)
        if args.loss_mode == "output-primary":
            target_lambda = min(
                args.condition_lambda_max,
                max(
                    args.condition_lambda_min,
                    args.condition_gradient_ratio * grid_norm / max(repr_norm, 1e-12),
                ),
            )
            auxiliary_lambda = (
                target_lambda
                if auxiliary_lambda == 0
                else args.condition_lambda_ema * auxiliary_lambda
                + (1 - args.condition_lambda_ema) * target_lambda
            )
            total = grid_loss + auxiliary_lambda * repr_loss
            lambda_name = "lambda_condition"
        else:
            target_lambda = min(
                args.grid_lambda_max,
                max(
                    args.grid_lambda_min,
                    args.grid_gradient_ratio * repr_norm / max(grid_norm, 1e-12),
                ),
            )
            auxiliary_lambda = (
                target_lambda
                if auxiliary_lambda == 0
                else args.grid_lambda_ema * auxiliary_lambda
                + (1 - args.grid_lambda_ema) * target_lambda
            )
            total = repr_loss + auxiliary_lambda * grid_loss
            lambda_name = "lambda_grid"
        total.backward()
        raw_grad = grad_norm(sec.parameters())
        torch.nn.utils.clip_grad_norm_(sec.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()

        values = {"total": total, "repr": repr_loss, "grid": grid_loss,
                  "lambda": torch.tensor(auxiliary_lambda, device=device), "cos": repr_parts["cosine"],
                  "grad": torch.tensor(raw_grad, device=device), "std_ratio": std_ratio}
        for key, value in values.items(): running[key] += reduced(value, world)
        count += 1
        if step == 1 or step % args.log_every == 0:
            now = time.monotonic(); avg = {key: value / count for key, value in running.items()}
            if main_rank:
                print(f"[training-loss] step={step}/{args.steps} total={avg['total']:.6f} repr={avg['repr']:.6f} grid_kd={avg['grid']:.6f}", flush=True)
                print(f"[train-state] step={step}/{args.steps} epoch={step / len(loader):.3f} grid_step={grid_step + 1}/{flow_steps} {lambda_name}={avg['lambda']:.5f} condition_grad_cos={grad_cos:.4f} std_ratio={avg['std_ratio']:.4f} sec_grad={avg['grad']:.4f} lr={scheduler.get_last_lr()[0]:.3e} step_time={(now-last_log)/count:.2f}s peak_cuda_rank0={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB", flush=True)
            running = {key: 0.0 for key in running}; count = 0; last_log = now
        if main_rank and (step % args.save_every == 0 or step == args.steps):
            torch.save({"format_version": 1, "model_name": "SEC284-L-full-inference-grid-KD", "step": step,
                        "sec284": unwrap(sec).state_dict(), "config": asdict(config),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "auxiliary_lambda": auxiliary_lambda, "args": vars(args)}, output / f"step-{step:06d}.pt")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
