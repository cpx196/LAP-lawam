#!/usr/bin/env python3
"""Fine-tune SEC284 on successful real-VLM inference-grid traces.

The Action Expert stays frozen.  Each update preserves the fixed-instruction
VLM condition and matches one randomly selected velocity on the teacher's
10-step flow trajectory.  Sampling one differentiable grid step bounds memory;
over time all steps receive unbiased supervision.
"""

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.sec284_data import SEC284LossWeights, sec284_distillation_loss
from tools.train_lap8_phase1 import grad_norm, load_action_expert


class GridTraceDataset(Dataset[dict[str, torch.Tensor]]):
    REQUIRED = (
        "features", "h_t", "h_t1", "teacher_condition", "state", "state_mask",
        "action_hz", "embodiment_id", "teacher_x_inputs", "teacher_velocities",
    )

    def __init__(self, roots: list[Path]) -> None:
        self.paths = sorted({path for root in roots for path in root.rglob("call-*.pt")})
        if not self.paths:
            raise FileNotFoundError(f"no call-*.pt traces under {roots}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        missing = [key for key in self.REQUIRED if key not in item]
        if missing:
            raise KeyError(f"{self.paths[index]} missing {missing}")
        result = {
            "features": item["features"].squeeze(0).float(),
            "h_t": item["h_t"].squeeze(0).float(),
            "h_t1": item["h_t1"].squeeze(0).float(),
            "teacher_condition": item["teacher_condition"].squeeze(0).float(),
            "state": item["state"].squeeze(0).float(),
            "state_mask": item["state_mask"].squeeze(0).bool(),
            "action_hz": item["action_hz"].reshape(-1)[0].float(),
            "embodiment_id": item["embodiment_id"].reshape(-1)[0].long(),
            "teacher_x_inputs": item["teacher_x_inputs"].squeeze(1).float(),
            "teacher_velocities": item["teacher_velocities"].squeeze(1).float(),
        }
        if result["features"].shape != (3, 256, 768):
            raise ValueError(f"bad feature shape in {self.paths[index]}: {result['features'].shape}")
        if result["teacher_condition"].shape != (284, 768):
            raise ValueError(
                f"trace is not fixed-instruction 284-token data: {self.paths[index]} "
                f"has {result['teacher_condition'].shape}"
            )
        return result


def setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, local_rank, torch.device("cuda", local_rank)


def unwrap(module: nn.Module) -> SEC284L:
    return module.module if isinstance(module, DDP) else module  # type: ignore[return-value]


def schedule(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def synchronized_gradient_stats(a: torch.Tensor, b: torch.Tensor, world: int) -> tuple[float, float, float]:
    values = torch.stack([
        a.detach().float().square().sum(), b.detach().float().square().sum(),
        (a.detach().float() * b.detach().float()).sum(),
    ])
    if world > 1:
        dist.all_reduce(values)
    na, nb = values[0].sqrt(), values[1].sqrt()
    return float(na), float(nb), float(values[2] / (na * nb).clamp_min(1e-12))


def reduced(value: torch.Tensor, world: int) -> float:
    value = value.detach().float().clone()
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, action="append", required=True)
    parser.add_argument("--sec284-checkpoint", default="outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt")
    parser.add_argument("--expert-checkpoint", default="cache/lap8_phase1_official_action_expert.pt")
    parser.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    parser.add_argument("--output-dir", default="outputs/sec284_inference_grid_kd")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1, help="local trace batch per GPU")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--min-lr", type=float, default=3e-7)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grid-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--grid-lambda-ema", type=float, default=0.9)
    parser.add_argument("--grid-lambda-min", type=float, default=1e-3)
    parser.add_argument("--grid-lambda-max", type=float, default=10.0)
    parser.add_argument("--gripper-weight", type=float, default=4.0)
    parser.add_argument("--xyz-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=284)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup()
    main_rank = rank == 0
    seed = args.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    dataset = GridTraceDataset(args.trace_root)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=args.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    epoch = 0

    checkpoint = torch.load(args.sec284_checkpoint, map_location="cpu", weights_only=False)
    config = SEC284Config(**checkpoint["config"])
    sec: nn.Module = SEC284L(config).to(device, torch.float32)
    sec.load_state_dict(checkpoint["sec284"], strict=True)
    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32).eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    if world > 1:
        sec = DDP(sec, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    trainable = list(sec.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, args.warmup_steps, args.steps, args.min_lr / args.lr)
    )
    variance = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)["position_variance"].to(device)
    repr_weights = SEC284LossWeights()
    grid_lambda = 0.0
    output = Path(args.output_dir)
    if main_rank:
        output.mkdir(parents=True, exist_ok=True)
        print(f"[data] successful_trace_calls={len(dataset)} world={world} local_batch={args.batch_size}", flush=True)
        print("[model] trainable=SEC284 frozen=ActionExpert supervision=fixed-condition+teacher-grid-velocity", flush=True)

    sec.train()
    running = {key: 0.0 for key in ("total", "repr", "grid", "lambda", "cos", "grad", "agree")}
    count = 0
    last = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); batch = next(iterator)
        get = lambda key, dtype=None: batch[key].to(device, dtype=dtype, non_blocking=True)
        features = get("features", torch.float32)
        teacher_condition = get("teacher_condition", torch.float32)
        num_grid = int(batch["teacher_x_inputs"].shape[1])
        grid_step = random.randrange(num_grid)
        forced_x = get("teacher_x_inputs", torch.float32).permute(1, 0, 2, 3).contiguous()
        teacher_velocity = get("teacher_velocities", torch.float32)[:, grid_step]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student = sec(features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
        mask = torch.ones(student.shape[:2], device=device, dtype=torch.bool)
        repr_parts = sec284_distillation_loss(student, teacher_condition, mask, variance, repr_weights)
        repr_loss = repr_parts["total"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, grid_trace = expert.sample_actions_cfg_train(
                h_t=get("h_t", torch.float32), h_t1_star=get("h_t1", torch.float32),
                h_vlm=None, h_lap=student, state=get("state", torch.float32),
                state_mask=get("state_mask", torch.bool), action_hz=get("action_hz", torch.float32),
                embodiment_id=get("embodiment_id", torch.long), attention_mask=mask,
                cfg_scale=float(expert.config.cfg_guidance_scale), num_inference_steps=num_grid,
                return_padded=False, return_trace=True, forced_x_inputs=forced_x,
                gradient_trace_step=grid_step,
            )
        predicted_velocity = grid_trace["velocities"][grid_step].float()
        valid = grid_trace["time_valid"].float().unsqueeze(-1)
        dim_weight = torch.ones(predicted_velocity.shape[-1], device=device)
        dim_weight[[0, 1, 2, 8, 9, 10]] = args.xyz_weight
        dim_weight[[7, 15]] = args.gripper_weight
        weights = valid * dim_weight.view(1, 1, -1)
        grid_loss = ((predicted_velocity - teacher_velocity).square() * weights).sum() / weights.sum().clamp_min(1)

        repr_grad = torch.autograd.grad(repr_loss, student, retain_graph=True)[0]
        grid_grad = torch.autograd.grad(grid_loss, student, retain_graph=True)[0]
        nr, ng, agreement = synchronized_gradient_stats(repr_grad, grid_grad, world)
        target = min(args.grid_lambda_max, max(args.grid_lambda_min, args.grid_gradient_ratio * nr / max(ng, 1e-12)))
        grid_lambda = target if grid_lambda == 0 else args.grid_lambda_ema * grid_lambda + (1 - args.grid_lambda_ema) * target
        total = repr_loss + grid_lambda * grid_loss
        total.backward()
        raw_grad = grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step(); scheduler.step()

        values = {"total": total, "repr": repr_loss, "grid": grid_loss,
                  "lambda": torch.tensor(grid_lambda, device=device), "cos": repr_parts["cosine"],
                  "grad": torch.tensor(raw_grad, device=device), "agree": torch.tensor(agreement, device=device)}
        for key, value in values.items(): running[key] += reduced(value, world)
        count += 1
        if step == 1 or step % args.log_every == 0:
            elapsed = time.monotonic() - last
            avg = {key: value / count for key, value in running.items()}
            if main_rank:
                print(f"[training-loss] step={step}/{args.steps} total={avg['total']:.6f} repr={avg['repr']:.6f} grid_kd={avg['grid']:.6f}", flush=True)
                print(f"[train-state] step={step}/{args.steps} grid_step={grid_step + 1}/{num_grid} lambda_grid={avg['lambda']:.5f} condition_grad_cos={avg['agree']:.4f} sec_grad={avg['grad']:.4f} lr={scheduler.get_last_lr()[0]:.3e} step_time={elapsed/count:.2f}s peak_cuda_rank0={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB", flush=True)
            running = {key: 0.0 for key in running}; count = 0; last = time.monotonic()
        if main_rank and (step % args.save_every == 0 or step == args.steps):
            torch.save({"format_version": 1, "model_name": "SEC284-L-inference-grid-KD", "step": step,
                        "sec284": unwrap(sec).state_dict(), "config": asdict(config),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "grid_lambda": grid_lambda, "args": vars(args)}, output / f"step-{step:06d}.pt")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
