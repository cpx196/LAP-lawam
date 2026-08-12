#!/usr/bin/env python3
"""Downstream-aware SEC284 training through a frozen LAP6/LaWM/Action Expert.

Only SEC284 is optimized.  Cached VLM conditions form the representation
teacher and, with identical flow noise/time, the frozen Expert's velocity
teacher.  Actions supervise flow matching but are never SEC284 inputs.
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
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.sec284_data import SEC284LossWeights, sec284_distillation_loss
from tools.train_lap10_alignment import LAP10Dataset
from tools.train_lap8_phase1 import grad_norm, load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, load_lawm_decoder


class ShardAwareDistributedBatchSampler(Sampler[list[int]]):
    """Keep each local batch shard-local while balancing complete DDP steps."""

    def __init__(self, dataset: LAP10Dataset, batch_size: int, world: int, rank: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.world = int(world)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.global_batches = (len(dataset) // self.batch_size) // self.world * self.world

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.global_batches // self.world

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        lengths = [int(x) for x in self.dataset.teacher.lengths]
        offsets = [int(x) for x in self.dataset.teacher.offsets]
        shard_order = torch.randperm(len(lengths), generator=generator).tolist()
        batches: list[list[int]] = []
        carry: list[int] = []
        for shard in shard_order:
            local = torch.randperm(lengths[shard], generator=generator).tolist()
            indices = carry + [offsets[shard] + item for item in local]
            complete = len(indices) // self.batch_size
            batches.extend(
                indices[start * self.batch_size : (start + 1) * self.batch_size]
                for start in range(complete)
            )
            carry = indices[complete * self.batch_size :]
        batches = batches[: self.global_batches]
        yield from batches[self.rank :: self.world]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local_rank, torch.device("cuda", local_rank)


def unwrap(model: nn.Module) -> SEC284L:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


def cosine_schedule(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = min(1.0, float(step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def dynamic_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve teacher cross-sample directions and standard deviation."""
    valid = mask.to(device=student.device, dtype=torch.float32).unsqueeze(-1)
    student = student.float() * valid
    teacher = teacher.float() * valid
    student_centered = student - student.mean(dim=0, keepdim=True)
    teacher_centered = teacher - teacher.mean(dim=0, keepdim=True)
    cosine = 1.0 - F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=-1, eps=1e-8
    ).mean()
    student_std = torch.sqrt(student_centered.square().sum() / valid.sum().clamp_min(1.0) + 1e-8)
    teacher_std = torch.sqrt(teacher_centered.square().sum() / valid.sum().clamp_min(1.0) + 1e-8)
    std_log_ratio = torch.log((student_std + 1e-6) / (teacher_std + 1e-6)).square()
    return cosine + std_log_ratio, cosine, student_std / teacher_std.clamp_min(1e-8)


def global_detached_mean(value: torch.Tensor, world: int) -> float:
    reduced = value.detach().float().clone()
    if world > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world
    return float(reduced.cpu())


def condition_gradient_stats(
    repr_gradient: torch.Tensor, behavior_gradient: torch.Tensor, world: int
) -> tuple[float, float, float]:
    """Return synchronized condition-gradient norms and cosine across ranks."""
    repr_fp32 = repr_gradient.detach().float()
    behavior_fp32 = behavior_gradient.detach().float()
    values = torch.stack(
        [
            repr_fp32.square().sum(),
            behavior_fp32.square().sum(),
            (repr_fp32 * behavior_fp32).sum(),
        ]
    )
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    repr_norm = values[0].sqrt()
    behavior_norm = values[1].sqrt()
    cosine = values[2] / (repr_norm * behavior_norm).clamp_min(1e-12)
    return float(repr_norm.cpu()), float(behavior_norm.cpu()), float(cosine.cpu())


def expert_inputs(batch_size: int, condition_mask: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.zeros(batch_size, 32, device=device),
        "action_hz": torch.full((batch_size,), 30.0, device=device),
        "embodiment_id": torch.ones(batch_size, device=device, dtype=torch.long),
        "state_mask": torch.zeros(batch_size, 32, device=device, dtype=torch.bool),
        "attention_mask": condition_mask,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    parser.add_argument("--sec284-checkpoint", default="outputs/sec284_l_bs32_3000step/step-003000.pt")
    parser.add_argument("--stage1-checkpoint", default="outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt")
    parser.add_argument("--expert-checkpoint", default="cache/lap8_phase1_official_action_expert.pt")
    parser.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    parser.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    parser.add_argument("--output-dir", default="outputs/sec284_frozen_expert_2000step")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32, help="local batch size per GPU")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dynamic-weight", type=float, default=0.1)
    parser.add_argument("--behavior-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--behavior-lambda-ema", type=float, default=0.9)
    parser.add_argument("--behavior-warmup-steps", type=int, default=100)
    parser.add_argument("--behavior-lambda-min", type=float, default=1e-3)
    parser.add_argument("--behavior-lambda-max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.log_every) < 1:
        raise ValueError("steps, batch-size, save-every and log-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup_distributed()
    main_rank = rank == 0
    seed_everything(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    dataset = LAP10Dataset(args, verbose=main_rank)
    sampler = ShardAwareDistributedBatchSampler(dataset, args.batch_size, world, rank, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    sec_obj = torch.load(args.sec284_checkpoint, map_location="cpu", weights_only=True)
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
    trainable = list(sec.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, args.warmup_steps, args.steps, args.min_lr / args.lr),
    )
    stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
    position_variance = stats["position_variance"].to(device, torch.float32)
    repr_weights = SEC284LossWeights()
    behavior_lambda = 0.0
    global_step = 0

    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap(sec).load_state_dict(resume["sec284"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        behavior_lambda = float(resume["behavior_lambda"])
        global_step = int(resume["step"])

    output = Path(args.output_dir)
    if main_rank:
        output.mkdir(parents=True, exist_ok=True)
        print(
            f"[model] SEC284={unwrap(sec).parameter_count:,} trainable={sum(p.numel() for p in trainable):,} "
            f"LAP6={count_parameters(lap6):,} LaWM={count_parameters(lawm):,} "
            f"Expert={count_parameters(expert):,}; frozen=LAP6,LaWM,Expert",
            flush=True,
        )
        print(
            f"[train] world={world} local_batch={args.batch_size} global_batch={world * args.batch_size} "
            f"steps={args.steps} precision=BF16-autocast samples={len(dataset):,}",
            flush=True,
        )

    def payload() -> dict:
        return {
            "format_version": 1,
            "model_name": "SEC284-L-frozen-expert",
            "step": global_step,
            "sec284": unwrap(sec).state_dict(),
            "config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "behavior_lambda": behavior_lambda,
            "args": vars(args),
        }

    sec.train()
    epoch = 0
    iterator = iter(loader)
    started = time.monotonic()
    last_log_time = started
    last_log_step = global_step
    running = {
        key: 0.0
        for key in (
            "total", "repr", "raw", "white", "cos", "dynamic", "dynamic_cos",
            "std_ratio", "behavior", "flow", "sec_grad", "condition_grad_repr",
            "condition_grad_behavior", "condition_grad_cos", "behavior_lambda",
        )
    }
    running_count = 0
    while global_step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)

        vision_t = batch["vision_t"].to(device, torch.float32, non_blocking=True)
        visual = torch.stack(
            [
                vision_t,
                batch["vision_left_t"].to(device, torch.float32, non_blocking=True),
                batch["vision_right_t"].to(device, torch.float32, non_blocking=True),
            ],
            dim=1,
        )
        state = batch["state_t"].to(device, torch.float32, non_blocking=True)
        actions = batch["actions"].to(device, torch.float32, non_blocking=True)
        actions_mask = batch["actions_mask"].to(device, torch.bool, non_blocking=True)
        teacher = batch["teacher_condition"].to(device, torch.float32, non_blocking=True)
        teacher_mask = batch.get("teacher_mask")
        if teacher_mask is None:
            teacher_mask = torch.ones(teacher.shape[:2], device=device, dtype=torch.bool)
        else:
            teacher_mask = teacher_mask.to(device, torch.bool, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_lap = lap6(visual, state)["z_lap"]
            h_t1 = lawm(vision_t, z_lap).squeeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student = sec(visual, torch.ones(visual.shape[:2], device=device, dtype=torch.bool))
        repr_parts = sec284_distillation_loss(student, teacher, teacher_mask, position_variance, repr_weights)
        dyn, dyn_cos, std_ratio = dynamic_loss(student, teacher, teacher_mask)
        repr_loss = repr_parts["total"] + args.dynamic_weight * dyn

        common = expert_inputs(actions.shape[0], teacher_mask, device)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_details = expert(
                h_t=vision_t, h_t1_star=h_t1, h_vlm=None, h_lap=teacher,
                actions=actions, actions_mask=actions_mask, return_training_details=True, **common,
            )
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_details = expert(
                h_t=vision_t, h_t1_star=h_t1, h_vlm=None, h_lap=student,
                actions=actions, actions_mask=actions_mask, return_training_details=True, **common,
            )
        valid = student_details["valid_weights"].float()
        behavior = (
            (student_details["pred_velocity"].float() - teacher_details["pred_velocity"].float()).square() * valid
        ).sum() / valid.sum().clamp_min(1.0)
        flow = student_details["loss"].float()

        repr_condition_gradient = torch.autograd.grad(
            repr_loss, student, retain_graph=True, create_graph=False
        )[0]
        behavior_condition_gradient = torch.autograd.grad(
            behavior, student, retain_graph=True, create_graph=False
        )[0]
        condition_grad_repr, condition_grad_behavior, condition_grad_cos = condition_gradient_stats(
            repr_condition_gradient, behavior_condition_gradient, world
        )
        lambda_target = args.behavior_gradient_ratio * condition_grad_repr / max(
            condition_grad_behavior, 1e-12
        )
        lambda_target = min(
            args.behavior_lambda_max, max(args.behavior_lambda_min, lambda_target)
        )
        behavior_lambda = (
            lambda_target
            if behavior_lambda == 0.0
            else args.behavior_lambda_ema * behavior_lambda
            + (1.0 - args.behavior_lambda_ema) * lambda_target
        )
        behavior_warmup = min(
            1.0, float(global_step + 1) / max(1, args.behavior_warmup_steps)
        )
        effective_behavior_lambda = behavior_lambda * behavior_warmup
        total = repr_loss + effective_behavior_lambda * behavior
        total.backward()
        raw_grad = grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        values = {
            "total": total.detach(), "repr": repr_loss.detach(), "raw": repr_parts["raw_mse"].detach(),
            "white": repr_parts["whitened_mse"].detach(), "cos": repr_parts["cosine"].detach(),
            "dynamic": dyn.detach(), "dynamic_cos": dyn_cos.detach(), "std_ratio": std_ratio.detach(),
            "behavior": behavior.detach(), "flow": flow.detach(),
            "sec_grad": torch.tensor(raw_grad, device=device),
            "condition_grad_repr": torch.tensor(condition_grad_repr, device=device),
            "condition_grad_behavior": torch.tensor(condition_grad_behavior, device=device),
            "condition_grad_cos": torch.tensor(condition_grad_cos, device=device),
            "behavior_lambda": torch.tensor(effective_behavior_lambda, device=device),
        }
        for key, value in values.items():
            running[key] += global_detached_mean(value, world)
        running_count += 1

        if global_step == 1 or global_step % args.log_every == 0:
            now = time.monotonic()
            averaged = {key: value / running_count for key, value in running.items()}
            step_time = (now - last_log_time) / max(1, global_step - last_log_step)
            eta = step_time * (args.steps - global_step) / 3600.0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            if main_rank:
                print(
                    f"[training-loss] step={global_step}/{args.steps} "
                    f"total={averaged['total']:.6f} repr={averaged['repr']:.6f} "
                    f"behavior_kd={averaged['behavior']:.6f} flow_monitor={averaged['flow']:.6f} "
                    f"raw_mse={averaged['raw']:.6f} white_mse={averaged['white']:.6f} "
                    f"cosine={averaged['cos']:.6f} dynamic={averaged['dynamic']:.6f} "
                    f"std_ratio={averaged['std_ratio']:.6f}",
                    flush=True,
                )
                print(
                    f"[train-state] step={global_step}/{args.steps} "
                    f"lambda_behavior={averaged['behavior_lambda']:.6f} "
                    f"condition_grad_repr={averaged['condition_grad_repr']:.6f} "
                    f"condition_grad_behavior={averaged['condition_grad_behavior']:.6f} "
                    f"condition_grad_cos={averaged['condition_grad_cos']:.6f} "
                    f"sec_grad={averaged['sec_grad']:.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"step_time={step_time:.2f}s eta={eta:.2f}h "
                    f"peak_cuda_rank0={peak:.2f}GiB",
                    flush=True,
                )
            running = {key: 0.0 for key in running}
            running_count = 0
            last_log_time, last_log_step = now, global_step

        if global_step % args.save_every == 0 or global_step == args.steps:
            if main_rank:
                torch.save(payload(), output / f"step-{global_step:06d}.pt")
                torch.save(payload(), output / "last.pt")
                print(f"[checkpoint] saved step={global_step}", flush=True)
            if world > 1:
                dist.barrier()

    if main_rank:
        print(f"[train] complete elapsed={(time.monotonic() - started) / 3600.0:.2f}h", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
