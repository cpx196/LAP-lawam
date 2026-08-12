#!/usr/bin/env python3
"""Adapt the Action Expert to a frozen SEC284 using inference-grid KD only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.lap_stage1 import LAP60M
from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.train_lap8_phase1 import grad_norm, load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, load_lawm_decoder
from tools.train_sec284_full_inference_grid_kd_ddp import FullGridDataset, setup


class ExpertGridKD(nn.Module):
    """DDP-visible wrapper around the trainable Action Expert."""

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        self.expert = expert

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1: torch.Tensor,
        condition: torch.Tensor,
        forced_x: torch.Tensor,
        grid_step: int,
        flow_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_t.shape[0]
        _, trace = self.expert.sample_actions_cfg_train(
            h_t=h_t,
            h_t1_star=h_t1,
            h_vlm=None,
            h_lap=condition,
            state=torch.zeros(batch_size, 32, device=h_t.device),
            state_mask=torch.zeros(batch_size, 32, device=h_t.device, dtype=torch.bool),
            action_hz=torch.full((batch_size,), 30.0, device=h_t.device),
            embodiment_id=torch.ones(batch_size, device=h_t.device, dtype=torch.long),
            attention_mask=torch.ones(
                condition.shape[:2], device=h_t.device, dtype=torch.bool
            ),
            cfg_scale=float(self.expert.config.cfg_guidance_scale),
            num_inference_steps=1,
            flow_total_steps=flow_steps,
            flow_step_offset=grid_step,
            return_padded=False,
            return_trace=True,
            forced_x_inputs=forced_x,
            gradient_trace_step=0,
        )
        return trace["velocities"][0].float(), trace["time_valid"]


def reduced(value: torch.Tensor, world: int) -> float:
    value = value.detach().float().clone()
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return float(value.cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-cache", default="cache/sec284_task14_inference_grid/train")
    p.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    p.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    p.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    p.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    p.add_argument(
        "--stage1-checkpoint",
        default="outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt",
    )
    p.add_argument(
        "--sec284-checkpoint",
        default="outputs/sec284_output_kd_primary_2000step/step-002000.pt",
    )
    p.add_argument("--expert-checkpoint", default="cache/lap8_phase1_official_action_expert.pt")
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--output-dir", default="outputs/sec284_expert_grid_kd_500step")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Absolute step offset for continuation runs; model weights are loaded via --expert-checkpoint.",
    )
    p.add_argument("--batch-size", type=int, default=8, help="local/per-GPU batch")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=284)
    p.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = setup()
    main_rank = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    dataset = FullGridDataset(args, verbose=main_rank)
    sampler = DistributedSampler(
        dataset, world, rank, shuffle=True, seed=args.seed, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(loader)

    sec_obj = torch.load(args.sec284_checkpoint, map_location="cpu", weights_only=False)
    sec_config = SEC284Config(**sec_obj["config"])
    sec = SEC284L(sec_config)
    sec.load_state_dict(sec_obj["sec284"], strict=True)
    sec = sec.to(device, torch.float32).eval()

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap6.load_state_dict(stage1["lap"], strict=True)
    lap6 = lap6.to(device, torch.float32).eval()
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml)
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    lawm = lawm.to(device, torch.float32).eval()

    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32)
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    for parameter in expert.enc_vlm.parameters():
        parameter.requires_grad_(False)
    model: nn.Module = ExpertGridKD(expert).to(device)
    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            # The h_lap-only CFG route intentionally leaves the Expert's
            # unconditional learned embedding unused on some flow steps.
            find_unused_parameters=True,
        )

    for module in (sec, lap6, lawm):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    output = Path(args.output_dir)
    if main_rank:
        output.mkdir(parents=True, exist_ok=True)
        print(
            f"[data] full_samples={len(dataset):,} world={world} "
            f"local_batch={args.batch_size} global_batch={world * args.batch_size}",
            flush=True,
        )
        print(
            f"[model] trainable=ActionExpert({sum(p.numel() for p in trainable):,}) "
            "frozen=SEC284,LAP6,LaWM,enc_vlm",
            flush=True,
        )
        print(
            f"[objective] grid_kd_only uniform_action_weights=True "
            f"balanced_grid_cycle=True lr={args.lr:.3e}",
            flush=True,
        )

    model.train()
    running_loss = 0.0
    running_grad = 0.0
    count = 0
    epoch = 0
    last_log = time.monotonic()
    total_steps = args.start_step + args.steps
    for local_step in range(1, args.steps + 1):
        step = args.start_step + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)

        visual = batch["visual_tokens"].to(device, torch.float32, non_blocking=True)
        state_t = batch["state_t"].to(device, torch.float32, non_blocking=True)
        forced_x_all = batch["teacher_x_inputs"].to(
            device, torch.float32, non_blocking=True
        ).permute(1, 0, 2, 3).contiguous()
        teacher_velocity_all = batch["teacher_velocities"].to(
            device, torch.float32, non_blocking=True
        )
        flow_steps = int(forced_x_all.shape[0])
        grid_step = (step - 1) % flow_steps
        forced_x = forced_x_all[grid_step : grid_step + 1]
        teacher_velocity = teacher_velocity_all[:, grid_step]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_lap = lap6(visual, state_t)["z_lap"]
            h_t1 = lawm(visual[:, 0], z_lap).squeeze(1)
            condition = sec(
                visual,
                torch.ones(visual.shape[:2], device=device, dtype=torch.bool),
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_velocity, time_valid = model(
                visual[:, 0], h_t1, condition, forced_x, grid_step, flow_steps
            )
        valid = time_valid.float().unsqueeze(-1).expand_as(predicted_velocity)
        grid_loss = (
            (predicted_velocity - teacher_velocity).square() * valid
        ).sum() / valid.sum().clamp_min(1.0)
        grid_loss.backward()
        raw_grad = grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        running_loss += reduced(grid_loss, world)
        running_grad += raw_grad
        count += 1
        if local_step == 1 or step % args.log_every == 0:
            now = time.monotonic()
            if main_rank:
                step_time = (now - last_log) / count
                eta_min = step_time * (total_steps - step) / 60.0
                print(
                f"[training-loss] step={step}/{total_steps} "
                    f"grid_kd={running_loss / count:.6f}",
                    flush=True,
                )
                print(
                    f"[train-state] step={step}/{total_steps} grid_step={grid_step + 1}/{flow_steps} "
                    f"expert_grad={running_grad / count:.4f} lr={args.lr:.3e} "
                    f"step_time={step_time:.2f}s eta={eta_min:.1f}min "
                    f"peak_cuda_rank0={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB",
                    flush=True,
                )
            running_loss = 0.0
            running_grad = 0.0
            count = 0
            last_log = now

        if main_rank and (step % args.save_every == 0 or step == total_steps):
            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "format_version": 1,
                    "model_name": "SEC284-frozen-Expert-grid-KD",
                    "step": step,
                    "expert": raw_model.expert.state_dict(),
                    "sec284_checkpoint": str(Path(args.sec284_checkpoint).resolve()),
                    "expert_checkpoint": str(Path(args.expert_checkpoint).resolve()),
                    "args": vars(args),
                },
                output / f"step-{step:06d}.pt",
            )
            print(f"[checkpoint] saved step={step}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
