#!/usr/bin/env python3
"""Jointly fine-tune LAP10V3 and the RoboTwin Action Expert in FP32 DDP.

The frozen path is DINO/LAP6 -> LaWM.  The trainable path is LAP7-LAP10
(``LAP10V3``) -> Action Expert.  The released VLM is not loaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
from tools.train_lap10_alignment import (
    DEFAULT_EXPERT,
    DEFAULT_LAP8,
    DEFAULT_STAGE1,
    DEFAULT_TEACHER_CACHE,
    LAP10Dataset,
    lr_lambda,
)
from tools.train_lap8_phase1 import grad_norm, load_action_expert
from tools.train_lap_stage1 import (
    DEFAULT_LAM_CKPT,
    DEFAULT_LAM_YAML,
    load_lawm_decoder,
)


class JointLAPExpert(nn.Module):
    """DDP-visible trainable graph for the LAP and Action Expert branches."""

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        lap_out = self.lap(visual, state)
        bsz = visual.shape[0]
        cond = lap_out["cond_lap"]
        flow = self.expert(
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
        )
        return lap_out, flow


def _load_lap10v3(args: argparse.Namespace) -> LAP10V3:
    lap8_obj = torch.load(args.lap8_checkpoint, map_location="cpu", weights_only=True)
    lap6_state = {
        key.removeprefix("lap6."): value
        for key, value in lap8_obj["lap8"].items()
        if key.startswith("lap6.")
    }
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap6.load_state_dict(lap6_state, strict=True)
    stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
    lap = LAP10V3(
        lap6,
        stats["position_mean"],
        output_tokens=args.output_tokens,
        view_dropout=args.view_dropout,
    )
    obj = torch.load(args.lap10v3_checkpoint, map_location="cpu", weights_only=True)
    state = obj.get("lap10v3")
    if state is None:
        raise KeyError(f"{args.lap10v3_checkpoint} does not contain `lap10v3`")
    lap.load_state_dict(state, strict=True)
    return lap


def _load_lawm(args: argparse.Namespace, stage1: dict, device: torch.device) -> nn.Module:
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml).to(device, torch.float32).eval()
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    for p in lawm.parameters():
        p.requires_grad_(False)
    return lawm


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Joint LAP/Expert training requires CUDA")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    dataset = LAP10Dataset(args, verbose=(rank == 0))
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    lap = _load_lap10v3(args)
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lawm = _load_lawm(args, stage1, device)
    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32)

    # There is no h_vlm input in this experiment; its 2048->768 projection
    # would otherwise be an unused DDP parameter.  The whole Action Expert
    # that consumes h_lap remains trainable.
    for p in expert.parameters():
        p.requires_grad_(True)
    for p in expert.enc_vlm.parameters():
        p.requires_grad_(False)

    joint = JointLAPExpert(lap.to(device, torch.float32), expert)
    joint = DDP(
        joint,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    lap_params = [p for p in joint.module.lap.parameters() if p.requires_grad]
    expert_params = [p for p in joint.module.expert.parameters() if p.requires_grad]
    trainable = lap_params + expert_params
    optimizer = torch.optim.AdamW(
        [
            {"params": lap_params, "lr": args.lap_lr},
            {"params": expert_params, "lr": args.expert_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )

    if rank == 0:
        print(
            f"[model] LAP10V3={count_parameters(joint.module.lap):,} "
            f"trainable={sum(p.numel() for p in lap_params):,} "
            f"ActionExpert={count_parameters(joint.module.expert):,} "
            f"trainable={sum(p.numel() for p in expert_params):,} "
            f"LaWM={count_parameters(lawm):,} frozen VLM=not-loaded",
            flush=True,
        )
        print(
            f"[train] world={world} precision=FP32 samples={len(dataset):,} "
            f"effective_batch={world * args.batch_size * args.grad_accumulation} "
            f"lap_lr={args.lap_lr:.2e} expert_lr={args.expert_lr:.2e}",
            flush=True,
        )

    joint.train()
    # LAP10V3.train() keeps LAP6 deterministic; enc_vlm is frozen and unused.
    iterator = iter(loader)
    epoch = 0
    sampler.set_epoch(epoch)
    last_t = time.perf_counter()
    last_step = 0
    keys = ("loss", "flow", "align_mse", "align_cos", "structure", "pred_rms", "teacher_rms")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        metrics = {k: 0.0 for k in keys}
        for micro in range(args.grad_accumulation):
            if micro < args.grad_accumulation - 1:
                joint.require_backward_grad_sync = False
            else:
                joint.require_backward_grad_sync = True
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            main = batch["vision_t"].to(device, torch.float32, non_blocking=True)
            visual = torch.stack(
                [
                    main,
                    batch["vision_left_t"].to(device, torch.float32, non_blocking=True),
                    batch["vision_right_t"].to(device, torch.float32, non_blocking=True),
                ],
                dim=1,
            )
            state = batch["state_t"].to(device, torch.float32, non_blocking=True)
            actions = batch["actions"].to(device, torch.float32, non_blocking=True)
            actions_mask = batch["actions_mask"].to(device, torch.bool, non_blocking=True)
            teacher = batch["teacher_condition"].to(device, torch.float32, non_blocking=True)

            # The LaWM branch is frozen.  Keeping this outside the DDP graph
            # also prevents its 230M decoder from accumulating gradients.
            with torch.no_grad():
                lap6 = joint.module.lap.lap6
                z_lap = lap6(visual, state)["z_lap"]
                h_t1 = lawm(main, z_lap).squeeze(1)

            lap_out, flow = joint(
                visual, state, main, h_t1.detach(), actions, actions_mask
            )
            pred = lap_out["cond_lap"]
            if pred.shape != teacher.shape:
                raise RuntimeError(f"condition mismatch {pred.shape} vs {teacher.shape}")
            align_mse = F.mse_loss(pred, teacher)
            align_cos = 1.0 - F.cosine_similarity(pred, teacher, dim=-1).mean()
            pred_n = F.normalize(pred, dim=-1)
            teacher_n = F.normalize(teacher, dim=-1)
            structure = F.mse_loss(
                pred_n @ pred_n.transpose(1, 2),
                teacher_n @ teacher_n.transpose(1, 2),
            )
            loss = (
                flow
                + args.align_mse_weight * align_mse
                + args.align_cos_weight * align_cos
                + args.structure_weight * structure
            )
            (loss / args.grad_accumulation).backward()
            values = (loss, flow, align_mse, align_cos, structure,
                      pred.square().mean().sqrt(), teacher.square().mean().sqrt())
            for key, value in zip(keys, values):
                metrics[key] += float(value.detach()) / args.grad_accumulation

        lap_grad = grad_norm(lap_params)
        expert_grad = grad_norm(expert_params)
        total_grad = grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            packed = torch.tensor(
                [metrics[k] for k in keys] + [lap_grad, expert_grad, total_grad],
                device=device,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.AVG)
            if rank == 0:
                now = time.perf_counter()
                step_time = (now - last_t) / max(1, step - last_step)
                eta = step_time * (args.steps - step) / 3600.0
                text = " ".join(
                    f"{k}={v:.6f}" for k, v in zip(keys, packed[:len(keys)].tolist())
                )
                print(
                    f"[train] step={step}/{args.steps} {text} "
                    f"lap_grad={packed[-3]:.4f} expert_grad={packed[-2]:.4f} "
                    f"total_grad={packed[-1]:.4f} "
                    f"lap_lr={scheduler.get_last_lr()[0]:.3e} "
                    f"expert_lr={scheduler.get_last_lr()[1]:.3e} "
                    f"step_time={step_time:.2f}s eta={eta:.2f}h "
                    f"peak_cuda={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB",
                    flush=True,
                )
                last_t, last_step = now, step

        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "lap10v3": joint.module.lap.state_dict(),
                        "expert": joint.module.expert.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                        "lap10v3_checkpoint": str(args.lap10v3_checkpoint),
                        "expert_checkpoint": str(args.expert_checkpoint),
                        "stage1_checkpoint": str(args.stage1_checkpoint),
                    },
                    output / f"lap10v3_expert_step{step:07d}.pt",
                )
                print(f"[train] checkpoint saved at step {step}", flush=True)
            dist.barrier()

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    p.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    p.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    p.add_argument("--teacher-cache", default=str(DEFAULT_TEACHER_CACHE))
    p.add_argument("--lap10v3-checkpoint", default="outputs/lap10v3_task14_1000step/lap10v3_step0001000.pt")
    p.add_argument("--lap8-checkpoint", default=str(DEFAULT_LAP8))
    p.add_argument("--stage1-checkpoint", default=str(DEFAULT_STAGE1))
    p.add_argument("--expert-checkpoint", default=str(DEFAULT_EXPERT))
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--teacher-stats", default="cache/lap10_task14_vlm_teacher_8192/position_stats.pt")
    p.add_argument("--output-tokens", type=int, default=284)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=2)
    p.add_argument("--lap-lr", type=float, default=2e-5)
    p.add_argument("--expert-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--view-dropout", type=float, default=0.1)
    p.add_argument("--align-mse-weight", type=float, default=0.1)
    p.add_argument("--align-cos-weight", type=float, default=0.02)
    p.add_argument("--structure-weight", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="outputs/lap10v3_expert_joint_task14_2000step")
    return p.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main(parse_args())
