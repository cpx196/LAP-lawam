#!/usr/bin/env python3
"""Materialize Frozen-Expert 10-step inference grids for the full SEC284 set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.lap_stage1 import LAP60M
from tools.train_lap10_alignment import LAP10Dataset
from tools.train_lap8_phase1 import load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, load_lawm_decoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    p.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    p.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    p.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    p.add_argument("--stage1-checkpoint", default="outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt")
    p.add_argument("--expert-checkpoint", default="cache/lap8_phase1_official_action_expert.pt")
    p.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    p.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    p.add_argument("--output-dir", default="cache/sec284_task14_inference_grid/train")
    p.add_argument("--batch-size", type=int, default=32, help="local/per-GPU batch size")
    p.add_argument("--cache-shard-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=284)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="debug smoke limit; 0 means full split")
    p.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    # LAP10Dataset is used only as the already-aligned full train view.  The
    # action tensors are not a training target here; they provide state_t and
    # the exact feature/teacher-cache ordering contract.
    dataset = LAP10Dataset(args, verbose=rank == 0)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_indices: list[torch.Tensor] = []
    pending_x: list[torch.Tensor] = []
    pending_v: list[torch.Tensor] = []
    written = 0
    shard_no = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal pending_indices, pending_x, pending_v, written, shard_no
        if not pending_indices:
            return
        indices = torch.cat(pending_indices, dim=0)
        payload = {
            "sample_index": indices.to(torch.int64),
            "teacher_x_inputs": torch.cat(pending_x, dim=0).to(torch.float16),
            "teacher_velocities": torch.cat(pending_v, dim=0).to(torch.float16),
            "flow_steps": int(pending_x[0].shape[1]),
            "action_horizon": int(pending_x[0].shape[2]),
            "action_dim": int(pending_x[0].shape[3]),
        }
        path = output_dir / f"rank-{rank:02d}-shard-{shard_no:05d}.pt"
        if path.exists() and not args.overwrite:
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if not torch.equal(existing["sample_index"], payload["sample_index"]):
                raise RuntimeError(f"existing shard has different sample_index: {path}")
        else:
            torch.save(payload, path)
        written += int(indices.numel())
        shard_no += 1
        pending_indices, pending_x, pending_v = [], [], []

    with torch.inference_mode():
        for batch_start, batch in enumerate(loader):
            if args.max_batches > 0 and batch_start >= args.max_batches:
                break
            vision_t = batch["vision_t"].to(device, torch.float32, non_blocking=True)
            visual = torch.stack([
                vision_t,
                batch["vision_left_t"].to(device, torch.float32, non_blocking=True),
                batch["vision_right_t"].to(device, torch.float32, non_blocking=True),
            ], dim=1)
            state = batch["state_t"].to(device, torch.float32, non_blocking=True)
            teacher = batch["teacher_condition"].to(device, torch.float32, non_blocking=True)
            batch_size = vision_t.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_lap = lap6(visual, state)["z_lap"]
                h_t1 = lawm(vision_t, z_lap).squeeze(1)
                _, trace = expert.sample_actions_cfg(
                    h_t=vision_t,
                    h_t1_star=h_t1,
                    h_vlm=None,
                    h_lap=teacher,
                    state=torch.zeros(batch_size, 32, device=device),
                    state_mask=torch.zeros(batch_size, 32, device=device, dtype=torch.bool),
                    action_hz=torch.full((batch_size,), 30.0, device=device),
                    embodiment_id=torch.ones(batch_size, device=device, dtype=torch.long),
                    attention_mask=torch.ones(batch_size, teacher.shape[1], device=device, dtype=torch.bool),
                    cfg_scale=float(expert.config.cfg_guidance_scale),
                    num_inference_steps=int(expert.config.num_inference_steps),
                    return_padded=False,
                    return_trace=True,
                )
            # DistributedSampler with shuffle=False assigns the deterministic
            # interleaved sequence rank, rank+world, ...; this is deliberately
            # reconstructed instead of relying on private sampler attributes.
            rank_indices = list(range(rank, len(dataset), world))
            offset = batch_start * args.batch_size
            global_indices = torch.tensor(
                rank_indices[offset : offset + batch_size], dtype=torch.int64
            )
            pending_indices.append(global_indices)
            # Expert traces are [flow_step,B,horizon,dim]; cache items are
            # [B,flow_step,horizon,dim] so DataLoader batches naturally.
            pending_x.append(trace["x_inputs"].permute(1, 0, 2, 3).detach().cpu())
            pending_v.append(trace["velocities"].permute(1, 0, 2, 3).detach().cpu())
            if sum(int(x.shape[0]) for x in pending_indices) >= args.cache_shard_size:
                flush()
            if rank == 0 and (batch_start == 0 or (batch_start + 1) % 10 == 0):
                elapsed = time.monotonic() - started
                done = written + sum(int(x.shape[0]) for x in pending_indices)
                rate = done / max(elapsed, 1e-6)
                eta = (len(dataset) - done * world) / max(rate * world, 1e-6) / 60.0
                print(f"[grid-cache] rank0 batch={batch_start + 1} local_done={done} rate={rate:.2f}/s eta_global={eta:.1f}min", flush=True)
    flush()
    metadata = {
        "split": "train", "samples": len(dataset), "world": world,
        "local_batch_size": args.batch_size, "flow_steps": int(expert.config.num_inference_steps),
        "action_horizon": int(expert.action_horizon), "action_dim": int(expert.config.action_dim),
        "fixed_instruction": "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad.",
    }
    if rank == 0:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[grid-cache] complete: {output_dir} samples={len(dataset)}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
