#!/usr/bin/env python3
"""Four-GPU FP32 DDP training for the unified LAP10V2 Expert conditioner."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.lap_stage2 import LAP8, LAP10V2, LAP10V3
from tools.train_lap10_alignment import (
    DEFAULT_EXPERT, DEFAULT_LAP8, DEFAULT_STAGE1, DEFAULT_TEACHER_CACHE,
    LAP10Dataset, lr_lambda,
)
from tools.train_lap8_phase1 import grad_norm, load_action_expert
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, diversity_loss, load_lawm_decoder


def main(args: argparse.Namespace) -> None:
    # Importing the project model stack may initialize this through Accelerate
    # when torchrun variables are already present.
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    dataset = LAP10Dataset(args, verbose=(rank == 0))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0,
                        pin_memory=True, drop_last=True)

    lap8_state = torch.load(args.lap8_checkpoint, map_location="cpu", weights_only=True)["lap8"]
    lap6 = LAP60M(num_views=3, view_dropout=0.2)
    if args.model_version == "v3":
        # V3 loads only the six-block Stage-1 trunk.  Its LAP7-LAP10 branch is
        # always fresh and never reads LAP8/LAP10 post-trunk weights.
        lap6_state = {
            key.removeprefix("lap6."): value for key, value in lap8_state.items()
            if key.startswith("lap6.")
        }
        lap6.load_state_dict(lap6_state, strict=True)
        stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
        model_core = LAP10V3(
            lap6, stats["position_mean"], output_tokens=args.output_tokens,
            view_dropout=0.0,
        )
    else:
        lap8 = LAP8(lap6, view_dropout=args.view_dropout)
        if args.init_mode == "scratch":
            # Reuse only the validated Stage-1 trunk.
            lap6_state = {
                key.removeprefix("lap6."): value
                for key, value in lap8_state.items()
                if key.startswith("lap6.")
            }
            lap8.lap6.load_state_dict(lap6_state, strict=True)
        else:
            lap8.load_state_dict(lap8_state, strict=True)
        model_core = LAP10V2(lap8, output_tokens=args.output_tokens)
        if args.init_mode == "legacy":
            old_state = torch.load(args.lap10_checkpoint, map_location="cpu", weights_only=True)["lap10"]
            model_core.load_from_lap10_state(old_state)
    model = model_core
    model = DDP(model.to(device, torch.float32), device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=False)

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml).to(device, torch.float32).eval()
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32).eval()
    for frozen in (lawm, expert):
        for p in frozen.parameters(): p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, args.warmup_steps, args.steps))
    if rank == 0:
        print(f"[model] LAP{args.model_version.upper()}={count_parameters(model.module):,} trainable={sum(p.numel() for p in trainable):,} init={args.init_mode} "
              f"LaWM={count_parameters(lawm):,} Expert={count_parameters(expert):,}", flush=True)
        print(f"[train] world={world} FP32 samples={len(dataset):,} effective_batch={world*args.batch_size*args.grad_accumulation}", flush=True)

    model.train(); iterator = iter(loader); epoch = 0; last_t = time.perf_counter(); last_step = 0
    keys = ("loss", "flow", "align_mse", "align_cos", "structure", "pred_rms", "teacher_rms")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True); metrics = {k: 0.0 for k in keys}
        for _ in range(args.grad_accumulation):
            try: batch = next(iterator)
            except StopIteration:
                epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); batch = next(iterator)
            main = batch["vision_t"].to(device, torch.float32, non_blocking=True)
            visual = torch.stack([main, batch["vision_left_t"].to(device, torch.float32, non_blocking=True),
                                  batch["vision_right_t"].to(device, torch.float32, non_blocking=True)], dim=1)
            state = batch["state_t"].to(device, torch.float32, non_blocking=True)
            actions = batch["actions"].to(device, torch.float32, non_blocking=True)
            actions_mask = batch["actions_mask"].to(device, torch.bool, non_blocking=True)
            teacher = batch["teacher_condition"].to(device, torch.float32, non_blocking=True)
            if args.model_version == "v3":
                model.module.view_dropout = 0.0 if step <= args.phase_a_steps else min(
                    args.view_dropout,
                    args.view_dropout * (step - args.phase_a_steps) / max(1, args.flow_ramp_steps),
                )
            out = model(visual, state); pred = out["cond_lap"] if args.model_version == "v3" else out["cond_lap10"]
            if pred.shape != teacher.shape: raise RuntimeError(f"condition mismatch {pred.shape} vs {teacher.shape}")
            mse = F.mse_loss(pred, teacher)
            cos = 1-F.cosine_similarity(pred, teacher, dim=-1).mean()
            if args.model_version == "v3":
                pred_n = F.normalize(pred, dim=-1)
                teacher_n = F.normalize(teacher, dim=-1)
                structure = F.mse_loss(pred_n @ pred_n.transpose(1, 2), teacher_n @ teacher_n.transpose(1, 2))
            else:
                structure = diversity_loss(out["scene_lap8"])
            flow_weight = 1.0
            if args.model_version == "v3":
                flow_weight = 0.0 if step <= args.phase_a_steps else min(
                    1.0, (step - args.phase_a_steps) / max(1, args.flow_ramp_steps)
                )
            flow = torch.zeros((), device=device)
            if flow_weight > 0.0:
                with torch.no_grad(): h_t1 = lawm(main, out["z_lap"]).squeeze(1)
                b = main.shape[0]
                flow = expert(h_t=main, h_t1_star=h_t1, h_vlm=None, h_lap=pred,
                              state=torch.zeros(b,32,device=device), actions=actions,
                              action_hz=torch.full((b,),30.0,device=device), embodiment_id=torch.ones(b,device=device,dtype=torch.long),
                              state_mask=torch.zeros(b,32,device=device,dtype=torch.bool), actions_mask=actions_mask,
                              attention_mask=torch.ones(b,pred.shape[1],device=device,dtype=torch.bool))
            loss = flow_weight*flow + args.align_mse_weight*mse + args.align_cos_weight*cos
            if args.model_version == "v3":
                loss = loss + args.structure_weight * structure
            else:
                loss = loss + args.diversity_weight * structure
            (loss/args.grad_accumulation).backward()
            vals=(loss,flow,mse,cos,structure,pred.square().mean().sqrt(),teacher.square().mean().sqrt())
            for k,v in zip(keys,vals): metrics[k] += float(v.detach())/args.grad_accumulation
        raw_grad = grad_norm(trainable); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); optimizer.step(); scheduler.step()
        if step == 1 or step % args.log_every == 0:
            packed=torch.tensor([metrics[k] for k in keys]+[raw_grad],device=device); dist.all_reduce(packed,op=dist.ReduceOp.AVG)
            if rank == 0:
                now=time.perf_counter(); dt=(now-last_t)/(step-last_step); eta=dt*(args.steps-step)/3600
                text=" ".join(f"{k}={v:.6f}" for k,v in zip(keys,packed.tolist()))
                print(f"[train] step={step}/{args.steps} {text} grad={packed[-1]:.4f} lr={scheduler.get_last_lr()[0]:.3e} step_time={dt:.2f}s eta={eta:.2f}h peak_cuda={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB",flush=True)
                last_t,last_step=now,step
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
                tag = "lap10v3" if args.model_version == "v3" else "lap10v2"
                torch.save({"step":step,tag:model.module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"args":vars(args),"lap8_checkpoint":str(args.lap8_checkpoint),"teacher_stats":str(args.teacher_stats)},out_dir/f"{tag}_step{step:07d}.pt")
                print(f"[train] checkpoint saved at step {step}",flush=True)
            dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-cache",default="cache/lap_stage1_task14"); p.add_argument("--wrist-cache",default="cache/lap_stage1_task14_wrist"); p.add_argument("--action-cache",default="cache/lap8_phase1_task14_actions"); p.add_argument("--teacher-cache",default=str(DEFAULT_TEACHER_CACHE))
    p.add_argument("--stage1-checkpoint",default=str(DEFAULT_STAGE1)); p.add_argument("--lap8-checkpoint",default=str(DEFAULT_LAP8)); p.add_argument("--lap10-checkpoint",default="outputs/lap10_alignment_task14_1000step/lap10_step0001000.pt"); p.add_argument("--init-mode",choices=("scratch","legacy"),default="scratch"); p.add_argument("--expert-checkpoint",default=str(DEFAULT_EXPERT)); p.add_argument("--lam-ckpt",default=str(DEFAULT_LAM_CKPT)); p.add_argument("--lam-yaml",default=str(DEFAULT_LAM_YAML))
    p.add_argument("--model-version",choices=("v2","v3"),default="v2"); p.add_argument("--teacher-stats",default="cache/lap10_task14_vlm_teacher_8192/position_stats.pt"); p.add_argument("--phase-a-steps",type=int,default=300); p.add_argument("--flow-ramp-steps",type=int,default=50); p.add_argument("--structure-weight",type=float,default=.1)
    p.add_argument("--output-dir",default="outputs/lap10v2_unified_task14_1000step"); p.add_argument("--steps",type=int,default=1000); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--grad-accumulation",type=int,default=2); p.add_argument("--output-tokens",type=int,default=284); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--warmup-steps",type=int,default=200); p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--grad-clip",type=float,default=1.); p.add_argument("--view-dropout",type=float,default=.2); p.add_argument("--align-mse-weight",type=float,default=1.); p.add_argument("--align-cos-weight",type=float,default=.1); p.add_argument("--diversity-weight",type=float,default=.01); p.add_argument("--log-every",type=int,default=10); p.add_argument("--save-every",type=int,default=250); p.add_argument("--seed",type=int,default=42); p.add_argument("--preload-cache",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--preload-teacher-cache",action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args()

if __name__ == "__main__": torch.set_float32_matmul_precision("high"); main(parse_args())
