#!/usr/bin/env python3
"""Cache VLM teacher conditions and train LAP10's aligned Expert interface."""

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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.server_policy import load_policy_from_checkpoint
from starVLA.model.framework.vlas.lawam import _cuda_autocast
from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.lap_stage2 import LAP8, LAP10
from tools.build_lap_multiview_cache import WRIST_KEYS, WristFrameReader
from tools.train_lap8_phase1 import (
    Phase1Dataset,
    TensorShardDataset,
    grad_norm,
    load_action_expert,
    off_diagonal_cosine,
)
from tools.train_lap_stage1 import (
    DEFAULT_LAM_CKPT,
    DEFAULT_LAM_YAML,
    Task14RawReader,
    diversity_loss,
    load_lawm_decoder,
    refs_from_manifest,
)


DEFAULT_POLICY = Path(
    "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
)
DEFAULT_STAGE1 = Path(
    "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
)
DEFAULT_LAP8 = Path(
    "outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt"
)
DEFAULT_EXPERT = Path("cache/lap8_phase1_official_action_expert.pt")
DEFAULT_TEACHER_CACHE = Path("cache/lap10_task14_vlm_teacher_8192")
DEFAULT_OUTPUT = Path("outputs/lap10_alignment_task14_1000step")
LANG = "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad."
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_teacher_cache(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher cache construction requires CUDA")
    device = torch.device("cuda")
    seed_everything(args.seed)
    manifest_path = Path(args.feature_cache) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = refs_from_manifest(manifest, args.split)
    if args.max_samples > 0:
        refs = refs[: args.max_samples]
    output_dir = Path(args.teacher_cache) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[teacher] samples={len(refs):,} split={args.split} dtype=float16", flush=True)
    main_reader = Task14RawReader(Path(args.dataset_root))
    wrist_reader = WristFrameReader(Path(args.dataset_root))
    policy = load_policy_from_checkpoint(
        str(args.official_policy), use_bf16=False, device="cuda"
    )
    backend = policy.policy_backend.eval()
    vlm_dtype = backend.model_cfg.vlm_dtype
    expected_tokens = int(args.output_tokens)

    written = 0
    started = time.perf_counter()
    for shard_start in range(0, len(refs), args.cache_shard_size):
        shard_index = shard_start // args.cache_shard_size
        shard_refs = refs[shard_start : shard_start + args.cache_shard_size]
        output_path = output_dir / f"shard-{shard_index:05d}.pt"
        if output_path.exists() and not args.overwrite_cache:
            existing = torch.load(output_path, map_location="cpu", weights_only=True)
            if int(existing["teacher_condition"].shape[0]) != len(shard_refs):
                raise RuntimeError(f"Existing teacher shard is misaligned: {output_path}")
            written += len(shard_refs)
            continue

        conditions: list[torch.Tensor] = []
        for batch_start in range(0, len(shard_refs), args.teacher_batch_size):
            batch_refs = shard_refs[batch_start : batch_start + args.teacher_batch_size]
            examples = []
            for ref in batch_refs:
                main = np.asarray(
                    main_reader.dataset.get_video(
                        ref.episode, "video.cam_high", ref.base_index
                    )
                )[0]
                wrists = [
                    np.asarray(
                        wrist_reader.dataset.get_video(
                            ref.episode, key, ref.base_index
                        )
                    )[0]
                    for key in WRIST_KEYS
                ]
                examples.append(
                    {
                        "lang": LANG,
                        "primary_image": [main],
                        "wrist_image": wrists,
                        "action_hz": 30.0,
                        "embodiment_id": 1,
                    }
                )
            prepared = policy.policy_infer_batch_builder.build_infer_batch(examples)
            act_query, flow_query = backend._prepare_queries(
                device=device, vlm_stage_dtype=vlm_dtype
            )
            with torch.inference_mode(), _cuda_autocast(vlm_dtype):
                vlm_out = backend._run_vlm_stage(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    pixel_values=prepared["pixel_values"],
                    image_grid_thw=prepared["image_grid_thw"],
                    act_placeholder_mask=prepared["act_placeholder_mask"],
                    flow_placeholder_mask=prepared["flow_placeholder_mask"],
                    act_query=act_query,
                    flow_query=flow_query,
                )
                condition = backend.flow._prepare_semantic_condition(
                    h_vlm=vlm_out["h_vlm"],
                    h_lap=None,
                    model_dtype=backend.flow._compute_dtype(),
                )
            if tuple(condition.shape[1:]) != (expected_tokens, 768):
                raise RuntimeError(
                    f"Teacher condition must be [B,{expected_tokens},768], got {tuple(condition.shape)}"
                )
            conditions.append(condition.detach().cpu().to(torch.float16))
            del prepared, vlm_out, condition

        torch.save(
            {"teacher_condition": torch.cat(conditions, dim=0)}, output_path
        )
        written += len(shard_refs)
        elapsed = time.perf_counter() - started
        rate = written / max(elapsed, 1e-6)
        eta = (len(refs) - written) / max(rate, 1e-6) / 60.0
        print(
            f"[teacher] {written}/{len(refs)} -> {output_path.name} "
            f"rate={rate:.2f} samples/s eta={eta:.1f}min",
            flush=True,
        )

    metadata = {
        "source_policy": str(Path(args.official_policy).resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "split": args.split,
        "samples": len(refs),
        "language": LANG,
        "shape_per_sample": [expected_tokens, 768],
        "dtype": "float16",
        "cache_shard_size": args.cache_shard_size,
    }
    Path(args.teacher_cache).mkdir(parents=True, exist_ok=True)
    (Path(args.teacher_cache) / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[teacher] complete: {args.teacher_cache}", flush=True)


class LAP10Dataset(Dataset):
    def __init__(self, args: argparse.Namespace, *, verbose: bool) -> None:
        self.phase1 = Phase1Dataset(
            Path(args.feature_cache) / "train",
            Path(args.wrist_cache) / "train",
            Path(args.action_cache) / "train",
            preload=args.preload_cache,
            verbose=verbose,
        )
        self.teacher = TensorShardDataset(
            Path(args.teacher_cache) / "train",
            preload=args.preload_teacher_cache,
            verbose=verbose,
        )
        if len(self.teacher) > len(self.phase1):
            raise RuntimeError(
                f"Teacher cache exceeds aligned Stage-1 cache: {len(self.teacher)} > {len(self.phase1)}"
            )

    def __len__(self) -> int:
        return len(self.teacher)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = self.phase1[index]
        result.update(self.teacher[index])
        return result


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LAP10 training requires CUDA")
    device = torch.device("cuda")
    seed_everything(args.seed)
    dataset = LAP10Dataset(args, verbose=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    lap8_obj = torch.load(args.lap8_checkpoint, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.2)
    lap8 = LAP8(lap6, view_dropout=args.view_dropout)
    lap8.load_state_dict(lap8_obj["lap8"], strict=True)
    lap10 = LAP10(lap8, output_tokens=args.output_tokens).to(device, torch.float32)

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml)
    lawm.load_state_dict(stage1["lawm_decoder"], strict=True)
    lawm = lawm.to(device, torch.float32).eval()
    expert = load_action_expert(Path(args.expert_checkpoint)).to(device, torch.float32).eval()
    for module in (lawm, expert):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    trainable = [parameter for parameter in lap10.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )
    total = count_parameters(lap10)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(
        f"[model] LAP10={total:,} trainable={trainable_count:,} "
        f"LaWM={count_parameters(lawm):,} Expert={count_parameters(expert):,} VLM=teacher-cache-only",
        flush=True,
    )
    print(
        f"[train] device={device} precision=FP32 samples={len(dataset):,} "
        f"effective_batch={args.batch_size * args.grad_accumulation}",
        flush=True,
    )

    iterator = iter(loader)
    lap10.train()
    started = time.perf_counter()
    last_time = started
    last_step = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        metrics = {
            "loss": 0.0,
            "flow": 0.0,
            "align_mse": 0.0,
            "align_cos": 0.0,
            "div": 0.0,
            "pred_rms": 0.0,
            "teacher_rms": 0.0,
        }
        for _ in range(args.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
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
            teacher = batch["teacher_condition"].to(
                device, torch.float32, non_blocking=True
            )

            lap_out = lap10(visual, state)
            prediction = lap_out["cond_lap10"]
            if prediction.shape != teacher.shape:
                raise RuntimeError(
                    f"LAP10/teacher mismatch: {tuple(prediction.shape)} vs {tuple(teacher.shape)}"
                )
            with torch.no_grad():
                h_t1 = lawm(vision_t, lap_out["z_lap"]).squeeze(1)
            batch_size = vision_t.shape[0]
            flow_loss = expert(
                h_t=vision_t,
                h_t1_star=h_t1,
                h_vlm=None,
                h_lap=prediction,
                state=torch.zeros(batch_size, 32, device=device),
                actions=actions,
                action_hz=torch.full((batch_size,), 30.0, device=device),
                embodiment_id=torch.ones(batch_size, device=device, dtype=torch.long),
                state_mask=torch.zeros(batch_size, 32, device=device, dtype=torch.bool),
                actions_mask=actions_mask,
                attention_mask=torch.ones(
                    batch_size, prediction.shape[1], device=device, dtype=torch.bool
                ),
            )
            align_mse = F.mse_loss(prediction, teacher)
            align_cos = 1.0 - F.cosine_similarity(
                prediction, teacher, dim=-1
            ).mean()
            div_loss = diversity_loss(lap_out["scene_lap8"])
            loss = (
                flow_loss
                + args.align_mse_weight * align_mse
                + args.align_cos_weight * align_cos
                + args.diversity_weight * div_loss
            )
            (loss / args.grad_accumulation).backward()
            values = {
                "loss": loss.detach(),
                "flow": flow_loss.detach(),
                "align_mse": align_mse.detach(),
                "align_cos": align_cos.detach(),
                "div": div_loss.detach(),
                "pred_rms": prediction.detach().square().mean().sqrt(),
                "teacher_rms": teacher.detach().square().mean().sqrt(),
            }
            for key, value in values.items():
                metrics[key] += float(value.cpu()) / args.grad_accumulation

        raw_grad = grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0:
            now = time.perf_counter()
            step_time = (now - last_time) / (step - last_step)
            eta = step_time * (args.steps - step) / 3600.0
            allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            print(
                f"[train] step={step}/{args.steps} "
                + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                + f" grad={raw_grad:.4f} lr={scheduler.get_last_lr()[0]:.3e} "
                + f"step_time={step_time:.2f}s eta={eta:.2f}h peak_cuda={allocated:.2f}GiB",
                flush=True,
            )
            last_time, last_step = now, step

        if (args.save_every > 0 and step % args.save_every == 0) or step == args.steps:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "lap10": lap10.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "lap8_checkpoint": str(args.lap8_checkpoint),
                },
                output / f"lap10_step{step:07d}.pt",
            )
            print(f"[train] checkpoint saved at step {step}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build_teacher_cache", "train"), required=True)
    parser.add_argument("--dataset-root", default="dataset/robotwin_merged")
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--action-cache", default="cache/lap8_phase1_task14_actions")
    parser.add_argument("--teacher-cache", default=str(DEFAULT_TEACHER_CACHE))
    parser.add_argument("--official-policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--stage1-checkpoint", default=str(DEFAULT_STAGE1))
    parser.add_argument("--lap8-checkpoint", default=str(DEFAULT_LAP8))
    parser.add_argument("--expert-checkpoint", default=str(DEFAULT_EXPERT))
    parser.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    parser.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int, default=8192)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--cache-shard-size", type=int, default=128)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=284)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preload-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--view-dropout", type=float, default=0.2)
    parser.add_argument("--align-mse-weight", type=float, default=1.0)
    parser.add_argument("--align-cos-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parsed = parse_args()
    if parsed.mode == "build_teacher_cache":
        build_teacher_cache(parsed)
    else:
        train(parsed)
