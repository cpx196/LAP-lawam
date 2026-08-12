#!/usr/bin/env python3
"""DDP training for pure SEC284-L VLM-condition distillation (no action path)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.sec284_data import SEC284Dataset, SEC284LossWeights, sec284_distillation_loss, sha256_file


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = min(1.0, float(step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("SEC284-L training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local_rank, torch.device("cuda", local_rank)


def unwrap(model: torch.nn.Module) -> SEC284L:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


class ShardAwareDistributedBatchSampler(Sampler[list[int]]):
    """Shuffle shards and samples while keeping I/O-local batches DDP-balanced."""

    def __init__(
        self,
        dataset: SEC284Dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        total_batches = len(dataset) // batch_size
        self.global_batches = total_batches - total_batches % num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.global_batches // self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.dataset.feature.lengths), generator=generator).tolist()
        indices: list[int] = []
        for shard in shard_order:
            start = self.dataset.feature.offsets[shard]
            length = self.dataset.feature.lengths[shard]
            local = torch.randperm(length, generator=generator).tolist()
            indices.extend(start + item for item in local)
        indices = indices[: self.global_batches * self.batch_size]
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        yield from batches[self.rank :: self.num_replicas]


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    position_variance: torch.Tensor,
    loss_weights: SEC284LossWeights,
    device: torch.device,
    world: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)  # total/raw/white/cos
    sample_count = torch.zeros(1, device=device, dtype=torch.float64)
    for batch in loader:
        visual = batch["visual_tokens"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        teacher = batch["teacher_condition"].to(device, non_blocking=True)
        teacher_mask = batch["teacher_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            student = model(visual, view_mask)
        losses = sec284_distillation_loss(student, teacher, teacher_mask, position_variance, loss_weights)
        count = float(visual.shape[0])
        totals += torch.tensor(
            [losses["total"].item(), losses["raw_mse"].item(), losses["whitened_mse"].item(), losses["cosine"].item()],
            device=device, dtype=torch.float64,
        ) * count
        sample_count += count
    if world > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    values = (totals / sample_count.clamp_min(1.0)).tolist()
    return dict(zip(("total", "raw_mse", "whitened_mse", "cosine"), values, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    parser.add_argument("--output-dir", default="outputs/sec284_l_task14_distill")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps; 0 uses all epochs")
    parser.add_argument("--save-every-steps", type=int, default=500, help="periodic checkpoint interval; 0 disables")
    parser.add_argument("--log-every-steps", type=int, default=10, help="print reduced train losses every N optimizer steps; 0 disables")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accumulation < 1 or args.max_steps < 0 or args.save_every_steps < 0 or args.log_every_steps < 0:
        raise ValueError("epochs, batch-size, grad-accumulation, max-steps, save-every-steps, and log-every-steps are invalid")
    rank, world, local_rank, device = setup_distributed()
    is_main = rank == 0
    seed_everything(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    train = SEC284Dataset(Path(args.feature_cache), Path(args.wrist_cache), Path(args.teacher_cache), "train", preload=args.preload)
    val = SEC284Dataset(Path(args.feature_cache), Path(args.wrist_cache), Path(args.teacher_cache), "val", preload=args.preload)
    train_sampler = ShardAwareDistributedBatchSampler(
        train, args.batch_size, world, rank, args.seed
    )
    val_sampler = DistributedSampler(val, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None
    train_loader = DataLoader(train, batch_sampler=train_sampler, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
    position_variance = stats["position_variance"].to(device=device, dtype=torch.float32)
    config = SEC284Config()
    model: torch.nn.Module = SEC284L(config).to(device)
    if unwrap(model).parameter_count != 76_624_896:
        raise RuntimeError(f"unexpected SEC284-L parameter count: {unwrap(model).parameter_count}")
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    updates_per_epoch = len(train_loader) // args.grad_accumulation
    if updates_per_epoch < 1:
        raise RuntimeError("effective batch is larger than one training epoch")
    total_updates = updates_per_epoch * args.epochs
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    warmup = max(1, round(total_updates * args.warmup_ratio))
    min_ratio = args.min_lr / args.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_lambda(step, warmup, total_updates, min_ratio))
    loss_weights = SEC284LossWeights()
    start_epoch, global_step, best, stale = 0, 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(checkpoint["sec284"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step = int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])
        best, stale = float(checkpoint["best_val_total"]), int(checkpoint.get("stale_epochs", 0))
    if is_main:
        print(f"[model] SEC284-L params={unwrap(model).parameter_count:,} world={world} train={len(train)} val={len(val)} effective_batch={world * args.batch_size * args.grad_accumulation}", flush=True)
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    def checkpoint_payload(epoch: int) -> dict:
        return {
            "format_version": 1, "model_name": "SEC284-L", "sec284": unwrap(model).state_dict(),
            "config": asdict(config), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "global_step": global_step, "best_val_total": best, "stale_epochs": stale,
            "teacher_stats_sha256": sha256_file(Path(args.teacher_stats)),
            "teacher_cache_metadata_sha256": sha256_file(Path(args.teacher_cache) / "metadata.json"),
            "manifest_sha256": sha256_file(Path(args.feature_cache) / "manifest.json"), "args": vars(args),
        }
    stop_requested = False
    log_running = torch.zeros(4, device=device, dtype=torch.float64)
    log_micro_count = 0
    train_started_at = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros(4, device=device, dtype=torch.float64)
        epoch_micro_count = 0
        update_count = 0
        for micro_step, batch in enumerate(train_loader, 1):
            if micro_step > updates_per_epoch * args.grad_accumulation:
                break
            visual = batch["visual_tokens"].to(device, non_blocking=True)
            view_mask = batch["view_mask"].to(device, non_blocking=True)
            teacher = batch["teacher_condition"].to(device, non_blocking=True)
            teacher_mask = batch["teacher_mask"].to(device, non_blocking=True)
            sync = micro_step % args.grad_accumulation == 0
            context = model.no_sync() if isinstance(model, DDP) and not sync else nullcontext()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    student = model(visual, view_mask)
                losses = sec284_distillation_loss(student, teacher, teacher_mask, position_variance, loss_weights)
                (losses["total"] / args.grad_accumulation).backward()
            batch_losses = torch.tensor([losses["total"].item(), losses["raw_mse"].item(), losses["whitened_mse"].item(), losses["cosine"].item()], device=device, dtype=torch.float64)
            running += batch_losses
            log_running += batch_losses
            epoch_micro_count += 1
            log_micro_count += 1
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
                global_step += 1
                if args.log_every_steps and global_step % args.log_every_steps == 0:
                    reduced = log_running.clone()
                    if world > 1:
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    values = (reduced / max(1, log_micro_count * world)).tolist()
                    if is_main:
                        print(
                            f"[step] {global_step}/{total_updates} total={values[0]:.6f} "
                            f"raw_mse={values[1]:.6f} white_mse={values[2]:.6f} "
                            f"cosine_loss={values[3]:.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                            f"elapsed={time.monotonic() - train_started_at:.1f}s",
                            flush=True,
                        )
                    log_running.zero_()
                    log_micro_count = 0
                if is_main and args.save_every_steps and global_step % args.save_every_steps == 0:
                    torch.save(checkpoint_payload(epoch), out_dir / f"step-{global_step:06d}.pt")
                if args.max_steps and global_step >= args.max_steps:
                    stop_requested = True
                    break
        if world > 1:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
        train_values = (running / max(1, epoch_micro_count * world)).tolist()
        val_values = evaluate_epoch(model, val_loader, position_variance, loss_weights, device, world)
        improved = val_values["total"] < best * 0.995
        if improved:
            best, stale = val_values["total"], 0
        else:
            stale += 1
        if is_main:
            record = {"epoch": epoch, "global_step": global_step, "lr": scheduler.get_last_lr()[0], "train": dict(zip(("total", "raw_mse", "whitened_mse", "cosine"), train_values, strict=True)), "val": val_values}
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            checkpoint = checkpoint_payload(epoch)
            torch.save(checkpoint, out_dir / "last.pt")
            if improved:
                torch.save(checkpoint, out_dir / "best.pt")
            print(f"[epoch] {epoch + 1}/{args.epochs} train={record['train']} val={val_values} best={best:.6f}", flush=True)
        if world > 1:
            dist.barrier()
        if stop_requested:
            if is_main:
                print(f"[train] reached max_steps={args.max_steps}", flush=True)
            break
        if epoch + 1 >= args.min_epochs and stale >= args.early_stop_patience:
            if is_main:
                print(f"[train] early stop at epoch={epoch + 1}", flush=True)
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
