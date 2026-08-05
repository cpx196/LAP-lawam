#!/usr/bin/env python3
"""Compute the fixed per-position teacher template for LAP10V3."""
from pathlib import Path
import argparse
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-cache", default="cache/lap10_task14_vlm_teacher_8192")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="cache/lap10_task14_vlm_teacher_8192/position_stats.pt")
    args = p.parse_args()
    paths = sorted((Path(args.teacher_cache) / args.split).glob("shard-*.pt"))
    if not paths:
        raise FileNotFoundError(f"No teacher shards under {Path(args.teacher_cache) / args.split}")
    total = None
    total_sq = None
    count = 0
    for index, path in enumerate(paths, 1):
        x = torch.load(path, map_location="cpu", weights_only=True)["teacher_condition"].float()
        if x.ndim != 3 or x.shape[1:] != (284, 768):
            raise RuntimeError(f"Unexpected teacher shape in {path}: {tuple(x.shape)}")
        if total is None:
            total = torch.zeros(284, 768, dtype=torch.float64)
            total_sq = torch.zeros_like(total)
        total += x.double().sum(0)
        total_sq += x.double().square().sum(0)
        count += int(x.shape[0])
        if index == 1 or index % 20 == 0 or index == len(paths):
            print(f"[stats] {index}/{len(paths)} shards samples={count}", flush=True)
    assert total is not None and total_sq is not None
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(0.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "position_mean": mean.float(),
        "position_std": variance.sqrt().float(),
        "num_samples": count,
        "source_cache": str(Path(args.teacher_cache).resolve()),
    }, output)
    print(f"[stats] position_mean_mse_baseline={variance.mean().item():.6f}")
    print(f"[stats] position_mean_rms={mean.float().square().mean().sqrt().item():.6f}")
    print(f"[stats] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
