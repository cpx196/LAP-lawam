#!/usr/bin/env python3
"""Compute train-only SEC284 teacher statistics and mean baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sec284_data import EXPECTED_CONDITION_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--output", default="cache/sec284_task14_teacher/train_stats.pt")
    args = parser.parse_args()
    paths = sorted((Path(args.teacher_cache) / "train").glob("shard-*.pt"))
    if not paths:
        raise FileNotFoundError("no train teacher shards found")
    total = torch.zeros(EXPECTED_CONDITION_SHAPE, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    for index, path in enumerate(paths, 1):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        x = payload["teacher_condition"].double()
        if tuple(x.shape[1:]) != EXPECTED_CONDITION_SHAPE:
            raise RuntimeError(f"unexpected teacher shape in {path}: {tuple(x.shape)}")
        total += x.sum(dim=0)
        total_sq += x.square().sum(dim=0)
        count += int(x.shape[0])
        if index == 1 or index % 20 == 0 or index == len(paths):
            print(f"[stats] {index}/{len(paths)} shards samples={count}", flush=True)
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(0.0)
    payload = {
        "position_mean": mean.float(),
        "position_variance": variance.float(),
        "position_std": variance.sqrt().float(),
        "global_mse_mean_baseline": variance.mean().float(),
        "teacher_rms": (total_sq / count).mean().sqrt().float(),
        "num_samples": count,
        "split": "train",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[stats] mean-baseline-mse={payload['global_mse_mean_baseline'].item():.6f}")
    print(f"[stats] wrote {output}")


if __name__ == "__main__":
    main()
