#!/usr/bin/env python3
"""Measure cross-sample SEC284 variance fidelity on a small CPU probe set."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.probe_sec284_sample import load_shard


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=284)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    feature_root, wrist_root = Path(args.feature_cache), Path(args.wrist_cache)
    teacher_root = Path(args.teacher_cache)
    manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    refs = manifest["splits"][args.split]["samples"]
    if not 2 <= args.samples <= len(refs):
        raise ValueError("samples must be between 2 and the split size")
    indices = sorted(random.Random(args.seed).sample(range(len(refs)), args.samples))
    shard_size = int(json.loads((teacher_root / "metadata.json").read_text())["shard_size"])
    by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for output_index, dataset_index in enumerate(indices):
        shard, local = divmod(dataset_index, shard_size)
        by_shard[shard].append((output_index, local))

    visuals: list[torch.Tensor | None] = [None] * len(indices)
    teachers: list[torch.Tensor | None] = [None] * len(indices)
    domains: list[str] = []
    for dataset_index in indices:
        domains.append(str(refs[dataset_index].get("domain", "unknown")))
    for shard, requested in by_shard.items():
        feature = load_shard(feature_root, args.split, shard)
        wrist = load_shard(wrist_root, args.split, shard)
        teacher = load_shard(teacher_root, args.split, shard)
        for output_index, local in requested:
            visuals[output_index] = torch.stack(
                [feature["vision_t"][local], wrist["vision_left_t"][local], wrist["vision_right_t"][local]],
                dim=0,
            ).float()
            teachers[output_index] = teacher["teacher_condition"][local].float()
    visual_batch = torch.stack([item for item in visuals if item is not None])
    teacher_batch = torch.stack([item for item in teachers if item is not None])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SEC284L(SEC284Config(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["sec284"], strict=True)
    students = []
    for start in range(0, len(visual_batch), args.batch_size):
        students.append(model(visual_batch[start : start + args.batch_size]).float())
    student_batch = torch.cat(students)

    teacher_centered = teacher_batch - teacher_batch.mean(dim=0, keepdim=True)
    student_centered = student_batch - student_batch.mean(dim=0, keepdim=True)
    teacher_variance = teacher_centered.square().mean(dim=0)
    student_variance = student_centered.square().mean(dim=0)
    centered_error = (student_centered - teacher_centered).square().mean()
    teacher_dynamic_power = teacher_variance.mean()
    variance_cosine = F.cosine_similarity(
        student_variance.flatten().unsqueeze(0), teacher_variance.flatten().unsqueeze(0)
    )[0]
    centered_sample_cosine = F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=-1
    )
    result = {
        "checkpoint_step": int(checkpoint["global_step"]),
        "split": args.split,
        "samples": args.samples,
        "seed": args.seed,
        "domains": {domain: domains.count(domain) for domain in sorted(set(domains))},
        "teacher_cross_sample_std_rms": float(teacher_dynamic_power.sqrt()),
        "student_cross_sample_std_rms": float(student_variance.mean().sqrt()),
        "student_teacher_std_ratio": float(
            student_variance.mean().sqrt() / teacher_dynamic_power.sqrt().clamp_min(1e-12)
        ),
        "variance_map_cosine": float(variance_cosine),
        "centered_dynamic_mse": float(centered_error),
        "centered_dynamic_r2": float(
            1.0 - centered_error / teacher_dynamic_power.clamp_min(1e-12)
        ),
        "centered_sample_cosine_mean": float(centered_sample_cosine.mean()),
        "centered_sample_cosine_p05": float(torch.quantile(centered_sample_cosine, 0.05)),
        "raw_mse": float(F.mse_loss(student_batch, teacher_batch)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
