#!/usr/bin/env python3
"""Evaluate SEC284-L condition distillation without invoking the action path."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.sec284_data import SEC284Dataset, SEC284LossWeights, bounded_variance_weights, sec284_distillation_loss


def scalar_metrics(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, variance: torch.Tensor) -> dict[str, torch.Tensor]:
    losses = sec284_distillation_loss(student, teacher, mask, variance, SEC284LossWeights())
    cosine = 1.0 - losses["cosine"]
    return {"raw_mse": losses["raw_mse"], "whitened_mse": losses["whitened_mse"], "token_cosine": cosine}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SEC284Config(**checkpoint["config"])
    model = SEC284L(config).to(device).eval()
    model.load_state_dict(checkpoint["sec284"], strict=True)
    dataset = SEC284Dataset(Path(args.feature_cache), Path(args.wrist_cache), Path(args.teacher_cache), args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
    variance = stats["position_variance"].to(device)
    mean = stats["position_mean"].to(device)
    metric_sums: dict[str, float] = defaultdict(float)
    domain_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    domain_counts: dict[str, int] = defaultdict(int)
    ablation_sums: dict[str, float] = defaultdict(float)
    ablation_counts: dict[str, int] = defaultdict(int)
    student_sum = torch.zeros_like(mean, dtype=torch.float64)
    student_square_sum = torch.zeros_like(mean, dtype=torch.float64)
    teacher_sum = torch.zeros_like(mean, dtype=torch.float64)
    teacher_square_sum = torch.zeros_like(mean, dtype=torch.float64)
    mean_baseline_sum = 0.0
    processed = 0
    for batch in loader:
        if args.max_samples and processed >= args.max_samples:
            break
        visual = batch["visual_tokens"].to(device)
        mask = batch["view_mask"].to(device)
        teacher = batch["teacher_condition"].to(device)
        teacher_mask = batch["teacher_mask"].to(device)
        if args.max_samples:
            keep = min(visual.shape[0], args.max_samples - processed)
            visual, mask, teacher, teacher_mask = visual[:keep], mask[:keep], teacher[:keep], teacher_mask[:keep]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            student = model(visual, mask)
        metric = scalar_metrics(student, teacher, teacher_mask, variance)
        batch_size = visual.shape[0]
        shuffled = None
        if batch_size > 1:
            shuffled_teacher = teacher.roll(shifts=1, dims=0)
            shuffled = F.mse_loss(student.float(), shuffled_teacher.float()).item()
        for view_index, name in enumerate(("main", "left", "right")):
            ablated_mask = mask.clone()
            ablated_mask[:, view_index] = False
            valid = ablated_mask.any(dim=1)
            if bool(valid.all()):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    ablated = model(visual, ablated_mask)
                ablation_sums[name] += F.mse_loss(ablated.float(), teacher.float()).item() * batch_size
                ablation_counts[name] += batch_size
        batch_row = {key: float(value.item()) for key, value in metric.items()}
        if shuffled is not None:
            batch_row["shuffle_teacher_mse"] = shuffled
        for key, value in batch_row.items():
            metric_sums[key] += value * batch_size
        for index, domain in enumerate(batch["domain"]):
            domain_counts[domain] += 1
            for key, value in batch_row.items():
                domain_sums[domain][key] += value
        student_fp32 = student.float()
        teacher_fp32 = teacher.float()
        student_sum += student_fp32.sum(dim=0, dtype=torch.float64)
        student_square_sum += student_fp32.square().sum(dim=0, dtype=torch.float64)
        teacher_sum += teacher_fp32.sum(dim=0, dtype=torch.float64)
        teacher_square_sum += teacher_fp32.square().sum(dim=0, dtype=torch.float64)
        mean_baseline_sum += F.mse_loss(mean.unsqueeze(0).expand_as(teacher_fp32), teacher_fp32).item() * batch_size
        processed += batch_size
    if not processed:
        raise RuntimeError("no evaluation samples were processed")
    aggregate = {key: value / processed for key, value in metric_sums.items()}
    mean_baseline = mean_baseline_sum / processed
    aggregate["dynamic_r2"] = 1.0 - aggregate["raw_mse"] / max(mean_baseline, 1e-12)
    student_variance = (student_square_sum / processed - (student_sum / processed).square()).clamp_min(0.0)
    teacher_variance = (teacher_square_sum / processed - (teacher_sum / processed).square()).clamp_min(0.0)
    aggregate["teacher_cross_sample_std_rms"] = float(teacher_variance.mean().sqrt().item())
    aggregate["student_cross_sample_std_rms"] = float(student_variance.mean().sqrt().item())
    aggregate["student_teacher_std_ratio"] = aggregate["student_cross_sample_std_rms"] / max(aggregate["teacher_cross_sample_std_rms"], 1e-12)
    result = {
        "split": args.split,
        "samples": processed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "aggregate": aggregate,
        "domains": {domain: {key: value / domain_counts[domain] for key, value in values.items()} for domain, values in domain_sums.items()},
        "view_ablation_raw_mse": {name: ablation_sums[name] / ablation_counts[name] for name in ablation_sums},
        "mean_only_baseline_mse": mean_baseline,
        "variance_weight_range": [float(bounded_variance_weights(variance).min()), float(bounded_variance_weights(variance).max())],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
