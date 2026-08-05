#!/usr/bin/env python3
"""Offline evaluation for a Stage-1 LAP + LaWM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.lap_stage1 import LAP60M, count_parameters
from tools.train_lap_stage1 import EEFHead, FeatureShardDataset, load_lawm_decoder


def per_sample_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().flatten(1).mean(dim=1)


def per_sample_diversity(scene_tokens: torch.Tensor) -> torch.Tensor:
    x = F.normalize(scene_tokens, dim=-1)
    gram = x @ x.transpose(1, 2)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return (gram - eye.unsqueeze(0)).square().flatten(1).mean(dim=1)


def quat_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    errors = []
    for start in (3, 11):
        p = F.normalize(pred[:, start : start + 4], dim=-1)
        t = F.normalize(target[:, start : start + 4], dim=-1)
        dot = (p * t).sum(dim=-1).abs().clamp(max=1.0)
        errors.append(2.0 * torch.acos(dot) * (180.0 / math.pi))
    return torch.stack(errors, dim=1).mean(dim=1)


def effective_rank(z: torch.Tensor) -> float:
    z = z.float() - z.float().mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh((z.T @ z) / max(1, z.shape[0] - 1)).clamp_min(0)
    if float(eig.sum()) <= 0:
        return 0.0
    p = eig / eig.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def evaluate_split(
    *,
    split: str,
    cache_dir: Path,
    manifest: dict,
    lap: nn.Module,
    eef_head: nn.Module,
    lawm: nn.Module,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
    dataset = FeatureShardDataset(cache_dir / split, preload=False, verbose=False)
    n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    dataset_view = Subset(dataset, range(n))
    loader = DataLoader(dataset_view, batch_size=batch_size, shuffle=False, num_workers=0)
    domains = [x["domain"] for x in manifest["splits"][split]["samples"][:n]]
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    z_lap_all: list[torch.Tensor] = []
    z_idm_all: list[torch.Tensor] = []
    first_input: tuple[torch.Tensor, torch.Tensor] | None = None
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            vision_t = batch["vision_t"].to(device)
            vision_t1 = batch["vision_t1"].to(device)
            z_idm = batch["z_idm"].to(device)
            state_t = batch["state_t"].to(device)
            state_t1 = batch["state_t1"].to(device)
            if first_input is None:
                first_input = (vision_t[:1].clone(), state_t[:1].clone())

            out = lap(vision_t, state_t)
            z_lap = out["z_lap"]
            student = lawm(vision_t, z_lap).squeeze(1)
            teacher = lawm(vision_t, z_idm).squeeze(1)
            shuffled = lawm(vision_t, torch.roll(z_lap, shifts=1, dims=0)).squeeze(1)
            zeroed = lawm(vision_t, torch.zeros_like(z_lap)).squeeze(1)
            state_pred = eef_head(out["pooled"])

            pred_raw = state_pred * state_std + state_mean
            target_raw = state_t1 * state_std + state_mean
            current_raw = state_t * state_std + state_mean
            values = {
                "latent_mse": per_sample_mse(z_lap, z_idm),
                "latent_cosine": F.cosine_similarity(z_lap.squeeze(1), z_idm.squeeze(1), dim=-1),
                "student_world_mse": per_sample_mse(student, vision_t1),
                "teacher_world_mse": per_sample_mse(teacher, vision_t1),
                "shuffle_world_mse": per_sample_mse(shuffled, vision_t1),
                "zero_world_mse": per_sample_mse(zeroed, vision_t1),
                "eef_norm_mse": per_sample_mse(state_pred, state_t1),
                "eef_baseline_norm_mse": per_sample_mse(state_t, state_t1),
                "eef_position_mae_m": (pred_raw[:, [0, 1, 2, 8, 9, 10]] - target_raw[:, [0, 1, 2, 8, 9, 10]]).abs().mean(1),
                "eef_baseline_position_mae_m": (current_raw[:, [0, 1, 2, 8, 9, 10]] - target_raw[:, [0, 1, 2, 8, 9, 10]]).abs().mean(1),
                "eef_gripper_mae": (pred_raw[:, [7, 15]] - target_raw[:, [7, 15]]).abs().mean(1),
                "eef_quaternion_error_deg": quat_error_deg(pred_raw, target_raw),
                "scene_diversity_loss": per_sample_diversity(out["scene_tokens"]),
            }
            batch_domains = domains[offset : offset + vision_t.shape[0]]
            offset += vision_t.shape[0]
            for i, domain in enumerate(batch_domains):
                for group in ("all", domain):
                    counts[group] += 1
                    for key, tensor in values.items():
                        sums[group][key] += float(tensor[i].cpu())
            z_lap_all.append(z_lap.squeeze(1).cpu())
            z_idm_all.append(z_idm.squeeze(1).cpu())

    results: dict[str, dict] = {}
    for group, count in counts.items():
        metrics = {key: value / count for key, value in sums[group].items()}
        student = metrics["student_world_mse"]
        teacher = metrics["teacher_world_mse"]
        metrics["student_teacher_ratio"] = student / teacher
        metrics["shuffle_increase_pct"] = 100.0 * (metrics["shuffle_world_mse"] / student - 1.0)
        metrics["zero_increase_pct"] = 100.0 * (metrics["zero_world_mse"] / student - 1.0)
        metrics["eef_mse_improvement_pct"] = 100.0 * (
            1.0 - metrics["eef_norm_mse"] / metrics["eef_baseline_norm_mse"]
        )
        metrics["eef_position_improvement_pct"] = 100.0 * (
            1.0 - metrics["eef_position_mae_m"] / metrics["eef_baseline_position_mae_m"]
        )
        results[group] = {"samples": count, **metrics}
    z_lap_cat = torch.cat(z_lap_all)
    z_idm_cat = torch.cat(z_idm_all)
    results["representation"] = {
        "z_lap_effective_rank": effective_rank(z_lap_cat),
        "z_idm_effective_rank": effective_rank(z_idm_cat),
        "z_lap_mean_dim_std": float(z_lap_cat.std(dim=0).mean()),
        "z_idm_mean_dim_std": float(z_idm_cat.std(dim=0).mean()),
    }
    results["elapsed_seconds"] = time.perf_counter() - started
    assert first_input is not None
    return results, first_input


def benchmark(lap: nn.Module, lawm: nn.Module, sample: tuple[torch.Tensor, torch.Tensor]) -> dict:
    vision_t, state_t = sample
    z = None
    with torch.inference_mode():
        for _ in range(50):
            z = lap(vision_t, state_t)["z_lap"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(300):
            z = lap(vision_t, state_t)["z_lap"]
        torch.cuda.synchronize()
        lap_ms = (time.perf_counter() - start) * 1000 / 300

        for _ in range(50):
            _ = lawm(vision_t, z).squeeze(1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(300):
            _ = lawm(vision_t, z).squeeze(1)
        torch.cuda.synchronize()
        lawm_ms = (time.perf_counter() - start) * 1000 / 300
    return {"lap_batch1_fp32_ms": lap_ms, "lawm_batch1_fp32_ms": lawm_ms, "combined_ms": lap_ms + lawm_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="cache/lap_stage1_task14")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--output", default="outputs/lap_stage1_task14/eval_step3000.json")
    parser.add_argument("--lam-ckpt", default="latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt")
    parser.add_argument("--lam-yaml", default="latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml")
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    lap = LAP60M()
    lap.load_state_dict(checkpoint["lap"], strict=True)
    eef_head = EEFHead()
    eef_head.load_state_dict(checkpoint["eef_head"], strict=True)
    lawm = load_lawm_decoder(args.lam_ckpt, args.lam_yaml)
    lawm.load_state_dict(checkpoint["lawm_decoder"], strict=True)
    del checkpoint
    lap = lap.to(device).eval()
    eef_head = eef_head.to(device).eval()
    lawm = lawm.to(device).eval()

    cache_dir = Path(args.cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    stats = json.loads((cache_dir / "state_stats.json").read_text())
    state_mean = torch.tensor(stats["mean"], device=device)
    state_std = torch.tensor(stats["std"], device=device)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "parameters": {
            "lap": count_parameters(lap),
            "eef_head": count_parameters(eef_head),
            "lawm": count_parameters(lawm),
        },
        "splits": {},
    }
    latency_sample = None
    for split, max_samples in (("train", args.train_samples), ("val", 0), ("test", 0)):
        print(f"[eval] {split} max_samples={max_samples or 'all'}", flush=True)
        result, sample = evaluate_split(
            split=split,
            cache_dir=cache_dir,
            manifest=manifest,
            lap=lap,
            eef_head=eef_head,
            lawm=lawm,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            batch_size=args.batch_size,
            max_samples=max_samples,
        )
        report["splits"][split] = result
        latency_sample = sample
        print(json.dumps(result, indent=2), flush=True)
    assert latency_sample is not None
    report["latency"] = benchmark(lap, lawm, latency_sample)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {output}")
    print(json.dumps(report["latency"], indent=2))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
