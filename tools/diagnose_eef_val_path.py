#!/usr/bin/env python3
"""Offline, reproducible EEF-delta validation and checkpoint-overlay comparison.

This script never creates a physical robot, camera, or motor-bus object. It only
loads the training dataset and a LaWAM checkpoint, then calls the same aligned
inference method used by the offline evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from plot_pretrained_eef_delta_three_cubes import ground_truth_delta_chunk, read_sidecar
from plot_three_cubes_action_curve import get_deterministic_sample, inverse_minmax, move_to_device
from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy, TrainerUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--eef-sidecar",
        type=Path,
        default=Path("dataset/Three_Cubes_1/derived/so101_eef"),
    )
    parser.add_argument("--val-local-index", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    return loaded.get("state_dict", loaded)


def digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values.astype(np.float32, copy=False))
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.checkpoint_dir / "config.yaml"
    stats_path = args.checkpoint_dir / "dataset_statistics.json"
    full_checkpoint = args.checkpoint_dir / "final_model" / "pytorch_model.pt"
    for path in (config_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = args.base_checkpoint or full_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.overlay and not args.overlay.is_file():
        raise FileNotFoundError(args.overlay)

    seed_everything(args.seed)
    cfg = OmegaConf.load(config_path)
    cfg.datasets.vla_data.data_root_dir = str(args.dataset_root)
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.val_num_workers = 0
    cfg.datasets.vla_data.drop_last = False
    cfg.datasets.vla_data.train_split_all = True
    policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
    collator = _build_latent_world_collator(cfg, policy_cfg=policy_cfg, training=False)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="all",
        balance_dataset_weights=True,
        framework_name=cfg.framework.name,
    )
    sample, meta = get_deterministic_sample(dataset, args.val_local_index)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch = move_to_device(collator([sample]), device)
    batch["embodiment_id"] = torch.full_like(batch["embodiment_id"], 31)

    model = apply_training_freeze_policy(build_framework(cfg), cfg)
    TrainerUtils.load_finetune_init_weights(model, checkpoint, load_pretrained_policy_flow=True)
    if args.overlay:
        missing, unexpected = model.load_state_dict(checkpoint_state_dict(args.overlay), strict=False)
        print(
            json.dumps(
                {
                    "overlay_missing_count": len(missing),
                    "overlay_unexpected_count": len(unexpected),
                    "overlay_missing_sample": missing[:10],
                    "overlay_unexpected_sample": unexpected[:10],
                },
                ensure_ascii=False,
            )
        )
    model.eval().to(device)
    with torch.inference_mode():
        prediction, target, mask = model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)

    valid = mask[0].detach().cpu().numpy().astype(bool)[:, 0]
    prediction = prediction[0].detach().float().cpu().numpy()[valid, :7]
    target = target[0].detach().float().cpu().numpy()[valid, :7]
    with stats_path.open(encoding="utf-8") as handle:
        action_stats = json.load(handle)["new_embodiment"]["action"]
    raw_prediction = inverse_minmax(prediction, action_stats)
    raw_target = inverse_minmax(target, action_stats)
    sidecar = read_sidecar(args.eef_sidecar)
    expected = ground_truth_delta_chunk(sidecar, args.val_local_index, len(prediction))
    sidecar_difference = np.abs(raw_target - expected)
    output = {
        "checkpoint": str(checkpoint),
        "overlay": str(args.overlay) if args.overlay else None,
        "seed": args.seed,
        "sample": meta,
        "valid_steps": int(len(prediction)),
        "validation_style_normalized_mse": float(np.mean((prediction - target) ** 2)),
        "validation_prediction_sha256": digest(prediction),
        "ground_truth_sha256": digest(target),
        "dataloader_vs_sidecar_max_abs": float(np.max(sidecar_difference)),
        "dataloader_vs_sidecar_first_step_max_abs": float(np.max(sidecar_difference[0])),
        "dataloader_vs_sidecar_later_steps_max_abs": float(np.max(sidecar_difference[1:])),
        "normalized_state": batch["state"][0].detach().float().cpu().numpy().tolist(),
        "unnormalized_action_first5": raw_target[:5].tolist(),
        "prediction_first5": raw_prediction[:5].tolist(),
        "ground_truth_first5": expected[:5].tolist(),
        "position_norm_mm_first5": (np.linalg.norm(raw_target[:5, :3], axis=1) * 1000.0).tolist(),
        "rotation_norm_deg_first5": (np.linalg.norm(raw_target[:5, 3:6], axis=1) * 180.0 / np.pi).tolist(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
