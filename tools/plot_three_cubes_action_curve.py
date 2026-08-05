from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import (
    apply_training_freeze_policy,
    TrainerUtils,
)


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def move_to_device(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def get_deterministic_sample(dataset, sample_index: int):
    """Fetch a deterministic local index from the underlying single dataset.

    LeRobotMixtureDataset.__getitem__ uses the provided index as an RNG seed in
    train/all mode, so dataset[idx] is not the same as row/local idx. For
    diagnostics we want exact local-index semantics.
    """
    if hasattr(dataset, "datasets") and len(getattr(dataset, "datasets")) == 1:
        single = dataset.datasets[0]
        trajectory_id, step = single.abs_index_to_episode_step(int(sample_index))
        selected_video_keys = list(single.modality_keys.get("video", []))
        raw_data = single.get_step_data(
            trajectory_id,
            step,
            modality_keys_override={"video": selected_video_keys},
        )
        transforms = single.transforms
        data = transforms(raw_data)
        sample = dataset._build_output_sample(single, data, selected_video_keys)
        return sample, {
            "trajectory_id": int(trajectory_id),
            "frame_index": int(step),
            "deterministic_local_index": int(sample_index),
        }
    sample = dataset[int(sample_index)]
    return sample, {"deterministic_local_index": int(sample_index)}


def inverse_minmax(norm_actions: np.ndarray, stats: dict) -> np.ndarray:
    mins = np.asarray(stats["min"], dtype=np.float32)
    maxs = np.asarray(stats["max"], dtype=np.float32)
    dims = min(norm_actions.shape[-1], mins.shape[0], maxs.shape[0])
    clipped = np.clip(norm_actions[..., :dims], -1.0, 1.0)
    return (clipped + 1.0) * 0.5 * (maxs[:dims] - mins[:dims]) + mins[:dims]


def write_curve_csv(path: Path, steps: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["t"]
        for name in JOINT_NAMES:
            header.extend([f"{name}_gt", f"{name}_pred"])
        writer.writerow(header)
        for t in range(len(steps)):
            row = [int(t)]
            for j in range(6):
                row.extend([float(target[t, j]), float(pred[t, j])])
            writer.writerow(row)


def plot_curve_grid(
    path: Path,
    *,
    steps: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    sample_index: int,
    value_space: str,
    ylabel: str,
    num_inference_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), dpi=150, sharex=True)
    axes = axes.reshape(-1)
    mae_per_dim = np.mean(np.abs(pred - target), axis=0)
    rmse_per_dim = np.sqrt(np.mean((pred - target) ** 2, axis=0))
    for j, ax in enumerate(axes[:6]):
        ax.plot(steps, target[:, j], label="GT / dataset", linewidth=2.0)
        ax.plot(steps, pred[:, j], label="Pred / checkpoint", linewidth=2.0, linestyle="--")
        ax.set_title(f"{JOINT_NAMES[j]}  MAE={mae_per_dim[j]:.3f}")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(ylabel)
        if value_space == "normalized":
            ax.set_ylim(-1.15, 1.15)
    axes[-2].set_xlabel("action step")
    axes[-1].set_xlabel("action step")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Three_Cubes_1 sample {sample_index}: {value_space} action curve vs 1000-step checkpoint prediction\n"
        f"valid horizon={len(steps)} steps, diffusion_steps={num_inference_steps}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)
    return mae_per_dim, rmse_per_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data/pxchen/LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/final_model/pytorch_model.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Build the model from --config and initialize it with this checkpoint via the "
            "same relaxed finetune loader used by training. Useful for evaluating a pretrain "
            "checkpoint whose own config uses an older framework name."
        ),
    )
    parser.add_argument(
        "--config",
        default="/data/pxchen/LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/config.yaml",
    )
    parser.add_argument(
        "--stats",
        default="/data/pxchen/LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/dataset_statistics.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/pxchen/LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/action_curve_compare",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated sample indices to plot. Overrides --sample-index when set.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
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
    if args.sample_indices:
        sample_indices = sorted({max(0, int(x.strip())) for x in args.sample_indices.split(",") if x.strip()})
    else:
        sample_indices = [max(0, int(args.sample_index))]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if args.init_checkpoint:
        model = build_framework(cfg)
        model = apply_training_freeze_policy(model, cfg)
        TrainerUtils.load_finetune_init_weights(
            model,
            args.init_checkpoint,
            load_pretrained_policy_flow=bool(getattr(cfg.trainer, "load_pretrained_policy_flow", True)),
        )
    else:
        model = baseframework.from_pretrained(args.checkpoint)
    model.eval().to(device)

    with open(args.stats, "r", encoding="utf-8") as f:
        stats_all = json.load(f)
    action_stats = stats_all["new_embodiment"]["action"]

    summaries = []
    dataset_len = len(dataset)
    for idx in sample_indices:
        if idx >= dataset_len:
            raise IndexError(f"sample index {idx} out of range for dataset length {dataset_len}")

        sample, sample_meta = get_deterministic_sample(dataset, idx)
        batch = collator([sample])
        batch = move_to_device(batch, device)
        pred, target, mask = model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
        pred = pred[0].detach().float().cpu().numpy()
        target = target[0].detach().float().cpu().numpy()
        mask = mask[0].detach().cpu().numpy().astype(bool)

        valid_t = mask[:, 0]
        pred_valid = pred[valid_t, :6]
        target_valid = target[valid_t, :6]
        steps = np.arange(pred_valid.shape[0])

        pred_raw = inverse_minmax(pred_valid, action_stats)
        target_raw = inverse_minmax(target_valid, action_stats)

        raw_csv_path = out_dir / f"action_curve_compare_sample{idx}_raw.csv"
        norm_csv_path = out_dir / f"action_curve_compare_sample{idx}_normalized.csv"
        # Keep backward-compatible raw CSV path from the first version of this diagnostic.
        legacy_raw_csv_path = out_dir / f"action_curve_compare_sample{idx}.csv"
        write_curve_csv(raw_csv_path, steps, target_raw, pred_raw)
        write_curve_csv(norm_csv_path, steps, target_valid, pred_valid)
        write_curve_csv(legacy_raw_csv_path, steps, target_raw, pred_raw)

        raw_png_path = out_dir / f"action_curve_compare_sample{idx}_raw.png"
        norm_png_path = out_dir / f"action_curve_compare_sample{idx}_normalized.png"
        # Keep backward-compatible raw PNG path from the first version of this diagnostic.
        legacy_raw_png_path = out_dir / f"action_curve_compare_sample{idx}.png"
        raw_mae_per_dim, raw_rmse_per_dim = plot_curve_grid(
            raw_png_path,
            steps=steps,
            target=target_raw,
            pred=pred_raw,
            sample_index=idx,
            value_space="raw motor",
            ylabel="raw motor value",
            num_inference_steps=args.num_inference_steps,
        )
        plot_curve_grid(
            legacy_raw_png_path,
            steps=steps,
            target=target_raw,
            pred=pred_raw,
            sample_index=idx,
            value_space="raw motor",
            ylabel="raw motor value",
            num_inference_steps=args.num_inference_steps,
        )
        norm_mae_per_dim, norm_rmse_per_dim = plot_curve_grid(
            norm_png_path,
            steps=steps,
            target=target_valid,
            pred=pred_valid,
            sample_index=idx,
            value_space="normalized",
            ylabel="normalized action value",
            num_inference_steps=args.num_inference_steps,
        )
        summary = {
            "sample_index": idx,
            "checkpoint": str(args.checkpoint),
            "valid_horizon_steps": int(len(steps)),
            "sample_meta": sample_meta,
            "joint_names": JOINT_NAMES,
            "raw_mae_per_dim": {name: float(raw_mae_per_dim[i]) for i, name in enumerate(JOINT_NAMES)},
            "raw_rmse_per_dim": {name: float(raw_rmse_per_dim[i]) for i, name in enumerate(JOINT_NAMES)},
            "raw_mean_mae": float(np.mean(raw_mae_per_dim)),
            "normalized_mae_per_dim": {
                name: float(norm_mae_per_dim[i]) for i, name in enumerate(JOINT_NAMES)
            },
            "normalized_rmse_per_dim": {
                name: float(norm_rmse_per_dim[i]) for i, name in enumerate(JOINT_NAMES)
            },
            "normalized_mean_mae": float(np.mean(norm_mae_per_dim)),
            "plot_path": str(legacy_raw_png_path),
            "csv_path": str(legacy_raw_csv_path),
            "raw_plot_path": str(raw_png_path),
            "raw_csv_path": str(raw_csv_path),
            "normalized_plot_path": str(norm_png_path),
            "normalized_csv_path": str(norm_csv_path),
        }
        # Backward-compatible aliases.
        summary["mae_per_dim"] = summary["raw_mae_per_dim"]
        summary["rmse_per_dim"] = summary["raw_rmse_per_dim"]
        summary["mean_mae"] = summary["raw_mean_mae"]
        summary_path = out_dir / f"action_curve_compare_sample{idx}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate = {
        "checkpoint": str(args.checkpoint),
        "sample_indices": [item["sample_index"] for item in summaries],
        "num_samples": len(summaries),
        "joint_names": JOINT_NAMES,
        "raw_mean_mae_across_samples": float(np.mean([item["raw_mean_mae"] for item in summaries])),
        "normalized_mean_mae_across_samples": float(
            np.mean([item["normalized_mean_mae"] for item in summaries])
        ),
        "raw_mae_per_sample": {str(item["sample_index"]): item["raw_mean_mae"] for item in summaries},
        "normalized_mae_per_sample": {
            str(item["sample_index"]): item["normalized_mean_mae"] for item in summaries
        },
        "raw_mae_per_dim_across_samples": {
            name: float(np.mean([item["raw_mae_per_dim"][name] for item in summaries])) for name in JOINT_NAMES
        },
        "normalized_mae_per_dim_across_samples": {
            name: float(np.mean([item["normalized_mae_per_dim"][name] for item in summaries]))
            for name in JOINT_NAMES
        },
        "summaries": summaries,
    }
    # Backward-compatible raw aliases.
    aggregate["mean_mae_across_samples"] = aggregate["raw_mean_mae_across_samples"]
    aggregate["mae_per_sample"] = aggregate["raw_mae_per_sample"]
    aggregate["mae_per_dim_across_samples"] = aggregate["raw_mae_per_dim_across_samples"]
    aggregate_path = out_dir / f"action_curve_compare_samples_{'_'.join(map(str, sample_indices))}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
