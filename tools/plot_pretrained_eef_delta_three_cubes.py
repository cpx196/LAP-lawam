#!/usr/bin/env python3
"""Compare pretrained Bridge EEF-delta predictions with Three_Cubes EEF targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_pretrained_eef_ik_three_cubes import decode_so101_eef_actions, transplant_action_head
from plot_three_cubes_action_curve import get_deterministic_sample, inverse_minmax, move_to_device
from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy, TrainerUtils


DEFAULT_SAMPLES = "0,500,1000,5000,10000,15000,20000,25000,30000,35000,40000,50000"
DELTA_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
ABSOLUTE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "/data/pxchen/LaWAM/results/Checkpoints/three_cubes/"
            "0728_144659+three_cubes_1_1k_8gpu_lawm_action/config.yaml"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=(
            "/data/pxchen/LaWAM/results/Checkpoints/pretrain/lawam_pretrain/"
            "final_model/pytorch_model.pt"
        ),
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default=None,
        help="Load a Three_Cubes EEF-delta checkpoint directly; skips Bridge head transplant.",
    )
    parser.add_argument(
        "--finetuned-stats",
        default=None,
        help="dataset_statistics.json saved with --finetuned-checkpoint.",
    )
    parser.add_argument(
        "--pretrained-stats",
        default=(
            "/data/pxchen/LaWAM/results/Checkpoints/pretrain/lawam_pretrain/"
            "dataset_statistics.json"
        ),
    )
    parser.add_argument(
        "--eef-sidecar",
        default="/data/pxchen/LaWAM/dataset/Three_Cubes_1/derived/so101_eef",
    )
    parser.add_argument("--sample-indices", default=DEFAULT_SAMPLES)
    parser.add_argument("--source-embodiment-id", type=int, default=18)
    parser.add_argument("--target-embodiment-id", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        default=(
            "results/Checkpoints/three_cubes/"
            "pretrained_bridge_eef_delta_vs_three_cubes"
        ),
    )
    return parser.parse_args()


def read_sidecar(root: Path) -> dict[str, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No EEF sidecar parquet files under {root / 'data'}")
    tables = [pq.read_table(path) for path in files]
    output = {
        "index": np.concatenate([table["index"].to_numpy() for table in tables]).astype(np.int64),
        "episode_index": np.concatenate(
            [table["episode_index"].to_numpy() for table in tables]
        ).astype(np.int64),
    }
    for name in (
        "observation.eef",
        "action.eef",
        "action.eef_delta_from_state",
        "action.eef_delta_sequence",
    ):
        output[name] = np.concatenate(
            [np.asarray(table[name].to_pylist(), dtype=np.float64) for table in tables]
        )
    if not np.array_equal(output["index"], np.arange(len(output["index"]))):
        raise ValueError("Expected contiguous sidecar indexes matching local Three_Cubes sample indexes")
    return output


def ground_truth_delta_chunk(sidecar: dict[str, np.ndarray], start: int, steps: int) -> np.ndarray:
    stop = start + steps
    if stop > len(sidecar["index"]):
        raise IndexError(f"Chunk [{start}, {stop}) exceeds EEF sidecar length")
    episode = sidecar["episode_index"][start]
    if not np.all(sidecar["episode_index"][start:stop] == episode):
        raise ValueError("Requested chunk crosses an episode boundary")
    delta = sidecar["action.eef_delta_sequence"][start:stop].copy()
    # A new policy chunk starts at the current observation, not at action[t-1].
    delta[0] = sidecar["action.eef_delta_from_state"][start]
    return delta


def integrate_deltas(start_eef: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    position = np.asarray(start_eef[:3], dtype=np.float64).copy()
    rotation = Rotation.from_rotvec(start_eef[3:6]).as_matrix()
    trajectory = np.empty((len(deltas), 7), dtype=np.float64)
    for step, delta in enumerate(deltas):
        position += delta[:3]
        rotation = rotation @ Rotation.from_rotvec(delta[3:6]).as_matrix()
        trajectory[step, :3] = position
        trajectory[step, 3:6] = Rotation.from_matrix(rotation).as_rotvec()
        trajectory[step, 6] = delta[6]
    return trajectory


def delta_display_values(values: np.ndarray) -> np.ndarray:
    displayed = values.copy()
    displayed[:, :3] *= 1000.0
    displayed[:, 3:6] = np.rad2deg(displayed[:, 3:6])
    return displayed


def plot_7d(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    names: list[str],
    units: list[str],
    title: str,
) -> None:
    steps = np.arange(len(target))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), dpi=150, sharex=True)
    axes = axes.reshape(-1)
    for index, axis in enumerate(axes[:7]):
        mae = np.mean(np.abs(prediction[:, index] - target[:, index]))
        axis.plot(steps, target[:, index], linewidth=2.0, label="GT")
        axis.plot(steps, prediction[:, index], "--", linewidth=2.0, label="Prediction")
        axis.set_title(f"{names[index]}  MAE={mae:.3f} {units[index]}")
        axis.set_ylabel(units[index])
        axis.grid(alpha=0.25)
    axes[7].axis("off")
    axes[0].legend(fontsize=8)
    axes[6].set_xlabel("action step")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)


def plot_trajectory_errors(path: Path, gt: np.ndarray, prediction: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_error = np.linalg.norm(prediction[:, :3] - gt[:, :3], axis=1) * 1000.0
    rotation_error = np.rad2deg(
        Rotation.from_matrix(Rotation.from_rotvec(gt[:, 3:6]).as_matrix().transpose(0, 2, 1)
                           @ Rotation.from_rotvec(prediction[:, 3:6]).as_matrix()).magnitude()
    )
    gripper_error = np.abs(prediction[:, 6] - gt[:, 6])
    steps = np.arange(len(gt))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=150, sharex=True)
    for axis, values, label in zip(
        axes,
        (position_error, rotation_error, gripper_error),
        ("position error (mm)", "orientation error (deg)", "gripper error (deg)"),
    ):
        axis.plot(steps, values, linewidth=2.0)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("action step")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)
    return position_error, rotation_error, gripper_error


def error_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def markdown_report(path: Path, aggregate: dict, samples: list[dict]) -> None:
    finetuned = aggregate.get("evaluation_mode") == "finetuned"
    lines = [
        "# Three_Cubes EEF Delta 离线预测对比" if finetuned else "# Pretrained Bridge EEF Delta 与 Three_Cubes 对比",
        "",
        "## 测试范围",
        "",
        (
            "这是后训练 checkpoint 的离线推理；输入当前图像和当前绝对 EEF state，输出 36 步 delta EEF。"
            if finetuned
            else "这是零样本离线推理：将 pretrained Bridge 的 action head 从 embodiment `18` 复制到 SO101 `31`，没有进行梯度训练。"
        ),
        (
            "预测 delta 与 GT delta 都从同一个当前绝对 EEF 起点累计；本测试不进行 IK。"
            if finetuned
            else "模型配置为 `use_state=false`，因此 Three_Cubes 的 joint state 只用于构造 GT 起点 EEF，并未输入模型。"
        ),
        "",
        "## GT Delta 定义",
        "",
        "每个 action chunk 的第 0 步 GT delta 是 `observation.eef -> action[0].eef`；后续步是 `action[t-1].eef -> action[t].eef`。",
        "GT 和预测都从同一个当前 `observation.eef` 开始累计，然后计算绝对 EEF 轨迹误差。",
        "",
        "## 汇总结果",
        "",
        f"- 样本数：`{aggregate['num_samples']}`；动作点数：`{aggregate['num_action_points']}`",
        f"- Delta 平移 MAE：`{aggregate['delta_translation_mae_mm']:.3f} mm`",
        f"- Delta 旋转 MAE：`{aggregate['delta_rotation_mae_deg']:.3f} deg`",
        f"- Delta 夹爪 MAE：`{aggregate['delta_gripper_mae_deg']:.3f} deg`",
        f"- 累计位置误差：中位数 `{aggregate['absolute_position_error_mm']['median']:.3f} mm`，p95 `{aggregate['absolute_position_error_mm']['p95']:.3f} mm`，最大 `{aggregate['absolute_position_error_mm']['max']:.3f} mm`",
        f"- 累计姿态误差：中位数 `{aggregate['absolute_orientation_error_deg']['median']:.3f} deg`，p95 `{aggregate['absolute_orientation_error_deg']['p95']:.3f} deg`，最大 `{aggregate['absolute_orientation_error_deg']['max']:.3f} deg`",
        "",
        "## 每个样本的图",
        "",
        "| 样本 | Delta 图 | 绝对 EEF 图 | 累计误差图 | Delta 平移 MAE (mm) | 最终位置误差 (mm) |",
        "|---:|---|---|---|---:|---:|",
    ]
    for sample in samples:
        lines.append(
            f"| {sample['sample_index']} | [{sample['delta_plot']}]({sample['delta_plot']}) | "
            f"[{sample['absolute_plot']}]({sample['absolute_plot']}) | "
            f"[{sample['error_plot']}]({sample['error_plot']}) | "
            f"{sample['delta_translation_mae_mm']:.3f} | {sample['final_position_error_mm']:.3f} |"
        )
    lines.extend([
        "",
        "## 如何看图",
        "",
        "Delta 图用于定位每一步的动作误差。累计误差图显示这些误差如何在 36 步 chunk 内累积；它们才直接对应 EEF -> IK 部署时的末端偏移。",
        "",
        "可机器读取的完整指标见 `aggregate.json`。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = read_sidecar(Path(args.eef_sidecar))

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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    model = apply_training_freeze_policy(build_framework(cfg), cfg)
    checkpoint = args.finetuned_checkpoint or args.pretrained_checkpoint
    TrainerUtils.load_finetune_init_weights(model, checkpoint, load_pretrained_policy_flow=True)
    copied_parameters = []
    if not args.finetuned_checkpoint:
        copied_parameters = transplant_action_head(model, args.source_embodiment_id, args.target_embodiment_id)
    model.eval().to(device)

    bridge_stats = None
    finetuned_stats = None
    if args.finetuned_checkpoint:
        if not args.finetuned_stats:
            raise ValueError("--finetuned-stats is required with --finetuned-checkpoint")
        with open(args.finetuned_stats, encoding="utf-8") as handle:
            finetuned_stats = json.load(handle)["new_embodiment"]["action"]
    else:
        with open(args.pretrained_stats, encoding="utf-8") as handle:
            bridge_stats = json.load(handle)["oxe_bridge"]["action"]
    with open(Path(args.eef_sidecar) / "meta" / "representation.json", encoding="utf-8") as handle:
        so101_stats = json.load(handle)["statistics"]["action.eef_delta_sequence"]

    sample_indices = [int(item) for item in args.sample_indices.split(",") if item.strip()]
    samples = []
    all_delta_error, all_position_error, all_rotation_error, all_gripper_error = [], [], [], []
    for sample_index in sample_indices:
        sample_seed = args.seed + sample_index
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        sample, sample_meta = get_deterministic_sample(dataset, sample_index)
        batch = move_to_device(collator([sample]), device)
        batch["embodiment_id"] = torch.full_like(batch["embodiment_id"], args.target_embodiment_id)
        prediction, _target, mask = model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
        valid_steps = mask[0].detach().cpu().numpy().astype(bool)[:, 0]
        prediction_normalized = prediction[0].detach().float().cpu().numpy()[valid_steps, :7]
        if args.finetuned_checkpoint:
            predicted_delta = inverse_minmax(prediction_normalized, finetuned_stats)
        else:
            predicted_delta = decode_so101_eef_actions(
                prediction_normalized,
                bridge_stats,
                so101_stats,
                "moment_match",
            )
        gt_delta = ground_truth_delta_chunk(sidecar, sample_index, len(predicted_delta))
        start_eef = sidecar["observation.eef"][sample_index]
        gt_absolute = integrate_deltas(start_eef, gt_delta)
        predicted_absolute = integrate_deltas(start_eef, predicted_delta)

        delta_plot = f"sample{sample_index}_delta_eef.png"
        absolute_plot = f"sample{sample_index}_absolute_eef.png"
        error_plot = f"sample{sample_index}_accumulated_error.png"
        plot_7d(
            out_dir / delta_plot,
            delta_display_values(gt_delta),
            delta_display_values(predicted_delta),
            DELTA_NAMES,
            ["mm", "mm", "mm", "deg", "deg", "deg", "deg"],
            f"Sample {sample_index}: GT vs predicted delta EEF",
        )
        plot_7d(
            out_dir / absolute_plot,
            gt_absolute,
            predicted_absolute,
            ABSOLUTE_NAMES,
            ["m", "m", "m", "rad", "rad", "rad", "deg"],
            f"Sample {sample_index}: accumulated absolute EEF trajectory",
        )
        position_error, rotation_error, gripper_error = plot_trajectory_errors(
            out_dir / error_plot,
            gt_absolute,
            predicted_absolute,
            f"Sample {sample_index}: accumulated EEF trajectory error",
        )
        delta_error = np.abs(delta_display_values(predicted_delta) - delta_display_values(gt_delta))
        all_delta_error.append(delta_error)
        all_position_error.append(position_error)
        all_rotation_error.append(rotation_error)
        all_gripper_error.append(gripper_error)
        samples.append(
            {
                "sample_index": sample_index,
                "sample_meta": sample_meta,
                "valid_horizon_steps": int(len(predicted_delta)),
                "delta_translation_mae_mm": float(delta_error[:, :3].mean()),
                "delta_rotation_mae_deg": float(delta_error[:, 3:6].mean()),
                "delta_gripper_mae_deg": float(delta_error[:, 6].mean()),
                "final_position_error_mm": float(position_error[-1]),
                "final_orientation_error_deg": float(rotation_error[-1]),
                "delta_plot": delta_plot,
                "absolute_plot": absolute_plot,
                "error_plot": error_plot,
            }
        )

    delta_error = np.concatenate(all_delta_error)
    aggregate = {
        "experiment": "finetuned EEF-delta prediction on Three_Cubes" if args.finetuned_checkpoint else "pretrained Bridge EEF-delta zero-shot prediction on Three_Cubes",
        "evaluation_mode": "finetuned" if args.finetuned_checkpoint else "pretrained_zero_shot",
        "checkpoint": str(checkpoint),
        "normalization_adapter": "SO101 minmax inverse" if args.finetuned_checkpoint else "Bridge minmax -> Bridge z-score -> SO101 EEF delta z-score -> clip",
        "copied_head_parameters": copied_parameters,
        "num_samples": len(samples),
        "num_action_points": int(len(delta_error)),
        "delta_mae_per_dimension": {
            name: float(delta_error[:, index].mean()) for index, name in enumerate(DELTA_NAMES)
        },
        "delta_translation_mae_mm": float(delta_error[:, :3].mean()),
        "delta_rotation_mae_deg": float(delta_error[:, 3:6].mean()),
        "delta_gripper_mae_deg": float(delta_error[:, 6].mean()),
        "absolute_position_error_mm": error_summary(np.concatenate(all_position_error)),
        "absolute_orientation_error_deg": error_summary(np.concatenate(all_rotation_error)),
        "absolute_gripper_error_deg": error_summary(np.concatenate(all_gripper_error)),
        "samples": samples,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_report(out_dir / "REPORT.md", aggregate, samples)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
