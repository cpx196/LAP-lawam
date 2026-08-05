#!/usr/bin/env python3
"""Zero-shot LaWAM OXE-Bridge EEF predictions converted to SO101 joints with IK."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy, TrainerUtils

from audit_so101_fk_ik import ARM_JOINT_NAMES, inverse_kinematics_synced
from plot_three_cubes_action_curve import get_deterministic_sample, inverse_minmax, move_to_device


JOINT_NAMES = [*ARM_JOINT_NAMES, "gripper"]
DEFAULT_SAMPLES = "0,500,1000,5000,10000,15000,20000,25000,30000,35000,40000,50000"


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
        "--so101-eef-stats",
        default="/data/pxchen/LaWAM/dataset/Three_Cubes_1/meta/so101_eef_delta_stats.json",
    )
    parser.add_argument(
        "--pretrained-stats",
        default="/data/pxchen/LaWAM/results/Checkpoints/pretrain/lawam_pretrain/dataset_statistics.json",
    )
    parser.add_argument(
        "--three-cubes-stats",
        default=(
            "/data/pxchen/LaWAM/results/Checkpoints/three_cubes/"
            "0728_144659+three_cubes_1_1k_8gpu_lawm_action/dataset_statistics.json"
        ),
    )
    parser.add_argument(
        "--urdf",
        default="/data/pxchen/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    )
    parser.add_argument("--sample-indices", default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--out-dir",
        default="results/Checkpoints/three_cubes/pretrained_bridge_head_so101_eef_stats_to_ik",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-embodiment-id", type=int, default=18, help="OXE Bridge source head.")
    parser.add_argument("--target-embodiment-id", type=int, default=31, help="SO101/new-embodiment head.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ik-iterations", type=int, default=3)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument(
        "--normalization-adapter",
        choices=("moment_match", "direct_minmax"),
        default="moment_match",
    )
    return parser.parse_args()


def decode_so101_eef_actions(
    normalized: np.ndarray,
    bridge_stats: dict,
    so101_stats: dict,
    adapter: str,
) -> np.ndarray:
    normalized = np.clip(np.asarray(normalized[..., :7], dtype=np.float32), -1.0, 1.0)
    # BridgeDataConfig flips the normalized gripper convention during training.
    normalized[..., 6] *= -1.0
    if adapter == "direct_minmax":
        return inverse_minmax(normalized, so101_stats)
    if adapter != "moment_match":
        raise ValueError(f"Unsupported normalization adapter: {adapter}")

    bridge_raw = inverse_minmax(normalized, bridge_stats).astype(np.float64)
    bridge_mean = np.asarray(bridge_stats["mean"], dtype=np.float64)
    bridge_std = np.asarray(bridge_stats["std"], dtype=np.float64)
    so101_mean = np.asarray(so101_stats["mean"], dtype=np.float64)
    so101_std = np.asarray(so101_stats["std"], dtype=np.float64)
    valid = bridge_std > 1e-12
    adapted = np.broadcast_to(so101_mean, bridge_raw.shape).copy()
    adapted[..., valid] += (
        (bridge_raw[..., valid] - bridge_mean[valid])
        / bridge_std[valid]
        * so101_std[valid]
    )
    return np.clip(
        adapted,
        np.asarray(so101_stats["min"], dtype=np.float64),
        np.asarray(so101_stats["max"], dtype=np.float64),
    )


def transplant_action_head(model, source_id: int, target_id: int) -> list[str]:
    """Copy a trained embodiment's learned action input/output projections to a new ID."""
    flow = model.policy_backend.flow
    layers = {
        "action_encoder.W1": flow.action_encoder.W1,
        "action_encoder.W2": flow.action_encoder.W2,
        "action_decoder.layer1": flow.action_decoder.layer1,
        "action_decoder.layer2": flow.action_decoder.layer2,
    }
    copied = []
    with torch.no_grad():
        for name, layer in layers.items():
            layer.W[target_id].copy_(layer.W[source_id])
            layer.b[target_id].copy_(layer.b[source_id])
            copied.extend([f"{name}.W[{source_id}->{target_id}]", f"{name}.b[{source_id}->{target_id}]"])
    return copied


def eef_deltas_to_so101_joints(
    deltas: np.ndarray,
    current_joint: np.ndarray,
    kinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_joint = np.asarray(current_joint, dtype=np.float64).copy()
    desired_pose = kinematics.forward_kinematics(seed_joint).copy()
    predicted_joints = []
    position_residual_mm = []
    orientation_residual_deg = []

    for delta in deltas:
        desired_pose[:3, 3] += delta[:3]
        desired_pose[:3, :3] = desired_pose[:3, :3] @ Rotation.from_rotvec(delta[3:6]).as_matrix()

        solved = inverse_kinematics_synced(
            kinematics,
            seed_joint,
            desired_pose,
            args.position_weight,
            args.orientation_weight,
            args.ik_iterations,
        )
        reached_pose = kinematics.forward_kinematics(solved).copy()
        position_residual_mm.append(float(np.linalg.norm(reached_pose[:3, 3] - desired_pose[:3, 3]) * 1000.0))
        relative_rotation = reached_pose[:3, :3].T @ desired_pose[:3, :3]
        orientation_residual_deg.append(
            float(np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude()))
        )

        solved = np.asarray(solved, dtype=np.float64).copy()
        solved[5] = float(delta[6])
        predicted_joints.append(solved[:6])
        seed_joint = solved

    return (
        np.asarray(predicted_joints),
        np.asarray(position_residual_mm),
        np.asarray(orientation_residual_deg),
    )


def joint_actions_to_eef_deltas(
    actions: np.ndarray,
    current_joint: np.ndarray,
    kinematics,
) -> np.ndarray:
    """Encode an absolute SO101 joint chunk with the same delta-EEF convention."""
    current_joint = np.asarray(current_joint, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    previous_pose = kinematics.forward_kinematics(current_joint).copy()
    deltas = []
    for action in actions:
        target_pose = kinematics.forward_kinematics(action).copy()
        delta = np.empty(7, dtype=np.float64)
        delta[:3] = target_pose[:3, 3] - previous_pose[:3, 3]
        delta[3:6] = Rotation.from_matrix(
            previous_pose[:3, :3].T @ target_pose[:3, :3]
        ).as_rotvec()
        delta[6] = action[5]
        deltas.append(delta)
        previous_pose = target_pose
    return np.asarray(deltas)


def residual_summary(values: np.ndarray, threshold: float | None = None) -> dict:
    summary = {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }
    if threshold is not None:
        above = values > threshold
        summary[f"above_{threshold:g}"] = int(np.count_nonzero(above))
        summary[f"above_{threshold:g}_ratio"] = float(np.mean(above))
    return summary


def write_csv(
    path: Path,
    target: np.ndarray,
    predicted: np.ndarray,
    eef_delta: np.ndarray,
    position_residual_mm: np.ndarray,
    orientation_residual_deg: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["t"]
        for name in JOINT_NAMES:
            header.extend([f"{name}_gt_deg", f"{name}_pred_deg"])
        header.extend(
            [
                "dx_m",
                "dy_m",
                "dz_m",
                "drx_rad",
                "dry_rad",
                "drz_rad",
                "so101_gripper_deg",
                "ik_position_residual_mm",
                "ik_orientation_residual_deg",
            ]
        )
        writer.writerow(header)
        for step in range(len(target)):
            row: list[float | int] = [step]
            for joint_index in range(6):
                row.extend([float(target[step, joint_index]), float(predicted[step, joint_index])])
            row.extend(float(value) for value in eef_delta[step])
            row.extend([float(position_residual_mm[step]), float(orientation_residual_deg[step])])
            writer.writerow(row)


def plot_joint_curves(
    path: Path,
    target: np.ndarray,
    predicted: np.ndarray,
    roundtrip: np.ndarray,
    current_joint: np.ndarray,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(len(target))
    mae = np.mean(np.abs(predicted - target), axis=0)
    rmse = np.sqrt(np.mean((predicted - target) ** 2, axis=0))
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), dpi=150, sharex=True)
    for joint_index, axis in enumerate(axes.reshape(-1)):
        axis.plot(steps, target[:, joint_index], linewidth=2.0, label="GT joint action")
        axis.plot(steps, predicted[:, joint_index], "--", linewidth=2.0, label="Pretrain EEF -> IK")
        axis.plot(
            steps,
            roundtrip[:, joint_index],
            ":",
            linewidth=1.5,
            color="0.25",
            label="GT joint -> FK/EEF -> IK",
        )
        axis.axhline(current_joint[joint_index], color="0.5", linewidth=1.0, linestyle=":", label="input state")
        axis.set_title(f"{JOINT_NAMES[joint_index]}  MAE={mae[joint_index]:.3f} deg")
        axis.set_ylabel("motor position (deg)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("action step")
    axes[-1, 1].set_xlabel("action step")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Three_Cubes sample {sample_index}: pretrained Bridge head + SO101 EEF stats -> IK\n"
        "Manual head transplant and rescaling; no gradient training",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path)
    plt.close(fig)
    return mae, rmse


def plot_overview(path: Path, summaries: list[dict]) -> None:
    rows, columns = 4, 3
    fig, axes = plt.subplots(rows, columns, figsize=(18, 18), dpi=140)
    for axis, summary in zip(axes.reshape(-1), summaries):
        target = np.load(summary["npz_path"])["target"]
        predicted = np.load(summary["npz_path"])["predicted"]
        steps = np.arange(len(target))
        axis.plot(steps, target[:, 1], linewidth=1.5, label="GT shoulder_lift")
        axis.plot(steps, predicted[:, 1], "--", linewidth=1.5, label="Pred shoulder_lift")
        axis.plot(steps, target[:, 2], linewidth=1.5, alpha=0.75, label="GT elbow_flex")
        axis.plot(steps, predicted[:, 2], "--", linewidth=1.5, alpha=0.75, label="Pred elbow_flex")
        axis.set_title(f"sample {summary['sample_index']}  mean MAE={summary['mean_mae_deg']:.2f} deg")
        axis.grid(alpha=0.2)
    axes.reshape(-1)[0].legend(fontsize=7)
    fig.suptitle("Pretrained Bridge head + SO101 EEF scaling -> IK: 12 Three_Cubes samples")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    TrainerUtils.load_finetune_init_weights(
        model,
        args.pretrained_checkpoint,
        load_pretrained_policy_flow=True,
    )
    copied_head_parameters = transplant_action_head(
        model,
        args.source_embodiment_id,
        args.target_embodiment_id,
    )
    model.eval().to(device)

    with open(args.so101_eef_stats, "r", encoding="utf-8") as handle:
        so101_eef_stats = json.load(handle)["action"]
    with open(args.pretrained_stats, "r", encoding="utf-8") as handle:
        bridge_stats = json.load(handle)["oxe_bridge"]["action"]
    with open(args.three_cubes_stats, "r", encoding="utf-8") as handle:
        three_stats = json.load(handle)["new_embodiment"]
    action_stats = three_stats["action"]
    state_stats = three_stats["state"]

    from lerobot.model.kinematics import RobotKinematics

    kinematics = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )

    sample_indices = [int(value.strip()) for value in args.sample_indices.split(",") if value.strip()]
    summaries = []
    for sample_index in sample_indices:
        sample_seed = args.seed + sample_index
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        sample, sample_meta = get_deterministic_sample(dataset, sample_index)
        batch = move_to_device(collator([sample]), device)
        batch["embodiment_id"] = torch.full_like(
            batch["embodiment_id"], args.target_embodiment_id
        )
        prediction, target, mask = model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
        prediction = prediction[0].detach().float().cpu().numpy()
        target = target[0].detach().float().cpu().numpy()
        mask = mask[0].detach().cpu().numpy().astype(bool)
        valid_steps = mask[:, 0]

        target_joint = inverse_minmax(target[valid_steps, :6], action_stats)
        normalized_state = batch["state"][0].detach().float().cpu().numpy()[:6]
        current_joint = inverse_minmax(normalized_state[None, :], state_stats)[0]
        so101_eef_action = decode_so101_eef_actions(
            prediction[valid_steps, :7],
            bridge_stats,
            so101_eef_stats,
            args.normalization_adapter,
        )
        predicted_joint, position_residual_mm, orientation_residual_deg = eef_deltas_to_so101_joints(
            so101_eef_action,
            current_joint,
            kinematics,
            args,
        )
        roundtrip_eef_action = joint_actions_to_eef_deltas(
            target_joint,
            current_joint,
            kinematics,
        )
        roundtrip_joint, roundtrip_position_mm, roundtrip_orientation_deg = (
            eef_deltas_to_so101_joints(
                roundtrip_eef_action,
                current_joint,
                kinematics,
                args,
            )
        )

        csv_path = out_dir / f"sample{sample_index}_joint_compare.csv"
        png_path = out_dir / f"sample{sample_index}_joint_compare.png"
        npz_path = out_dir / f"sample{sample_index}_joint_compare.npz"
        write_csv(
            csv_path,
            target_joint,
            predicted_joint,
            so101_eef_action,
            position_residual_mm,
            orientation_residual_deg,
        )
        np.savez_compressed(
            npz_path,
            target=target_joint,
            predicted=predicted_joint,
            roundtrip=roundtrip_joint,
            predicted_position_residual_mm=position_residual_mm,
            predicted_orientation_residual_deg=orientation_residual_deg,
            roundtrip_position_residual_mm=roundtrip_position_mm,
            roundtrip_orientation_residual_deg=roundtrip_orientation_deg,
        )
        mae, rmse = plot_joint_curves(
            png_path,
            target_joint,
            predicted_joint,
            roundtrip_joint,
            current_joint,
            sample_index,
        )
        roundtrip_mae = np.mean(np.abs(roundtrip_joint - target_joint), axis=0)
        summary = {
            "sample_index": sample_index,
            "sample_meta": sample_meta,
            "inference_seed": sample_seed,
            "valid_horizon_steps": int(len(target_joint)),
            "current_joint_deg": current_joint.tolist(),
            "mae_deg_per_joint": {name: float(mae[i]) for i, name in enumerate(JOINT_NAMES)},
            "rmse_deg_per_joint": {name: float(rmse[i]) for i, name in enumerate(JOINT_NAMES)},
            "mean_mae_deg": float(mae.mean()),
            "ik_position_residual_mm": residual_summary(position_residual_mm, 10.0),
            "ik_orientation_residual_deg": residual_summary(orientation_residual_deg),
            "ground_truth_fk_eef_ik_roundtrip": {
                "mean_mae_deg": float(roundtrip_mae.mean()),
                "mae_deg_per_joint": {
                    name: float(roundtrip_mae[i]) for i, name in enumerate(JOINT_NAMES)
                },
                "position_residual_mm": residual_summary(roundtrip_position_mm, 10.0),
                "orientation_residual_deg": residual_summary(roundtrip_orientation_deg),
            },
            "plot_path": str(png_path),
            "csv_path": str(csv_path),
            "npz_path": str(npz_path),
        }
        summaries.append(summary)
        (out_dir / f"sample{sample_index}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    overview_path = out_dir / "joint_compare_12samples_overview.png"
    plot_overview(overview_path, summaries)
    result_arrays = [np.load(item["npz_path"]) for item in summaries]
    predicted_position_all = np.concatenate(
        [item["predicted_position_residual_mm"] for item in result_arrays]
    )
    predicted_orientation_all = np.concatenate(
        [item["predicted_orientation_residual_deg"] for item in result_arrays]
    )
    roundtrip_position_all = np.concatenate(
        [item["roundtrip_position_residual_mm"] for item in result_arrays]
    )
    roundtrip_orientation_all = np.concatenate(
        [item["roundtrip_orientation_residual_deg"] for item in result_arrays]
    )
    aggregate = {
        "experiment": "lawam_pretrain Bridge head transplanted to SO101 EEF stats -> IK -> joint",
        "assumptions": {
            "source_embodiment_id": args.source_embodiment_id,
            "target_embodiment_id": args.target_embodiment_id,
            "eef_layout": "[dx,dy,dz,drx,dry,drz,gripper]",
            "translation_integration": "base-frame addition",
            "rotation_integration": "current_R @ delta_R",
            "normalization": "Bridge output distribution adapted to SO101 EEF statistics; gripper convention inverted",
            "normalization_adapter": args.normalization_adapter,
        },
        "copied_head_parameters": copied_head_parameters,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "so101_eef_stats": args.so101_eef_stats,
        "urdf": args.urdf,
        "sample_indices": sample_indices,
        "base_seed": args.seed,
        "num_samples": len(summaries),
        "mean_mae_deg_across_samples": float(np.mean([item["mean_mae_deg"] for item in summaries])),
        "mae_deg_per_joint_across_samples": {
            name: float(np.mean([item["mae_deg_per_joint"][name] for item in summaries]))
            for name in JOINT_NAMES
        },
        "ik_residual_across_all_steps": {
            "position_mm": residual_summary(predicted_position_all, 10.0),
            "orientation_deg": residual_summary(predicted_orientation_all),
        },
        "ground_truth_fk_eef_ik_roundtrip_across_samples": {
            "mean_mae_deg": float(
                np.mean(
                    [item["ground_truth_fk_eef_ik_roundtrip"]["mean_mae_deg"] for item in summaries]
                )
            ),
            "position_mm": residual_summary(roundtrip_position_all, 10.0),
            "orientation_deg": residual_summary(roundtrip_orientation_all),
        },
        "overview_path": str(overview_path),
        "summaries": summaries,
    }
    aggregate_path = out_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
