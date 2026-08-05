#!/usr/bin/env python3
"""Compare saved EEF-delta predictions with FK labels rebuilt from raw joints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_so101_fk_ik import ARM_JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(root: Path) -> dict[str, np.ndarray]:
    files = sorted(root.glob("predictions_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"No prediction shards in {root}")
    keys = ("normalized_prediction", "raw_prediction", "sample_index", "episode_index", "frame_index")
    parts = {key: [] for key in keys}
    for path in files:
        with np.load(path) as shard:
            for key in keys:
                parts[key].append(shard[key])
    result = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(result["sample_index"])
    result = {key: value[order] for key, value in result.items()}
    if len(np.unique(result["sample_index"])) != len(result["sample_index"]):
        raise ValueError("Prediction sample indexes are not unique")
    return result


def load_raw_dataset(root: Path) -> dict[str, np.ndarray]:
    columns = ["index", "episode_index", "frame_index", "observation.state", "action"]
    tables = [pq.read_table(path, columns=columns) for path in sorted((root / "data").glob("chunk-*/*.parquet"))]
    if not tables:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    output = {
        "index": np.concatenate([np.asarray(table["index"]) for table in tables]).astype(np.int64),
        "episode_index": np.concatenate([np.asarray(table["episode_index"]) for table in tables]).astype(np.int64),
        "frame_index": np.concatenate([np.asarray(table["frame_index"]) for table in tables]).astype(np.int64),
        "state": np.concatenate([np.asarray(table["observation.state"].to_pylist()) for table in tables]).astype(np.float64),
        "action": np.concatenate([np.asarray(table["action"].to_pylist()) for table in tables]).astype(np.float64),
    }
    if not np.array_equal(output["index"], np.arange(len(output["index"]))):
        raise ValueError("Raw dataset indexes must be contiguous and equal to row positions")
    return output


def fk_pose(kinematics: RobotKinematics, joint: np.ndarray) -> np.ndarray:
    # The recording follower calibration already expresses these values in the
    # URDF convention, so the correct motor-to-URDF offset is zero.
    return kinematics.forward_kinematics(joint).copy()


def relative_delta(reference: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
    result = np.empty(7, dtype=np.float64)
    result[:3] = target[:3, 3] - reference[:3, 3]
    result[3:6] = Rotation.from_matrix(reference[:3, :3].T @ target[:3, :3]).as_rotvec()
    result[6] = gripper
    return result


def rebuild_chunk(
    raw: dict[str, np.ndarray], kinematics: RobotKinematics, start: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    stop = start + horizon
    if stop > len(raw["index"]):
        raise IndexError(f"Chunk [{start}, {stop}) exceeds dataset")
    if not np.all(raw["episode_index"][start:stop] == raw["episode_index"][start]):
        raise ValueError(f"Chunk at {start} crosses an episode boundary")
    state_pose = fk_pose(kinematics, raw["state"][start])
    action_poses = [fk_pose(kinematics, joint) for joint in raw["action"][start:stop]]
    gt = np.empty((horizon, 7), dtype=np.float64)
    reference = state_pose
    for step, target in enumerate(action_poses):
        gt[step] = relative_delta(reference, target, raw["action"][start + step, 5])
        reference = target
    state_eef = np.empty(7, dtype=np.float64)
    state_eef[:3] = state_pose[:3, 3]
    state_eef[3:6] = Rotation.from_matrix(state_pose[:3, :3]).as_rotvec()
    state_eef[6] = raw["state"][start, 5]
    return state_eef, gt


def integrate(start_eef: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = start_eef[:3].copy()
    orientation = Rotation.from_rotvec(start_eef[3:6]).as_matrix()
    positions, orientations, grippers = [], [], []
    for delta in deltas:
        position = position + delta[:3]
        orientation = orientation @ Rotation.from_rotvec(delta[3:6]).as_matrix()
        positions.append(position.copy())
        orientations.append(orientation.copy())
        grippers.append(delta[6])
    return np.asarray(positions), np.asarray(orientations), np.asarray(grippers)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def phase_metrics(phase: int, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    select = arrays["phase"] == phase
    return {
        "samples": int(np.sum(select)),
        "normalized_mse": float(np.mean(arrays["normalized_squared_error"][select])),
        "delta_translation_vector_mae_mm": float(np.mean(arrays["delta_position_error_mm"][select])),
        "delta_rotation_angle_mae_deg": float(np.mean(arrays["delta_rotation_error_deg"][select])),
        "delta_gripper_mae_deg": float(np.mean(arrays["delta_gripper_error_deg"][select])),
        "trajectory_position_error_mm": summary(arrays["position_error_mm"][select]),
        "trajectory_rotation_error_deg": summary(arrays["rotation_error_deg"][select]),
        "final_position_error_mm": summary(arrays["position_error_mm"][select, -1]),
        "final_rotation_error_deg": summary(arrays["rotation_error_deg"][select, -1]),
    }


def plot_step_curves(path: Path, arrays: dict[str, np.ndarray]) -> None:
    steps = np.arange(1, arrays["position_error_mm"].shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160, sharex=True)
    series = (
        (arrays["delta_position_error_mm"], "Delta translation vector error", "mm"),
        (arrays["delta_rotation_error_deg"], "Delta rotation angle error", "deg"),
        (arrays["position_error_mm"], "Accumulated position error", "mm"),
        (arrays["rotation_error_deg"], "Accumulated orientation error", "deg"),
    )
    for axis, (values, title, unit) in zip(axes.reshape(-1), series):
        median = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25, 75], axis=0)
        p95 = np.percentile(values, 95, axis=0)
        axis.fill_between(steps, q25, q75, alpha=0.25, label="25-75%")
        axis.plot(steps, median, linewidth=2, label="median")
        axis.plot(steps, p95, "--", linewidth=1.5, label="p95")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("action step")
    axes[1, 1].set_xlabel("action step")
    fig.suptitle("Three_Cubes: prediction error across a 36-step chunk")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path: Path, metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    lines = [
        "# Three_Cubes EEF 阶段二离线比较",
        "",
        "## 比较口径",
        "",
        "阶段一保存的 400 个预测与原始 `Three_Cubes_1` joint 数据重新计算的 GT 比较。GT 不读取旧 EEF sidecar；使用采集 follower calibration 的零 offset 关节表示直接做 FK。每个 36-step chunk 的第 0 步为 `observation.state(t) -> action(t)`，后续为 `action(t+k-1) -> action(t+k)`。本测试完全离线，不运行 IK，也不连接机械臂。",
        "",
        "## 总体结果",
        "",
        f"- 样本：`{metrics['samples']}`；动作点：`{metrics['action_points']}`",
        f"- 归一化 MSE：`{overall['normalized_mse']:.6f}`",
        f"- 单步 delta 平移向量误差均值：`{overall['delta_translation_vector_mae_mm']:.3f} mm`",
        f"- 单步 delta 旋转角误差均值：`{overall['delta_rotation_angle_mae_deg']:.3f} deg`",
        f"- 单步夹爪误差均值：`{overall['delta_gripper_mae_deg']:.3f} deg`",
        f"- 累计位置误差：中位数 `{overall['trajectory_position_error_mm']['median']:.3f} mm`，p95 `{overall['trajectory_position_error_mm']['p95']:.3f} mm`",
        f"- 第 36 步位置误差：中位数 `{overall['final_position_error_mm']['median']:.3f} mm`，p95 `{overall['final_position_error_mm']['p95']:.3f} mm`",
        f"- 第 36 步姿态误差：中位数 `{overall['final_rotation_error_deg']['median']:.3f} deg`，p95 `{overall['final_rotation_error_deg']['p95']:.3f} deg`",
        "",
        "![36-step error curves](error_by_step.png)",
        "",
        "## 分阶段结果",
        "",
        "每条 episode 依次取起始、约 1/3、约 2/3、末段四个位置。",
        "",
        "| 阶段 | 样本 | normalized MSE | delta 平移 (mm) | delta 旋转 (deg) | 第36步位置中位数/p95 (mm) | 第36步姿态中位数/p95 (deg) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in range(4):
        item = metrics["phases"][str(phase)]
        lines.append(
            f"| {phase + 1} | {item['samples']} | {item['normalized_mse']:.6f} | "
            f"{item['delta_translation_vector_mae_mm']:.3f} | {item['delta_rotation_angle_mae_deg']:.3f} | "
            f"{item['final_position_error_mm']['median']:.3f} / {item['final_position_error_mm']['p95']:.3f} | "
            f"{item['final_rotation_error_deg']['median']:.3f} / {item['final_rotation_error_deg']['p95']:.3f} |"
        )
    lines.extend([
        "",
        "完整数值见 `metrics.json`，逐样本结果见 `per_sample.csv`。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.prediction_dir)
    raw = load_raw_dataset(args.dataset)
    if predictions["raw_prediction"].shape[1:] != (args.horizon, 7):
        raise ValueError(f"Unexpected prediction shape: {predictions['raw_prediction'].shape}")

    with (args.checkpoint_dir / "dataset_statistics.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)["new_embodiment"]["action"]
    low = np.asarray(stats["min"], dtype=np.float64)[:7]
    high = np.asarray(stats["max"], dtype=np.float64)[:7]
    scale = high - low
    if np.any(scale == 0):
        raise ValueError("Action normalization range contains zero-width dimensions")

    kinematics = RobotKinematics(
        urdf_path=str(args.urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )
    gt_chunks, starts, phases = [], [], []
    for sample, episode, frame in zip(
        predictions["sample_index"], predictions["episode_index"], predictions["frame_index"]
    ):
        if raw["episode_index"][sample] != episode or raw["frame_index"][sample] != frame:
            raise ValueError(f"Prediction metadata mismatch at sample {sample}")
        start, gt = rebuild_chunk(raw, kinematics, int(sample), args.horizon)
        starts.append(start)
        gt_chunks.append(gt)
        episode_frames = raw["frame_index"][raw["episode_index"] == episode]
        valid_count = int(episode_frames.max()) - args.horizon + 2
        anchors = np.unique(np.linspace(0, valid_count - 1, 4, dtype=np.int64))
        matches = np.flatnonzero(anchors == frame)
        if len(matches) != 1:
            raise ValueError(f"Cannot assign phase for episode={episode}, frame={frame}, anchors={anchors}")
        phases.append(int(matches[0]))
    gt = np.asarray(gt_chunks)
    starts = np.asarray(starts)
    prediction = predictions["raw_prediction"].astype(np.float64)
    gt_normalized = (gt - low) / scale * 2.0 - 1.0

    count, horizon, _ = prediction.shape
    delta_position_error = np.linalg.norm(prediction[:, :, :3] - gt[:, :, :3], axis=2) * 1000.0
    gt_delta_rotation = Rotation.from_rotvec(gt[:, :, 3:6].reshape(-1, 3)).as_matrix()
    pred_delta_rotation = Rotation.from_rotvec(prediction[:, :, 3:6].reshape(-1, 3)).as_matrix()
    delta_rotation_error = np.rad2deg(
        Rotation.from_matrix(np.transpose(gt_delta_rotation, (0, 2, 1)) @ pred_delta_rotation).magnitude()
    ).reshape(count, horizon)
    delta_gripper_error = np.abs(prediction[:, :, 6] - gt[:, :, 6])

    position_error = np.empty((count, horizon), dtype=np.float64)
    rotation_error = np.empty((count, horizon), dtype=np.float64)
    for index in range(count):
        gt_pos, gt_rot, _ = integrate(starts[index], gt[index])
        pred_pos, pred_rot, _ = integrate(starts[index], prediction[index])
        position_error[index] = np.linalg.norm(pred_pos - gt_pos, axis=1) * 1000.0
        rotation_error[index] = np.rad2deg(
            Rotation.from_matrix(np.transpose(gt_rot, (0, 2, 1)) @ pred_rot).magnitude()
        )

    arrays = {
        "phase": np.asarray(phases),
        "normalized_squared_error": (predictions["normalized_prediction"].astype(np.float64) - gt_normalized) ** 2,
        "delta_position_error_mm": delta_position_error,
        "delta_rotation_error_deg": delta_rotation_error,
        "delta_gripper_error_deg": delta_gripper_error,
        "position_error_mm": position_error,
        "rotation_error_deg": rotation_error,
    }
    all_phases = np.ones(count, dtype=bool)
    overall = {
        "normalized_mse": float(np.mean(arrays["normalized_squared_error"])),
        "delta_translation_vector_mae_mm": float(np.mean(delta_position_error)),
        "delta_rotation_angle_mae_deg": float(np.mean(delta_rotation_error)),
        "delta_gripper_mae_deg": float(np.mean(delta_gripper_error)),
        "trajectory_position_error_mm": summary(position_error),
        "trajectory_rotation_error_deg": summary(rotation_error),
        "final_position_error_mm": summary(position_error[all_phases, -1]),
        "final_rotation_error_deg": summary(rotation_error[all_phases, -1]),
    }
    metrics = {
        "prediction_dir": str(args.prediction_dir),
        "source_dataset": str(args.dataset),
        "checkpoint_dir": str(args.checkpoint_dir),
        "urdf": str(args.urdf),
        "motor_to_urdf_offset_deg": [0.0] * 5,
        "gt_rebuilt_from_raw_joint": True,
        "samples": count,
        "action_points": count * horizon,
        "overall": overall,
        "phases": {str(phase): phase_metrics(phase, arrays) for phase in range(4)},
        "per_step": {
            "position_error_median_mm": np.median(position_error, axis=0).tolist(),
            "position_error_p95_mm": np.percentile(position_error, 95, axis=0).tolist(),
            "rotation_error_median_deg": np.median(rotation_error, axis=0).tolist(),
            "rotation_error_p95_deg": np.percentile(rotation_error, 95, axis=0).tolist(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "episode_index", "frame_index", "phase", "normalized_mse", "final_position_error_mm", "final_rotation_error_deg"])
        for index in range(count):
            writer.writerow([
                int(predictions["sample_index"][index]),
                int(predictions["episode_index"][index]),
                int(predictions["frame_index"][index]),
                int(phases[index]) + 1,
                float(np.mean(arrays["normalized_squared_error"][index])),
                float(position_error[index, -1]),
                float(rotation_error[index, -1]),
            ])
    np.savez_compressed(args.output_dir / "comparison_arrays.npz", gt_delta=gt, start_eef=starts, **arrays)
    plot_step_curves(args.output_dir / "error_by_step.png", arrays)
    write_report(args.output_dir / "REPORT.md", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
