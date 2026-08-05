#!/usr/bin/env python3
"""Score a live-observation seed sweep against training starts and a known successful chunk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation

from audit_so101_fk_ik import ARM_JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--success-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def integrate(deltas: np.ndarray) -> tuple[np.ndarray, Rotation, np.ndarray]:
    positions = np.cumsum(deltas[:, :3], axis=0)
    matrices = []
    current = np.eye(3)
    for delta in deltas:
        current = current @ Rotation.from_rotvec(delta[3:6]).as_matrix()
        matrices.append(current.copy())
    return positions, Rotation.from_matrix(np.stack(matrices)), deltas[:, 6]


def load_training(dataset: Path, kinematics: RobotKinematics) -> tuple[np.ndarray, Rotation, np.ndarray]:
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    tables = [pq.read_table(path, columns=columns) for path in sorted((dataset / "data").glob("chunk-*/*.parquet"))]
    episodes = np.concatenate([np.asarray(table["episode_index"]) for table in tables]).astype(np.int64)
    frames = np.concatenate([np.asarray(table["frame_index"]) for table in tables]).astype(np.int64)
    states = np.concatenate([np.asarray(table["observation.state"].to_pylist()) for table in tables]).astype(np.float64)
    actions = np.concatenate([np.asarray(table["action"].to_pylist()) for table in tables]).astype(np.float64)

    all_positions, all_rotations, all_grippers = [], [], []
    for episode in np.unique(episodes):
        indexes = np.flatnonzero(episodes == episode)
        by_frame = {int(frames[index]): int(index) for index in indexes}
        if any(frame not in by_frame for frame in range(36)):
            continue
        start = by_frame[0]
        start_pose = kinematics.forward_kinematics(states[start])
        positions, matrices, grippers = [], [], []
        for frame in range(36):
            index = by_frame[frame]
            target = kinematics.forward_kinematics(actions[index])
            positions.append(target[:3, 3] - start_pose[:3, 3])
            matrices.append(start_pose[:3, :3].T @ target[:3, :3])
            grippers.append(actions[index, 5])
        all_positions.append(positions)
        all_rotations.append(matrices)
        all_grippers.append(grippers)
    return (
        np.asarray(all_positions),
        Rotation.from_matrix(np.asarray(all_rotations).reshape(-1, 3, 3)),
        np.asarray(all_grippers),
    )


def rotation_errors(candidate: Rotation, references: Rotation, episodes: int) -> np.ndarray:
    candidate_matrices = candidate.as_matrix()
    reference_matrices = references.as_matrix().reshape(episodes, 36, 3, 3)
    relative = np.einsum("tji,etjk->etik", candidate_matrices, reference_matrices)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(episodes, 36) * 180.0 / np.pi


def main() -> None:
    args = parse_args()
    shards = sorted(args.sweep_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No sweep shards in {args.sweep_dir}")
    parts = [np.load(path) for path in shards]
    seeds = np.concatenate([part["seeds"] for part in parts])
    predictions = np.concatenate([part["clipped_prediction"] for part in parts])
    raw_predictions = np.concatenate([part["raw_prediction"] for part in parts])
    clipped_count = np.concatenate([part["clipped_count"] for part in parts])
    order = np.argsort(seeds)
    seeds, predictions, raw_predictions, clipped_count = (
        seeds[order], predictions[order], raw_predictions[order], clipped_count[order]
    )
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in sweep")

    kinematics = RobotKinematics(
        urdf_path=str(args.urdf), target_frame_name="gripper_frame_link", joint_names=ARM_JOINT_NAMES
    )
    train_pos, train_rot_flat, train_grip = load_training(args.dataset, kinematics)
    episodes = len(train_pos)
    train_final = train_pos[:, -1] * 1000.0
    covariance = np.cov(train_final.T) + np.eye(3) * 1e-6
    covariance_inv = np.linalg.inv(covariance)
    train_final_mean = train_final.mean(axis=0)

    success_rows = [json.loads(line) for line in args.success_log.open()]
    success_delta = np.asarray(
        [row["eef_delta"] for row in success_rows if row.get("event") == "action" and row.get("chunk") == 0][:36],
        dtype=np.float64,
    )
    if success_delta.shape != (36, 7):
        raise ValueError(f"Expected successful first chunk [36,7], got {success_delta.shape}")
    success_pos, success_rot, success_grip = integrate(success_delta)

    rows = []
    for seed, prediction, raw, clipped in zip(seeds, predictions, raw_predictions, clipped_count):
        pos, rot, grip = integrate(prediction.astype(np.float64))
        pos_error = np.linalg.norm(train_pos - pos[None], axis=2) * 1000.0
        ori_error = rotation_errors(rot, train_rot_flat, episodes)
        grip_error = np.abs(train_grip - grip[None])
        pos_mean = pos_error.mean(axis=1)
        ori_mean = ori_error.mean(axis=1)
        grip_mean = grip_error.mean(axis=1)
        demo_composite = pos_mean + ori_mean + 0.25 * grip_mean
        nearest = int(np.argmin(demo_composite))

        success_pos_error = np.linalg.norm(pos - success_pos, axis=1) * 1000.0
        success_ori_error = (
            (rot.inv() * success_rot).magnitude() * 180.0 / np.pi
        )
        success_grip_error = np.abs(grip - success_grip)
        success_composite = float(
            success_pos_error.mean() + success_ori_error.mean() + 0.25 * success_grip_error.mean()
        )
        final_mm = pos[-1] * 1000.0
        centered = final_mm - train_final_mean
        mahalanobis = float(np.sqrt(centered @ covariance_inv @ centered))
        envelope_violations = int(
            np.sum((pos < train_pos.min(axis=0)) | (pos > train_pos.max(axis=0)))
        )
        rows.append(
            {
                "seed": int(seed),
                "demo_score": float(demo_composite[nearest]),
                "nearest_demo_episode": nearest,
                "demo_position_mean_mm": float(pos_mean[nearest]),
                "demo_orientation_mean_deg": float(ori_mean[nearest]),
                "demo_gripper_mean_deg": float(grip_mean[nearest]),
                "success_score": success_composite,
                "success_position_mean_mm": float(success_pos_error.mean()),
                "success_orientation_mean_deg": float(success_ori_error.mean()),
                "success_gripper_mean_deg": float(success_grip_error.mean()),
                "final_x_mm": float(final_mm[0]),
                "final_y_mm": float(final_mm[1]),
                "final_z_mm": float(final_mm[2]),
                "final_mahalanobis": mahalanobis,
                "trajectory_envelope_violations": envelope_violations,
                "max_step_translation_mm": float(np.linalg.norm(prediction[:, :3], axis=1).max() * 1000.0),
                "clipped_values": int(clipped[:6].sum()),
                "raw_final_x_mm": float(raw[:, 0].sum() * 1000.0),
                "raw_final_y_mm": float(raw[:, 1].sum() * 1000.0),
                "raw_final_z_mm": float(raw[:, 2].sum() * 1000.0),
            }
        )

    demo_order = {row_index: rank for rank, row_index in enumerate(np.argsort([row["demo_score"] for row in rows]))}
    success_order = {
        row_index: rank for rank, row_index in enumerate(np.argsort([row["success_score"] for row in rows]))
    }
    mahal_order = {
        row_index: rank for rank, row_index in enumerate(np.argsort([row["final_mahalanobis"] for row in rows]))
    }
    for index, row in enumerate(rows):
        row["combined_rank"] = float(demo_order[index] + success_order[index] + 0.5 * mahal_order[index])
    ranked = sorted(rows, key=lambda row: (row["combined_rank"], row["demo_score"]))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ranking.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    summary = {
        "seeds": len(seeds),
        "training_episodes": episodes,
        "training_final_xyz_mm": {
            "min": train_final.min(axis=0).tolist(),
            "p05": np.percentile(train_final, 5, axis=0).tolist(),
            "median": np.median(train_final, axis=0).tolist(),
            "p95": np.percentile(train_final, 95, axis=0).tolist(),
            "max": train_final.max(axis=0).tolist(),
        },
        "success_final_xyz_mm": (success_pos[-1] * 1000.0).tolist(),
        "top_candidates": ranked[:10],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fixed-observation seed sweep",
        "",
        f"Scored {len(seeds)} deterministic seeds against {episodes} training first chunks and the 2026-07-31 successful first chunk.",
        "",
        "| rank | seed | combined | demo score | demo pos mm | demo ori deg | success pos mm | final xyz mm | envelope violations |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {rank} | {row['seed']} | {row['combined_rank']:.1f} | {row['demo_score']:.2f} | "
            f"{row['demo_position_mean_mm']:.2f} | {row['demo_orientation_mean_deg']:.2f} | "
            f"{row['success_position_mean_mm']:.2f} | "
            f"[{row['final_x_mm']:.1f}, {row['final_y_mm']:.1f}, {row['final_z_mm']:.1f}] | "
            f"{row['trajectory_envelope_violations']} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
