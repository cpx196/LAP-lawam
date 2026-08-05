#!/usr/bin/env python3
"""Create an SO101 EEF sidecar dataset from Three_Cubes joint trajectories.

The source LeRobot dataset remains untouched.  Each output parquet has the
same rows and ``index`` values as its source parquet, so it can be joined
directly for plotting or used to build an EEF-native training dataset later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation

from audit_so101_fk_ik import ARM_JOINT_NAMES


EEF_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
DELTA_EEF_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1/derived/so101_eef"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/data/pxchen/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
    )
    return parser.parse_args()


def poses_from_joints(joints: np.ndarray, kinematics: RobotKinematics) -> np.ndarray:
    poses = np.empty((len(joints), 4, 4), dtype=np.float64)
    for row, joint in enumerate(joints):
        poses[row] = kinematics.forward_kinematics(joint).copy()
    return poses


def pose_to_eef(pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    output = np.empty((len(pose), 7), dtype=np.float64)
    output[:, :3] = pose[:, :3, 3]
    output[:, 3:6] = Rotation.from_matrix(pose[:, :3, :3]).as_rotvec()
    output[:, 6] = gripper
    return output


def relative_delta(reference: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
    delta = np.empty(7, dtype=np.float64)
    delta[:3] = target[:3, 3] - reference[:3, 3]
    delta[3:6] = Rotation.from_matrix(reference[:3, :3].T @ target[:3, :3]).as_rotvec()
    delta[6] = gripper
    return delta


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=values.shape[1])


def statistics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    source_files = sorted((args.dataset / "data").glob("chunk-*/*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No source parquet files under {args.dataset / 'data'}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    kinematics = RobotKinematics(
        urdf_path=str(args.urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )
    all_state_eef, all_action_eef, all_state_delta, all_sequence_delta = [], [], [], []
    previous_action_pose: dict[int, np.ndarray] = {}

    for source_path in source_files:
        table = pq.read_table(
            source_path,
            columns=["index", "episode_index", "frame_index", "timestamp", "observation.state", "action"],
        )
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)

        state_pose = poses_from_joints(states, kinematics)
        action_pose = poses_from_joints(actions, kinematics)
        state_eef = pose_to_eef(state_pose, states[:, 5])
        action_eef = pose_to_eef(action_pose, actions[:, 5])
        state_delta = np.empty((len(actions), 7), dtype=np.float64)
        sequence_delta = np.empty((len(actions), 7), dtype=np.float64)
        for row, episode in enumerate(episodes):
            state_delta[row] = relative_delta(state_pose[row], action_pose[row], actions[row, 5])
            reference = previous_action_pose.get(int(episode), state_pose[row])
            sequence_delta[row] = relative_delta(reference, action_pose[row], actions[row, 5])
            previous_action_pose[int(episode)] = action_pose[row].copy()

        relative_path = source_path.relative_to(args.dataset / "data")
        output_path = args.output / "data" / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_table = pa.table(
            {
                "index": pa.array(indices),
                "episode_index": pa.array(episodes),
                "frame_index": pa.array(frame_indices),
                "timestamp": pa.array(timestamps),
                "observation.eef": fixed_list(state_eef),
                "action.eef": fixed_list(action_eef),
                "action.eef_delta_from_state": fixed_list(state_delta),
                "action.eef_delta_sequence": fixed_list(sequence_delta),
            }
        )
        pq.write_table(output_table, output_path, compression="zstd")
        all_state_eef.append(state_eef)
        all_action_eef.append(action_eef)
        all_state_delta.append(state_delta)
        all_sequence_delta.append(sequence_delta)
        print(f"wrote {output_path} ({len(output_table)} rows)")

    state_eef = np.concatenate(all_state_eef)
    action_eef = np.concatenate(all_action_eef)
    state_delta = np.concatenate(all_state_delta)
    sequence_delta = np.concatenate(all_sequence_delta)
    metadata = {
        "source_dataset": str(args.dataset),
        "urdf": str(args.urdf),
        "num_frames": int(len(state_eef)),
        "row_alignment": "Each row is aligned to the source data row by index.",
        "representations": {
            "observation.eef": {
                "layout": EEF_NAMES,
                "meaning": "Absolute gripper-frame pose in SO101 base frame plus absolute gripper motor position.",
            },
            "action.eef": {
                "layout": EEF_NAMES,
                "meaning": "Absolute target gripper-frame pose in SO101 base frame plus absolute gripper motor target.",
            },
            "action.eef_delta_from_state": {
                "layout": DELTA_EEF_NAMES,
                "meaning": "Current observation pose to this row's target pose. Useful for one-step plots.",
            },
            "action.eef_delta_sequence": {
                "layout": DELTA_EEF_NAMES,
                "meaning": "Previous target pose to this row's target pose; the first row of each episode uses observation pose. Use this for action chunks and EEF policy training.",
            },
            "translation": "Difference in SO101 base-frame meters.",
            "rotation": "Rotation vector of reference_R.T @ target_R in radians.",
            "gripper": "Absolute SO101 motor position in degrees.",
        },
        "statistics": {
            "observation.eef": statistics(state_eef),
            "action.eef": statistics(action_eef),
            "action.eef_delta_from_state": statistics(state_delta),
            "action.eef_delta_sequence": statistics(sequence_delta),
        },
    }
    meta_dir = args.output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "representation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "num_frames": metadata["num_frames"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
