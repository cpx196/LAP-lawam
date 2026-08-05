#!/usr/bin/env python3
"""Build SO101 delta-EEF action statistics from joint state/action pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation

from audit_so101_fk_ik import ARM_JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/data/pxchen/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1/meta/so101_eef_delta_stats.json"),
    )
    return parser.parse_args()


def load_joint_pairs(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, episodes = [], [], []
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset / 'data'}")
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        states.extend(table["observation.state"].to_pylist())
        actions.extend(table["action"].to_pylist())
        episodes.extend(table["episode_index"].to_pylist())
    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        np.asarray(episodes, dtype=np.int64),
    )


def compute_pose_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    kinematics: RobotKinematics,
) -> tuple[np.ndarray, np.ndarray]:
    state_poses = np.empty((len(actions), 4, 4), dtype=np.float64)
    action_poses = np.empty((len(actions), 4, 4), dtype=np.float64)
    for index, (state, action) in enumerate(zip(states, actions)):
        state_poses[index] = kinematics.forward_kinematics(state).copy()
        action_poses[index] = kinematics.forward_kinematics(action).copy()
    return state_poses, action_poses


def compute_chunk_delta_eef(
    state_poses: np.ndarray,
    action_poses: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    first_delta = np.empty((len(actions), 7), dtype=np.float64)
    first_delta[:, :3] = action_poses[:, :3, 3] - state_poses[:, :3, 3]
    first_delta[:, 3:6] = Rotation.from_matrix(
        np.transpose(state_poses[:, :3, :3], (0, 2, 1)) @ action_poses[:, :3, :3]
    ).as_rotvec()
    first_delta[:, 6] = actions[:, 5]

    successive_delta = np.empty((len(actions), 7), dtype=np.float64)
    successive_delta[:] = np.nan
    same_episode = episodes[1:] == episodes[:-1]
    target_indices = np.flatnonzero(same_episode) + 1
    previous_indices = target_indices - 1
    successive_delta[target_indices, :3] = (
        action_poses[target_indices, :3, 3] - action_poses[previous_indices, :3, 3]
    )
    successive_delta[target_indices, 3:6] = Rotation.from_matrix(
        np.transpose(action_poses[previous_indices, :3, :3], (0, 2, 1))
        @ action_poses[target_indices, :3, :3]
    ).as_rotvec()
    successive_delta[target_indices, 6] = actions[target_indices, 5]

    episode_ranges = []
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        if len(indices):
            episode_ranges.append((int(indices[0]), int(indices[-1]) + 1))

    output_size = sum(
        min(horizon, episode_end - start)
        for episode_start, episode_end in episode_ranges
        for start in range(episode_start, episode_end)
    )
    result = np.empty((output_size, 7), dtype=np.float64)
    cursor = 0
    for episode_start, episode_end in episode_ranges:
        for start in range(episode_start, episode_end):
            stop = min(start + horizon, episode_end)
            result[cursor] = first_delta[start]
            cursor += 1
            future_count = stop - start - 1
            if future_count > 0:
                result[cursor : cursor + future_count] = successive_delta[start + 1 : stop]
                cursor += future_count
    if cursor != output_size:
        raise RuntimeError(f"Delta EEF allocation mismatch: wrote {cursor}, expected {output_size}")
    return result


def statistics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "mask": [True] * values.shape[1],
    }


def main() -> None:
    args = parse_args()
    states, actions, episodes = load_joint_pairs(args.dataset)
    kinematics = RobotKinematics(
        urdf_path=str(args.urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )
    state_poses, action_poses = compute_pose_arrays(states, actions, kinematics)
    action_horizon = 36
    delta_eef = compute_chunk_delta_eef(
        state_poses,
        action_poses,
        actions,
        episodes,
        action_horizon,
    )
    payload = {
        "representation": {
            "layout": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            "translation": "target_position - current_position in SO101 base frame, meters",
            "rotation": "rotvec(current_R.T @ target_R), radians",
            "gripper": "absolute SO101 motor position, degrees",
            "source_pair": "chunk step 0 is observation.state -> action[0]; later steps are action[t-1] -> action[t]",
            "action_horizon": action_horizon,
            "normalization": "min_max to [-1, 1]; normalized gripper is inverted for Bridge-head compatibility",
        },
        "num_frames": int(len(actions)),
        "num_chunk_action_tokens": int(len(delta_eef)),
        "urdf": str(args.urdf),
        "action": statistics(delta_eef),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
