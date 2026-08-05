#!/usr/bin/env python3
"""Audit SO101 joint data against a URDF with an FK -> IK round trip."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument("--ik-iterations", type=int, default=3)
    parser.add_argument("--position-threshold-mm", type=float, default=10.0)
    parser.add_argument("--orientation-threshold-deg", type=float, default=10.0)
    return parser.parse_args()


def read_joint_data(dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset / 'data'}")

    states, actions = [], []
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        states.extend(table["observation.state"].to_pylist())
        actions.extend(table["action"].to_pylist())
    return np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)


def read_urdf_limits(urdf: Path) -> np.ndarray:
    joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf).getroot().findall("joint")}
    limits = []
    for name in ARM_JOINT_NAMES:
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"Joint {name!r} has no limit in {urdf}")
        limits.append([np.rad2deg(float(limit.attrib["lower"])), np.rad2deg(float(limit.attrib["upper"]))])
    return np.asarray(limits)


def rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    relative_rotation = actual[:3, :3].T @ target[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude()))


def inverse_kinematics_synced(
    kinematics: RobotKinematics,
    current_joint: np.ndarray,
    target_pose: np.ndarray,
    position_weight: float,
    orientation_weight: float,
    iterations: int,
) -> np.ndarray:
    """Work around LeRobot 0.3.3 not updating kinematics after setting its IK seed."""
    solved_joint = current_joint.copy()
    for _ in range(iterations):
        for index, name in enumerate(ARM_JOINT_NAMES):
            kinematics.robot.set_joint(name, np.deg2rad(solved_joint[index]))
        kinematics.robot.update_kinematics()
        solved_joint = kinematics.inverse_kinematics(
            solved_joint,
            target_pose,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
    return solved_joint


def percentile_text(values: np.ndarray, scale: float = 1.0) -> str:
    p50, p95, maximum = np.percentile(values * scale, [50, 95, 100])
    return f"p50={p50:.4f}, p95={p95:.4f}, max={maximum:.4f}"


def print_ranges(label: str, values: np.ndarray, limits: np.ndarray) -> None:
    print(f"\n{label} ranges and URDF-limit violations:")
    for index, name in enumerate(ARM_JOINT_NAMES):
        column = values[:, index]
        lower, upper = limits[index]
        violations = np.count_nonzero((column < lower) | (column > upper))
        print(
            f"  {name:15s} data=[{column.min():8.3f}, {column.max():8.3f}] deg  "
            f"URDF=[{lower:8.3f}, {upper:8.3f}] deg  "
            f"outside={violations:6d}/{len(column)} ({100.0 * violations / len(column):6.2f}%)"
        )


def main() -> None:
    args = parse_args()
    states, actions = read_joint_data(args.dataset)
    if states.shape[1] < 6 or actions.shape[1] < 6:
        raise ValueError(f"Expected 6D state/action, got {states.shape} and {actions.shape}")

    limits = read_urdf_limits(args.urdf)
    print(f"frames: {len(actions)}")
    print_ranges("state", states, limits)
    print_ranges("action", actions, limits)

    sample_count = min(args.max_samples, len(actions))
    sample_indices = np.unique(np.linspace(0, len(actions) - 1, sample_count, dtype=int))
    kinematics = RobotKinematics(
        urdf_path=str(args.urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )

    position_errors = []
    orientation_errors = []
    joint_errors = []
    exact_init_position_errors = []
    sampled_limit_violations = []
    for index in sample_indices:
        target_joint = actions[index]
        # Placo may expose a matrix backed by mutable robot state.
        target_pose = kinematics.forward_kinematics(target_joint).copy()
        sampled_limit_violations.append(
            np.any((target_joint[:5] < limits[:, 0]) | (target_joint[:5] > limits[:, 1]))
        )

        solved_joint = inverse_kinematics_synced(
            kinematics,
            states[index],
            target_pose,
            args.position_weight,
            args.orientation_weight,
            args.ik_iterations,
        )
        reached_pose = kinematics.forward_kinematics(solved_joint).copy()
        position_errors.append(np.linalg.norm(reached_pose[:3, 3] - target_pose[:3, 3]))
        orientation_errors.append(rotation_error_deg(reached_pose, target_pose))
        joint_errors.append(np.abs(solved_joint[:5] - target_joint[:5]))

        exact_init_joint = inverse_kinematics_synced(
            kinematics,
            target_joint,
            target_pose,
            args.position_weight,
            args.orientation_weight,
            args.ik_iterations,
        )
        exact_init_pose = kinematics.forward_kinematics(exact_init_joint).copy()
        exact_init_position_errors.append(
            np.linalg.norm(exact_init_pose[:3, 3] - target_pose[:3, 3])
        )

    position_errors = np.asarray(position_errors)
    orientation_errors = np.asarray(orientation_errors)
    joint_errors = np.asarray(joint_errors)
    exact_init_position_errors = np.asarray(exact_init_position_errors)
    sampled_limit_violations = np.asarray(sampled_limit_violations)

    position_failures = position_errors * 1000.0 > args.position_threshold_mm
    orientation_failures = orientation_errors > args.orientation_threshold_deg
    print(f"\nFK -> IK audit on {len(sample_indices)} evenly spaced frames:")
    print(f"  deployment init (state), position mm: {percentile_text(position_errors, 1000.0)}")
    print(f"  deployment init (state), orientation deg: {percentile_text(orientation_errors)}")
    print(f"  exact-target init, position mm: {percentile_text(exact_init_position_errors, 1000.0)}")
    if np.any(~sampled_limit_violations):
        print(
            "  exact-target init, in-limit actions only, position mm: "
            f"{percentile_text(exact_init_position_errors[~sampled_limit_violations], 1000.0)}"
        )
    print(
        "  sampled actions outside at least one URDF limit: "
        f"{sampled_limit_violations.sum()}/{len(sampled_limit_violations)} "
        f"({100.0 * sampled_limit_violations.mean():.2f}%)"
    )
    print(
        f"  above {args.position_threshold_mm:g} mm: "
        f"{position_failures.sum()}/{len(position_errors)} ({100.0 * position_failures.mean():.2f}%)"
    )
    print(
        f"  above {args.orientation_threshold_deg:g} deg: "
        f"{orientation_failures.sum()}/{len(orientation_errors)} ({100.0 * orientation_failures.mean():.2f}%)"
    )
    print("  absolute joint error versus source action (deg):")
    for joint_index, name in enumerate(ARM_JOINT_NAMES):
        print(f"    {name:15s} {percentile_text(joint_errors[:, joint_index])}")


if __name__ == "__main__":
    main()
