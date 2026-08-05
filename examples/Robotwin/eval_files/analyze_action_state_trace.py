#!/usr/bin/env python3
"""Analyze joint and EEF-action continuity at RoboTwin chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _split_stats(values: np.ndarray, boundary_mask: np.ndarray) -> dict[str, object]:
    boundary = _stats(values[boundary_mask])
    interior = _stats(values[~boundary_mask])
    boundary_mean = float(boundary["mean"])
    interior_mean = float(interior["mean"])
    return {
        "boundary": boundary,
        "interior": interior,
        "boundary_mean_over_interior_mean": (
            boundary_mean / interior_mean if interior_mean > 0.0 else None
        ),
    }


def _quat_angle_deg(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous = previous / np.clip(np.linalg.norm(previous, axis=1, keepdims=True), 1e-12, None)
    current = current / np.clip(np.linalg.norm(current, axis=1, keepdims=True), 1e-12, None)
    dots = np.abs(np.sum(previous * current, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def _add_boundaries(ax: plt.Axes, boundary_steps: np.ndarray) -> None:
    for step in boundary_steps:
        ax.axvline(step, color="#c43c39", linewidth=0.7, alpha=0.55)


def _plot_joint_positions(
    output_path: Path, steps: np.ndarray, joints: np.ndarray, boundary_steps: np.ndarray
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for arm, ax in enumerate(axes):
        start = arm * 7
        for index in range(7):
            label = f"joint_{index + 1}" if index < 6 else "gripper"
            ax.plot(steps, joints[:, start + index], linewidth=1.0, label=label)
        _add_boundaries(ax, boundary_steps)
        ax.set_ylabel(f"Arm {arm + 1} joint position")
        ax.grid(alpha=0.2)
        ax.legend(ncol=7, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("Measured joint positions (red lines: new action chunk)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_actions(
    output_path: Path, steps: np.ndarray, actions: np.ndarray, boundary_steps: np.ndarray
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    colors = ["#2878b5", "#9ac9db", "#f8ac8c", "#c82423"]
    for arm in range(2):
        offset = arm * 8
        for index, name in enumerate(("x", "y", "z")):
            axes[0, arm].plot(steps, actions[:, offset + index], label=name, linewidth=1.0)
        for index, name in enumerate(("q0", "q1", "q2", "q3")):
            axes[1, arm].plot(
                steps,
                actions[:, offset + 3 + index],
                label=name,
                linewidth=1.0,
                color=colors[index],
            )
        axes[2, arm].plot(steps, actions[:, offset + 7], label="gripper", linewidth=1.0)
        for row in range(3):
            _add_boundaries(axes[row, arm], boundary_steps)
            axes[row, arm].grid(alpha=0.2)
            axes[row, arm].legend(fontsize=8, loc="upper right")
        axes[0, arm].set_title(f"Arm {arm + 1}")
        axes[2, arm].set_xlabel("Simulation step")
    axes[0, 0].set_ylabel("EEF position")
    axes[1, 0].set_ylabel("EEF quaternion")
    axes[2, 0].set_ylabel("Gripper")
    fig.suptitle("Executed EEF actions (red lines: new action chunk)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_transition_metrics(
    output_path: Path,
    transition_steps: np.ndarray,
    joint_delta_l2: np.ndarray,
    joint_slope_change_l2: np.ndarray,
    action_position_delta: list[np.ndarray],
    boundary_steps: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(transition_steps, joint_delta_l2, linewidth=1.0)
    axes[0].set_ylabel("||delta joint||2")
    axes[1].plot(transition_steps[1:], joint_slope_change_l2, linewidth=1.0)
    axes[1].set_ylabel("||delta joint[t]\n- delta joint[t-1]||2")
    for arm, values in enumerate(action_position_delta):
        axes[2].plot(transition_steps, values, linewidth=1.0, label=f"arm_{arm + 1}")
    axes[2].set_ylabel("EEF target position delta")
    axes[2].set_xlabel("Simulation step")
    axes[2].legend()
    for ax in axes:
        _add_boundaries(ax, boundary_steps)
        ax.grid(alpha=0.2)
    fig.suptitle("Per-step changes (red lines: new action chunk)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze(trace_path: Path, output_dir: Path) -> dict[str, object]:
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if len(records) < 3:
        raise ValueError(f"Need at least three trace records, got {len(records)}")

    steps = np.asarray([record["sim_step"] for record in records], dtype=np.int64)
    joints = np.asarray([record["joint_position"] for record in records], dtype=np.float64)
    actions = np.asarray([record["action"] for record in records], dtype=np.float64)
    model_query = np.asarray([record["model_query"] for record in records], dtype=bool)
    if joints.ndim != 2 or actions.shape != (len(records), 16):
        raise ValueError(f"Unexpected trace shapes: joints={joints.shape}, actions={actions.shape}")

    transition_steps = steps[1:]
    boundary_mask = model_query[1:]
    boundary_steps = transition_steps[boundary_mask]
    joint_delta = np.diff(joints, axis=0)
    joint_delta_l2 = np.linalg.norm(joint_delta, axis=1)
    joint_delta_max_abs = np.max(np.abs(joint_delta), axis=1)
    joint_slope_change_l2 = np.linalg.norm(np.diff(joint_delta, axis=0), axis=1)
    arm_joint_indices = np.asarray([*range(6), *range(7, 13)], dtype=np.int64)
    gripper_joint_indices = np.asarray([6, 13], dtype=np.int64)
    arm_joint_delta = joint_delta[:, arm_joint_indices]
    arm_joint_delta_l2 = np.linalg.norm(arm_joint_delta, axis=1)
    arm_joint_slope_change_l2 = np.linalg.norm(np.diff(arm_joint_delta, axis=0), axis=1)
    gripper_joint_delta_max_abs = np.max(np.abs(joint_delta[:, gripper_joint_indices]), axis=1)
    slope_boundary_mask = model_query[2:]

    action_position_delta = []
    action_rotation_delta_deg = []
    action_gripper_delta = []
    for offset in (0, 8):
        action_position_delta.append(
            np.linalg.norm(np.diff(actions[:, offset : offset + 3], axis=0), axis=1)
        )
        action_rotation_delta_deg.append(
            _quat_angle_deg(actions[:-1, offset + 3 : offset + 7], actions[1:, offset + 3 : offset + 7])
        )
        action_gripper_delta.append(np.abs(np.diff(actions[:, offset + 7], axis=0)))

    chunk_records = [record for record in records if record.get("action_chunk") is not None]
    chunk_lengths = [len(record["action_chunk"]) for record in chunk_records]
    report: dict[str, object] = {
        "trace": str(trace_path),
        "num_steps": len(records),
        "joint_dimensions": int(joints.shape[1]),
        "action_dimensions": int(actions.shape[1]),
        "policy_use_state_values": sorted({bool(record["policy_use_state"]) for record in records}),
        "model_query_steps": [int(value) for value in steps[model_query]],
        "chunk_boundary_steps": [int(value) for value in boundary_steps],
        "chunk_lengths": chunk_lengths,
        "joint_position_transition_l2": _split_stats(joint_delta_l2, boundary_mask),
        "joint_position_transition_max_abs": _split_stats(joint_delta_max_abs, boundary_mask),
        "joint_slope_change_l2": _split_stats(joint_slope_change_l2, slope_boundary_mask),
        "arm_joints_only_transition_l2": _split_stats(arm_joint_delta_l2, boundary_mask),
        "arm_joints_only_slope_change_l2": _split_stats(
            arm_joint_slope_change_l2, slope_boundary_mask
        ),
        "measured_gripper_transition_max_abs": _split_stats(
            gripper_joint_delta_max_abs, boundary_mask
        ),
        "eef_action": {},
    }
    eef_report = report["eef_action"]
    assert isinstance(eef_report, dict)
    for arm in range(2):
        eef_report[f"arm_{arm + 1}"] = {
            "position_transition_l2": _split_stats(action_position_delta[arm], boundary_mask),
            "rotation_transition_deg": _split_stats(action_rotation_delta_deg[arm], boundary_mask),
            "gripper_transition_abs": _split_stats(action_gripper_delta[arm], boundary_mask),
        }

    boundary_details = []
    for index in np.flatnonzero(boundary_mask):
        trace_index = index + 1
        boundary_details.append(
            {
                "step": int(steps[trace_index]),
                "joint_delta_l2": float(joint_delta_l2[index]),
                "joint_delta_max_abs": float(joint_delta_max_abs[index]),
                "joint_slope_change_l2": float(joint_slope_change_l2[index - 1]) if index > 0 else None,
                "arm_joints_only_delta_l2": float(arm_joint_delta_l2[index]),
                "arm_joints_only_slope_change_l2": (
                    float(arm_joint_slope_change_l2[index - 1]) if index > 0 else None
                ),
                "measured_gripper_delta_max_abs": float(gripper_joint_delta_max_abs[index]),
                "arm_1_action_position_delta": float(action_position_delta[0][index]),
                "arm_2_action_position_delta": float(action_position_delta[1][index]),
                "arm_1_action_rotation_delta_deg": float(action_rotation_delta_deg[0][index]),
                "arm_2_action_rotation_delta_deg": float(action_rotation_delta_deg[1][index]),
                "arm_1_gripper_delta": float(action_gripper_delta[0][index]),
                "arm_2_gripper_delta": float(action_gripper_delta[1][index]),
            }
        )
    report["boundary_details"] = boundary_details

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunk_boundary_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    )
    _plot_joint_positions(output_dir / "joint_position_trace.png", steps, joints, boundary_steps)
    _plot_actions(output_dir / "eef_action_trace.png", steps, actions, boundary_steps)
    _plot_transition_metrics(
        output_dir / "chunk_boundary_transitions.png",
        transition_steps,
        joint_delta_l2,
        joint_slope_change_l2,
        action_position_delta,
        boundary_steps,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.trace.parent / "trace_analysis"
    report = analyze(args.trace, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
