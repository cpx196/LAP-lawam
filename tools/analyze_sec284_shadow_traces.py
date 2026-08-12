#!/usr/bin/env python3
"""Aggregate fixed-instruction SEC284 shadow traces and plot divergence by replan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_split(path: Path) -> list[dict]:
    summary = path / "traces" / "summary.jsonl"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    return [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "shadow_trace_metrics.png"
    splits = {name: load_split(args.root / name) for name in ("clean", "randomized")}

    report = {}
    for name, rows in splits.items():
        worst = max(rows, key=lambda row: float(row["grid_velocity_mse"]))
        report[name] = {
            "calls": len(rows),
            "condition_mse": mean(rows, "condition_prefix_mse"),
            "condition_cosine": mean(rows, "condition_prefix_cosine"),
            "action_mse": mean(rows, "action_mse"),
            "xyz_mse": mean(rows, "xyz_mse"),
            "gripper_sign_agreement": mean(rows, "gripper_sign_agreement"),
            "grid_velocity_mse": mean(rows, "grid_velocity_mse"),
            "worst_grid_call": int(worst["call"]),
            "worst_grid_velocity_mse": float(worst["grid_velocity_mse"]),
            "mean_grid_velocity_mse_per_step": np.mean(
                [row["grid_velocity_mse_per_step"] for row in rows], axis=0
            ).tolist(),
        }
    (args.root / "shadow_trace_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, rows in splits.items():
        calls = [row["call"] for row in rows]
        axes[0].plot(calls, [row["condition_prefix_cosine"] for row in rows], "o-", label=name)
        axes[1].semilogy(calls, [row["action_mse"] for row in rows], "o-", label=name)
        axes[1].semilogy(calls, [row["grid_velocity_mse"] for row in rows], "s--", label=f"{name} grid")
        per_step = np.mean([row["grid_velocity_mse_per_step"] for row in rows], axis=0)
        axes[2].semilogy(np.arange(1, len(per_step) + 1), per_step, "o-", label=name)
    axes[0].set(title="Condition cosine", xlabel="Replan call", ylabel="cosine")
    axes[1].set(title="Downstream divergence", xlabel="Replan call", ylabel="MSE")
    axes[2].set(title="Teacher-grid velocity error", xlabel="Flow step", ylabel="MSE")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(json.dumps(report, indent=2))
    print(f"plot={output}")


if __name__ == "__main__":
    main()
