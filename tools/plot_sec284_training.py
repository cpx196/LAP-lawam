#!/usr/bin/env python3
"""Plot SEC284 train/validation and variance-probe curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_PATTERN = re.compile(
    r"\[step\] (?P<step>\d+)/\d+ total=(?P<total>[\d.e+-]+) "
    r"raw_mse=(?P<raw>[\d.e+-]+) white_mse=(?P<white>[\d.e+-]+) "
    r"cosine_loss=(?P<cos>[\d.e+-]+) lr=(?P<lr>[\d.e+-]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--variance-json", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    step_rows = [
        {"step": int(match["step"]), **{key: float(match[key]) for key in ("total", "raw", "white", "cos", "lr")}}
        for line in Path(args.train_log).read_text(encoding="utf-8").splitlines()
        if (match := STEP_PATTERN.search(line))
    ]
    epoch_rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    variance_rows = sorted(
        [json.loads(Path(path).read_text()) for path in args.variance_json],
        key=lambda row: row["checkpoint_step"],
    )
    if not step_rows or not epoch_rows:
        raise RuntimeError("missing train or validation metrics")

    steps = [row["step"] for row in step_rows]
    val_steps = [row["global_step"] for row in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, [row["total"] for row in step_rows], label="train total", linewidth=1.5)
    ax.plot(steps, [row["raw"] for row in step_rows], label="train raw MSE", linewidth=1.2)
    ax.plot(steps, [row["white"] for row in step_rows], label="train whitened MSE", linewidth=1.2)
    ax.scatter(val_steps, [row["val"]["total"] for row in epoch_rows], label="val total", s=22, zorder=3)
    ax.set_yscale("log")
    ax.set_title("Distillation losses (log scale)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, [row["raw"] for row in step_rows], label="train raw MSE", color="tab:blue")
    ax.plot(val_steps, [row["val"]["raw_mse"] for row in epoch_rows], "o-", label="val raw MSE", color="tab:orange")
    ax.set_xlim(400, 3050)
    ax.set_ylim(0.045, 0.13)
    ax.set_title("Late-stage raw MSE and LR")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("raw MSE")
    ax.grid(alpha=0.25)
    lr_ax = ax.twinx()
    lr_ax.plot(steps, [row["lr"] for row in step_rows], "--", color="0.45", alpha=0.8, label="learning rate")
    lr_ax.set_ylabel("learning rate", color="0.35")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = lr_ax.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, [1.0 - row["cos"] for row in step_rows], label="train token cosine")
    ax.plot(val_steps, [1.0 - row["val"]["cosine"] for row in epoch_rows], "o-", label="val token cosine")
    ax.axhline(0.95, color="tab:red", linestyle=":", label="acceptance threshold 0.95")
    ax.set_ylim(0.5, 0.97)
    ax.set_title("Token cosine similarity")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cosine similarity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if variance_rows:
        variance_steps = [row["checkpoint_step"] for row in variance_rows]
        ax.plot(variance_steps, [row["student_teacher_std_ratio"] for row in variance_rows], "o-", label="std ratio")
        ax.plot(variance_steps, [row["centered_dynamic_r2"] for row in variance_rows], "o-", label="centered dynamic R²")
        ax.plot(variance_steps, [row["variance_map_cosine"] for row in variance_rows], "o-", label="variance-map cosine")
        ax.axhline(1.0, color="0.3", linestyle=":", linewidth=1)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Cross-sample dynamic fidelity (fixed 32-test probe)")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("metric")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("SEC284-L distillation: 3000-step run", fontsize=15)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
