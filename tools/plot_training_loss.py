#!/usr/bin/env python3
"""Plot LaWAM training and validation losses from one or more text logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metric(block: str, name: str) -> float | None:
    match = re.search(rf"'{re.escape(name)}':\s*([0-9.eE+-]+)", block)
    return float(match.group(1)) if match else None


def parse_logs(paths: list[Path]) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    pattern = re.compile(r"\[RANK 0\] Step\s+(\d+), Loss:.*?(?=\[RANK 0\] Step|\Z)", re.S)
    for path in paths:
        text = path.read_text(errors="replace").replace("\r", "\n")
        for match in pattern.finditer(text):
            step = int(match.group(1))
            block = match.group(0)
            values = {
                key: value
                for key in ("train_loss_total", "train_loss_flow", "val_mse_score")
                if (value := metric(block, key)) is not None
            }
            records[step] = {**records.get(step, {}), **values}
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = parse_logs(args.logs)
    if not records:
        raise RuntimeError("No loss records found")

    fig, (train_ax, val_ax) = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)
    for key, label in (("train_loss_total", "train total"), ("train_loss_flow", "train flow/action")):
        points = [(step, row[key]) for step, row in sorted(records.items()) if key in row]
        train_ax.plot([x for x, _ in points], [y for _, y in points], marker="o", markersize=3, label=label)
    train_ax.set_ylabel("loss")
    train_ax.set_yscale("log")
    train_ax.grid(alpha=0.25)
    train_ax.legend()
    train_ax.set_title("Three_Cubes EEF-delta training loss")

    val = [(step, row["val_mse_score"]) for step, row in sorted(records.items()) if "val_mse_score" in row]
    val_ax.plot([x for x, _ in val], [y for _, y in val], marker="o", linewidth=2, label="validation action MSE")
    for step, value in val:
        val_ax.annotate(f"{value:.4g}", (step, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    val_ax.set_xlabel("optimizer step")
    val_ax.set_ylabel("MSE")
    val_ax.grid(alpha=0.25)
    val_ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
