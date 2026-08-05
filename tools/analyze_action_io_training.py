#!/usr/bin/env python3
"""Measure embodiment-specific action encoder/decoder parameter drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYERS = {
    "encoder.W1": "policy_backend.flow.action_encoder.W1",
    "encoder.W2": "policy_backend.flow.action_encoder.W2",
    "decoder.layer1": "policy_backend.flow.action_decoder.layer1",
    "decoder.layer2": "policy_backend.flow.action_decoder.layer2",
    "state_encoder.W1": "policy_backend.flow.enc_state.W1",
    "state_encoder.W2": "policy_backend.flow.enc_state.W2",
}


def summarize(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    value = value.float()
    delta = value - reference
    reference_l2 = float(torch.linalg.vector_norm(reference))
    delta_l2 = float(torch.linalg.vector_norm(delta))
    return {
        "reference_l2": reference_l2,
        "delta_l2": delta_l2,
        "relative_l2_percent": 100.0 * delta_l2 / max(reference_l2, 1e-12),
        "mean_abs_change": float(delta.abs().mean()),
        "max_abs_change": float(delta.abs().max()),
    }


def load_layer(handle, prefix: str, embodiment_id: int) -> torch.Tensor:
    return torch.cat(
        [
            handle.get_tensor(f"{prefix}.W")[embodiment_id].reshape(-1),
            handle.get_tensor(f"{prefix}.b")[embodiment_id].reshape(-1),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step500", type=Path, required=True)
    parser.add_argument("--step1500", type=Path, required=True)
    parser.add_argument("--source-id", type=int, default=18)
    parser.add_argument("--target-id", type=int, default=31)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with safe_open(args.step500, framework="pt", device="cpu") as ckpt500, safe_open(
        args.step1500, framework="pt", device="cpu"
    ) as ckpt1500:
        for name, prefix in LAYERS.items():
            source500 = load_layer(ckpt500, prefix, args.source_id)
            target500 = load_layer(ckpt500, prefix, args.target_id)
            source1500 = load_layer(ckpt1500, prefix, args.source_id)
            target1500 = load_layer(ckpt1500, prefix, args.target_id)
            copied_from_source = not name.startswith("state_encoder")
            results[name] = {
                "initialization": "copied from id 18" if copied_from_source else "independent id 31 state-head initialization",
                "first_500_steps_target_vs_source": summarize(source500, target500) if copied_from_source else None,
                "continued_500_to_1500": summarize(target500, target1500),
                "full_1500_steps_target_vs_source": summarize(source1500, target1500) if copied_from_source else None,
                "source_18_drift_500_to_1500": summarize(source500, source1500),
            }

        encoder_w1_500 = ckpt500.get_tensor(f"{LAYERS['encoder.W1']}.W")[args.target_id]
        encoder_w1_1500 = ckpt1500.get_tensor(f"{LAYERS['encoder.W1']}.W")[args.target_id]
        decoder_w_500 = ckpt500.get_tensor(f"{LAYERS['decoder.layer2']}.W")[args.target_id]
        decoder_w_1500 = ckpt1500.get_tensor(f"{LAYERS['decoder.layer2']}.W")[args.target_id]
        decoder_b_500 = ckpt500.get_tensor(f"{LAYERS['decoder.layer2']}.b")[args.target_id]
        decoder_b_1500 = ckpt1500.get_tensor(f"{LAYERS['decoder.layer2']}.b")[args.target_id]
        dimension_breakdown = {
            "encoder_W1_real_7_inputs": summarize(encoder_w1_500[:7], encoder_w1_1500[:7]),
            "encoder_W1_padded_25_inputs": summarize(encoder_w1_500[7:], encoder_w1_1500[7:]),
            "decoder_layer2_real_7_outputs": summarize(
                torch.cat([decoder_w_500[:, :7].reshape(-1), decoder_b_500[:7]]),
                torch.cat([decoder_w_1500[:, :7].reshape(-1), decoder_b_1500[:7]]),
            ),
            "decoder_layer2_padded_25_outputs": summarize(
                torch.cat([decoder_w_500[:, 7:].reshape(-1), decoder_b_500[7:]]),
                torch.cat([decoder_w_1500[:, 7:].reshape(-1), decoder_b_1500[7:]]),
            ),
        }

    output = {
        "source_id": args.source_id,
        "target_id": args.target_id,
        "interpretation": "target id 31 was copied exactly from source id 18 before step 0",
        "layers": results,
        "dimension_breakdown_500_to_1500": dimension_breakdown,
    }
    (args.out_dir / "action_io_parameter_drift.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    names = [name for name in results if not name.startswith("state_encoder")]
    first = [results[name]["first_500_steps_target_vs_source"]["relative_l2_percent"] for name in names]
    continued = [results[name]["continued_500_to_1500"]["relative_l2_percent"] for name in names]
    full = [results[name]["full_1500_steps_target_vs_source"]["relative_l2_percent"] for name in names]
    x = np.arange(len(names))
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 6), dpi=160)
    axis.bar(x - width, first, width, label="step 0->500 (id31 vs id18)")
    axis.bar(x, continued, width, label="step 500->1500")
    axis.bar(x + width, full, width, label="step 0->1500 (id31 vs id18)")
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylabel("relative L2 parameter change (%)")
    axis.set_title("SO101 id=31 action/state encoder and action decoder training")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "action_io_parameter_drift.png")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
