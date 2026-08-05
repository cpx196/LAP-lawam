#!/usr/bin/env python3
"""Run deterministic LaWAM seed sweeps on one frozen SO101 observation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-script", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--mixed-precision", choices=("fp32", "bf16", "fp16", "no"), default="fp32")
    return parser.parse_args()


def load_deploy(path: Path):
    spec = importlib.util.spec_from_file_location("lawam_snapshot_deploy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import deployment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    deploy = load_deploy(args.deploy_script)

    with np.load(args.snapshot) as snap:
        joint = np.asarray(snap["joint_state"], dtype=np.float64)
        frames = {key: np.asarray(snap[key], dtype=np.uint8) for key in deploy.ROBOT_CAM_KEYS}
    if joint.shape != (6,):
        raise ValueError(f"Expected 6D joint state, got {joint.shape}")

    kinematics = RobotKinematics(
        urdf_path=str(args.urdf),
        target_frame_name="gripper_frame_link",
        joint_names=deploy.ARM_JOINT_NAMES,
    )
    pose = kinematics.forward_kinematics(joint)
    eef_state = np.empty(7, dtype=np.float32)
    eef_state[:3] = pose[:3, 3]
    eef_state[3:6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    eef_state[6] = joint[5]

    policy_args = SimpleNamespace(
        policy_ckpt_path=str(args.checkpoint),
        unnorm_key=None,
        device=args.device,
        mixed_precision=args.mixed_precision,
        guidance_scale=None,
        num_inference_steps=args.num_inference_steps,
        actions_per_chunk=36,
    )
    policy, action_stats, state_stats = deploy.load_policy(policy_args)
    obs = {key: value for key, value in frames.items()}
    example = deploy.build_lawam_example(obs, args.task, eef_state, state_stats, 30.0)
    quantiles = deploy.load_eef_delta_quantiles(args.representation)

    all_seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    seeds = [seed for index, seed in enumerate(all_seeds) if index % args.num_shards == args.shard_id]
    raw_predictions: list[np.ndarray] = []
    clipped_predictions: list[np.ndarray] = []
    clipped_counts: list[np.ndarray] = []
    inference_ms: list[float] = []
    started = time.monotonic()
    for number, seed in enumerate(seeds, start=1):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        infer_started = time.perf_counter()
        raw = deploy.predict_chunk(policy, example, policy_args, action_stats)
        clipped, mask = deploy.limit_eef_deltas_for_chunk(
            raw, quantiles, first_delta_reference="observation"
        )
        elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
        raw_predictions.append(raw)
        clipped_predictions.append(clipped)
        clipped_counts.append(mask.sum(axis=0))
        inference_ms.append(elapsed_ms)
        print(
            json.dumps(
                {
                    "event": "prediction",
                    "shard": args.shard_id,
                    "done": number,
                    "total": len(seeds),
                    "seed": seed,
                    "infer_ms": round(elapsed_ms, 1),
                    "final_xyz_mm": np.round(clipped[:, :3].sum(axis=0) * 1000.0, 2).tolist(),
                }
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        seeds=np.asarray(seeds, dtype=np.int64),
        raw_prediction=np.stack(raw_predictions),
        clipped_prediction=np.stack(clipped_predictions),
        clipped_count=np.stack(clipped_counts),
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
        joint_state=joint,
        eef_state=eef_state,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "shard": args.shard_id,
                "samples": len(seeds),
                "elapsed_s": round(time.monotonic() - started, 1),
                "output": str(args.output),
                "eef_state": eef_state.tolist(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
