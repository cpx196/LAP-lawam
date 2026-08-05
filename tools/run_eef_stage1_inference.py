#!/usr/bin/env python3
"""Run sharded Three_Cubes EEF inference and save predictions without scoring."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from plot_three_cubes_action_curve import get_deterministic_sample, inverse_minmax, move_to_device
from starVLA.dataloader import _build_latent_world_collator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy, TrainerUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_samples(dataset: Path, horizon: int, samples_per_episode: int) -> list[dict[str, int]]:
    columns = ["index", "episode_index", "frame_index"]
    tables = [pq.read_table(path, columns=columns) for path in sorted((dataset / "data").glob("chunk-*/*.parquet"))]
    if not tables:
        raise FileNotFoundError(f"No parquet files under {dataset / 'data'}")
    indexes = np.concatenate([np.asarray(table["index"]) for table in tables]).astype(np.int64)
    episodes = np.concatenate([np.asarray(table["episode_index"]) for table in tables]).astype(np.int64)
    frames = np.concatenate([np.asarray(table["frame_index"]) for table in tables]).astype(np.int64)
    entries: list[dict[str, int]] = []
    for episode in np.unique(episodes):
        positions = np.flatnonzero(episodes == episode)
        valid = positions[frames[positions] <= frames[positions].max() - horizon + 1]
        if not len(valid):
            continue
        selected = np.unique(np.linspace(0, len(valid) - 1, samples_per_episode, dtype=np.int64))
        for offset in selected:
            row = int(valid[offset])
            entries.append(
                {
                    "sample_index": int(indexes[row]),
                    "episode_index": int(episodes[row]),
                    "frame_index": int(frames[row]),
                }
            )
    return entries


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    config_path = args.checkpoint_dir / "config.yaml"
    stats_path = args.checkpoint_dir / "dataset_statistics.json"
    checkpoint_path = args.checkpoint_dir / "final_model" / "pytorch_model.pt"
    for path in (config_path, stats_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    all_entries = select_samples(
        args.dataset_root / "Three_Cubes_1",
        args.horizon,
        args.samples_per_episode,
    )
    entries = [entry for entry in all_entries if entry["episode_index"] % args.num_shards == args.shard_id]
    print(
        json.dumps(
            {
                "event": "selection",
                "shard": args.shard_id,
                "samples": len(entries),
                "episodes": len({entry["episode_index"] for entry in entries}),
                "total_samples": len(all_entries),
            }
        ),
        flush=True,
    )

    seed_everything(args.seed)
    cfg = OmegaConf.load(config_path)
    cfg.datasets.vla_data.data_root_dir = str(args.dataset_root)
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.val_num_workers = 0
    cfg.datasets.vla_data.drop_last = False
    cfg.datasets.vla_data.train_split_all = True
    policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
    collator = _build_latent_world_collator(cfg, policy_cfg=policy_cfg, training=False)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="all",
        balance_dataset_weights=True,
        framework_name=cfg.framework.name,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = apply_training_freeze_policy(build_framework(cfg), cfg)
    TrainerUtils.load_finetune_init_weights(model, checkpoint_path, load_pretrained_policy_flow=True)
    model.eval().to(device)
    with stats_path.open(encoding="utf-8") as handle:
        action_stats = json.load(handle)["new_embodiment"]["action"]

    normalized_predictions = []
    raw_predictions = []
    sample_indexes = []
    episode_indexes = []
    frame_indexes = []
    inference_ms = []
    started = time.monotonic()
    for number, entry in enumerate(entries, start=1):
        sample_seed = args.seed + entry["sample_index"]
        seed_everything(sample_seed)
        sample, _meta = get_deterministic_sample(dataset, entry["sample_index"])
        batch = move_to_device(collator([sample]), device)
        batch["embodiment_id"] = torch.full_like(batch["embodiment_id"], 31)
        infer_started = time.perf_counter()
        with torch.inference_mode():
            prediction, _target, mask = model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
        elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
        valid = mask[0].detach().cpu().numpy().astype(bool)[:, 0]
        normalized = prediction[0].detach().float().cpu().numpy()[valid, :7]
        if len(normalized) != args.horizon:
            raise RuntimeError(
                f"sample {entry['sample_index']} returned {len(normalized)} actions, expected {args.horizon}"
            )
        raw = inverse_minmax(normalized, action_stats).astype(np.float32)
        normalized_predictions.append(normalized.astype(np.float32))
        raw_predictions.append(raw)
        sample_indexes.append(entry["sample_index"])
        episode_indexes.append(entry["episode_index"])
        frame_indexes.append(entry["frame_index"])
        inference_ms.append(elapsed_ms)
        if number == 1 or number % 10 == 0 or number == len(entries):
            duration = time.monotonic() - started
            rate = number / duration
            eta = (len(entries) - number) / rate if rate else 0.0
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "shard": args.shard_id,
                        "done": number,
                        "total": len(entries),
                        "sample_index": entry["sample_index"],
                        "infer_ms": round(elapsed_ms, 1),
                        "eta_s": round(eta, 1),
                    }
                ),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        normalized_prediction=np.stack(normalized_predictions),
        raw_prediction=np.stack(raw_predictions),
        sample_index=np.asarray(sample_indexes, dtype=np.int64),
        episode_index=np.asarray(episode_indexes, dtype=np.int64),
        frame_index=np.asarray(frame_indexes, dtype=np.int64),
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
        seed_base=np.asarray(args.seed, dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "shard": args.shard_id,
                "samples": len(entries),
                "output": str(args.output),
                "elapsed_s": round(time.monotonic() - started, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
