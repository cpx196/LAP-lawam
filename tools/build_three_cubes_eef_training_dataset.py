#!/usr/bin/env python3
"""Build a trainable LeRobot SO101 EEF-delta dataset from aligned sidecars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EEF_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
DELTA_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1"),
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1/derived/so101_eef"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/pxchen/LaWAM/dataset/Three_Cubes_1_EEF"),
    )
    parser.add_argument("--validation-episodes", type=int, default=10)
    return parser.parse_args()


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=values.shape[1])


def statistics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def modality_metadata() -> dict:
    return {
        "state": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.eef",
            },
            "eef_orientation_rotvec": {
                "start": 3,
                "end": 6,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.eef",
            },
            "gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.eef",
            },
        },
        "action": {
            "eef_delta_position": {
                "start": 0,
                "end": 3,
                "absolute": False,
                "dtype": "float32",
                "original_key": "action.eef_delta_sequence",
            },
            "eef_delta_orientation_rotvec": {
                "start": 3,
                "end": 6,
                "absolute": False,
                "dtype": "float32",
                "original_key": "action.eef_delta_sequence",
            },
            "gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action.eef_delta_sequence",
            },
            "eef_delta_from_state_position": {
                "start": 0,
                "end": 3,
                "absolute": False,
                "dtype": "float32",
                "original_key": "action.eef_delta_from_state",
            },
            "eef_delta_from_state_orientation_rotvec": {
                "start": 3,
                "end": 6,
                "absolute": False,
                "dtype": "float32",
                "original_key": "action.eef_delta_from_state",
            },
            "gripper_from_state": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action.eef_delta_from_state",
            },
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
            "right": {"original_key": "observation.images.right"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def main() -> None:
    args = parse_args()
    source_files = sorted((args.source / "data").glob("chunk-*/*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No source parquet files under {args.source / 'data'}")

    all_episodes: list[np.ndarray] = []
    all_state: list[np.ndarray] = []
    all_from_state: list[np.ndarray] = []
    all_sequence: list[np.ndarray] = []

    for source_path in source_files:
        relative_path = source_path.relative_to(args.source / "data")
        sidecar_path = args.sidecar / "data" / relative_path
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing aligned EEF sidecar: {sidecar_path}")

        source = pq.read_table(
            source_path,
            columns=["index", "episode_index", "frame_index", "timestamp", "task_index"],
        )
        sidecar = pq.read_table(
            sidecar_path,
            columns=[
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "observation.eef",
                "action.eef_delta_from_state",
                "action.eef_delta_sequence",
            ],
        )
        for key in ("index", "episode_index", "frame_index"):
            if not np.array_equal(np.asarray(source[key]), np.asarray(sidecar[key])):
                raise ValueError(f"Source/sidecar alignment mismatch for {relative_path}: {key}")

        state = np.asarray(sidecar["observation.eef"].to_pylist(), dtype=np.float32)
        from_state = np.asarray(sidecar["action.eef_delta_from_state"].to_pylist(), dtype=np.float32)
        sequence = np.asarray(sidecar["action.eef_delta_sequence"].to_pylist(), dtype=np.float32)
        output_table = pa.table(
            {
                "index": source["index"],
                "episode_index": source["episode_index"],
                "frame_index": source["frame_index"],
                "timestamp": source["timestamp"],
                "task_index": source["task_index"],
                "observation.eef": fixed_list(state),
                "action.eef_delta_from_state": fixed_list(from_state),
                "action.eef_delta_sequence": fixed_list(sequence),
            }
        )
        output_path = args.output / "data" / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(output_table, output_path, compression="zstd")

        all_episodes.append(np.asarray(source["episode_index"], dtype=np.int64))
        all_state.append(state)
        all_from_state.append(from_state)
        all_sequence.append(sequence)
        print(f"wrote {output_path} ({len(output_table)} rows)")

    episodes = np.concatenate(all_episodes)
    state = np.concatenate(all_state)
    from_state = np.concatenate(all_from_state)
    sequence = np.concatenate(all_sequence)
    unique_episodes = np.unique(episodes)
    if args.validation_episodes <= 0 or args.validation_episodes >= len(unique_episodes):
        raise ValueError(
            f"validation_episodes must be in [1, {len(unique_episodes) - 1}], "
            f"got {args.validation_episodes}."
        )
    validation_ids = unique_episodes[-args.validation_episodes :]
    train_mask = ~np.isin(episodes, validation_ids)

    meta_dir = args.output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source / "meta" / "episodes", meta_dir / "episodes", dirs_exist_ok=True)
    shutil.copy2(args.source / "meta" / "tasks.parquet", meta_dir / "tasks.parquet")

    with (args.source / "meta" / "info.json").open("r", encoding="utf-8") as file:
        info = json.load(file)
    info["robot_type"] = "so101_eef_delta"
    info["features"] = {
        key: value
        for key, value in info["features"].items()
        if key not in {"observation.state", "action"}
    }
    info["features"].update(
        {
            "observation.eef": {"dtype": "float32", "shape": [7], "names": EEF_NAMES},
            "action.eef_delta_from_state": {
                "dtype": "float32",
                "shape": [7],
                "names": DELTA_NAMES,
            },
            "action.eef_delta_sequence": {
                "dtype": "float32",
                "shape": [7],
                "names": DELTA_NAMES,
            },
        }
    )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (meta_dir / "modality.json").write_text(
        json.dumps(modality_metadata(), indent=2), encoding="utf-8"
    )

    train_sequence_and_first = np.concatenate((sequence[train_mask], from_state[train_mask]), axis=0)
    stats = {
        "observation.eef": statistics(state[train_mask]),
        "action.eef_delta_sequence": statistics(train_sequence_and_first),
        "action.eef_delta_from_state": statistics(from_state[train_mask]),
    }
    stats_json = json.dumps(stats, indent=2)
    (meta_dir / "stats.json").write_text(stats_json, encoding="utf-8")
    (meta_dir / "stats_gr00t.json").write_text(stats_json, encoding="utf-8")
    shutil.copy2(args.sidecar / "meta" / "representation.json", meta_dir / "representation.json")

    video_link = args.output / "videos"
    if video_link.exists() or video_link.is_symlink():
        if not video_link.is_symlink() or video_link.resolve() != (args.source / "videos").resolve():
            raise FileExistsError(f"Refusing to replace existing video path: {video_link}")
    else:
        os.symlink((args.source / "videos").resolve(), video_link, target_is_directory=True)

    split_manifest = {
        "strategy": "episode_tail",
        "train_episode_ids": unique_episodes[: -args.validation_episodes].tolist(),
        "validation_episode_ids": validation_ids.tolist(),
        "train_frames": int(train_mask.sum()),
        "validation_frames": int((~train_mask).sum()),
        "normalization_statistics": "Computed from train episodes only.",
    }
    (meta_dir / "split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(json.dumps(split_manifest, indent=2))


if __name__ == "__main__":
    main()
