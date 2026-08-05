#!/usr/bin/env python3
"""Build current-frame left/right wrist DINO features aligned to Stage-1 cache."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import imagenet_normalize_
from starVLA.dataloader.gr00t_lerobot.data_config import RobotwinEEFDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoTransform
from tools.train_lap_stage1 import DEFAULT_LAM_CKPT, DEFAULT_LAM_YAML, SampleRef, refs_from_manifest


WRIST_KEYS = ("video.cam_left_wrist", "video.cam_right_wrist")


class WristFrameReader:
    def __init__(self, dataset_root: Path) -> None:
        modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=list(WRIST_KEYS))
        }
        self.dataset = LeRobotSingleDataset(
            dataset_root,
            modality_configs,
            EmbodimentTag.AGILEX,
            mode="all",
            video_backend="pyav",
        )
        base_transform = RobotwinEEFDataConfig().transform(image_hw=(256, 256))
        video_transforms = []
        for transform in base_transform.transforms:
            if not isinstance(transform, VideoTransform):
                continue
            transform_copy = copy.deepcopy(transform)
            transform_copy.apply_to = list(WRIST_KEYS)
            video_transforms.append(transform_copy)
        self.video_transform = ComposedModalityTransform(transforms=video_transforms)
        self.video_transform.set_metadata(self.dataset.metadata)
        self.video_transform.eval()

    def get(self, ref: SampleRef) -> torch.Tensor:
        raw: dict[str, Any] = {
            key: self.dataset.get_video(ref.episode, key, ref.base_index)
            for key in WRIST_KEYS
        }
        data = self.video_transform(raw)
        frames = []
        for key in WRIST_KEYS:
            value = data[key]
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(np.asarray(value))
            value = value.to(dtype=torch.float32).div_(255.0)
            imagenet_normalize_(value)
            if value.shape != (1, 3, 256, 256):
                raise RuntimeError(f"Unexpected {key} shape: {tuple(value.shape)}")
            frames.append(value[0])
        return torch.stack(frames, dim=0)  # [2,3,256,256]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/robotwin_merged")
    parser.add_argument("--main-cache-dir", default="cache/lap_stage1_task14")
    parser.add_argument("--output-dir", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--lam-ckpt", default=str(DEFAULT_LAM_CKPT))
    parser.add_argument("--lam-yaml", default=str(DEFAULT_LAM_YAML))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--cache-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_cache_dir = Path(args.main_cache_dir)
    output_dir = Path(args.output_dir)
    manifest_path = main_cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reader = WristFrameReader(Path(args.dataset_root))
    model = load_latent_action_model(args.lam_ckpt, args.lam_yaml).to(device).eval()
    cache_dtype = torch.float32 if args.cache_dtype == "float32" else torch.float16
    splits = (args.split,) if args.split != "all" else ("train", "val", "test")
    print(f"[wrist-cache] device={device} dtype={cache_dtype} views={WRIST_KEYS}", flush=True)

    for split in splits:
        refs = refs_from_manifest(manifest, split)
        if args.max_samples > 0:
            refs = refs[: args.max_samples]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        expected_main_shards = sorted((main_cache_dir / split).glob("shard-*.pt"))
        for shard_start in range(0, len(refs), args.shard_size):
            shard_index = shard_start // args.shard_size
            shard_refs = refs[shard_start : shard_start + args.shard_size]
            path = split_dir / f"shard-{shard_index:05d}.pt"
            if path.exists() and not args.overwrite:
                print(f"[wrist-cache] skip existing {path}", flush=True)
                continue
            left: list[torch.Tensor] = []
            right: list[torch.Tensor] = []
            for batch_start in range(0, len(shard_refs), args.batch_size):
                batch_refs = shard_refs[batch_start : batch_start + args.batch_size]
                videos = torch.stack([reader.get(ref) for ref in batch_refs]).to(device)
                with torch.inference_mode():
                    features = model.extract_vision_features(videos).float()
                if features.shape[1:] != (2, 256, 768):
                    raise RuntimeError(f"Unexpected wrist feature shape: {tuple(features.shape)}")
                left.append(features[:, 0].cpu().to(cache_dtype))
                right.append(features[:, 1].cpu().to(cache_dtype))
            shard = {
                "vision_left_t": torch.cat(left, dim=0),
                "vision_right_t": torch.cat(right, dim=0),
            }
            if len(shard["vision_left_t"]) != len(shard_refs):
                raise RuntimeError("Wrist shard/sample count mismatch")
            if args.max_samples <= 0 and shard_index < len(expected_main_shards):
                main = torch.load(expected_main_shards[shard_index], map_location="cpu", weights_only=True)
                if len(main["vision_t"]) != len(shard_refs):
                    raise RuntimeError(
                        f"Alignment mismatch at {split}/shard-{shard_index:05d}: "
                        f"main={len(main['vision_t'])}, wrist={len(shard_refs)}"
                    )
                del main
            torch.save(shard, path)
            print(
                f"[wrist-cache] wrote {path} samples={len(shard_refs)} "
                f"progress={min(shard_start + len(shard_refs), len(refs))}/{len(refs)}",
                flush=True,
            )

    metadata = {
        "source_manifest": str(manifest_path.resolve()),
        "source_main_cache": str(main_cache_dir.resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "views": list(WRIST_KEYS),
        "frame_offset": 0,
        "feature_shape_per_view": [256, 768],
        "dtype": args.cache_dtype,
        "shard_size": args.shard_size,
        "manifest_seed": manifest.get("seed"),
        "task": manifest.get("task"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[wrist-cache] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
