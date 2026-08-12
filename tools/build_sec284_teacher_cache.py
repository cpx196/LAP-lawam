#!/usr/bin/env python3
"""Build full train/val/test fixed-prompt VLM conditions for SEC284-L.

This script is intentionally a cache construction utility, not a training
entrypoint.  It runs the official policy only to materialize the frozen target
at the Action Expert semantic-condition boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.server_policy import load_policy_from_checkpoint
from starVLA.model.framework.vlas.lawam import _cuda_autocast
from tools.build_lap_multiview_cache import WRIST_KEYS, WristFrameReader
from tools.sec284_data import EXPECTED_CONDITION_SHAPE, sha256_file
from tools.train_lap_stage1 import Task14RawReader


INSTRUCTION = "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad."


def prompt_hash() -> str:
    return hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def preprocessing_fingerprint() -> dict[str, str]:
    """Record the source code defining raw-camera loading and policy batch preparation."""
    paths = {
        "raw_reader": ROOT / "tools/train_lap_stage1.py",
        "wrist_reader": ROOT / "tools/build_lap_multiview_cache.py",
        "policy_batch_builder": ROOT / "deployment/model_server/server_policy.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def refs_for_split(manifest: dict, split: str) -> list[dict]:
    try:
        refs = manifest["splits"][split]["samples"]
    except KeyError as exc:
        raise KeyError(f"missing split={split!r} in manifest") from exc
    if not refs:
        raise ValueError(f"empty manifest split={split!r}")
    return refs


def cache_split(
    *,
    split: str,
    refs: list[dict],
    output_root: Path,
    main_reader: Task14RawReader,
    wrist_reader: WristFrameReader,
    policy: object,
    shard_size: int,
    batch_size: int,
    overwrite: bool,
    shard_start: int = 0,
    shard_end: int | None = None,
) -> None:
    backend = policy.policy_backend.eval()
    device = torch.device("cuda")
    vlm_dtype = backend.model_cfg.vlm_dtype
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total_shards = (len(refs) + shard_size - 1) // shard_size
    if not 0 <= shard_start <= total_shards:
        raise ValueError(f"shard_start must be in [0,{total_shards}], got {shard_start}")
    if shard_end is None:
        shard_end = total_shards
    if not shard_start <= shard_end <= total_shards:
        raise ValueError(f"shard_end must be in [{shard_start},{total_shards}], got {shard_end}")
    for shard_index in range(shard_start, shard_end):
        start = shard_index * shard_size
        shard_refs = refs[start : start + shard_size]
        path = split_dir / f"shard-{shard_index:05d}.pt"
        if path.exists() and not overwrite:
            existing = torch.load(path, map_location="cpu", weights_only=True)
            expected = len(shard_refs)
            if int(existing["teacher_condition"].shape[0]) != expected:
                raise RuntimeError(f"existing shard length mismatch: {path}")
            if not torch.equal(existing["episode_id"], torch.tensor([r["episode"] for r in shard_refs])):
                raise RuntimeError(f"existing shard episode alignment mismatch: {path}")
            if not torch.equal(existing["base_index"], torch.tensor([r["base_index"] for r in shard_refs])):
                raise RuntimeError(f"existing shard base-index alignment mismatch: {path}")
            print(f"[teacher] {split} skip {path.name}", flush=True)
            continue
        conditions: list[torch.Tensor] = []
        for batch_start in range(0, len(shard_refs), batch_size):
            batch_refs = shard_refs[batch_start : batch_start + batch_size]
            examples = []
            for ref in batch_refs:
                main = np.asarray(
                    main_reader.dataset.get_video(ref["episode"], "video.cam_high", ref["base_index"])
                )[0]
                wrists = [
                    np.asarray(wrist_reader.dataset.get_video(ref["episode"], key, ref["base_index"]))[0]
                    for key in WRIST_KEYS
                ]
                examples.append(
                    {
                        "lang": INSTRUCTION,
                        "primary_image": [main],
                        "wrist_image": wrists,
                        "action_hz": 30.0,
                        "embodiment_id": 1,
                    }
                )
            prepared = policy.policy_infer_batch_builder.build_infer_batch(examples)
            act_query, flow_query = backend._prepare_queries(device=device, vlm_stage_dtype=vlm_dtype)
            with torch.inference_mode(), _cuda_autocast(vlm_dtype):
                vlm_out = backend._run_vlm_stage(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    pixel_values=prepared["pixel_values"],
                    image_grid_thw=prepared["image_grid_thw"],
                    act_placeholder_mask=prepared["act_placeholder_mask"],
                    flow_placeholder_mask=prepared["flow_placeholder_mask"],
                    act_query=act_query,
                    flow_query=flow_query,
                )
                condition = backend.flow._prepare_semantic_condition(
                    h_vlm=vlm_out["h_vlm"], h_lap=None, model_dtype=backend.flow._compute_dtype()
                )
            if tuple(condition.shape[1:]) != EXPECTED_CONDITION_SHAPE:
                raise RuntimeError(f"teacher shape mismatch: {tuple(condition.shape)}")
            conditions.append(condition.detach().cpu().to(torch.float16))
        payload = {
            "teacher_condition": torch.cat(conditions, dim=0),
            "teacher_mask": torch.ones(len(shard_refs), EXPECTED_CONDITION_SHAPE[0], dtype=torch.bool),
            "episode_id": torch.tensor([ref["episode"] for ref in shard_refs], dtype=torch.int64),
            "base_index": torch.tensor([ref["base_index"] for ref in shard_refs], dtype=torch.int64),
        }
        torch.save(payload, path)
        print(f"[teacher] {split} {start + len(shard_refs)}/{len(refs)} -> {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--dataset-root", default="dataset/robotwin_merged")
    parser.add_argument("--official-policy", default="results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt")
    parser.add_argument("--policy-config", default="results/Checkpoints/robotwin/lawam_robotwin_sft_release/config.yaml")
    parser.add_argument("--output", default="cache/sec284_task14_teacher")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--shard-start", type=int, default=0, help="inclusive shard index; only valid with one --split")
    parser.add_argument("--shard-end", type=int, default=None, help="exclusive shard index; only valid with one --split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SEC284 teacher cache construction requires CUDA")
    if args.shard_size < 1 or args.teacher_batch_size < 1:
        raise ValueError("shard and batch sizes must be positive")
    if args.split == "all" and (args.shard_start != 0 or args.shard_end is not None):
        raise ValueError("--shard-start/--shard-end require --split train, val, or test")
    feature_cache = Path(args.feature_cache)
    manifest_path = feature_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    policy_path = Path(args.official_policy)
    config_path = Path(args.policy_config)
    metadata = {
        "instruction": INSTRUCTION,
        "prompt_hash": prompt_hash(),
        "source_policy": str(policy_path.resolve()),
        "source_policy_sha256": sha256_file(policy_path),
        "policy_config": str(config_path.resolve()),
        "policy_config_sha256": sha256_file(config_path),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "camera_preprocessing_fingerprint": preprocessing_fingerprint(),
        "shape_per_sample": list(EXPECTED_CONDITION_SHAPE),
        "dtype": "float16",
        "shard_size": args.shard_size,
        "creation_command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "teacher_condition_boundary": "official VLM output after Action Expert.enc_vlm",
    }
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    metadata["splits"] = {split: len(refs_for_split(manifest, split)) for split in ("train", "val", "test")}
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "prompt_hash",
            "source_policy_sha256",
            "policy_config_sha256",
            "source_manifest_sha256",
            "camera_preprocessing_fingerprint",
            "shape_per_sample",
            "dtype",
            "shard_size",
            "teacher_condition_boundary",
        )
        changed = [key for key in immutable_keys if existing.get(key) != metadata.get(key)]
        if changed:
            raise RuntimeError(
                "existing teacher cache provenance differs in " + ", ".join(changed) + "; use a new output path or --overwrite"
            )
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    policy = load_policy_from_checkpoint(str(policy_path), use_bf16=False, device="cuda")
    main_reader = Task14RawReader(Path(args.dataset_root))
    wrist_reader = WristFrameReader(Path(args.dataset_root))
    for split in splits:
        cache_split(
            split=split,
            refs=refs_for_split(manifest, split),
            output_root=output,
            main_reader=main_reader,
            wrist_reader=wrist_reader,
            policy=policy,
            shard_size=args.shard_size,
            batch_size=args.teacher_batch_size,
            overwrite=args.overwrite,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
        )


if __name__ == "__main__":
    main()
