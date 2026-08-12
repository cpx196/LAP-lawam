#!/usr/bin/env python3
"""Probe one SEC284 checkpoint/sample entirely on CPU."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.sec284 import SEC284Config, SEC284L


def load_shard(root: Path, split: str, shard: int) -> dict[str, torch.Tensor]:
    return torch.load(
        root / split / f"shard-{shard:05d}.pt", map_location="cpu", weights_only=True
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-cache", default="cache/lap_stage1_task14")
    parser.add_argument("--wrist-cache", default="cache/lap_stage1_task14_wrist")
    parser.add_argument("--teacher-cache", default="cache/sec284_task14_teacher")
    parser.add_argument("--teacher-stats", default="cache/sec284_task14_teacher/train_stats.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    feature_root, wrist_root = Path(args.feature_cache), Path(args.wrist_cache)
    teacher_root = Path(args.teacher_cache)
    manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    refs = manifest["splits"][args.split]["samples"]
    if args.index is None:
        index = random.Random(args.seed).randrange(len(refs)) if args.seed is not None else secrets.randbelow(len(refs))
    else:
        index = args.index
    if not 0 <= index < len(refs):
        raise IndexError(f"index {index} outside split size {len(refs)}")

    metadata = json.loads((teacher_root / "metadata.json").read_text(encoding="utf-8"))
    shard_size = int(metadata["shard_size"])
    shard_index, local_index = divmod(index, shard_size)
    feature = load_shard(feature_root, args.split, shard_index)
    wrist = load_shard(wrist_root, args.split, shard_index)
    teacher_payload = load_shard(teacher_root, args.split, shard_index)
    ref = refs[index]
    if int(teacher_payload["episode_id"][local_index]) != int(ref["episode"]):
        raise RuntimeError("teacher episode identity mismatch")
    if int(teacher_payload["base_index"][local_index]) != int(ref["base_index"]):
        raise RuntimeError("teacher base-index identity mismatch")

    visual = torch.stack(
        [
            feature["vision_t"][local_index],
            wrist["vision_left_t"][local_index],
            wrist["vision_right_t"][local_index],
        ],
        dim=0,
    ).float().unsqueeze(0)
    teacher = teacher_payload["teacher_condition"][local_index].float()
    mask = teacher_payload["teacher_mask"][local_index].bool()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SEC284L(SEC284Config(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["sec284"], strict=True)
    student = model(visual)[0].float()

    selected_student, selected_teacher = student[mask], teacher[mask]
    token_cosine = F.cosine_similarity(selected_student, selected_teacher, dim=-1)
    raw_mse = F.mse_loss(selected_student, selected_teacher)
    flat_cosine = F.cosine_similarity(
        selected_student.flatten().unsqueeze(0), selected_teacher.flatten().unsqueeze(0)
    )[0]
    stats = torch.load(args.teacher_stats, map_location="cpu", weights_only=True)
    mean_baseline_mse = F.mse_loss(stats["position_mean"][mask].float(), selected_teacher)
    result = {
        "checkpoint_step": int(checkpoint["global_step"]),
        "split": args.split,
        "split_index": index,
        "episode_id": int(ref["episode"]),
        "base_index": int(ref["base_index"]),
        "domain": str(ref.get("domain", "unknown")),
        "raw_mse": float(raw_mse),
        "token_cosine_mean": float(token_cosine.mean()),
        "token_cosine_std": float(token_cosine.std(unbiased=False)),
        "token_cosine_p05": float(torch.quantile(token_cosine, 0.05)),
        "flat_cosine": float(flat_cosine),
        "mean_baseline_mse": float(mean_baseline_mse),
        "mse_improvement_vs_mean": float(1.0 - raw_mse / mean_baseline_mse.clamp_min(1e-12)),
        "student_teacher_rms_ratio": float(
            selected_student.square().mean().sqrt()
            / selected_teacher.square().mean().sqrt().clamp_min(1e-12)
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
