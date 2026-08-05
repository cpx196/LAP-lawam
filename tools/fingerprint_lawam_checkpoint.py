#!/usr/bin/env python3
"""Create reproducible SHA256 fingerprints for LaWAM checkpoint subtrees."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def digest(state_dict: dict[str, torch.Tensor], predicate) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    keys = [key for key in sorted(state_dict) if predicate(key)]
    for key in keys:
        tensor = state_dict[key].detach().cpu().contiguous()
        header = f"{key}\t{tensor.dtype}\t{tuple(tensor.shape)}\n".encode("utf-8")
        hasher.update(header)
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest(), len(keys), sum(state_dict[key].numel() for key in keys)


def main() -> None:
    args = parse_args()
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state_dict = loaded.get("state_dict", loaded)
    groups = {
        "vlm": lambda key: key.startswith("policy_backend.vlm."),
        "lam_non_decoder": lambda key: key.startswith("policy_backend.lam.") and ".decoder." not in key,
        "lam_decoder": lambda key: key.startswith("policy_backend.lam.decoder."),
        "flow": lambda key: key.startswith("policy_backend.flow."),
    }
    for name, predicate in groups.items():
        sha, count, numel = digest(state_dict, predicate)
        print(f"{name}\tsha256={sha}\tkeys={count}\tnumel={numel}")


if __name__ == "__main__":
    main()
