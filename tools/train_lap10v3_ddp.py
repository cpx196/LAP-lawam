#!/usr/bin/env python3
"""DDP entry point for the LAP10V3 1000-step training plan."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_lap10v2_ddp import main, parse_args


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.model_version = "v3"
    args.init_mode = "scratch"
    args.output_dir = "outputs/lap10v3_task14_1000step" if args.output_dir == "outputs/lap10v2_unified_task14_1000step" else args.output_dir
    main(args)
