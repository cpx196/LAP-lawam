#!/usr/bin/env bash
# Expert-only inference-grid KD on GPUs 1,3,4,5; save at 250 and 500.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
TORCHRUN=/home/pxchen/miniconda3/envs/flashwam/bin/torchrun
OUTPUT=${OUTPUT:-$ROOT/outputs/sec284_expert_grid_kd_500step}

mkdir -p "$OUTPUT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1,3,4,5
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1

"$TORCHRUN" --standalone --nproc_per_node=4 \
  tools/train_sec284_expert_grid_kd_ddp.py \
  --grid-cache cache/sec284_task14_inference_grid/train \
  --sec284-checkpoint outputs/sec284_output_kd_primary_2000step/step-002000.pt \
  --output-dir "$OUTPUT" \
  --steps 500 \
  --batch-size 8 \
  --save-every 250 \
  --log-every 10 \
  --lr 1e-7 \
  2>&1 | tee "$OUTPUT/train.log"
