#!/usr/bin/env bash
# Frozen-Expert output-KD: GPUs 3/4/5/6, local batch 32, 2000 steps.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
TORCHRUN=/home/pxchen/miniconda3/envs/flashwam/bin/torchrun
OUTPUT=${OUTPUT:-$ROOT/outputs/sec284_output_kd_primary_2000step}

mkdir -p "$OUTPUT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3,4,5,6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$TORCHRUN" --standalone --nproc_per_node=4 \
  tools/train_sec284_full_inference_grid_kd_ddp.py \
  --grid-cache cache/sec284_task14_inference_grid/train \
  --sec284-checkpoint outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt \
  --output-dir "$OUTPUT" \
  --loss-mode output-primary \
  --uniform-action-weights \
  --condition-gradient-ratio 0.1 \
  --dynamic-weight 0.1 \
  --steps 2000 \
  --batch-size 32 \
  --save-every 500 \
  --log-every 10 \
  --lr 3e-6 \
  --min-lr 3e-7 \
  --warmup-steps 100 \
  2>&1 | tee "$OUTPUT/train.log"
