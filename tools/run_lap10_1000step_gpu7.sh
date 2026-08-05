#!/usr/bin/env bash
set -euo pipefail

cd /data/pxchen/LaWAM

export CUDA_VISIBLE_DEVICES=7
export NO_ALBUMENTATIONS_UPDATE=1

PYTHON=/home/pxchen/miniconda3/envs/flashwam/bin/python
RUN_DIR=/data/pxchen/LaWAM/outputs/lap10_alignment_task14_1000step
TEACHER_CACHE=/data/pxchen/LaWAM/cache/lap10_task14_vlm_teacher_8192
LOG_DIR=/data/pxchen/LaWAM/logs/lap10_alignment_task14_1000step

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

"${PYTHON}" tools/train_lap10_alignment.py \
  --mode build_teacher_cache \
  --teacher-cache "${TEACHER_CACHE}" \
  --max-samples 8192 \
  --teacher-batch-size 4 \
  --cache-shard-size 128 \
  2>&1 | tee "${LOG_DIR}/teacher_cache.log"

"${PYTHON}" tools/train_lap10_alignment.py \
  --mode train \
  --teacher-cache "${TEACHER_CACHE}" \
  --output-dir "${RUN_DIR}" \
  --steps 1000 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --warmup-steps 200 \
  --lr 1e-4 \
  --save-every 250 \
  --log-every 10 \
  --preload-cache \
  --preload-teacher-cache \
  2>&1 | tee "${LOG_DIR}/train.log"
