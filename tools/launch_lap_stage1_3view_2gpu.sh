#!/usr/bin/env bash
set -euo pipefail

cd /data/pxchen/LaWAM

CACHE_PID="${CACHE_PID:-2338605}"
MAIN_CACHE="cache/lap_stage1_task14"
WRIST_CACHE="cache/lap_stage1_task14_wrist"
OUTPUT_DIR="outputs/lap_stage1_task14_3view_joint_3000step"

echo "[launcher] waiting for complete three-view cache (producer pid=${CACHE_PID})"
while [[ ! -f "${WRIST_CACHE}/metadata.json" ]]; do
  if ! kill -0 "${CACHE_PID}" 2>/dev/null; then
    echo "[launcher] cache producer exited without metadata.json" >&2
    exit 1
  fi
  train_shards=$(find "${WRIST_CACHE}/train" -maxdepth 1 -name 'shard-*.pt' | wc -l)
  echo "[launcher] wrist cache train shards=${train_shards}/189"
  sleep 30
done

for split_spec in train:189 val:14 test:14; do
  split="${split_spec%%:*}"
  expected="${split_spec##*:}"
  actual=$(find "${WRIST_CACHE}/${split}" -maxdepth 1 -name 'shard-*.pt' | wc -l)
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "[launcher] incomplete ${split} wrist cache: ${actual}/${expected}" >&2
    exit 1
  fi
done

echo "[launcher] cache complete; starting FP32 joint LAP+LaWM training on physical GPUs 6,7"
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

exec /home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  tools/train_lap_stage1.py \
  --mode train \
  --phase 2 \
  --cache-dir "${MAIN_CACHE}" \
  --wrist-cache-dir "${WRIST_CACHE}" \
  --view-dropout 0.2 \
  --steps 3000 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --log-every 20 \
  --save-every 1000 \
  --output-dir "${OUTPUT_DIR}"
