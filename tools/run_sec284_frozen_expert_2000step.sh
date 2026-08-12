#!/usr/bin/env bash
# Launch frozen-Expert SEC284 training. BATCH_SIZE is local/per-GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu_set="${GPU_SET:-1,3,4,5}"
IFS=',' read -r -a gpu_array <<< "$gpu_set"
nproc="${NPROC:-${#gpu_array[@]}}"
output_dir="${OUTPUT_DIR:-outputs/sec284_frozen_expert_behavior_kd_2000step}"
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="$gpu_set"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

echo "[launcher] physical_gpus=$gpu_set nproc=$nproc local_batch=32 global_batch=$((32 * nproc))"
exec /home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
    --standalone --nproc-per-node="$nproc" \
    tools/train_sec284_frozen_expert_ddp.py \
    --steps 2000 \
    --batch-size 32 \
    --save-every 500 \
    --log-every 10 \
    --output-dir "$output_dir" \
    "$@"
