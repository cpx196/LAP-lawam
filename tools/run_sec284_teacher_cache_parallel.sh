#!/usr/bin/env bash
# Build the complete SEC284 teacher cache on three non-overlapping GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
cache_root="${1:-cache/sec284_task14_teacher}"
log_dir="${2:-outputs/sec284_teacher_cache_logs}"
mkdir -p "$log_dir"

run_builder() {
    local gpu="$1"
    local label="$2"
    shift 2
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n flashwam --no-capture-output \
        python tools/build_sec284_teacher_cache.py --output "$cache_root" \
        --shard-size 128 --teacher-batch-size 4 "$@" 2>&1 | tee -a "$log_dir/$label.log"
}

worker_1() {
    run_builder 1 gpu1 --split train --shard-start 0 --shard-end 63
    run_builder 1 gpu1 --split val --shard-start 0 --shard-end 14
}

worker_4() {
    run_builder 4 gpu4 --split train --shard-start 63 --shard-end 126
    run_builder 4 gpu4 --split test --shard-start 0 --shard-end 14
}

worker_5() {
    run_builder 5 gpu5 --split train --shard-start 126 --shard-end 189
}

worker_1 & pid_1=$!
worker_4 & pid_4=$!
worker_5 & pid_5=$!
cleanup() {
    local status=$?
    if (( status != 0 )); then
        kill "$pid_1" "$pid_4" "$pid_5" 2>/dev/null || true
        wait || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
wait "$pid_1"
wait "$pid_4"
wait "$pid_5"
echo "[teacher] complete: $cache_root"
