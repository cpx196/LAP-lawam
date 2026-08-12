#!/usr/bin/env bash
# Follow the three parallel SEC284 teacher-cache builders.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${1:-$repo_root/outputs/sec284_teacher_cache_logs}"
tail -n "${TAIL_LINES:-40}" -F "$log_dir/gpu1.log" "$log_dir/gpu4.log" "$log_dir/gpu5.log"
