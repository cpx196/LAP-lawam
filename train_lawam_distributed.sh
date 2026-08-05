#!/usr/bin/env bash
set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DEFAULT_DATASET_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)/datasets"
DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${DEFAULT_DATASET_ROOT}}"

# ---------------------- Default configuration ----------------------
DEFAULT_CONFIG_YAML="starVLA/config/training/train_libero.yaml"
DEFAULT_ACCELERATE_CONFIG="starVLA/config/accelerate/ddp_bf16.yaml"

CONFIG_YAML="${CONFIG_YAML:-${DEFAULT_CONFIG_YAML}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_yaml=*)
      CONFIG_YAML="${1#*=}"
      shift
      ;;
    --config_yaml)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config_yaml" >&2
        exit 1
      fi
      CONFIG_YAML="$2"
      shift 2
      ;;
    --accelerate_config=*)
      ACCELERATE_CONFIG="${1#*=}"
      shift
      ;;
    --accelerate_config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --accelerate_config" >&2
        exit 1
      fi
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    *)
      if [[ "${1}" != --* && "${CONFIG_YAML}" == "${DEFAULT_CONFIG_YAML}" ]]; then
        CONFIG_YAML="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

# ---------------------- Runtime environment -----------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1
export KMP_BLOCKTIME=0
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLE_STATS=true
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=7200
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export DEEPSPEED_LOG_LEVEL="${DEEPSPEED_LOG_LEVEL:-error}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$(dirname "${TRANSFORMERS_CACHE}")"
fi
export HF_HOME="${HF_HOME:-${DATASET_ROOT_DIR}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

mkdir -p "${TRITON_CACHE_DIR}/autotune"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# ---------------------- Checks and config paths --------------------
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config file not found: ${CONFIG_YAML}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

if [[ "${CONFIG_YAML}" = /* ]]; then
  CONFIG_PATH="${CONFIG_YAML}"
else
  if command -v realpath >/dev/null 2>&1; then
    CONFIG_PATH="$(realpath -m "${CONFIG_YAML}")"
  else
    CONFIG_PATH="${REPO_ROOT}/${CONFIG_YAML}"
  fi
fi

# ---------------------- GPU discovery ------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS=0
  for GPU_ID in "${VISIBLE_GPU_IDS[@]}"; do
    if [[ -n "${GPU_ID//[[:space:]]/}" ]]; then
      NUM_GPUS="$(( NUM_GPUS + 1 ))"
    fi
  done
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"
else
  NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Failed to detect visible GPU count, got: ${NUM_GPUS}" >&2
  exit 1
fi

# ---------------------- Distributed params -------------------------
# Priority:
# 1. Explicit NNODES/NODE_RANK
# 2. Alternate env names NUM_MACHINES/MACHINE_RANK
# 3. Compatibility fallback from WORLD_SIZE/RANK
RAW_WORLD_SIZE="${WORLD_SIZE:-}"
NNODES="${NNODES:-${NUM_MACHINES:-}}"
NODE_RANK="${NODE_RANK:-${MACHINE_RANK:-${RANK:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NNODES}" ]]; then
  if [[ "${RAW_WORLD_SIZE}" =~ ^[0-9]+$ ]] && [[ "${RAW_WORLD_SIZE}" -gt 0 ]]; then
    if [[ "${RAW_WORLD_SIZE}" -gt "${NUM_GPUS}" ]] && (( RAW_WORLD_SIZE % NUM_GPUS == 0 )); then
      NNODES="$(( RAW_WORLD_SIZE / NUM_GPUS ))"
    else
      NNODES="${RAW_WORLD_SIZE}"
    fi
  else
    NNODES=1
  fi
fi

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [[ "${NNODES}" -lt 1 ]]; then
  echo "Invalid nnodes value: ${NNODES}" >&2
  exit 1
fi

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ "${NODE_RANK}" -lt 0 ]]; then
  echo "Invalid node rank value: ${NODE_RANK}" >&2
  exit 1
fi

if [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "Node rank ${NODE_RANK} must be in [0, ${NNODES})" >&2
  exit 1
fi

export WORLD_SIZE="$(( NNODES * NUM_GPUS ))"

# ---------------------- Logging -----------------------------------
TIMESTAMP="${LAWAM_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LAWAM_RUN_TIMESTAMP="${TIMESTAMP}"
LOG_DIR="${REPO_ROOT}/logs/train_lawam_ddp/${TIMESTAMP}"
if [[ "${NODE_RANK}" == "0" ]]; then
  LOG_FILE="${LOG_DIR}/train.log"
else
  LOG_FILE="${LOG_DIR}/train.node${NODE_RANK}.log"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${REPO_ROOT}/wandb"

if [[ "${NODE_RANK}" == "0" ]]; then
  cp -f "${CONFIG_PATH}" "${LOG_DIR}/$(basename "${CONFIG_PATH}")"
fi

echo "Starting LaWAM distributed training"
echo "Config YAML: ${CONFIG_PATH}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "Local GPU count: ${NUM_GPUS}"
echo "Num machines: ${NNODES}"
echo "Node rank: ${NODE_RANK}"
echo "Main process IP: ${MASTER_ADDR}"
echo "Main process port: ${MASTER_PORT}"
echo "World size: ${WORLD_SIZE}"
echo "HF cache root: ${HF_HOME}"
echo "Log dir: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --machine_rank "${NODE_RANK}" \
  --num_machines "${NNODES}" \
  --num_processes "${WORLD_SIZE}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_PATH}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
