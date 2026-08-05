#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${LAWAM_ROOT:-${SCRIPT_DIR}}"
ARIA2_ARGS=(
  -c
  -x 8
  -s 8
  -k 1M
  --file-allocation=none
  --auto-file-renaming=false
  --allow-overwrite=true
  --summary-interval=10
  --console-log-level=notice
  --download-result=full
)

download_file() {
  local repo="$1"
  local file_path="$2"
  local out_dir="$3"
  local out_name="$4"

  mkdir -p "${out_dir}"
  echo
  echo "==> ${repo}/${file_path}"
  echo "    -> ${out_dir}/${out_name}"
  aria2c "${ARIA2_ARGS[@]}" \
    --dir="${out_dir}" \
    --out="${out_name}" \
    "https://huggingface.co/${repo}/resolve/main/${file_path}"
}

download_repo() {
  local repo="$1"
  local out_dir="$2"
  shift 2

  echo
  echo "################################################################################"
  echo "# Downloading ${repo}"
  echo "# Target: ${out_dir}"
  echo "################################################################################"

  local file_path
  for file_path in "$@"; do
    download_file "${repo}" "${file_path}" "${out_dir}" "${file_path}"
  done
}

download_repo "Qwen/Qwen3-VL-2B-Instruct" \
  "${ROOT_DIR}/results/Checkpoints/qwen3_weights" \
  ".gitattributes" \
  "README.md" \
  "chat_template.json" \
  "config.json" \
  "generation_config.json" \
  "merges.txt" \
  "model.safetensors" \
  "preprocessor_config.json" \
  "tokenizer.json" \
  "tokenizer_config.json" \
  "video_preprocessor_config.json" \
  "vocab.json"

download_repo "jialei02/lawam_lam" \
  "${ROOT_DIR}/latent_action_model/logs/dino_large_vae/lam_release" \
  ".gitattributes" \
  "README.md" \
  "checkpoints/pytorch_model.pt" \
  "dino_large_vae.yaml"

download_repo "jialei02/lawam_pretrain" \
  "${ROOT_DIR}/results/Checkpoints/pretrain/lawam_pretrain" \
  ".gitattributes" \
  "README.md" \
  "config.yaml" \
  "dataset_statistics.json" \
  "final_model/pytorch_model.pt"

download_repo "jialei02/lawam_libero_sft_release" \
  "${ROOT_DIR}/results/Checkpoints/libero/lawam_libero_sft_release" \
  ".gitattributes" \
  "README.md" \
  "config.yaml" \
  "dataset_statistics.json" \
  "final_model/pytorch_model.pt"

download_repo "jialei02/lawam_robotwin_sft_release" \
  "${ROOT_DIR}/results/Checkpoints/robotwin/lawam_robotwin_sft_release" \
  ".gitattributes" \
  "README.md" \
  "config.yaml" \
  "dataset_statistics.json" \
  "final_model/pytorch_model.pt"
