#!/usr/bin/env bash
# Paired 1+1 baseline check for the exact seeds used by the SEC284 smoke test.
# official: real VLM condition -> released LaWAM Action Expert.
# lap8: original LAP8 no-VLM condition -> released LaWAM Action Expert.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
ROBOTWIN=/data/pxchen/RoboTwin
SERVER_PY=/home/pxchen/miniconda3/envs/flashwam/bin/python
SIM_PY=/home/pxchen/miniconda3/envs/robotwin/bin/python
POLICY_CKPT="$ROOT/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
LAP8_CKPT="$ROOT/outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt"
OUT_ROOT="$ROOT/results/eval_runs/sec284_paired_baselines_seed0_1x"
TASK=move_pillbottle_pad
SERVER_GPU=${SERVER_GPU:-1}
SIM_GPU=${SIM_GPU:-5}

mkdir -p "$OUT_ROOT"
server_pid=""

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap cleanup EXIT INT TERM

start_server() {
    local variant="$1"
    local port="$2"
    local log="$OUT_ROOT/server_${variant}.log"
    if [[ "$variant" == "official" ]]; then
        CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy \
            --ckpt_path "$POLICY_CKPT" --port "$port" --idle_timeout -1 \
            >"$log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy_lap8_no_vlm \
            --ckpt-path "$POLICY_CKPT" --lap8-checkpoint "$LAP8_CKPT" \
            --host 0.0.0.0 --port "$port" --idle-timeout -1 \
            >"$log" 2>&1 &
    fi
    server_pid=$!
    for _ in $(seq 1 240); do
        if "$SIM_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$port), timeout=1); s.close()" 2>/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -n 120 "$log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "Timed out waiting for ${variant} policy server" >&2
    exit 1
}

run_split() {
    local variant="$1"
    local label="$2"
    local task_config="$3"
    local port="$4"
    local run_root="$OUT_ROOT/${variant}_${label}"
    mkdir -p "$run_root"
    CUDA_VISIBLE_DEVICES="$SIM_GPU" \
    ROBOTWIN_PATH="$ROBOTWIN" \
    ROBOTWIN_PYTHON="$SIM_PY" \
    ROBOTWIN_EVAL_ROOT="$run_root" \
    ROBOTWIN_TEST_NUM=1 \
    ROBOTWIN_NUM_SLOTS=1 \
    ROBOTWIN_SAVE_VIDEO=1 \
    ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1 \
    "$SIM_PY" "$ROOT/examples/Robotwin/eval_files/robotwin_batch_bridge.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --overrides \
        --task_name "$TASK" --task_config "$task_config" \
        --ckpt_setting "${variant}_${label}" --seed 0 \
        --policy_name model2robotwin_interface --host 127.0.0.1 --port "$port" \
        --policy_ckpt_path "$POLICY_CKPT" --replan_steps 36 \
        |& tee "$run_root/eval.log"
}

for variant in official lap8; do
    port=11049
    [[ "$variant" == "lap8" ]] && port=11050
    start_server "$variant" "$port"
    run_split "$variant" clean demo_clean "$port"
    run_split "$variant" randomized demo_randomized "$port"
    cleanup
done
