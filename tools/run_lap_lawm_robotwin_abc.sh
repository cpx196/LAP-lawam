#!/usr/bin/env bash
# A/B/C RoboTwin ablation for move_pillbottle_pad.
# A: official VLM latent action -> official LaWM
# B: Stage-1 LAP latent action -> official LaWM
# C: Stage-1 LAP latent action -> jointly trained Stage-1 LaWM
set -euo pipefail

ROOT=/data/pxchen/LaWAM
ROBOTWIN=/data/pxchen/RoboTwin
SERVER_PY=/home/pxchen/miniconda3/envs/flashwam/bin/python
SIM_PY=/home/pxchen/miniconda3/envs/robotwin/bin/python
POLICY_CKPT="$ROOT/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
STAGE1_CKPT="$ROOT/outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
STATE_STATS="$ROOT/cache/lap_stage1_task14/state_stats.json"
OUT_ROOT="$ROOT/results/eval_runs/lap_lawm_ablation/abc_5x_clean_random_fp32"
TASK=move_pillbottle_pad
SERVER_GPU=6
SIM_GPU=7

mkdir -p "$OUT_ROOT"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_server() {
    local mode="$1"
    local port="$2"
    local log="$OUT_ROOT/server_${mode}.log"
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy_lap_lawm_ablation \
        --mode "$mode" --host 0.0.0.0 --port "$port" --idle-timeout -1 \
        --ckpt-path "$POLICY_CKPT" --lap-checkpoint "$STAGE1_CKPT" --lap-state-stats "$STATE_STATS" \
        >"$log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 180); do
        if "$SIM_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$port), timeout=1); s.close()" 2>/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -n 80 "$log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "Timed out waiting for ${mode} server" >&2
    exit 1
}

run_split() {
    local label="$1"
    local config="$2"
    local port="$3"
    local run_root="$OUT_ROOT/$label"
    mkdir -p "$run_root"
    CUDA_VISIBLE_DEVICES="$SIM_GPU" \
    ROBOTWIN_PATH="$ROBOTWIN" \
    ROBOTWIN_PYTHON="$SIM_PY" \
    ROBOTWIN_EVAL_ROOT="$run_root" \
    ROBOTWIN_TEST_NUM=5 \
    ROBOTWIN_NUM_SLOTS=1 \
    ROBOTWIN_SAVE_VIDEO=1 \
    ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1 \
    "$SIM_PY" "$ROOT/examples/Robotwin/eval_files/robotwin_batch_bridge.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --overrides \
        --task_name "$TASK" --task_config "$config" --ckpt_setting "$label" --seed 0 \
        --policy_name model2robotwin_interface --host 127.0.0.1 --port "$port" \
        --policy_ckpt_path "$POLICY_CKPT" \
        |& tee "$run_root/bridge.log"
}

for spec in "baseline:A" "lap_official:B" "lap_joint:C"; do
    mode="${spec%%:*}"
    tag="${spec##*:}"
    port=$((11010 + ${#tag}))
    start_server "$mode" "$port"
    run_split "${tag}_${mode}_clean" demo_clean "$port"
    run_split "${tag}_${mode}_randomized" demo_randomized "$port"
    cleanup
    server_pid=""
done
