#!/usr/bin/env bash
# LAP10V3 + jointly fine-tuned Action Expert: 5 clean + 5 randomized, FP32.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
ROBOTWIN=/data/pxchen/RoboTwin
SERVER_PY=/home/pxchen/miniconda3/envs/flashwam/bin/python
SIM_PY=/home/pxchen/miniconda3/envs/robotwin/bin/python
POLICY_CKPT="$ROOT/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
JOINT_CKPT="$ROOT/outputs/lap10v3_expert_joint_task14_2000step/lap10v3_expert_step0002000.pt"
REPLAN_STEPS=${REPLAN_STEPS:-36}
OUT_ROOT=${OUT_ROOT:-"$ROOT/results/eval_runs/lap10v3_expert_joint_2000step_replan${REPLAN_STEPS}_5x_clean_random_fp32"}
TASK=move_pillbottle_pad
PORT=${PORT:-11044}
SERVER_GPU=${SERVER_GPU:-0}
SIM_GPU=${SIM_GPU:-1}

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

CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy_lap8_no_vlm \
    --ckpt-path "$POLICY_CKPT" --joint-checkpoint "$JOINT_CKPT" \
    --host 0.0.0.0 --port "$PORT" --idle-timeout -1 \
    >"$OUT_ROOT/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 180); do
    if "$SIM_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT), timeout=1); s.close()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 120 "$OUT_ROOT/server.log" >&2 || true
        exit 1
    fi
    sleep 2
done
if ! kill -0 "$server_pid" 2>/dev/null; then
    exit 1
fi

run_split() {
    local label="$1"
    local task_config="$2"
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
    ROBOTWIN_REPLAN_STEPS="$REPLAN_STEPS" \
    "$SIM_PY" "$ROOT/examples/Robotwin/eval_files/robotwin_batch_bridge.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --overrides \
        --task_name "$TASK" --task_config "$task_config" \
        --ckpt_setting "lap10v3_expert_joint_2000_${label}" --seed 0 \
        --policy_name model2robotwin_interface --host 127.0.0.1 --port "$PORT" \
        --policy_ckpt_path "$POLICY_CKPT" \
        --replan_steps "$REPLAN_STEPS" \
        |& tee "$run_root/bridge.log"
}

run_split clean demo_clean
run_split randomized demo_randomized
