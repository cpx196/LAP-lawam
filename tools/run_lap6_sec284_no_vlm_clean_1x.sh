#!/usr/bin/env bash
# Single clean downstream episode for a chosen SEC284 Expert checkpoint.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
ROBOTWIN=/data/pxchen/RoboTwin
SERVER_PY=/home/pxchen/miniconda3/envs/flashwam/bin/python
SIM_PY=/home/pxchen/miniconda3/envs/robotwin/bin/python
POLICY_CKPT="$ROOT/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
STAGE1_CKPT="$ROOT/outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
SEC284_CKPT=${SEC284_CKPT:-"$ROOT/outputs/sec284_output_kd_primary_2000step/step-002000.pt"}
STATE_STATS="$ROOT/cache/lap_stage1_task14/state_stats.json"
EXPERT_CKPT=${EXPERT_CKPT:?Set EXPERT_CKPT to a deploy-compatible Action Expert checkpoint}
LAM_CKPT="$ROOT/latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
LAM_YAML="$ROOT/latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT for this evaluation}
LABEL=${LABEL:-sec284_expert_clean}
ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM:-1}
SERVER_GPU=${SERVER_GPU:-6}
SIM_GPU=${SIM_GPU:-2}
PORT=${PORT:-11054}

mkdir -p "$OUT_ROOT"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy_lap6_sec284_no_vlm \
    --host 0.0.0.0 --port "$PORT" --idle-timeout -1 \
    --ckpt-path "$POLICY_CKPT" \
    --stage1-checkpoint "$STAGE1_CKPT" \
    --sec284-checkpoint "$SEC284_CKPT" \
    --state-stats "$STATE_STATS" \
    --expert-checkpoint "$EXPERT_CKPT" \
    --lam-checkpoint "$LAM_CKPT" --lam-yaml "$LAM_YAML" \
    >"$OUT_ROOT/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
    if "$SIM_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT), timeout=1); s.close()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 160 "$OUT_ROOT/server.log" >&2 || true
        exit 1
    fi
    sleep 2
done
if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 160 "$OUT_ROOT/server.log" >&2 || true
    exit 1
fi

run_root="$OUT_ROOT/clean"
mkdir -p "$run_root"
CUDA_VISIBLE_DEVICES="$SIM_GPU" \
ROBOTWIN_PATH="$ROBOTWIN" \
ROBOTWIN_PYTHON="$SIM_PY" \
ROBOTWIN_EVAL_ROOT="$run_root" \
ROBOTWIN_TEST_NUM="$ROBOTWIN_TEST_NUM" \
ROBOTWIN_NUM_SLOTS=1 \
ROBOTWIN_SAVE_VIDEO=1 \
ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1 \
"$SIM_PY" "$ROOT/examples/Robotwin/eval_files/robotwin_batch_bridge.py" \
    --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --overrides \
    --task_name move_pillbottle_pad --task_config demo_clean \
    --ckpt_setting "$LABEL" --seed 0 \
    --policy_name model2robotwin_interface --host 127.0.0.1 --port "$PORT" \
    --policy_ckpt_path "$POLICY_CKPT" --replan_steps 36 \
    |& tee "$run_root/eval.log"
