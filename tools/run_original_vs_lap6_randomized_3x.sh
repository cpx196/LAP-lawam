#!/usr/bin/env bash
# Three randomized RoboTwin episodes for each requested variant:
#   original: released LaWAM (official VLM -> LaWM -> Action Expert)
#   lap6:     LAP6 -> released LaWM, with the official VLM still conditioning
#             the released Action Expert (lap_official)
# Deliberately does not load or invoke any SEC284 checkpoint or server mode.
set -euo pipefail

ROOT=/data/pxchen/LaWAM
ROBOTWIN=/data/pxchen/RoboTwin
SERVER_PY=/home/pxchen/miniconda3/envs/flashwam/bin/python
SIM_PY=/home/pxchen/miniconda3/envs/robotwin/bin/python
POLICY_CKPT="$ROOT/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
LAP6_CKPT="$ROOT/outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
STATE_STATS="$ROOT/cache/lap_stage1_task14/state_stats.json"
OUT_ROOT=${OUT_ROOT:-"$ROOT/results/eval_runs/original_vs_lap6_randomized_seed0_3x_20260813"}
TASK=move_pillbottle_pad
TEST_NUM=${TEST_NUM:-3}
VARIANTS=${VARIANTS:-"original lap6"}
SERVER_WAIT_ATTEMPTS=${SERVER_WAIT_ATTEMPTS:-240}
SERVER_GPU=${SERVER_GPU:-6}
SIM_GPU=${SIM_GPU:-7}

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

wait_for_server() {
    local port="$1"
    local log="$2"
    for _ in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
        if "$SIM_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$port), timeout=1); s.close()" 2>/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -n 160 "$log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "Timed out waiting for policy server on port $port" >&2
    exit 1
}

start_server() {
    local variant="$1"
    local port="$2"
    local log="$OUT_ROOT/server_${variant}.log"
    if [[ "$variant" == "original" ]]; then
        CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy \
            --ckpt_path "$POLICY_CKPT" --port "$port" --idle_timeout -1 \
            >"$log" 2>&1 &
    elif [[ "$variant" == "lap6" ]]; then
        CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$SERVER_PY" -m deployment.model_server.server_policy_lap_lawm_ablation \
            --mode lap_official --host 0.0.0.0 --port "$port" --idle-timeout -1 \
            --ckpt-path "$POLICY_CKPT" --lap-checkpoint "$LAP6_CKPT" \
            --lap-state-stats "$STATE_STATS" >"$log" 2>&1 &
    else
        echo "Unknown variant: $variant" >&2
        exit 2
    fi
    server_pid=$!
    wait_for_server "$port" "$log"
}

run_randomized() {
    local variant="$1"
    local port="$2"
    local run_root="$OUT_ROOT/$variant"
    mkdir -p "$run_root"
    CUDA_VISIBLE_DEVICES="$SIM_GPU" \
    ROBOTWIN_PATH="$ROBOTWIN" \
    ROBOTWIN_PYTHON="$SIM_PY" \
    ROBOTWIN_EVAL_ROOT="$run_root" \
    ROBOTWIN_TEST_NUM="$TEST_NUM" \
    ROBOTWIN_START_SEED="${ROBOTWIN_START_SEED:-100000}" \
    ROBOTWIN_NUM_SLOTS=1 \
    ROBOTWIN_SAVE_VIDEO=1 \
    ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1 \
    "$SIM_PY" "$ROOT/examples/Robotwin/eval_files/robotwin_batch_bridge.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --overrides \
        --task_name "$TASK" --task_config demo_randomized \
        --ckpt_setting "${variant}_randomized_${TEST_NUM}x" --seed 0 \
        --policy_name model2robotwin_interface --host 127.0.0.1 --port "$port" \
        --policy_ckpt_path "$POLICY_CKPT" --replan_steps 36 \
        |& tee "$run_root/eval.log"
}

for variant in $VARIANTS; do
    if [[ "$variant" == "original" ]]; then
        port=11161
    elif [[ "$variant" == "lap6" ]]; then
        port=11162
    else
        echo "Unknown requested variant: $variant" >&2
        exit 2
    fi
    start_server "$variant" "$port"
    run_randomized "$variant" "$port"
    cleanup
done

find "$OUT_ROOT" -type f -name '*.mp4' -print | sort >"$OUT_ROOT/videos.txt"
echo "All requested runs completed. Videos:"
cat "$OUT_ROOT/videos.txt"
