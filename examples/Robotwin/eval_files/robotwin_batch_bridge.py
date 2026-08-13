#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.Robotwin.eval_files.model2robotwin_interface import (  # noqa: E402
    flatten_robotwin_endpose_state,
    get_model,
)
from examples.Robotwin.eval_files.robotwin_eval_common import (  # noqa: E402
    build_run_group,
    build_run_tag,
    derive_ckpt_alias,
    ensure_runtime_paths,
    load_config_with_overrides,
    normalize_bool_env,
    sanitize_component,
    write_json,
)


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if isinstance(data, str) and not getattr(stream, "isatty", lambda: False)():
                stream.write(strip_ansi(data))
            else:
                stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@dataclass
class CandidateInfo:
    seed: int
    episode_info: dict[str, Any]


@dataclass
class ObservationRequest:
    seed: int
    episode_id: int
    instruction: str
    requested_at: float
    must_query: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched Robotwin evaluation for a single task.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_save_dir(usr_args: dict[str, Any], current_time: str) -> Path:
    eval_root = Path(
        os.getenv(
            "ROBOTWIN_EVAL_ROOT",
            "results/eval_runs/robotwin",
        )
    ).expanduser().resolve()
    ckpt_alias = derive_ckpt_alias(usr_args["policy_ckpt_path"])
    run_group = build_run_group(ckpt_alias, str(usr_args["task_config"]))
    run_tag = build_run_tag(current_time)
    task_name = sanitize_component(str(usr_args["task_name"]))
    return eval_root / run_group / run_tag / "tasks" / task_name


def write_eval_meta(save_dir: Path, usr_args: dict[str, Any], current_time: str, num_slots: int) -> None:
    meta = {
        "timestamp": current_time,
        "save_dir": str(save_dir),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "task_name": usr_args.get("task_name"),
        "task_config": usr_args.get("task_config"),
        "ckpt_setting": usr_args.get("ckpt_setting"),
        "policy_name": usr_args.get("policy_name"),
        "instruction_type": usr_args.get("instruction_type"),
        "seed": usr_args.get("seed"),
        "policy_ckpt_path": usr_args.get("policy_ckpt_path"),
        "host": usr_args.get("host"),
        "port": usr_args.get("port"),
        "planner_backend": os.getenv("ROBOTWIN_PLANNER_BACKEND", "auto"),
        "test_num": os.getenv("ROBOTWIN_TEST_NUM"),
        "num_slots": int(num_slots),
        "argv": sys.argv,
        "usr_args": usr_args,
    }
    write_json(save_dir / "meta.json", meta)


def resolve_num_slots() -> int:
    raw_value = os.getenv("ROBOTWIN_NUM_SLOTS", os.getenv("ROBOTWIN_BATCH_SIZE", "1"))
    num_slots = int(raw_value)
    if num_slots <= 0:
        raise ValueError(f"`ROBOTWIN_NUM_SLOTS` must be > 0, got {num_slots}.")
    return num_slots


def resolve_batch_wait_sec() -> float:
    return max(0.0, float(os.getenv("ROBOTWIN_BATCH_WAIT_MS", "0")) / 1000.0)


def resolve_seed_log_every() -> int:
    return max(0, int(os.getenv("ROBOTWIN_LOG_SEED_SEARCH_EVERY", "5")))


def resolve_skip_get_obs_within_replan() -> bool:
    return normalize_bool_env(os.getenv("ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN"), default=True)


def _plan_action_request(
    *,
    model: Any,
    slot_id: int,
    instruction: str,
    skip_get_obs_within_replan: bool,
) -> tuple[bool, bool]:
    if not skip_get_obs_within_replan:
        return True, False
    needs_query = bool(model.needs_query(slot_id=slot_id, task_description=instruction))
    return needs_query, needs_query


def _observation_requires_model_query(
    *,
    model: Any,
    slot_id: int,
    instruction: str,
    must_query: bool,
) -> bool:
    if must_query:
        return True
    return bool(model.needs_query(slot_id=slot_id, task_description=instruction))


def prepare_task_args(
    usr_args: dict[str, Any],
    *,
    save_dir: Path,
    robotwin_path: str,
) -> tuple[dict[str, Any], str]:
    ensure_runtime_paths(robotwin_path, STARVLA_ROOT)

    from envs import CONFIGS_PATH  # type: ignore
    from script.eval_policy import get_camera_config, get_embodiment_config  # type: ignore
    import yaml

    task_config = str(usr_args["task_config"])
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = str(usr_args["task_name"])
    task_args["task_config"] = task_config
    task_args["ckpt_setting"] = str(usr_args["ckpt_setting"])
    task_args["save_path"] = str(save_dir)
    task_args["eval_mode"] = True
    task_args["save_data"] = False

    embodiment_type = task_args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name: str) -> str:
        robot_file = embodiment_types[embodiment_name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment files for {embodiment_name}")
        return robot_file

    head_camera_type = task_args["camera"]["head_camera_type"]
    camera_cfg = get_camera_config(head_camera_type)
    task_args["head_camera_h"] = camera_cfg["h"]
    task_args["head_camera_w"] = camera_cfg["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unexpected embodiment config: {embodiment_type}")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])

    usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])

    requested_eval_video = usr_args.get("eval_video_log", None)
    if requested_eval_video is not None:
        save_video = normalize_bool_env(requested_eval_video, default=False)
    elif "ROBOTWIN_SAVE_VIDEO" in os.environ:
        save_video = normalize_bool_env(os.environ.get("ROBOTWIN_SAVE_VIDEO"), default=False)
    else:
        save_video = False
    task_args["eval_video_log"] = bool(save_video)
    task_args["eval_video_save_dir"] = str(save_dir) if task_args["eval_video_log"] else None

    return task_args, f"{camera_cfg['w']}x{camera_cfg['h']}"


def _select_instruction(
    *,
    task_name: str,
    instruction_type: str,
    episode_info: dict[str, Any],
    seed: int,
) -> str:
    from generate_episode_instructions import generate_episode_descriptions  # type: ignore

    descriptions = generate_episode_descriptions(task_name, [episode_info], 1)
    if not descriptions:
        return ""
    instruction_pool = descriptions[0].get(str(instruction_type), [])
    if not instruction_pool:
        instruction_pool = descriptions[0].get("unseen", []) or descriptions[0].get("seen", [])
    if not instruction_pool:
        return ""
    if len(instruction_pool) == 1:
        return str(instruction_pool[0])
    rng = np.random.default_rng(int(seed))
    selected_idx = int(rng.integers(0, len(instruction_pool)))
    return str(instruction_pool[selected_idx])


def _configure_warp_cache_for_slot(*, task_args: dict[str, Any], slot_id: int) -> None:
    del task_args, slot_id
    cache_root = Path(
        os.environ.get(
            "ROBOTWIN_SHARED_WARP_CACHE_DIR",
            str(Path.home() / ".cache" / "robotwin_warp_shared"),
        )
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(cache_root)


def _open_ffmpeg(video_dir: Path, video_size: str, episode_id: int) -> subprocess.Popen[bytes]:
    video_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_dir / f"episode{episode_id}.mp4"),
        ],
        stdin=subprocess.PIPE,
    )


def _try_seed_once(
    task_env: Any,
    task_args: dict[str, Any],
    *,
    seed: int,
    episode_probe_id: int,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    from envs.utils.create_actor import UnStableError  # type: ignore

    probe_args = dict(task_args)
    probe_args["render_freq"] = 0
    probe_args["eval_video_save_dir"] = None
    try:
        task_env.setup_demo(now_ep_num=episode_probe_id, seed=seed, is_test=True, **probe_args)
        episode_info = task_env.play_once()
        task_env.close_env()
    except UnStableError:
        try:
            task_env.close_env()
        except Exception:
            pass
        return False, None, None
    except Exception as exc:
        try:
            task_env.close_env()
        except Exception:
            pass
        return False, None, traceback.format_exc() or str(exc)

    valid = bool(task_env.plan_success and task_env.check_success())
    return valid, episode_info, None


def _slot_worker_main(
    slot_id: int,
    conn,
    *,
    task_name: str,
    instruction_type: str,
    task_args: dict[str, Any],
    video_size: str,
    robotwin_path: str,
    env_action_type: str,
) -> None:
    _configure_warp_cache_for_slot(task_args=task_args, slot_id=slot_id)
    ensure_runtime_paths(robotwin_path, STARVLA_ROOT)
    from envs.utils.create_actor import UnStableError  # type: ignore
    from script.eval_policy import class_decorator  # type: ignore

    task_env = class_decorator(task_name)
    pending_candidate: Optional[CandidateInfo] = None
    active_episode_id: Optional[int] = None

    def send_action_request(seed: int) -> None:
        if active_episode_id is None:
            raise RuntimeError(f"Slot {slot_id} has no active episode for action request.")
        conn.send(
            {
                "type": "action_request",
                "slot_id": slot_id,
                "episode_id": active_episode_id,
                "seed": int(seed),
                "instruction": str(task_env.get_instruction()),
                "step": int(task_env.take_action_cnt),
            }
        )

    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")

            if cmd == "try_seed":
                seed = int(msg["seed"])
                valid, episode_info, error_text = _try_seed_once(
                    task_env,
                    task_args,
                    seed=seed,
                    episode_probe_id=slot_id,
                )
                if valid and episode_info is not None:
                    pending_candidate = CandidateInfo(seed=seed, episode_info=dict(episode_info.get("info", {})))
                    conn.send({"type": "candidate_valid", "slot_id": slot_id, "seed": seed})
                else:
                    conn.send(
                        {
                            "type": "candidate_invalid",
                            "slot_id": slot_id,
                            "seed": seed,
                            "error": error_text,
                        }
                    )
                continue

            if cmd == "reject_candidate":
                pending_candidate = None
                continue

            if cmd == "accept_candidate":
                if pending_candidate is None:
                    raise RuntimeError(f"Slot {slot_id} received accept without pending candidate.")
                active_episode_id = int(msg["episode_id"])
                seed = int(pending_candidate.seed)

                try:
                    task_env.setup_demo(now_ep_num=active_episode_id, seed=seed, is_test=True, **task_args)
                    step_limit_override = os.getenv("ROBOTWIN_STEP_LIMIT_OVERRIDE")
                    if step_limit_override:
                        task_env.step_lim = int(step_limit_override)
                        if task_env.step_lim <= 0:
                            raise ValueError("ROBOTWIN_STEP_LIMIT_OVERRIDE must be positive")
                    instruction = _select_instruction(
                        task_name=task_name,
                        instruction_type=instruction_type,
                        episode_info=pending_candidate.episode_info,
                        seed=seed,
                    )
                    task_env.set_instruction(instruction=instruction)
                    if task_args.get("eval_video_log", False) and task_args.get("eval_video_save_dir"):
                        ffmpeg = _open_ffmpeg(Path(task_args["eval_video_save_dir"]), video_size, active_episode_id)
                        task_env._set_eval_video_ffmpeg(ffmpeg)
                    pending_candidate = None
                    send_action_request(seed)
                except UnStableError:
                    try:
                        task_env.close_env()
                    except Exception:
                        pass
                    pending_candidate = None
                    active_episode_id = None
                    conn.send(
                        {
                            "type": "candidate_setup_invalid",
                            "slot_id": slot_id,
                            "episode_id": int(msg["episode_id"]),
                            "seed": seed,
                            "error": traceback.format_exc(),
                        }
                    )
                except Exception:
                    try:
                        task_env.close_env()
                    except Exception:
                        pass
                    pending_candidate = None
                    active_episode_id = None
                    conn.send(
                        {
                            "type": "candidate_setup_invalid",
                            "slot_id": slot_id,
                            "episode_id": int(msg["episode_id"]),
                            "seed": seed,
                            "error": traceback.format_exc(),
                        }
                    )
                continue

            if cmd == "action":
                if active_episode_id is None:
                    raise RuntimeError(f"Slot {slot_id} received action without active episode.")
                action = np.asarray(msg["action"], dtype=np.float32)
                try:
                    task_env.take_action(action, action_type=env_action_type)
                except Exception:
                    error_text = traceback.format_exc()
                    try:
                        if getattr(task_env, "eval_video_path", None) is not None:
                            task_env._del_eval_video_ffmpeg()
                    except Exception:
                        pass
                    try:
                        task_env.close_env(clear_cache=((active_episode_id + 1) % int(task_args["clear_cache_freq"]) == 0))
                    except Exception:
                        pass
                    conn.send(
                        {
                            "type": "episode_done",
                            "slot_id": slot_id,
                            "episode_id": active_episode_id,
                            "seed": int(msg["seed"]),
                            "success": False,
                            "steps": int(getattr(task_env, "take_action_cnt", 0)),
                            "error": error_text,
                        }
                    )
                    active_episode_id = None
                    continue

                if bool(task_env.eval_success) or int(task_env.take_action_cnt) >= int(task_env.step_lim):
                    success = bool(task_env.eval_success)
                    steps = int(task_env.take_action_cnt)
                    try:
                        if getattr(task_env, "eval_video_path", None) is not None:
                            task_env._del_eval_video_ffmpeg()
                    except Exception:
                        pass
                    task_env.close_env(clear_cache=((active_episode_id + 1) % int(task_args["clear_cache_freq"]) == 0))
                    conn.send(
                        {
                            "type": "episode_done",
                            "slot_id": slot_id,
                            "episode_id": active_episode_id,
                            "seed": int(msg["seed"]),
                            "success": success,
                            "steps": steps,
                        }
                    )
                    active_episode_id = None
                else:
                    send_action_request(seed=int(msg["seed"]))
                continue

            if cmd == "get_observation":
                if active_episode_id is None:
                    raise RuntimeError(f"Slot {slot_id} received get_observation without active episode.")
                conn.send(
                    {
                        "type": "observation_ready",
                        "slot_id": slot_id,
                        "episode_id": active_episode_id,
                        "seed": int(msg["seed"]),
                        "instruction": str(task_env.get_instruction()),
                        "observation": task_env.get_obs(),
                        "step": int(task_env.take_action_cnt),
                    }
                )
                continue

            if cmd == "shutdown":
                break

            raise RuntimeError(f"Unsupported slot worker command: {cmd}")
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception:
        try:
            conn.send({"type": "slot_error", "slot_id": slot_id, "error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if active_episode_id is not None:
                if getattr(task_env, "eval_video_path", None) is not None:
                    task_env._del_eval_video_ffmpeg()
                task_env.close_env()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _print_task_config(task_args: dict[str, Any]) -> None:
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(task_args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(task_args["domain_randomization"]["random_background"]))
    if task_args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(task_args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(task_args["domain_randomization"]["random_light"]))
    if task_args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(task_args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(task_args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(task_args["domain_randomization"]["random_head_camera_dis"]))
    print("\033[94mHead Camera Config:\033[0m " + str(task_args["camera"]["head_camera_type"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(task_args["camera"]["wrist_camera_type"]))
    print("\n==================================")


def run_batched_eval(usr_args: dict[str, Any]) -> int:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    robotwin_path = os.environ.get("ROBOTWIN_PATH", "../RoboTwin")
    save_dir = build_save_dir(usr_args, current_time)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = save_dir / "run.log"
    num_slots = resolve_num_slots()
    batch_wait_sec = resolve_batch_wait_sec()
    seed_log_every = resolve_seed_log_every()
    skip_get_obs_within_replan = resolve_skip_get_obs_within_replan()
    test_num = int(usr_args.get("test_num", os.getenv("ROBOTWIN_TEST_NUM", 100)))
    seed = int(usr_args.get("seed", 0))
    instruction_type = str(usr_args.get("instruction_type", "unseen"))

    log_file = open(run_log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    model = None
    parent_conns = {}
    retired_conns: dict[int, Any] = {}
    slot_processes: list[mp.Process] = []
    try:
        task_args, video_size = prepare_task_args(usr_args, save_dir=save_dir, robotwin_path=robotwin_path)
        model = get_model(usr_args)
        usr_args["resolved_robotwin_mode"] = getattr(model, "robotwin_mode", None)
        usr_args["resolved_env_action_type"] = getattr(model, "env_action_type", None)
        usr_args["resolved_replan_steps"] = getattr(model, "replan_steps", None)
        usr_args["resolved_action_ensemble"] = getattr(model, "action_ensemble", None)
        usr_args["resolved_action_ensemble_alpha"] = getattr(model, "action_ensemble_alpha", None)
        usr_args["skip_get_obs_within_replan"] = bool(skip_get_obs_within_replan)
        write_eval_meta(save_dir, usr_args, current_time, num_slots)
        _print_task_config(task_args)
        print(f"skip_get_obs_within_replan: {skip_get_obs_within_replan}")

        ctx = mp.get_context(os.getenv("ROBOTWIN_MP_START_METHOD", "spawn"))
        slot_state: dict[int, dict[str, Any]] = {}
        for slot_id in range(num_slots):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_slot_worker_main,
                kwargs={
                    "slot_id": slot_id,
                    "conn": child_conn,
                    "task_name": str(usr_args["task_name"]),
                    "instruction_type": instruction_type,
                    "task_args": task_args,
                    "video_size": video_size,
                    "robotwin_path": robotwin_path,
                    "env_action_type": str(model.env_action_type),
                },
            )
            proc.start()
            child_conn.close()
            parent_conns[slot_id] = parent_conn
            slot_processes.append(proc)
            slot_state[slot_id] = {"mode": "idle", "episode_id": None, "seed": None}

        next_episode_id = 0
        outstanding_episodes = 0
        completed_episodes = 0
        success_count = 0
        next_seed = int(os.getenv("ROBOTWIN_START_SEED", 100000 * (1 + seed)))
        pending_examples: dict[int, dict[str, Any]] = {}
        pending_observation_requests: dict[int, ObservationRequest] = {}
        episode_records: list[dict[str, Any]] = []
        invalid_seed_counts: dict[int, int] = {slot_id: 0 for slot_id in range(num_slots)}
        start_time = time.time()
        perf_counters = {
            "model_queries": 0,
            "cached_action_hits": 0,
            "obs_fetch_count": 0,
            "obs_skip_count": 0,
            "obs_wait_sec": 0.0,
            "infer_wait_sec": 0.0,
        }
        trace_enabled = normalize_bool_env(os.getenv("ROBOTWIN_TRACE_ACTION_STATE"), default=False)
        trace_path = save_dir / "action_state_trace.jsonl"

        def _trace_array(value: Any) -> Optional[list[float]]:
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            return array.tolist()

        def write_action_trace(
            *,
            slot_id: int,
            pending: dict[str, Any],
            action: Any,
            model_query: bool,
        ) -> None:
            if not trace_enabled:
                return
            snapshot = model.get_action_trace_snapshot(slot_id=slot_id)
            raw_actions = snapshot.get("raw_actions")
            record = {
                "slot_id": int(slot_id),
                "episode_id": int(pending["episode_id"]),
                "seed": int(pending["seed"]),
                "sim_step": int(pending["sim_step"]),
                "model_query": bool(model_query),
                "policy_use_state": bool(getattr(model, "use_state", False)),
                "policy_input_state": _trace_array(pending.get("policy_input_state")),
                "joint_position": _trace_array(pending.get("joint_position")),
                "endpose_state": _trace_array(pending.get("endpose_state")),
                "action": _trace_array(action),
                "action_cursor": int(snapshot["action_cursor"]),
                "executed_steps": int(snapshot["executed_steps"]),
            }
            if model_query and raw_actions is not None:
                record["action_chunk"] = np.asarray(raw_actions, dtype=np.float32).tolist()
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        def detach_slot(slot_id: int, *, mode: str) -> None:
            conn = parent_conns.pop(slot_id, None)
            if conn is not None:
                retired_conns[slot_id] = conn
            pending_examples.pop(slot_id, None)
            pending_observation_requests.pop(slot_id, None)
            slot_state[slot_id]["mode"] = mode

        def retire_slot(slot_id: int, *, reject_candidate: bool = False) -> None:
            conn = parent_conns.get(slot_id)
            if conn is None:
                pending_examples.pop(slot_id, None)
                slot_state[slot_id]["mode"] = "stopping"
                return
            if reject_candidate:
                try:
                    conn.send({"cmd": "reject_candidate"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            try:
                conn.send({"cmd": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            detach_slot(slot_id, mode="stopping")

        def issue_try_seed(slot_id: int) -> None:
            nonlocal next_seed
            parent_conns[slot_id].send({"cmd": "try_seed", "seed": next_seed})
            slot_state[slot_id] = {"mode": "seeking", "episode_id": None, "seed": next_seed}
            next_seed += 1

        for slot_id in range(num_slots):
            issue_try_seed(slot_id)

        while completed_episodes < test_num:
            live_conns = [conn for conn in parent_conns.values()]
            if not live_conns:
                raise RuntimeError("All slot workers exited before evaluation completed.")

            ready = mp_wait(live_conns, timeout=1.0)
            for conn in ready:
                slot_id = next(candidate_slot_id for candidate_slot_id, candidate_conn in parent_conns.items() if candidate_conn == conn)
                try:
                    message = conn.recv()
                except EOFError as exc:
                    if slot_state.get(slot_id, {}).get("mode") in {"stopping", "stopped"}:
                        detach_slot(slot_id, mode="stopped")
                        continue
                    detach_slot(slot_id, mode="crashed")
                    raise RuntimeError(f"Slot {slot_id} closed its pipe unexpectedly.") from exc
                msg_type = str(message["type"])

                if msg_type == "candidate_invalid":
                    invalid_seed_counts[slot_id] = invalid_seed_counts.get(slot_id, 0) + 1
                    if message.get("error"):
                        print(f"[slot={slot_id}] seed={message['seed']} invalid due to error")
                        print(str(message["error"]).rstrip())
                    elif seed_log_every > 0 and invalid_seed_counts[slot_id] % seed_log_every == 0:
                        print(
                            f"[slot={slot_id}] searched {invalid_seed_counts[slot_id]} invalid seeds so far; "
                            f"latest seed={message['seed']}"
                        )
                    if completed_episodes + outstanding_episodes < test_num:
                        issue_try_seed(slot_id)
                    else:
                        retire_slot(slot_id)
                    continue

                if msg_type == "candidate_valid":
                    invalid_seed_counts[slot_id] = 0
                    if completed_episodes + outstanding_episodes >= test_num:
                        retire_slot(slot_id, reject_candidate=True)
                        continue
                    episode_id = next_episode_id
                    next_episode_id += 1
                    outstanding_episodes += 1
                    model.reset(slot_id=slot_id)
                    parent_conns[slot_id].send({"cmd": "accept_candidate", "episode_id": episode_id})
                    slot_state[slot_id] = {
                        "mode": "starting",
                        "episode_id": episode_id,
                        "seed": int(message["seed"]),
                    }
                    continue

                if msg_type == "candidate_setup_invalid":
                    outstanding_episodes = max(0, outstanding_episodes - 1)
                    slot_state[slot_id] = {"mode": "seeking", "episode_id": None, "seed": None}
                    print(
                        f"[slot={slot_id}] seed={message['seed']} rejected during episode setup; "
                        "retrying with a new seed"
                    )
                    if message.get("error"):
                        print(str(message["error"]).rstrip())
                    if completed_episodes + outstanding_episodes < test_num:
                        issue_try_seed(slot_id)
                    else:
                        retire_slot(slot_id)
                    continue

                if msg_type == "action_request":
                    slot_state[slot_id]["mode"] = "waiting_action"
                    slot_state[slot_id]["episode_id"] = int(message["episode_id"])
                    slot_state[slot_id]["seed"] = int(message["seed"])
                    instruction = str(message["instruction"])
                    should_fetch_obs, must_query = _plan_action_request(
                        model=model,
                        slot_id=slot_id,
                        instruction=instruction,
                        skip_get_obs_within_replan=skip_get_obs_within_replan,
                    )
                    if not should_fetch_obs:
                        perf_counters["cached_action_hits"] += 1
                        perf_counters["obs_skip_count"] += 1
                        parent_conns[slot_id].send(
                            {
                                "cmd": "action",
                                "action": model.step_cached(slot_id=slot_id, task_description=instruction),
                                "seed": int(message["seed"]),
                            }
                        )
                        slot_state[slot_id]["mode"] = "running"
                        continue
                    pending_observation_requests[slot_id] = ObservationRequest(
                        seed=int(message["seed"]),
                        episode_id=int(message["episode_id"]),
                        instruction=instruction,
                        requested_at=time.perf_counter(),
                        must_query=bool(must_query),
                    )
                    parent_conns[slot_id].send({"cmd": "get_observation", "seed": int(message["seed"])})
                    slot_state[slot_id]["mode"] = "waiting_obs"
                    continue

                if msg_type == "observation_ready":
                    observation_request = pending_observation_requests.pop(slot_id, None)
                    if observation_request is None:
                        raise RuntimeError(f"Slot {slot_id} returned observation without a pending request.")
                    if int(message["episode_id"]) != observation_request.episode_id:
                        raise RuntimeError(
                            f"Slot {slot_id} observation episode mismatch: "
                            f"{message['episode_id']} vs {observation_request.episode_id}"
                        )
                    if int(message["seed"]) != observation_request.seed:
                        raise RuntimeError(
                            f"Slot {slot_id} observation seed mismatch: "
                            f"{message['seed']} vs {observation_request.seed}"
                        )
                    perf_counters["obs_fetch_count"] += 1
                    perf_counters["obs_wait_sec"] += time.perf_counter() - observation_request.requested_at
                    instruction = observation_request.instruction
                    observation = message["observation"]
                    example = model.build_example(instruction, observation)
                    trace_pending = {
                        "seed": int(message["seed"]),
                        "episode_id": int(message["episode_id"]),
                        "sim_step": int(message["step"]),
                        "policy_input_state": example.get("state"),
                        "joint_position": observation.get("joint_action", {}).get("vector"),
                        "endpose_state": flatten_robotwin_endpose_state(observation),
                    }
                    requires_query = _observation_requires_model_query(
                        model=model,
                        slot_id=slot_id,
                        instruction=instruction,
                        must_query=observation_request.must_query,
                    )
                    if requires_query:
                        pending_examples[slot_id] = {
                            "slot_id": slot_id,
                            "seed": int(message["seed"]),
                            "episode_id": int(message["episode_id"]),
                            "sim_step": int(message["step"]),
                            "example": example,
                            **trace_pending,
                        }
                        slot_state[slot_id]["mode"] = "waiting_infer"
                        continue
                    perf_counters["cached_action_hits"] += 1
                    action = model.step_cached(slot_id=slot_id, task_description=instruction)
                    write_action_trace(
                        slot_id=slot_id,
                        pending=trace_pending,
                        action=action,
                        model_query=False,
                    )
                    parent_conns[slot_id].send(
                        {
                            "cmd": "action",
                            "action": action,
                            "seed": int(message["seed"]),
                        }
                    )
                    slot_state[slot_id]["mode"] = "running"
                    continue

                if msg_type == "episode_done":
                    completed_episodes += 1
                    outstanding_episodes = max(0, outstanding_episodes - 1)
                    success = bool(message["success"])
                    success_count += int(success)
                    record = {
                        "slot_id": slot_id,
                        "episode_id": int(message["episode_id"]),
                        "seed": int(message["seed"]),
                        "success": success,
                        "steps": int(message.get("steps", 0)),
                    }
                    if message.get("error"):
                        record["error"] = str(message["error"])
                    episode_records.append(record)
                    pending_examples.pop(slot_id, None)
                    pending_observation_requests.pop(slot_id, None)
                    rate = success_count / max(1, completed_episodes)
                    print(
                        f"\033[93m{usr_args['task_name']}\033[0m | "
                        f"\033[94m{usr_args['policy_name']}\033[0m | "
                        f"\033[92m{usr_args['task_config']}\033[0m | "
                        f"\033[91m{usr_args['ckpt_setting']}\033[0m\n"
                        f"Success rate: \033[96m{success_count}/{completed_episodes}\033[0m => "
                        f"\033[95m{round(rate * 100, 1)}%\033[0m, "
                        f"progress: \033[96m{completed_episodes}/{test_num}\033[0m, "
                        f"current seed: \033[90m{message['seed']}\033[0m\n"
                    )
                    if completed_episodes + outstanding_episodes < test_num:
                        issue_try_seed(slot_id)
                    else:
                        retire_slot(slot_id)
                    continue

                if msg_type == "slot_error":
                    raise RuntimeError(f"Slot {slot_id} crashed:\n{message['error']}")

                raise RuntimeError(f"Unknown worker message: {message}")

            if pending_examples and batch_wait_sec > 0:
                time.sleep(batch_wait_sec)

            if pending_examples:
                slot_ids = sorted(pending_examples)
                examples = [pending_examples[slot_id]["example"] for slot_id in slot_ids]
                infer_t0 = time.perf_counter()
                actions = model.step_batch(examples, slot_ids=slot_ids)
                perf_counters["infer_wait_sec"] += time.perf_counter() - infer_t0
                perf_counters["model_queries"] += len(examples)
                for slot_id, action in zip(slot_ids, actions):
                    write_action_trace(
                        slot_id=slot_id,
                        pending=pending_examples[slot_id],
                        action=action,
                        model_query=True,
                    )
                    parent_conns[slot_id].send(
                        {
                            "cmd": "action",
                            "action": action,
                            "seed": int(pending_examples[slot_id]["seed"]),
                        }
                    )
                    slot_state[slot_id]["mode"] = "running"
                pending_examples.clear()

        elapsed = time.time() - start_time
        success_rate = success_count / max(1, test_num)
        summary = {
            "task_name": str(usr_args["task_name"]),
            "task_config": str(usr_args["task_config"]),
            "n_episodes": int(test_num),
            "num_slots": int(num_slots),
            "successes": int(success_count),
            "success_rate": float(success_rate),
            "elapsed_sec": float(elapsed),
            "skip_get_obs_within_replan": bool(skip_get_obs_within_replan),
            "model_queries": int(perf_counters["model_queries"]),
            "cached_action_hits": int(perf_counters["cached_action_hits"]),
            "obs_fetch_count": int(perf_counters["obs_fetch_count"]),
            "obs_skip_count": int(perf_counters["obs_skip_count"]),
            "obs_wait_sec": float(perf_counters["obs_wait_sec"]),
            "infer_wait_sec": float(perf_counters["infer_wait_sec"]),
            "episodes": sorted(episode_records, key=lambda item: int(item["episode_id"])),
        }
        write_json(save_dir / "summary.json", summary)
        with open(save_dir / "_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {current_time}\n\n")
            f.write(f"Instruction Type: {instruction_type}\n\n")
            f.write(f"{float(success_rate)}\n")
        print(f"Data has been saved to {save_dir / '_result.txt'}")
        return 0
    finally:
        if model is not None:
            try:
                model.close()
            except Exception:
                pass
        for slot_id, conn in parent_conns.items():
            try:
                conn.send({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        for conn in retired_conns.values():
            try:
                conn.close()
            except Exception:
                pass
        for proc in slot_processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main() -> int:
    parsed = parse_args()
    usr_args = load_config_with_overrides(parsed.config, parsed.overrides)
    return run_batched_eval(usr_args)


if __name__ == "__main__":
    raise SystemExit(main())
