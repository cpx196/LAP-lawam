#!/usr/bin/env python3
"""Serve real-VLM LAP6 actions while tracing SEC284 on identical observations.

The environment always executes the official VLM-conditioned Action Expert
output.  SEC284 runs in shadow mode with the same LAP6/LaWM visual future and
the same initial flow noise.  Per-replan tensors are saved for inference-grid
diagnostics and later teacher-forced velocity distillation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import torch

from deployment.model_server.server_policy import (
    build_policy_server_metadata,
    load_policy_from_checkpoint,
)
from deployment.model_server.server_policy_lap_lawm_ablation import (
    DEFAULT_LAP,
    DEFAULT_STATE_STATS,
    LAPLaWMHybridPolicy,
    _build_wrist_tensor,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_infer_output
from starVLA.model.framework.vlas.lawam import _cuda_autocast
from starVLA.model.sec284 import SEC284Config, SEC284L


LOGGER = logging.getLogger(__name__)
DEFAULT_POLICY = "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
DEFAULT_SEC284 = "outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt"
DEFAULT_FIXED_INSTRUCTION = (
    "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad."
)


def _cpu_half(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float16) if value.is_floating_point() else value.detach().cpu()


class LAP6SEC284ShadowTracePolicy(LAPLaWMHybridPolicy):
    def __init__(
        self,
        policy,
        *,
        lap_checkpoint: str | Path,
        state_stats: str | Path,
        sec284_checkpoint: str | Path,
        trace_dir: str | Path,
        fixed_instruction: str,
    ) -> None:
        super().__init__(
            policy,
            mode="lap_official",
            lap_checkpoint=lap_checkpoint,
            state_stats=state_stats,
        )
        device = next(self.backend.parameters()).device
        sec_obj = torch.load(sec284_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.sec284 = SEC284L(SEC284Config(**sec_obj["config"]))
        self.sec284.load_state_dict(sec_obj["sec284"], strict=True)
        self.sec284 = self.sec284.to(device=device, dtype=torch.float32).eval()
        for parameter in self.sec284.parameters():
            parameter.requires_grad_(False)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.trace_dir / "summary.jsonl"
        self.calls = 0
        self.sec284_checkpoint = str(Path(sec284_checkpoint).resolve())
        self.fixed_instruction = str(fixed_instruction)
        LOGGER.info(
            "Loaded SEC284 shadow checkpoint=%s trace_dir=%s",
            self.sec284_checkpoint,
            self.trace_dir.resolve(),
        )

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        started = time.perf_counter()
        original_instructions = [str(example.get("lang", "")) for example in examples]
        effective_examples = [
            {**example, "lang": self.fixed_instruction} for example in examples
        ]
        batch = self.policy.policy_infer_batch_builder.build_infer_batch(effective_examples)
        wrist = _build_wrist_tensor(examples, device=batch["primary_image"].device)
        batch["lap_visual"] = torch.cat([batch["primary_image"].unsqueeze(1), wrist], dim=1)
        prepared = self.backend._prepare_infer_batch(batch=batch)
        shared = self.backend._run_shared_encoding_infer(
            prepared_batch=prepared,
            source="LAP6SEC284ShadowTracePolicy.predict_action",
            lam_features_with_no_grad=False,
            profile=None,
        )
        features = self._last_lap_features
        student_condition = self.sec284(features)
        teacher_condition = self.backend.flow._prepare_semantic_condition(
            h_vlm=shared.h_vlm,
            h_lap=None,
            model_dtype=self.backend.flow._compute_dtype(),
        )
        if (
            student_condition.shape[0] != teacher_condition.shape[0]
            or student_condition.shape[2] != teacher_condition.shape[2]
        ):
            raise RuntimeError(
                f"SEC/VLM condition batch-width mismatch: {tuple(student_condition.shape)} vs "
                f"{tuple(teacher_condition.shape)}"
            )

        flow = self.backend.flow
        guidance_scale = kwargs.get("guidance_scale")
        if guidance_scale is None:
            guidance_scale = float(flow.config.cfg_guidance_scale)
        num_steps = kwargs.get("num_inference_steps")
        if num_steps is None:
            num_steps = int(flow.config.num_inference_steps)
        teacher_mask = prepared["attention_mask"] == 1
        student_mask = torch.ones(
            student_condition.shape[:2], device=student_condition.device, dtype=torch.bool
        )
        common = {
            "h_t": shared.h_t,
            "h_t1_star": shared.h_t1_pred,
            "state": prepared["state"],
            "state_mask": prepared["state_mask"],
            "action_hz": prepared["action_hz"],
            "embodiment_id": prepared["embodiment_id"],
            "cfg_scale": float(guidance_scale),
            "num_inference_steps": int(num_steps),
            "return_padded": False,
            "return_trace": True,
        }

        cpu_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state(student_condition.device)
        with _cuda_autocast(torch.float32):
            teacher_actions, teacher_trace = flow.sample_actions_cfg(
                h_vlm=shared.h_vlm,
                h_lap=None,
                attention_mask=teacher_mask,
                **common,
            )
        cpu_after_teacher = torch.get_rng_state()
        cuda_after_teacher = torch.cuda.get_rng_state(student_condition.device)

        torch.set_rng_state(cpu_before)
        torch.cuda.set_rng_state(cuda_before, student_condition.device)
        with _cuda_autocast(torch.float32):
            student_actions, student_trace = flow.sample_actions_cfg(
                h_vlm=None,
                h_lap=student_condition,
                attention_mask=student_mask,
                **common,
            )
            _, student_teacher_grid_trace = flow.sample_actions_cfg(
                h_vlm=None,
                h_lap=student_condition,
                attention_mask=student_mask,
                forced_x_inputs=teacher_trace["x_inputs"],
                **common,
            )
        # Shadow work must not perturb the next teacher action's initial noise.
        torch.set_rng_state(cpu_after_teacher)
        torch.cuda.set_rng_state(cuda_after_teacher, student_condition.device)

        teacher_fp32 = teacher_actions.float()
        student_fp32 = student_actions.float()
        action_diff = student_fp32 - teacher_fp32
        xyz_indices = torch.tensor([0, 1, 2, 8, 9, 10], device=action_diff.device)
        gripper_indices = torch.tensor([7, 15], device=action_diff.device)
        teacher_grid_velocity = teacher_trace["velocities"].float()
        student_grid_velocity = student_teacher_grid_trace["velocities"].float()
        velocity_mse_per_step = (
            student_grid_velocity - teacher_grid_velocity
        ).square().mean(dim=(1, 2, 3))
        # Runtime VLM length follows the tokenized instruction; SEC284 keeps a
        # fixed 284-token interface.  Prefix metrics are diagnostic only.  The
        # action and forced-grid velocity comparisons below remain exact.
        common_tokens = min(student_condition.shape[1], teacher_condition.shape[1])
        student_common = student_condition[:, :common_tokens].float()
        teacher_common = teacher_condition[:, :common_tokens].float()
        condition_mse = (student_common - teacher_common).square().mean()
        condition_cosine = torch.nn.functional.cosine_similarity(
            student_common, teacher_common, dim=-1
        ).mean()

        self.calls += 1
        trace_path = self.trace_dir / f"call-{self.calls:05d}.pt"
        payload = {
            "call": self.calls,
            "original_instruction": original_instructions,
            "instruction": [self.fixed_instruction] * len(examples),
            "student_condition_tokens": int(student_condition.shape[1]),
            "teacher_condition_tokens": int(teacher_condition.shape[1]),
            "condition_common_tokens": int(common_tokens),
            "features": _cpu_half(features),
            "h_t": _cpu_half(shared.h_t),
            "h_t1": _cpu_half(shared.h_t1_pred),
            "teacher_condition": _cpu_half(teacher_condition),
            "student_condition": _cpu_half(student_condition),
            "state": _cpu_half(prepared["state"]),
            "state_mask": prepared["state_mask"].detach().cpu(),
            "action_hz": prepared["action_hz"].detach().cpu(),
            "embodiment_id": prepared["embodiment_id"].detach().cpu(),
            "teacher_actions": _cpu_half(teacher_actions),
            "student_actions": _cpu_half(student_actions),
            "teacher_x_inputs": _cpu_half(teacher_trace["x_inputs"]),
            "teacher_velocities": _cpu_half(teacher_trace["velocities"]),
            "student_x_inputs": _cpu_half(student_trace["x_inputs"]),
            "student_velocities": _cpu_half(student_trace["velocities"]),
            "student_teacher_grid_velocities": _cpu_half(
                student_teacher_grid_trace["velocities"]
            ),
            "time_grid": teacher_trace["time_grid"].detach().cpu(),
            "sec284_checkpoint": self.sec284_checkpoint,
        }
        torch.save(payload, trace_path)
        summary = {
            "call": self.calls,
            "original_instruction": original_instructions,
            "instruction": [self.fixed_instruction] * len(examples),
            "student_condition_tokens": int(student_condition.shape[1]),
            "teacher_condition_tokens": int(teacher_condition.shape[1]),
            "condition_common_tokens": int(common_tokens),
            "condition_prefix_mse": float(condition_mse.cpu()),
            "condition_prefix_cosine": float(condition_cosine.cpu()),
            "action_mse": float(action_diff.square().mean().cpu()),
            "xyz_mse": float(action_diff.index_select(-1, xyz_indices).square().mean().cpu()),
            "gripper_mse": float(
                action_diff.index_select(-1, gripper_indices).square().mean().cpu()
            ),
            "gripper_sign_agreement": float(
                (
                    student_fp32.index_select(-1, gripper_indices) >= 0
                )
                .eq(teacher_fp32.index_select(-1, gripper_indices) >= 0)
                .float()
                .mean()
                .cpu()
            ),
            "grid_velocity_mse": float(velocity_mse_per_step.mean().cpu()),
            "grid_velocity_mse_per_step": [
                float(value) for value in velocity_mse_per_step.cpu().tolist()
            ],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "trace": trace_path.name,
        }
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        LOGGER.info(
            "shadow call=%d cond_mse=%.6f action_mse=%.6f xyz_mse=%.6f "
            "gripper_agree=%.3f grid_velocity_mse=%.6f latency=%.1fms",
            self.calls,
            summary["condition_prefix_mse"],
            summary["action_mse"],
            summary["xyz_mse"],
            summary["gripper_sign_agreement"],
            summary["grid_velocity_mse"],
            summary["latency_ms"],
        )
        return map_policy_infer_output(teacher_actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", default=DEFAULT_POLICY)
    parser.add_argument("--lap-checkpoint", default=DEFAULT_LAP)
    parser.add_argument("--lap-state-stats", default=DEFAULT_STATE_STATS)
    parser.add_argument("--sec284-checkpoint", default=DEFAULT_SEC284)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--fixed-instruction", default=DEFAULT_FIXED_INSTRUCTION)
    parser.add_argument("--port", type=int, default=11053)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy_from_checkpoint(args.ckpt_path, use_bf16=False, device="cuda")
    served = LAP6SEC284ShadowTracePolicy(
        policy,
        lap_checkpoint=args.lap_checkpoint,
        state_stats=args.lap_state_stats,
        sec284_checkpoint=args.sec284_checkpoint,
        trace_dir=args.trace_dir,
        fixed_instruction=args.fixed_instruction,
    )
    metadata = build_policy_server_metadata(
        policy,
        ckpt_path=args.ckpt_path,
        server_type="lap6_sec284_shadow_trace",
        env="robotwin",
        supported_eval_envs=["robotwin"],
        extra_metadata={"trace_dir": str(Path(args.trace_dir).resolve())},
    )
    LOGGER.info("Serving real-VLM actions with SEC284 shadow trace on %s:%d", args.host, args.port)
    WebsocketPolicyServer(
        policy=served,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    torch.set_float32_matmul_precision("high")
    main()
