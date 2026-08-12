#!/usr/bin/env python3
"""RoboTwin policy server for isolating the LAP -> LaWM replacement path.

Modes:
  baseline     Official LaWAM unchanged.
  lap_official VLM still conditions Action Expert, while LAP supplies LaWM's
               latent action; the official LaWM decoder is retained.
  lap_joint    Same as lap_official, but uses the jointly trained Stage-1 LaWM
               decoder paired with LAP.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch

from deployment.model_server.server_policy import build_policy_server_metadata, load_policy_from_checkpoint
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.latent_world.batch_utils import (
    imagenet_normalize_image_batch_,
    prepare_frame_spatial_uint8,
)
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_infer_output
from starVLA.model.framework.vlas.lawam import PolicyEncodingState, _cuda_autocast
from starVLA.model.lap_stage1 import LAP60M


LOGGER = logging.getLogger(__name__)
DEFAULT_POLICY = "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
DEFAULT_LAP = "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
DEFAULT_STATE_STATS = "cache/lap_stage1_task14/state_stats.json"


def _load_state_stats(path: str | Path, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mean = torch.tensor(payload["mean"], device=device, dtype=dtype)
    std = torch.tensor(payload["std"], device=device, dtype=dtype).clamp_min(1e-6)
    if tuple(mean.shape) != (16,) or tuple(std.shape) != (16,):
        raise ValueError(f"Expected 16-D LAP EEF stats, got mean={tuple(mean.shape)} std={tuple(std.shape)}")
    return mean, std


def _unit_quaternion(state: torch.Tensor) -> torch.Tensor:
    state = state.clone()
    for start in (3, 11):
        quat = state[..., start : start + 4]
        state[..., start : start + 4] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return state


def _build_wrist_tensor(examples: Sequence[dict[str, Any]], *, device: torch.device) -> torch.Tensor:
    per_example: list[torch.Tensor] = []
    for index, example in enumerate(examples):
        wrist = example.get("wrist_image")
        if not isinstance(wrist, (list, tuple)) or len(wrist) != 2:
            raise ValueError(
                f"LAP ablation requires exactly [left_wrist,right_wrist] images; sample {index} got "
                f"{0 if wrist is None else len(wrist) if isinstance(wrist, (list, tuple)) else type(wrist)}."
            )
        frames = [
            prepare_frame_spatial_uint8(np.asarray(frame), target_hw=(256, 256))
            for frame in wrist
        ]
        per_example.append(torch.stack(frames, dim=0))
    wrist_batch = torch.stack(per_example, dim=0).to(device=device, dtype=torch.float32).div_(255.0)
    flat = wrist_batch.flatten(0, 1)
    imagenet_normalize_image_batch_(flat)
    return flat.view_as(wrist_batch)


class LAPLaWMHybridPolicy:
    """Wrap an official policy, replacing only its VLM-to-LaWM latent path."""

    def __init__(self, policy, *, mode: str, lap_checkpoint: str | Path, state_stats: str | Path) -> None:
        if mode not in {"lap_official", "lap_joint"}:
            raise ValueError(f"Hybrid policy mode must be lap_official/lap_joint, got {mode!r}")
        self.policy = policy
        self.mode = str(mode)
        self.backend = self.policy.policy_backend
        device = next(self.backend.parameters()).device
        model_dtype = next(self.backend.parameters()).dtype

        stage1 = torch.load(lap_checkpoint, map_location="cpu", weights_only=False)
        self.lap = LAP60M(num_views=3, view_dropout=0.0)
        self.lap.load_state_dict(stage1["lap"], strict=True)
        self.lap = self.lap.to(device=device, dtype=model_dtype).eval()
        self.state_mean, self.state_std = _load_state_stats(state_stats, device, model_dtype)
        self.lap_checkpoint = str(Path(lap_checkpoint).resolve())
        self.state_stats = str(Path(state_stats).resolve())

        if self.mode == "lap_joint":
            decoder_state = stage1.get("lawm_decoder")
            if decoder_state is None:
                raise ValueError("Stage-1 checkpoint has no jointly trained `lawm_decoder` state.")
            self.backend.lam.decoder.load_state_dict(decoder_state, strict=True)
            LOGGER.info("Loaded joint Stage-1 LaWM decoder from %s", self.lap_checkpoint)

        # Preserve the original backend's full VLM -> h_vlm -> Action Expert path.
        # Only the shared encoding implementation is swapped so h_t1 is decoded
        # from z_lap rather than the VLM action query output.
        self.backend._run_shared_encoding_infer = MethodType(self._run_shared_encoding_infer, self.backend)

    def _run_shared_encoding_infer(self, backend, *, prepared_batch, source: str, lam_features_with_no_grad: bool, profile=None):
        del lam_features_with_no_grad
        device = prepared_batch["input_ids"].device
        vlm_dtype = backend.model_cfg.vlm_dtype
        act_query, flow_query = backend._prepare_queries(device=device, vlm_stage_dtype=vlm_dtype)

        stage_start = time.perf_counter()
        with _cuda_autocast(vlm_dtype):
            vlm_out = backend._run_vlm_stage(
                input_ids=prepared_batch["input_ids"],
                attention_mask=prepared_batch["attention_mask"],
                pixel_values=prepared_batch["pixel_values"],
                image_grid_thw=prepared_batch["image_grid_thw"],
                act_placeholder_mask=prepared_batch["act_placeholder_mask"],
                flow_placeholder_mask=prepared_batch["flow_placeholder_mask"],
                act_query=act_query,
                flow_query=flow_query,
            )
        if profile is not None:
            profile["qwen_vlm_ms"] = (time.perf_counter() - stage_start) * 1000.0

        lap_visual = prepared_batch.get("lap_visual")
        if lap_visual is None or tuple(lap_visual.shape[1:]) != (3, 3, 256, 256):
            raise ValueError(f"Missing/malformed lap_visual, got {None if lap_visual is None else tuple(lap_visual.shape)}")
        raw_state = prepared_batch["state"][:, :16]
        if raw_state.shape[-1] != 16:
            raise ValueError(f"LAP requires raw 16-D EEF state, got {tuple(raw_state.shape)}")
        lap_state = (_unit_quaternion(raw_state) - self.state_mean) / self.state_std

        with _cuda_autocast(torch.bfloat16):
            stage_start = time.perf_counter()
            features = backend.lam.extract_vision_features(lap_visual)
            if profile is not None:
                profile["dino_lam_ms"] = (time.perf_counter() - stage_start) * 1000.0
            if tuple(features.shape[1:]) != (3, 256, 768):
                raise ValueError(f"Expected 3-view DINO features [B,3,256,768], got {tuple(features.shape)}")
            # Shadow-trace policies reuse the exact DINO tensor that drove
            # LAP6, avoiding a second feature extraction or image mismatch.
            self._last_lap_features = features.detach()
            h_t = features[:, 0]
            stage_start = time.perf_counter()
            z_lap = self.lap(features, lap_state.to(dtype=features.dtype))["z_lap"]
            h_t1_pred = backend._decode_future_tokens_strict_single_query(
                h_t=h_t,
                pred_action_emb=z_lap.to(dtype=h_t.dtype),
                source=source,
            )
            if profile is not None:
                profile["lap_plus_lawm_ms"] = (time.perf_counter() - stage_start) * 1000.0

        return PolicyEncodingState(
            h_vlm=vlm_out["h_vlm"],
            pred_action_emb=z_lap,
            h_t=h_t,
            h_t1_pred=h_t1_pred,
            h_t1_gt=h_t,
            h_t_original=h_t,
        )

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        batch = self.policy.policy_infer_batch_builder.build_infer_batch(examples)
        wrist = _build_wrist_tensor(examples, device=batch["primary_image"].device)
        batch["lap_visual"] = torch.cat([batch["primary_image"].unsqueeze(1), wrist], dim=1)
        actions = self.backend.predict_action(
            batch=batch,
            guidance_scale=kwargs.get("guidance_scale"),
            num_inference_steps=kwargs.get("num_inference_steps"),
        )
        return map_policy_infer_output(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", default=DEFAULT_POLICY)
    parser.add_argument("--mode", choices=["baseline", "lap_official", "lap_joint"], required=True)
    parser.add_argument("--lap-checkpoint", default=DEFAULT_LAP)
    parser.add_argument("--lap-state-stats", default=DEFAULT_STATE_STATS)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy_from_checkpoint(args.ckpt_path, use_bf16=bool(args.use_bf16), device="cuda")
    if args.mode == "baseline":
        served_policy = policy
    else:
        served_policy = LAPLaWMHybridPolicy(
            policy,
            mode=args.mode,
            lap_checkpoint=args.lap_checkpoint,
            state_stats=args.lap_state_stats,
        )
    metadata = build_policy_server_metadata(
        policy,
        ckpt_path=args.ckpt_path,
        server_type="lawam_lap_lawm_ablation",
        env="robotwin",
        supported_eval_envs=["robotwin"],
        extra_metadata={
            "ablation_mode": args.mode,
            "requires_raw_eef": args.mode != "baseline",
            "lap_checkpoint": str(Path(args.lap_checkpoint).resolve()) if args.mode != "baseline" else None,
        },
    )
    LOGGER.info("Serving ablation mode=%s on host=%s port=%d", args.mode, args.host, args.port)
    WebsocketPolicyServer(
        policy=served_policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
