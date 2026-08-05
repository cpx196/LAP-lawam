#!/usr/bin/env python3
"""VLM-free RoboTwin server for LAP8 or the 284-token LAP10 checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from latent_action_model.core.lam_model import load_latent_action_model
from starVLA.model.framework.latent_world.batch_utils import (
    imagenet_normalize_image_batch_,
    prepare_frame_spatial_uint8,
)
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_infer_output
from starVLA.model.lap_stage1 import LAP60M, count_parameters
from starVLA.model.lap_stage2 import LAP8, LAP10, LAP10V3
from tools.train_lap8_phase1 import load_action_expert


LOGGER = logging.getLogger(__name__)
DEFAULT_POLICY = "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
DEFAULT_STAGE1 = "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
DEFAULT_LAP8 = "outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt"
DEFAULT_LAP10 = "outputs/lap10_alignment_task14_1000step/lap10_step0001000.pt"
DEFAULT_LAP10V3_STATS = "cache/lap10_task14_vlm_teacher_8192/position_stats.pt"
DEFAULT_STATE_STATS = "cache/lap_stage1_task14/state_stats.json"
DEFAULT_EXPERT = "cache/lap8_phase1_official_action_expert.pt"
DEFAULT_LAM_CKPT = "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
DEFAULT_LAM_YAML = "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"


def _load_state_stats(path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mean = torch.tensor(payload["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(payload["std"], device=device, dtype=torch.float32).clamp_min(1e-6)
    if tuple(mean.shape) != (16,) or tuple(std.shape) != (16,):
        raise ValueError(f"Expected 16-D EEF stats, got mean={tuple(mean.shape)} std={tuple(std.shape)}")
    return mean, std


def _unit_quaternion(state: torch.Tensor) -> torch.Tensor:
    state = state.clone()
    for start in (3, 11):
        quat = state[..., start : start + 4]
        state[..., start : start + 4] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return state


def _build_inputs(
    examples: Sequence[dict[str, Any]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    views: list[torch.Tensor] = []
    states: list[np.ndarray] = []
    action_hz: list[float] = []
    embodiment_ids: list[int] = []
    for index, example in enumerate(examples):
        primary = example.get("primary_image")
        wrists = example.get("wrist_image")
        if not isinstance(primary, (list, tuple)) or len(primary) < 1:
            raise ValueError(f"Sample {index} has no primary image")
        if not isinstance(wrists, (list, tuple)) or len(wrists) != 2:
            raise ValueError(f"Sample {index} requires [left_wrist,right_wrist]")
        raw_state = np.asarray(example.get("state"), dtype=np.float32).reshape(-1)
        if raw_state.shape != (16,):
            raise ValueError(f"Sample {index} requires raw 16-D EEF state, got {raw_state.shape}")
        frames = [primary[0], wrists[0], wrists[1]]
        views.append(
            torch.stack(
                [prepare_frame_spatial_uint8(np.asarray(frame), target_hw=(256, 256)) for frame in frames],
                dim=0,
            )
        )
        states.append(raw_state)
        action_hz.append(float(example.get("action_hz", 30.0)))
        embodiment_ids.append(int(example.get("embodiment_id", 1)))

    visual = torch.stack(views).to(device=device, dtype=torch.float32).div_(255.0)
    flat = visual.flatten(0, 1)
    imagenet_normalize_image_batch_(flat)
    visual = flat.view_as(visual)
    state = torch.from_numpy(np.stack(states)).to(device=device, dtype=torch.float32)
    hz = torch.tensor(action_hz, device=device, dtype=torch.float32)
    embodiment = torch.tensor(embodiment_ids, device=device, dtype=torch.long)
    return visual, state, hz, embodiment


class LAP8NoVLMPolicy:
    """DINO -> LAP8 -> LaWM -> released Action Expert, with no VLM allocation."""

    def __init__(
        self,
        *,
        stage1_checkpoint: str | Path,
        lap8_checkpoint: str | Path,
        lap10_checkpoint: str | Path | None,
        lap10v3_checkpoint: str | Path | None,
        joint_checkpoint: str | Path | None,
        lap10v3_stats: str | Path,
        state_stats: str | Path,
        expert_checkpoint: str | Path,
        lam_checkpoint: str | Path,
        lam_yaml: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        started = time.perf_counter()

        stage1 = torch.load(stage1_checkpoint, map_location="cpu", weights_only=True)
        lap8_obj = torch.load(lap8_checkpoint, map_location="cpu", weights_only=True)
        joint_obj = (
            torch.load(joint_checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if joint_checkpoint is not None else None
        )
        if lap10_checkpoint is not None and lap10v3_checkpoint is not None:
            raise ValueError("Choose exactly one of LAP10 or LAP10V3")
        if joint_checkpoint is not None and (lap10_checkpoint is not None or lap10v3_checkpoint is not None):
            raise ValueError("--joint-checkpoint already supplies LAP10V3; do not also pass LAP10/LAP10V3")
        lap6 = LAP60M(num_views=3, view_dropout=0.0)
        if lap10v3_checkpoint is not None or joint_obj is not None:
            lap6.load_state_dict({
                key.removeprefix("lap6."): value
                for key, value in lap8_obj["lap8"].items()
                if key.startswith("lap6.")
            }, strict=True)
            stats = torch.load(lap10v3_stats, map_location="cpu", weights_only=True)
            lap = LAP10V3(lap6, stats["position_mean"], view_dropout=0.0)
            lap10v3_obj = joint_obj if joint_obj is not None else torch.load(
                lap10v3_checkpoint, map_location="cpu", weights_only=True
            )
            lap.load_state_dict(lap10v3_obj["lap10v3"], strict=True)
            self.lap_mode = "lap10v3_joint" if joint_obj is not None else "lap10v3"
        else:
            lap8 = LAP8(lap6, view_dropout=0.0)
            lap8.load_state_dict(lap8_obj["lap8"], strict=True)
            self.lap_mode = "lap8"
        if lap10_checkpoint is not None:
            lap10_obj = torch.load(lap10_checkpoint, map_location="cpu", weights_only=True)
            lap = LAP10(lap8, output_tokens=284)
            lap.load_state_dict(lap10_obj["lap10"], strict=True)
            self.lap_mode = "lap10"
        self.lap = lap.to(self.device, torch.float32).eval()

        self.lam = load_latent_action_model(lam_checkpoint, lam_yaml)
        self.lam.decoder.load_state_dict(stage1["lawm_decoder"], strict=True)
        self.lam = self.lam.to(self.device, torch.float32).eval()

        self.expert = load_action_expert(Path(expert_checkpoint)).to(self.device, torch.float32).eval()
        if joint_obj is not None:
            self.expert.load_state_dict(joint_obj["expert"], strict=True)
            LOGGER.info("Loaded jointly fine-tuned Action Expert from %s", joint_checkpoint)
            del joint_obj
        for module in (self.lap, self.lam, self.expert):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.state_mean, self.state_std = _load_state_stats(state_stats, self.device)
        self.calls = 0
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        LOGGER.info(
            "Loaded FP32 no-VLM policy in %.1fs: %s=%s LAM=%s Expert=%s; CUDA allocated=%.2fGiB reserved=%.2fGiB",
            time.perf_counter() - started,
            self.lap_mode.upper(),
            f"{count_parameters(self.lap):,}",
            f"{count_parameters(self.lam):,}",
            f"{count_parameters(self.expert):,}",
            allocated,
            reserved,
        )

    @staticmethod
    def _decode(decoder, h_t: torch.Tensor, z_lap: torch.Tensor) -> torch.Tensor:
        decoded = decoder(h_t, z_lap)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() == 4:
            decoded = decoded[:, 0] if decoded.shape[1] == 1 else decoded[:, -1]
        if decoded.ndim != 3:
            raise ValueError(f"Expected decoded future tokens [B,K,D], got {tuple(decoded.shape)}")
        return decoded

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        del kwargs
        started = time.perf_counter()
        visual, raw_state, action_hz, embodiment = _build_inputs(examples, device=self.device)
        normalized_state = (_unit_quaternion(raw_state) - self.state_mean) / self.state_std

        features = self.lam.extract_vision_features(visual)
        if tuple(features.shape[1:]) != (3, 256, 768):
            raise ValueError(f"Expected DINO features [B,3,256,768], got {tuple(features.shape)}")
        h_t = features[:, 0]
        lap_out = self.lap(features, normalized_state)
        h_t1 = self._decode(self.lam.decoder, h_t, lap_out["z_lap"])
        batch_size = h_t.shape[0]
        cond_lap = lap_out["cond_lap10"] if self.lap_mode == "lap10" else lap_out["cond_lap"]
        actions = self.expert.sample_actions_cfg(
            h_t=h_t,
            h_t1_star=h_t1,
            h_vlm=None,
            h_lap=cond_lap,
            state=torch.zeros(batch_size, 32, device=self.device, dtype=torch.float32),
            state_mask=torch.zeros(batch_size, 32, device=self.device, dtype=torch.bool),
            action_hz=action_hz,
            embodiment_id=embodiment,
            cfg_scale=1.0,
            num_inference_steps=10,
            attention_mask=torch.ones(batch_size, cond_lap.shape[1], device=self.device, dtype=torch.bool),
            return_padded=False,
        )
        torch.cuda.synchronize(self.device)
        self.calls += 1
        if self.calls <= 3 or self.calls % 20 == 0:
            LOGGER.info(
                "infer call=%d batch=%d output=%s latency=%.1fms peak_cuda=%.2fGiB",
                self.calls,
                batch_size,
                tuple(actions.shape),
                (time.perf_counter() - started) * 1000.0,
                torch.cuda.max_memory_allocated(self.device) / 1024**3,
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Action Expert returned NaN/Inf")
        return map_policy_infer_output(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", default=DEFAULT_POLICY, help="Official checkpoint path used only for client metadata")
    parser.add_argument("--stage1-checkpoint", default=DEFAULT_STAGE1)
    parser.add_argument("--lap8-checkpoint", default=DEFAULT_LAP8)
    parser.add_argument("--lap10-checkpoint", default=None, help="Enable LAP10 and use this 284-token checkpoint for the Expert condition")
    parser.add_argument("--lap10v3-checkpoint", default=None, help="Enable LAP10V3 284-token Expert condition")
    parser.add_argument("--joint-checkpoint", default=None, help="Load jointly fine-tuned LAP10V3 and Action Expert")
    parser.add_argument("--lap10v3-stats", default=DEFAULT_LAP10V3_STATS)
    parser.add_argument("--state-stats", default=DEFAULT_STATE_STATS)
    parser.add_argument("--expert-checkpoint", default=DEFAULT_EXPERT)
    parser.add_argument("--lam-checkpoint", default=DEFAULT_LAM_CKPT)
    parser.add_argument("--lam-yaml", default=DEFAULT_LAM_YAML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11018)
    parser.add_argument("--idle-timeout", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LAP8NoVLMPolicy(
        stage1_checkpoint=args.stage1_checkpoint,
        lap8_checkpoint=args.lap8_checkpoint,
        lap10_checkpoint=args.lap10_checkpoint,
        lap10v3_checkpoint=args.lap10v3_checkpoint,
        joint_checkpoint=args.joint_checkpoint,
        lap10v3_stats=args.lap10v3_stats,
        state_stats=args.state_stats,
        expert_checkpoint=args.expert_checkpoint,
        lam_checkpoint=args.lam_checkpoint,
        lam_yaml=args.lam_yaml,
    )
    metadata = {
        "env": "robotwin",
        "server_type": "lap10v3_expert_joint_no_vlm" if args.joint_checkpoint else ("lap10v3_no_vlm" if args.lap10v3_checkpoint else ("lap10_no_vlm" if args.lap10_checkpoint else "lap8_phase1_no_vlm")),
        "supported_eval_envs": ["robotwin"],
        "ckpt_path": str(Path(args.ckpt_path).expanduser().resolve()),
        "framework_name": "LaWAM-LAP10V3-ExpertJoint-no-VLM" if args.joint_checkpoint else ("LaWAM-LAP10V3-no-VLM" if args.lap10v3_checkpoint else ("LaWAM-LAP10-no-VLM" if args.lap10_checkpoint else "LaWAM-LAP8-no-VLM")),
        "requires_raw_eef": True,
        "vlm_loaded": False,
        "precision": "fp32",
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "lap8_checkpoint": str(Path(args.lap8_checkpoint).resolve()),
        "lap10_checkpoint": str(Path(args.lap10_checkpoint).resolve()) if args.lap10_checkpoint else None,
        "lap10v3_checkpoint": str(Path(args.lap10v3_checkpoint).resolve()) if args.lap10v3_checkpoint else None,
        "joint_checkpoint": str(Path(args.joint_checkpoint).resolve()) if args.joint_checkpoint else None,
    }
    variant = "LAP10V3+ExpertJoint" if args.joint_checkpoint else ("LAP10V3" if args.lap10v3_checkpoint else ("LAP10" if args.lap10_checkpoint else "LAP8"))
    LOGGER.info("Serving %s no-VLM policy on %s:%d", variant, args.host, args.port)
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    torch.set_float32_matmul_precision("high")
    main()
