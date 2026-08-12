#!/usr/bin/env python3
"""VLM-free RoboTwin server using only the Stage-1 LAP6 path and SEC284.

This server intentionally does not load a LAP8 checkpoint and never produces a
LAP8 latent.  The route is:

    three RGB views -> DINO -> LAP6 z_lap -> official LaWM decoder
                              + SEC284(features) -> Action Expert

The released Action Expert receives ``h_vlm=None`` and SEC284 as ``h_lap``.
"""

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
from starVLA.model.sec284 import SEC284Config, SEC284L
from tools.train_lap8_phase1 import load_action_expert


LOGGER = logging.getLogger(__name__)
DEFAULT_POLICY = "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
DEFAULT_STAGE1 = "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
DEFAULT_SEC284 = "outputs/sec284_l_bs32_3000step/step-003000.pt"
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


class LAP6SEC284NoVLMPolicy:
    """Stage-1 LAP6 + SEC284 condition + released Action Expert, no VLM."""

    def __init__(
        self,
        *,
        stage1_checkpoint: str | Path,
        sec284_checkpoint: str | Path,
        state_stats: str | Path,
        expert_checkpoint: str | Path,
        lam_checkpoint: str | Path,
        lam_yaml: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        started = time.perf_counter()

        # The Stage-1 artifact contains the trained LAP6 and official LaWM
        # decoder.  No LAP8 state or model is loaded in this process.
        stage1 = torch.load(stage1_checkpoint, map_location="cpu", weights_only=True)
        if "lap" not in stage1 or "lawm_decoder" not in stage1:
            raise ValueError("Stage-1 checkpoint must contain `lap` and `lawm_decoder` states")
        self.lap = LAP60M(num_views=3, view_dropout=0.0)
        self.lap.load_state_dict(stage1["lap"], strict=True)
        self.lap = self.lap.to(self.device, torch.float32).eval()

        sec_obj = torch.load(sec284_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if "config" not in sec_obj or "sec284" not in sec_obj:
            raise ValueError("SEC284 checkpoint must contain `config` and `sec284` states")
        self.sec284 = SEC284L(SEC284Config(**sec_obj["config"]))
        self.sec284.load_state_dict(sec_obj["sec284"], strict=True)
        self.sec284 = self.sec284.to(self.device, torch.float32).eval()

        self.lam = load_latent_action_model(lam_checkpoint, lam_yaml)
        self.lam.decoder.load_state_dict(stage1["lawm_decoder"], strict=True)
        self.lam = self.lam.to(self.device, torch.float32).eval()

        self.expert = load_action_expert(Path(expert_checkpoint)).to(self.device, torch.float32).eval()
        modules = [self.lap, self.sec284, self.lam, self.expert]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.state_mean, self.state_std = _load_state_stats(state_stats, self.device)
        self.calls = 0
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        LOGGER.info(
            "Loaded LAP6+SEC284 no-VLM policy in %.1fs: LAP6=%s SEC284=%s LAM=%s Expert=%s; "
            "CUDA allocated=%.2fGiB reserved=%.2fGiB; no LAP8 loaded",
            time.perf_counter() - started,
            f"{count_parameters(self.lap):,}",
            f"{count_parameters(self.sec284):,}",
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
        cond_lap = self.sec284(features)
        batch_size = h_t.shape[0]
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
    parser.add_argument("--sec284-checkpoint", default=DEFAULT_SEC284)
    parser.add_argument("--state-stats", default=DEFAULT_STATE_STATS)
    parser.add_argument("--expert-checkpoint", default=DEFAULT_EXPERT)
    parser.add_argument("--lam-checkpoint", default=DEFAULT_LAM_CKPT)
    parser.add_argument("--lam-yaml", default=DEFAULT_LAM_YAML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11052)
    parser.add_argument("--idle-timeout", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LAP6SEC284NoVLMPolicy(
        stage1_checkpoint=args.stage1_checkpoint,
        sec284_checkpoint=args.sec284_checkpoint,
        state_stats=args.state_stats,
        expert_checkpoint=args.expert_checkpoint,
        lam_checkpoint=args.lam_checkpoint,
        lam_yaml=args.lam_yaml,
    )
    metadata = {
        "env": "robotwin",
        "server_type": "lap6_sec284_no_vlm",
        "supported_eval_envs": ["robotwin"],
        "ckpt_path": str(Path(args.ckpt_path).expanduser().resolve()),
        "framework_name": "LaWAM-LAP6-SEC284-no-VLM",
        "requires_raw_eef": True,
        "vlm_loaded": False,
        "precision": "fp32",
        "lap_module": "LAP6",
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).expanduser().resolve()),
        "sec284_checkpoint": str(Path(args.sec284_checkpoint).expanduser().resolve()),
        "lap8_checkpoint": None,
    }
    LOGGER.info("Serving LAP6+SEC284 no-VLM policy on %s:%d (no LAP8)", args.host, args.port)
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
