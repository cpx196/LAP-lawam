#!/usr/bin/env python3
"""Compare official VLM and VLM-free LAP8 conditions on fresh Task-14 data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.server_policy import load_policy_from_checkpoint
from starVLA.model.framework.latent_world.batch_utils import (
    imagenet_normalize_image_batch_,
    prepare_frame_spatial_uint8,
)
from starVLA.model.lap_stage1 import LAP60M
from starVLA.model.lap_stage2 import LAP8
from tools.build_lap_multiview_cache import WRIST_KEYS, WristFrameReader
from tools.train_lap8_phase1 import load_action_expert
from tools.train_lap_stage1 import Task14RawReader, refs_from_manifest


DEVICE = torch.device("cuda")
POLICY = ROOT / "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
STAGE1 = ROOT / "outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt"
LAP8_CKPT = ROOT / "outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt"
STATE_STATS = ROOT / "cache/lap_stage1_task14/state_stats.json"
DATASET = ROOT / "dataset/robotwin_merged"
OUTPUT = ROOT / "results/diagnostics/vlm_vs_lap8_task14"
LANG = "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad."


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x, dim=tuple(range(1, x.ndim))))


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    gram = x @ x.transpose(-1, -2)
    mask = ~torch.eye(gram.shape[-1], device=x.device, dtype=torch.bool).unsqueeze(0)
    return gram.masked_select(mask).reshape(x.shape[0], -1).mean(-1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "cache/lap_stage1_task14/manifest.json").read_text())
    refs_all = refs_from_manifest(manifest, "test")
    sample_indices = np.linspace(0, len(refs_all) - 1, 4, dtype=int).tolist()
    refs = [refs_all[index] for index in sample_indices]

    print("[1/6] reading fresh Task-14 samples", flush=True)
    main_reader = Task14RawReader(DATASET)
    wrist_reader = WristFrameReader(DATASET)
    examples = []
    raw_states = []
    raw_views = []
    for ref in refs:
        main = np.asarray(main_reader.dataset.get_video(ref.episode, "video.cam_high", ref.base_index))[0]
        wrists = [
            np.asarray(wrist_reader.dataset.get_video(ref.episode, key, ref.base_index))[0]
            for key in WRIST_KEYS
        ]
        state = main_reader.get_raw_states(ref)[0].numpy().astype(np.float32)
        examples.append(
            {
                "lang": LANG,
                "primary_image": [main],
                "wrist_image": wrists,
                "state": state,
                "action_hz": 30.0,
                "embodiment_id": 1,
            }
        )
        raw_states.append(state)
        raw_views.append((main, *wrists))
    print("refs=", [(r.episode, r.base_index, r.domain) for r in refs], flush=True)

    print("[2/6] loading official FP32 policy and VLM", flush=True)
    policy = load_policy_from_checkpoint(str(POLICY), use_bf16=False, device="cuda")
    backend = policy.policy_backend.eval()

    print("[3/6] extracting official VLM conditions", flush=True)
    batch = policy.policy_infer_batch_builder.build_infer_batch(examples)
    with torch.inference_mode():
        shared = backend._run_shared_encoding_infer(
            prepared_batch=batch, source="condition_diagnostic", lam_features_with_no_grad=True
        )
        flow = backend.flow
        model_dtype = flow._compute_dtype()
        cond_vlm = flow._prepare_semantic_condition(
            h_vlm=shared.h_vlm, h_lap=None, model_dtype=model_dtype
        ).float()
    flow_mask = batch["flow_placeholder_mask"].bool()
    flow_queries = int(backend.flow_action_query.shape[0])
    cond_vlm_flow = cond_vlm[flow_mask].reshape(len(refs), flow_queries, -1)
    print(
        "h_vlm=", tuple(shared.h_vlm.shape),
        "cond_vlm_all=", tuple(cond_vlm.shape),
        "cond_vlm_flow=", tuple(cond_vlm_flow.shape),
        flush=True,
    )

    print("[4/6] extracting LAP8 conditions on identical images/states", flush=True)
    lap8_obj = torch.load(LAP8_CKPT, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap8 = LAP8(lap6, view_dropout=0.0)
    lap8.load_state_dict(lap8_obj["lap8"], strict=True)
    lap8 = lap8.to(DEVICE, torch.float32).eval()
    state_stats = json.loads(STATE_STATS.read_text())
    state_mean = torch.tensor(state_stats["mean"], device=DEVICE, dtype=torch.float32)
    state_std = torch.tensor(state_stats["std"], device=DEVICE, dtype=torch.float32).clamp_min(1e-6)

    visual = torch.stack(
        [
            torch.stack(
                [prepare_frame_spatial_uint8(np.asarray(frame), target_hw=(256, 256)) for frame in views]
            )
            for views in raw_views
        ]
    ).to(DEVICE, torch.float32).div_(255.0)
    flat = visual.flatten(0, 1)
    imagenet_normalize_image_batch_(flat)
    visual = flat.view_as(visual)
    states = torch.tensor(np.stack(raw_states), device=DEVICE, dtype=torch.float32)
    for start in (3, 11):
        quat = states[:, start : start + 4]
        states[:, start : start + 4] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    lap_state = (states - state_mean) / state_std
    with torch.inference_mode():
        features = backend.lam.extract_vision_features(visual).float()
        lap_out = lap8(features, lap_state)
        cond_lap = lap_out["cond_lap"].float()
    print("DINO=", tuple(features.shape), "cond_lap=", tuple(cond_lap.shape), flush=True)

    print("[5/6] computing representation and matched-action metrics", flush=True)
    metrics = {
        "refs": [
            {"episode": r.episode, "base_index": r.base_index, "domain": r.domain}
            for r in refs
        ],
        "language": LANG,
        "shapes": {
            "h_vlm": list(shared.h_vlm.shape),
            "cond_vlm_all": list(cond_vlm.shape),
            "cond_vlm_flow": list(cond_vlm_flow.shape),
            "cond_lap": list(cond_lap.shape),
        },
        "all_vlm_rms": rms(cond_vlm).cpu().tolist(),
        "flow_vlm_rms": rms(cond_vlm_flow).cpu().tolist(),
        "lap8_rms": rms(cond_lap).cpu().tolist(),
        "flow_vlm_mean": cond_vlm_flow.mean(dim=(1, 2)).cpu().tolist(),
        "lap8_mean": cond_lap.mean(dim=(1, 2)).cpu().tolist(),
        "flow_vlm_std": cond_vlm_flow.std(dim=(1, 2)).cpu().tolist(),
        "lap8_std": cond_lap.std(dim=(1, 2)).cpu().tolist(),
        "paired_token_cosine": cosine(cond_vlm_flow, cond_lap).mean(dim=1).cpu().tolist(),
        "paired_token_mse": ((cond_vlm_flow - cond_lap) ** 2).mean(dim=(1, 2)).cpu().tolist(),
        "within_token_cosine_vlm_flow": off_diagonal(cond_vlm_flow).cpu().tolist(),
        "within_token_cosine_lap8": off_diagonal(cond_lap).cpu().tolist(),
    }

    zero_state = torch.zeros(len(refs), 32, device=DEVICE, dtype=torch.float32)
    zero_mask = torch.zeros(len(refs), 32, device=DEVICE, dtype=torch.bool)
    hz = torch.full((len(refs),), 30.0, device=DEVICE)
    embodiment = torch.ones(len(refs), device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        torch.manual_seed(1234)
        action_vlm = flow.sample_actions_cfg(
            h_t=shared.h_t.float(), h_t1_star=shared.h_t1_pred.float(),
            h_vlm=shared.h_vlm.float(), h_lap=None, state=zero_state,
            state_mask=zero_mask, action_hz=hz, embodiment_id=embodiment,
            cfg_scale=1.0, num_inference_steps=10,
            attention_mask=batch["attention_mask"], return_padded=False,
        )
        torch.manual_seed(1234)
        action_lap = flow.sample_actions_cfg(
            h_t=shared.h_t.float(), h_t1_star=shared.h_t1_pred.float(),
            h_vlm=None, h_lap=cond_lap, state=zero_state,
            state_mask=zero_mask, action_hz=hz, embodiment_id=embodiment,
            cfg_scale=1.0, num_inference_steps=10,
            attention_mask=torch.ones(len(refs), cond_lap.shape[1], device=DEVICE, dtype=torch.bool),
            return_padded=False,
        )
    metrics["action_shapes"] = {"vlm": list(action_vlm.shape), "lap8": list(action_lap.shape)}
    metrics["matched_action_rms_vlm"] = rms(action_vlm).cpu().tolist()
    metrics["matched_action_rms_lap8"] = rms(action_lap).cpu().tolist()
    metrics["matched_action_mse"] = ((action_vlm - action_lap) ** 2).mean(dim=(1, 2)).cpu().tolist()
    metrics["matched_action_mae"] = torch.abs(action_vlm - action_lap).mean(dim=(1, 2)).cpu().tolist()

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUTPUT / "conditions.npz",
        cond_vlm_flow=cond_vlm_flow.cpu().numpy(),
        cond_lap=cond_lap.cpu().numpy(),
        actions_vlm=action_vlm.cpu().numpy(),
        actions_lap8=action_lap.cpu().numpy(),
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print("saved", OUTPUT, flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
