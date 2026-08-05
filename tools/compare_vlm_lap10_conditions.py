#!/usr/bin/env python3
"""Small held-out A-vs-C diagnostic: official VLM condition versus LAP10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.server_policy import load_policy_from_checkpoint
from starVLA.model.framework.latent_world.batch_utils import (
    imagenet_normalize_image_batch_,
    prepare_frame_spatial_uint8,
)
from starVLA.model.lap_stage1 import LAP60M
from starVLA.model.lap_stage2 import LAP8, LAP10
from tools.build_lap_multiview_cache import WRIST_KEYS, WristFrameReader
from tools.train_lap_stage1 import Task14RawReader, refs_from_manifest


POLICY = ROOT / "results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
LAP8_CKPT = ROOT / "outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt"
LAP10_CKPT = ROOT / "outputs/lap10_alignment_task14_1000step/lap10_step0001000.pt"
STATE_STATS = ROOT / "cache/lap_stage1_task14/state_stats.json"
DATASET = ROOT / "dataset/robotwin_merged"
MANIFEST = ROOT / "cache/lap_stage1_task14/manifest.json"
LANG = "Use the left arm to pick and place the orange bottle for pills or liquid onto the pad."


def per_sample_rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=tuple(range(1, x.ndim))).sqrt()


def mean_std(x: torch.Tensor) -> dict[str, float]:
    return {"mean": float(x.mean().cpu()), "std": float(x.std(unbiased=False).cpu())}


def load_lap10(device: torch.device) -> LAP10:
    lap8_state = torch.load(LAP8_CKPT, map_location="cpu", weights_only=True)
    lap6 = LAP60M(num_views=3, view_dropout=0.0)
    lap8 = LAP8(lap6, view_dropout=0.0)
    lap8.load_state_dict(lap8_state["lap8"], strict=True)
    lap10 = LAP10(lap8, output_tokens=284)
    lap10_state = torch.load(LAP10_CKPT, map_location="cpu", weights_only=True)
    lap10.load_state_dict(lap10_state["lap10"], strict=True)
    return lap10.to(device=device, dtype=torch.float32).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--balanced-domains", action="store_true", help="sample clean/randomized evenly when possible")
    parser.add_argument("--output", default="results/diagnostics/vlm_vs_lap10_task14")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    device = torch.device("cuda")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_refs = refs_from_manifest(manifest, args.split)
    if args.balanced_domains:
        grouped: dict[str, list] = {}
        for ref in all_refs:
            grouped.setdefault(ref.domain, []).append(ref)
        domains = ("clean", "randomized")
        base, remainder = divmod(args.samples, len(domains))
        refs = []
        for position, domain in enumerate(domains):
            candidates = grouped.get(domain, [])
            wanted = min(len(candidates), base + (1 if position < remainder else 0))
            if wanted:
                indices = np.linspace(0, len(candidates) - 1, wanted, dtype=int)
                refs.extend(candidates[int(index)] for index in indices)
        if len(refs) < args.samples:
            selected = {(ref.episode, ref.base_index) for ref in refs}
            refs.extend(ref for ref in all_refs if (ref.episode, ref.base_index) not in selected)
            refs = refs[: args.samples]
    else:
        indices = np.linspace(0, len(all_refs) - 1, min(args.samples, len(all_refs)), dtype=int)
        refs = [all_refs[int(index)] for index in indices]

    print(f"[1/5] loading {len(refs)} held-out {args.split} samples", flush=True)
    main_reader = Task14RawReader(DATASET)
    wrist_reader = WristFrameReader(DATASET)
    examples, raw_states, raw_views = [], [], []
    for ref in refs:
        main_view = np.asarray(main_reader.dataset.get_video(ref.episode, "video.cam_high", ref.base_index))[0]
        wrists = [
            np.asarray(wrist_reader.dataset.get_video(ref.episode, key, ref.base_index))[0]
            for key in WRIST_KEYS
        ]
        state = main_reader.get_raw_states(ref)[0].numpy().astype(np.float32)
        examples.append({
            "lang": LANG,
            "primary_image": [main_view],
            "wrist_image": wrists,
            "state": state,
            "action_hz": 30.0,
            "embodiment_id": 1,
        })
        raw_states.append(state)
        raw_views.append((main_view, *wrists))

    print("[2/5] running official VLM branch", flush=True)
    policy = load_policy_from_checkpoint(str(POLICY), use_bf16=False, device="cuda")
    backend = policy.policy_backend.eval()
    prepared = policy.policy_infer_batch_builder.build_infer_batch(examples)
    with torch.inference_mode():
        shared = backend._run_shared_encoding_infer(
            prepared_batch=prepared, source="lap10_diagnostic", lam_features_with_no_grad=True
        )
        flow = backend.flow
        cond_vlm = flow._prepare_semantic_condition(
            h_vlm=shared.h_vlm, h_lap=None, model_dtype=flow._compute_dtype()
        ).float()

    print("[3/5] running LAP10 on the identical images and EEF states", flush=True)
    lap10 = load_lap10(device)
    visual = torch.stack([
        torch.stack([
            prepare_frame_spatial_uint8(np.asarray(frame), target_hw=(256, 256))
            for frame in views
        ])
        for views in raw_views
    ]).to(device=device, dtype=torch.float32).div_(255.0)
    flat_visual = visual.flatten(0, 1)
    imagenet_normalize_image_batch_(flat_visual)
    visual = flat_visual.view_as(visual)
    states = torch.tensor(np.stack(raw_states), device=device, dtype=torch.float32)
    for start in (3, 11):
        quat = states[:, start : start + 4]
        states[:, start : start + 4] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    state_stats = json.loads(STATE_STATS.read_text(encoding="utf-8"))
    state_mean = torch.tensor(state_stats["mean"], device=device, dtype=torch.float32)
    state_std = torch.tensor(state_stats["std"], device=device, dtype=torch.float32).clamp_min(1e-6)
    with torch.inference_mode():
        dino = backend.lam.extract_vision_features(visual).float()
        lap_out = lap10(dino, (states - state_mean) / state_std)
        cond_lap10 = lap_out["cond_lap10"].float()
    if cond_vlm.shape != cond_lap10.shape:
        raise RuntimeError(f"condition shape mismatch: VLM={tuple(cond_vlm.shape)} LAP10={tuple(cond_lap10.shape)}")

    print("[4/5] sampling Expert actions with matched VLM subgoal and diffusion noise", flush=True)
    batch_size = len(refs)
    zero_state = torch.zeros(batch_size, 32, device=device, dtype=torch.float32)
    zero_mask = torch.zeros(batch_size, 32, device=device, dtype=torch.bool)
    hz = torch.full((batch_size,), 30.0, device=device)
    embodiment = torch.ones(batch_size, device=device, dtype=torch.long)
    with torch.inference_mode():
        torch.manual_seed(1234)
        actions_vlm = flow.sample_actions_cfg(
            h_t=shared.h_t.float(), h_t1_star=shared.h_t1_pred.float(),
            h_vlm=shared.h_vlm.float(), h_lap=None, state=zero_state, state_mask=zero_mask,
            action_hz=hz, embodiment_id=embodiment, cfg_scale=1.0, num_inference_steps=10,
            attention_mask=prepared["attention_mask"], return_padded=False,
        )
        torch.manual_seed(1234)
        actions_lap10 = flow.sample_actions_cfg(
            h_t=shared.h_t.float(), h_t1_star=shared.h_t1_pred.float(),
            h_vlm=None, h_lap=cond_lap10, state=zero_state, state_mask=zero_mask,
            action_hz=hz, embodiment_id=embodiment, cfg_scale=1.0, num_inference_steps=10,
            attention_mask=torch.ones(batch_size, cond_lap10.shape[1], device=device, dtype=torch.bool),
            return_padded=False,
        )

    token_cos = F.cosine_similarity(cond_vlm, cond_lap10, dim=-1)
    token_mse = (cond_vlm - cond_lap10).square().mean(dim=(1, 2))
    action_delta = actions_vlm - actions_lap10
    action_mse = action_delta.square().mean(dim=(1, 2))
    action_mae = action_delta.abs().mean(dim=(1, 2))
    action_cos = F.cosine_similarity(actions_vlm.flatten(1), actions_lap10.flatten(1), dim=-1)
    metrics = {
        "split": args.split,
        "samples": len(refs),
        "language": LANG,
        "refs": [{"episode": ref.episode, "base_index": ref.base_index, "domain": ref.domain} for ref in refs],
        "shapes": {"vlm": list(cond_vlm.shape), "lap10": list(cond_lap10.shape), "actions": list(actions_vlm.shape)},
        "condition": {
            "token_cosine": mean_std(token_cos),
            "token_mse": mean_std(token_mse),
            "vlm_rms": mean_std(per_sample_rms(cond_vlm)),
            "lap10_rms": mean_std(per_sample_rms(cond_lap10)),
        },
        "matched_action": {
            "cosine": mean_std(action_cos),
            "mse": mean_std(action_mse),
            "mae": mean_std(action_mae),
            "vlm_rms": mean_std(per_sample_rms(actions_vlm)),
            "lap10_rms": mean_std(per_sample_rms(actions_lap10)),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "outputs.npz",
        cond_vlm=cond_vlm.cpu().numpy(), cond_lap10=cond_lap10.cpu().numpy(),
        actions_vlm=actions_vlm.cpu().numpy(), actions_lap10=actions_lap10.cpu().numpy(),
    )
    print("[5/5] result", json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
