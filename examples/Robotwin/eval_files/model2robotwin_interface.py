from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


def invert_gripper_value(gripper: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(gripper, dtype=np.float32)


def normalize_quaternion_value(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[None, :]

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    normalized = np.zeros_like(quat, dtype=np.float32)
    valid_rows = np.isfinite(norm[:, 0]) & (norm[:, 0] > 1e-8)
    if np.any(valid_rows):
        normalized[valid_rows] = quat[valid_rows] / norm[valid_rows]
    invalid = ~valid_rows
    if np.any(invalid):
        normalized[invalid, 0] = 1.0
    return normalized


_ROBOTWIN_JOINT_DATA_MIXES = {"robotwin_joint"}
_ROBOTWIN_EEF_DATA_MIXES = {
    "robotwin_eef": 50.0,
    "robotwin_eef_all": 50.0,
    "robotwin_eef_30hz": 30.0,
}


@dataclass(frozen=True)
class RobotwinControlSpec:
    data_mix: str
    mode: str
    env_action_type: str
    action_hz: float
    state_gripper_indices: tuple[int, ...]
    passthrough_indices: tuple[int, ...]
    gripper_is_binary: bool


def resolve_robotwin_control_from_data_mix(data_mix: Any) -> RobotwinControlSpec:
    normalized = str(data_mix).strip()
    if normalized in _ROBOTWIN_JOINT_DATA_MIXES:
        return RobotwinControlSpec(
            data_mix=normalized,
            mode="joint",
            env_action_type="qpos",
            action_hz=30.0,
            state_gripper_indices=(6, 13),
            passthrough_indices=(),
            gripper_is_binary=True,
        )
    if normalized in _ROBOTWIN_EEF_DATA_MIXES:
        return RobotwinControlSpec(
            data_mix=normalized,
            mode="eef",
            env_action_type="ee",
            action_hz=_ROBOTWIN_EEF_DATA_MIXES[normalized],
            state_gripper_indices=(7, 15),
            passthrough_indices=(3, 4, 5, 6, 11, 12, 13, 14),
            gripper_is_binary=False,
        )
    raise ValueError(
        "Unsupported Robotwin data_mix for eval mode resolution: "
        f"{normalized!r}. Expected one of {sorted(_ROBOTWIN_JOINT_DATA_MIXES | set(_ROBOTWIN_EEF_DATA_MIXES))}."
    )


def _extract_robotwin_data_mix(model_config: dict[str, Any]) -> str:
    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets` dict; cannot resolve Robotwin action mode.")
    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets.vla_data`; cannot resolve Robotwin action mode.")
    data_mix = vla_data_cfg.get("data_mix", None)
    if data_mix is None:
        raise ValueError("Checkpoint config is missing `datasets.vla_data.data_mix`; cannot resolve Robotwin action mode.")
    return str(data_mix)


def flatten_robotwin_endpose_state(observation: dict[str, Any]) -> np.ndarray:
    endpose = observation.get("endpose", None)
    if not isinstance(endpose, dict):
        raise KeyError("Robotwin EEF eval requires observation['endpose'].")

    required_keys = (
        "left_endpose",
        "left_gripper",
        "right_endpose",
        "right_gripper",
    )
    missing = [key for key in required_keys if key not in endpose]
    if missing:
        raise KeyError(f"Robotwin EEF eval observation is missing endpose keys: {missing}")

    state = np.concatenate(
        [
            np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1),
            np.asarray([endpose["left_gripper"]], dtype=np.float32).reshape(-1),
            np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1),
            np.asarray([endpose["right_gripper"]], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )
    return np.asarray(state, dtype=np.float32)


def build_robotwin_example(
    task_description: str,
    observation: dict[str, Any],
    *,
    robotwin_mode: str = "joint",
) -> dict[str, Any]:
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    if robotwin_mode == "joint":
        state = observation.get("joint_action", {}).get("vector", None)
    elif robotwin_mode == "eef":
        state = flatten_robotwin_endpose_state(observation)
    else:
        raise ValueError(f"Unsupported Robotwin mode: {robotwin_mode!r}")
    return {
        "lang": str(task_description),
        "image": [head_img, left_img, right_img],
        "state": state,
    }


@dataclass
class _SlotState:
    task_description: str | None = None
    raw_actions: Optional[np.ndarray] = None
    action_cursor: int = 0
    executed_steps: int = 0
    action_ensemble_history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, task_description: str | None = None) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.action_cursor = 0
        self.executed_steps = 0
        self.action_ensemble_history.clear()

    def needs_query(self) -> bool:
        return self.raw_actions is None or self.action_cursor >= int(self.raw_actions.shape[0])


class ModelClient:
    _DEFAULT_EMBODIMENT_ID = 1
    _DEFAULT_IMAGE_SIZE = (256, 256)
    _DEFAULT_BINARY_THRESHOLD = 0.49

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        image_size: Optional[Sequence[int]] = None,
        replan_steps: Optional[int | str] = None,
        action_ensemble: Optional[bool | str] = None,
        action_ensemble_alpha: Optional[float | str] = None,
        action_reorder: Optional[Sequence[int] | str] = None,
        host: str = "127.0.0.1",
        port: int = 5694,
    ) -> None:
        self.policy_ckpt_path = Path(policy_ckpt_path).expanduser().resolve()
        self.model_config, self.norm_stats = read_mode_config(self.policy_ckpt_path)
        framework_name = self.model_config.get("framework", {}).get("name", "")
        self.framework_name = str(framework_name)
        self.data_mix = _extract_robotwin_data_mix(self.model_config)
        self.control_spec = resolve_robotwin_control_from_data_mix(self.data_mix)
        self.robotwin_mode = self.control_spec.mode
        self.env_action_type = self.control_spec.env_action_type
        self.state_gripper_indices = self.control_spec.state_gripper_indices
        self.passthrough_indices = self.control_spec.passthrough_indices
        self.gripper_is_binary = bool(self.control_spec.gripper_is_binary)
        flow_cfg = self.model_config.get("framework", {}).get("action_model", {}).get("flow_cfg", {})
        self.use_state = bool(flow_cfg.get("use_state", True))

        self.unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        self.dataset_stats = self.norm_stats[self.unnorm_key]
        self.action_norm_stats = self.dataset_stats["action"]
        self.state_norm_stats = self.dataset_stats.get("state", None)
        self.action_dim = int(np.asarray(self.action_norm_stats["min"], dtype=np.float32).shape[0])
        self.action_binary_indices = self._resolve_action_binary_indices(
            self.action_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_binary_indices = self._resolve_state_binary_indices(
            self.state_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_invert_indices = self._resolve_state_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )
        self.action_invert_indices = self._resolve_action_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )

        resolved_image_size = self._resolve_image_size(self.model_config, image_size)
        self.image_size = [int(resolved_image_size[0]), int(resolved_image_size[1])]
        self.action_hz = float(self.control_spec.action_hz)
        if self.action_hz <= 0.0:
            raise ValueError(f"`action_hz` must be > 0, got {self.action_hz}.")
        self.replan_steps = self._resolve_replan_steps(replan_steps)
        self.action_ensemble = self._resolve_action_ensemble_enabled(action_ensemble)
        self.action_ensemble_alpha = self._resolve_action_ensemble_alpha(action_ensemble_alpha)
        if self.action_ensemble and self.replan_steps is None:
            raise ValueError("`action_ensemble` requires a positive `replan_steps` value.")
        self.action_reorder = self._resolve_action_reorder(action_reorder, action_dim=self.action_dim)
        self.embodiment_id = int(self._DEFAULT_EMBODIMENT_ID)
        self.client = WebsocketClientPolicy(host, port)
        self.server_metadata = self.client.get_server_metadata() or {}
        # The official Robotwin policy does not use state in its flow head.
        # A LAP/LaWM ablation server nevertheless needs the raw 16-D EEF state
        # for LAP, so this capability is advertised explicitly by the server.
        self.requires_raw_eef = bool(self.server_metadata.get("requires_raw_eef", False))
        self._validate_server_metadata()
        self._slot_states: dict[int, _SlotState] = {}

        print(
            "*** "
            f"framework: {self.framework_name}, "
            f"data_mix: {self.data_mix}, "
            f"robotwin_mode: {self.robotwin_mode}, "
            f"unnorm_key: {self.unnorm_key}, "
            f"use_state: {self.use_state}, "
            f"image_size: {self.image_size}, "
            f"action_hz: {self.action_hz}, "
            f"replan_steps: {self.replan_steps}, "
            f"action_ensemble: {self.action_ensemble}, "
            f"action_ensemble_alpha: {self.action_ensemble_alpha}, "
            f"action_reorder: {self.action_reorder}, "
            f"embodiment_id: {self.embodiment_id}, "
            f"env_action_type: {self.env_action_type}, "
            f"state_gripper_indices: {self.state_gripper_indices}, "
            f"passthrough_indices: {self.passthrough_indices}, "
            f"action_binary_indices: {self.action_binary_indices}, "
            f"action_invert_indices: {self.action_invert_indices}, "
            "***"
        )

    def close(self) -> None:
        self.client.close()

    def reset(self, task_description: str = "", slot_id: int = 0) -> None:
        self._get_slot_state(slot_id).reset(task_description=task_description)

    def reset_slots(self, slot_ids: Optional[Iterable[int]] = None) -> None:
        if slot_ids is None:
            self._slot_states.clear()
            return
        for slot_id in slot_ids:
            self._slot_states.pop(int(slot_id), None)

    def needs_query(self, slot_id: int = 0, task_description: Optional[str] = None) -> bool:
        slot_state = self._get_slot_state(slot_id)
        if task_description is not None:
            resolved_task = str(task_description)
            if resolved_task != slot_state.task_description:
                slot_state.reset(task_description=resolved_task)
        return slot_state.needs_query()

    def step_cached(self, slot_id: int = 0, task_description: Optional[str] = None) -> np.ndarray:
        if self.needs_query(slot_id=slot_id, task_description=task_description):
            raise RuntimeError(f"Slot {slot_id} does not have any cached actions available.")
        return self._pop_slot_action(slot_id)

    def get_action_trace_snapshot(self, slot_id: int = 0) -> dict[str, Any]:
        """Return a copy of the current chunk state for optional rollout diagnostics."""
        slot_state = self._get_slot_state(int(slot_id))
        raw_actions = None
        if slot_state.raw_actions is not None:
            raw_actions = np.asarray(slot_state.raw_actions, dtype=np.float32).copy()
        return {
            "raw_actions": raw_actions,
            "action_cursor": int(slot_state.action_cursor),
            "executed_steps": int(slot_state.executed_steps),
        }

    def step(self, example: dict[str, Any], step: int = 0, slot_id: int = 0) -> np.ndarray:
        del step
        return self.step_batch([example], slot_ids=[slot_id])[0]

    def build_example(self, task_description: str, observation: dict[str, Any]) -> dict[str, Any]:
        return build_robotwin_example(
            task_description,
            observation,
            robotwin_mode=self.robotwin_mode,
        )

    def step_batch(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        slot_ids: Optional[Sequence[int]] = None,
    ) -> list[np.ndarray]:
        if slot_ids is None:
            slot_ids = list(range(len(examples)))
        if len(examples) != len(slot_ids):
            raise ValueError(f"`examples` and `slot_ids` must have the same length, got {len(examples)} vs {len(slot_ids)}.")

        prepared_examples: list[dict[str, Any]] = []
        slots_to_query: list[int] = []
        query_examples: list[dict[str, Any]] = []
        output_actions: list[np.ndarray | None] = [None] * len(examples)

        for idx, (example, slot_id) in enumerate(zip(examples, slot_ids)):
            if not isinstance(example, dict):
                raise TypeError(f"`example` must be dict, got {type(example)}.")
            slot_id = int(slot_id)
            slot_state = self._get_slot_state(slot_id)

            prepared = self._prepare_example(example)
            prepared_examples.append(prepared)
            task_description = str(prepared["lang"])
            if task_description != slot_state.task_description:
                slot_state.reset(task_description=task_description)

            if slot_state.needs_query():
                slots_to_query.append(idx)
                query_examples.append(self._build_infer_example(prepared))

        if query_examples:
            response = self.client.predict_action({"examples": query_examples})
            try:
                normalized_actions = response["data"]["normalized_actions"]
            except KeyError as exc:
                raise KeyError(f"Key `normalized_actions` not found in response: {response}") from exc

            normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
            if normalized_actions.ndim == 2:
                normalized_actions = normalized_actions[None, :, :]
            if normalized_actions.ndim != 3:
                raise ValueError(
                    f"`normalized_actions` must have shape [B, T, D] or [T, D], got {normalized_actions.shape}."
                )

            if int(normalized_actions.shape[0]) != len(query_examples):
                raise ValueError(
                    f"Expected {len(query_examples)} batched predictions, got shape {normalized_actions.shape}."
                )

            for batch_idx, example_idx in enumerate(slots_to_query):
                slot_id = int(slot_ids[example_idx])
                slot_state = self._get_slot_state(slot_id)
                raw_actions = self.unnormalize_actions(
                    normalized_actions=normalized_actions[batch_idx],
                    action_norm_stats=self.action_norm_stats,
                    gripper_indices=self.action_binary_indices,
                    passthrough_indices=self.passthrough_indices,
                    invert_indices=self.action_invert_indices,
                )
                if getattr(self, "action_ensemble", False):
                    raw_actions = self._build_action_ensemble_cache(slot_state, raw_actions)
                else:
                    raw_actions = self._truncate_action_chunk(raw_actions, replan_steps=self.replan_steps)
                if int(raw_actions.shape[0]) <= 0:
                    raise RuntimeError("Inference returned empty action chunk.")
                slot_state.raw_actions = raw_actions
                slot_state.action_cursor = 0

        for idx, slot_id in enumerate(slot_ids):
            output_actions[idx] = self._pop_slot_action(int(slot_id))

        return [np.asarray(action, dtype=np.float32) for action in output_actions if action is not None]

    def _pop_slot_action(self, slot_id: int) -> np.ndarray:
        slot_state = self._get_slot_state(int(slot_id))
        if slot_state.raw_actions is None or slot_state.action_cursor >= int(slot_state.raw_actions.shape[0]):
            raise RuntimeError(f"Slot {slot_id} does not have any cached actions after inference.")
        current_action = np.asarray(slot_state.raw_actions[slot_state.action_cursor], dtype=np.float32)
        slot_state.action_cursor += 1
        slot_state.executed_steps += 1
        return self._remap_action_for_robotwin_env(current_action)

    def _get_slot_state(self, slot_id: int) -> _SlotState:
        slot_id = int(slot_id)
        if slot_id not in self._slot_states:
            self._slot_states[slot_id] = _SlotState()
        return self._slot_states[slot_id]

    def _prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        images = example.get("image", None)
        if not isinstance(images, (list, tuple)) or len(images) == 0:
            raise ValueError("Robotwin example must contain non-empty `image` list.")
        resized_images = [self._resize_image(np.asarray(image)) for image in images]
        return {
            "lang": str(example.get("lang", "")),
            "image": resized_images,
            "state": self._prepare_state(example.get("state", None)),
        }

    def _build_infer_example(self, example: dict[str, Any]) -> dict[str, Any]:
        images = list(example["image"])
        infer_example: dict[str, Any] = {
            "lang": str(example["lang"]),
            "primary_image": [images[0]],
            "embodiment_id": int(self.embodiment_id),
            "action_hz": float(self.action_hz),
        }
        wrist_images = list(images[1:])
        if wrist_images:
            infer_example["wrist_image"] = wrist_images
        if example.get("state", None) is not None:
            infer_example["state"] = np.asarray(example["state"], dtype=np.float32)
        return infer_example

    def _prepare_state(self, state: Any) -> Optional[np.ndarray]:
        if state is None:
            return None
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 2 and state_array.shape[0] == 1:
            state_array = state_array[0]
        if state_array.ndim != 1:
            raise ValueError(f"`state` must have shape [D], got {state_array.shape}.")
        if self.requires_raw_eef:
            return state_array
        if not self.use_state:
            return None
        if self.state_norm_stats is None:
            return state_array
        return self.normalize_state(
            state=state_array,
            state_norm_stats=self.state_norm_stats,
            binary_indices=self.state_binary_indices,
            passthrough_indices=self.passthrough_indices,
            invert_indices=self.state_invert_indices,
            binary_threshold=self._DEFAULT_BINARY_THRESHOLD,
        )

    @staticmethod
    def normalize_state(
        state: np.ndarray,
        state_norm_stats: Dict[str, np.ndarray],
        binary_indices: tuple[int, ...],
        passthrough_indices: tuple[int, ...] = (),
        invert_indices: tuple[int, ...] = (),
        binary_threshold: float = _DEFAULT_BINARY_THRESHOLD,
    ) -> np.ndarray:
        high_key = "max" if "max" in state_norm_stats else "q99"
        low_key = "min" if "min" in state_norm_stats else "q01"
        state_high = np.asarray(state_norm_stats[high_key], dtype=np.float32)
        state_low = np.asarray(state_norm_stats[low_key], dtype=np.float32)

        normalized = np.asarray(state, dtype=np.float32).copy()
        stats_dim = int(state_high.shape[0])
        if normalized.shape[-1] != stats_dim:
            if normalized.shape[-1] < stats_dim:
                raise ValueError(
                    "`state` dim does not match checkpoint statistics: "
                    f"state_dim={normalized.shape[-1]}, stats_dim={stats_dim}."
                )
            normalized = normalized[..., :stats_dim]

        binary_idx = ModelClient._bounded_indices(normalized.shape[-1], binary_indices)
        passthrough_idx = ModelClient._bounded_indices(normalized.shape[-1], passthrough_indices)
        continuous_mask = np.ones(stats_dim, dtype=bool)
        if binary_idx.size > 0:
            continuous_mask[binary_idx] = False
        if passthrough_idx.size > 0:
            continuous_mask[passthrough_idx] = False

        denom = state_high - state_low
        valid = np.abs(denom) > 1e-12
        valid &= continuous_mask
        if np.any(valid):
            normalized[..., valid] = (normalized[..., valid] - state_low[valid]) / denom[valid] * 2.0 - 1.0
            normalized[..., valid] = np.clip(normalized[..., valid], -1.0, 1.0)
        zero_mask = ~valid & continuous_mask
        if np.any(zero_mask):
            normalized[..., zero_mask] = 0.0
        if binary_idx.size > 0:
            normalized[..., binary_idx] = (normalized[..., binary_idx] > binary_threshold).astype(np.float32)
        invert_idx = ModelClient._bounded_indices(normalized.shape[-1], invert_indices)
        if invert_idx.size > 0:
            binary_invert_idx = np.intersect1d(invert_idx, binary_idx, assume_unique=False)
            continuous_invert_idx = np.setdiff1d(invert_idx, binary_idx, assume_unique=False)
            if binary_invert_idx.size > 0:
                normalized[..., binary_invert_idx] = invert_gripper_value(
                    normalized[..., binary_invert_idx]
                )
            if continuous_invert_idx.size > 0:
                normalized[..., continuous_invert_idx] = -normalized[..., continuous_invert_idx]

        return normalized

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        gripper_indices: tuple[int, ...] = (),
        passthrough_indices: tuple[int, ...] = (),
        invert_indices: tuple[int, ...] = (),
        binary_threshold: float = _DEFAULT_BINARY_THRESHOLD,
    ) -> np.ndarray:
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"
        action_high = np.asarray(action_norm_stats[high_key], dtype=np.float32)
        action_low = np.asarray(action_norm_stats[low_key], dtype=np.float32)
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))

        normalized_actions = np.array(normalized_actions, dtype=np.float32, copy=True)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        stats_dim = int(action_high.shape[0])
        if normalized_actions.shape[-1] != stats_dim:
            if normalized_actions.shape[-1] < stats_dim:
                raise ValueError(
                    "`normalized_actions` dim does not match checkpoint statistics: "
                    f"action_dim={normalized_actions.shape[-1]}, stats_dim={stats_dim}."
                )
            normalized_actions = normalized_actions[..., :stats_dim]

        gripper_idx = ModelClient._bounded_indices(stats_dim, gripper_indices)
        passthrough_idx = ModelClient._bounded_indices(stats_dim, passthrough_indices)
        denorm_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if passthrough_idx.size > 0:
            denorm_mask[passthrough_idx] = False
        if gripper_idx.size > 0:
            normalized_actions[:, gripper_idx] = np.where(
                normalized_actions[:, gripper_idx] < binary_threshold,
                0.0,
                1.0,
            )

        actions = normalized_actions.copy()
        if np.any(denorm_mask):
            clipped = np.clip(normalized_actions[:, denorm_mask], -1.0, 1.0)
            actions[:, denorm_mask] = (
                0.5 * (clipped + 1.0) * (action_high[denorm_mask] - action_low[denorm_mask])
                + action_low[denorm_mask]
            )
        if passthrough_idx.size > 0:
            actions[:, passthrough_idx] = normalized_actions[:, passthrough_idx]
        invert_idx = ModelClient._bounded_indices(stats_dim, invert_indices)
        if invert_idx.size > 0:
            actions[:, invert_idx] = invert_gripper_value(actions[:, invert_idx])
        ModelClient._normalize_robotwin_eef_quaternions_(actions, passthrough_indices)
        return actions

    @staticmethod
    def _normalize_robotwin_eef_quaternions_(
        actions: np.ndarray,
        passthrough_indices: tuple[int, ...] = (),
    ) -> None:
        passthrough_set = set(int(idx) for idx in passthrough_indices)
        quaternion_groups = ((3, 4, 5, 6), (11, 12, 13, 14))
        for group in quaternion_groups:
            if not set(group).issubset(passthrough_set):
                continue
            if actions.shape[-1] <= group[-1]:
                continue
            actions[:, group[0] : group[-1] + 1] = normalize_quaternion_value(
                actions[:, group[0] : group[-1] + 1]
            )

    @staticmethod
    def _bounded_indices(dim: int, indices: tuple[int, ...] | list[int] | np.ndarray) -> np.ndarray:
        resolved = np.asarray(indices, dtype=np.int64).reshape(-1)
        return resolved[(resolved >= 0) & (resolved < int(dim))]

    @staticmethod
    def _infer_binary_indices(state_norm_stats: Optional[dict]) -> tuple[int, ...]:
        if state_norm_stats is None:
            return ()
        mask = state_norm_stats.get("mask", None)
        if mask is not None:
            low_key = "min" if "min" in state_norm_stats else "q01"
            state_low = np.asarray(state_norm_stats[low_key], dtype=np.float32)
            dim = int(state_low.shape[0])
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] == dim:
                return tuple(np.where(~mask_arr)[0].tolist())
        return ()

    @classmethod
    def _resolve_state_binary_indices(
        cls,
        state_norm_stats: Optional[dict],
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
        gripper_is_binary: bool,
    ) -> tuple[int, ...]:
        inferred = cls._infer_binary_indices(state_norm_stats)
        if robotwin_mode in {"joint", "eef"} and gripper_is_binary:
            return tuple(sorted(set(inferred) | set(state_gripper_indices)))
        return inferred

    @staticmethod
    def _resolve_state_invert_indices(
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if robotwin_mode in {"joint", "eef"}:
            return tuple(state_gripper_indices)
        return ()

    @staticmethod
    def _resolve_action_invert_indices(
        robotwin_mode: str,
        action_gripper_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if robotwin_mode in {"joint", "eef"}:
            return tuple(action_gripper_indices)
        return ()

    @classmethod
    def _resolve_action_binary_indices(
        cls,
        action_norm_stats: dict,
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
        gripper_is_binary: bool,
    ) -> tuple[int, ...]:
        expected = tuple(int(idx) for idx in state_gripper_indices)
        inferred = cls._infer_binary_indices(action_norm_stats)
        legacy = cls._legacy_action_gripper_indices(robotwin_mode)
        if inferred == legacy:
            raise ValueError(
                "Robotwin checkpoint statistics use legacy action order. "
                f"mode={robotwin_mode}, legacy_gripper_indices={legacy}, expected={expected}. "
                "Regenerate `dataset_statistics.json` before running benchmark."
            )
        if gripper_is_binary and inferred in {(), expected}:
            return expected
        if not gripper_is_binary and inferred in {(), expected}:
            return inferred
        raise ValueError(
            "Robotwin checkpoint statistics have unexpected action.mask layout. "
            f"mode={robotwin_mode}, inferred_gripper_indices={inferred}, expected={expected}. "
            "Regenerate `dataset_statistics.json` before running benchmark."
        )

    @staticmethod
    def _legacy_action_gripper_indices(robotwin_mode: str) -> tuple[int, ...]:
        if robotwin_mode == "joint":
            return (12, 13)
        if robotwin_mode == "eef":
            return (14, 15)
        raise ValueError(f"Unsupported Robotwin mode for legacy layout detection: {robotwin_mode}")

    def _validate_server_metadata(self) -> None:
        metadata = self.client.get_server_metadata() or {}
        server_ckpt = metadata.get("ckpt_path", None)
        if server_ckpt is None:
            return
        server_ckpt_path = Path(str(server_ckpt)).expanduser().resolve()
        if server_ckpt_path != self.policy_ckpt_path:
            raise ValueError(
                "Checkpoint mismatch between eval client and server: "
                f"client={self.policy_ckpt_path}, server={server_ckpt_path}."
            )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _resolve_action_reorder(
        action_reorder: Optional[Sequence[int] | str],
        *,
        action_dim: int,
    ) -> Optional[tuple[int, ...]]:
        if action_reorder is None:
            return None
        if isinstance(action_reorder, str):
            stripped = action_reorder.strip()
            if not stripped:
                return None
            resolved = tuple(int(part.strip()) for part in stripped.split(","))
        else:
            resolved = tuple(int(part) for part in action_reorder)
        if len(resolved) != int(action_dim):
            raise ValueError(
                f"`action_reorder` must contain exactly {action_dim} indices, got {resolved}."
            )
        if tuple(sorted(resolved)) != tuple(range(int(action_dim))):
            raise ValueError(
                "`action_reorder` must be a permutation of action indices "
                f"[0, {action_dim - 1}], got {resolved}."
            )
        return resolved

    def _remap_action_for_robotwin_env(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if self.action_reorder is None:
            return action_array
        return np.asarray(action_array[list(self.action_reorder)], dtype=np.float32)

    def _build_action_ensemble_cache(self, slot_state: _SlotState, raw_actions: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"`raw_actions` must have shape [T, D], got {action_chunk.shape}.")
        if int(action_chunk.shape[0]) <= 0:
            raise RuntimeError("Inference returned empty action chunk.")
        if self.replan_steps is None:
            raise ValueError("`action_ensemble` requires a positive `replan_steps` value.")
        query_interval = int(self.replan_steps)
        if int(action_chunk.shape[0]) < query_interval:
            raise ValueError(
                "`action_ensemble` requires model action chunk length to be >= replan_steps, "
                f"got chunk_len={int(action_chunk.shape[0])}, replan_steps={query_interval}."
            )

        current_step = int(slot_state.executed_steps)
        slot_state.action_ensemble_history = [
            item
            for item in slot_state.action_ensemble_history
            if int(item["start_step"]) + int(item["actions"].shape[0]) > current_step
        ]
        slot_state.action_ensemble_history.append(
            {
                "start_step": current_step,
                "actions": action_chunk,
            }
        )

        ensembled_actions = [
            self._ensemble_action_for_step(slot_state.action_ensemble_history, current_step + offset)
            for offset in range(query_interval)
        ]
        return np.asarray(ensembled_actions, dtype=np.float32)

    def _ensemble_action_for_step(self, action_history: list[dict[str, Any]], step: int) -> np.ndarray:
        relevant_preds = []
        for item in action_history:
            idx = int(step) - int(item["start_step"])
            actions = np.asarray(item["actions"], dtype=np.float32)
            if 0 <= idx < int(actions.shape[0]):
                relevant_preds.append(actions[idx])
        if not relevant_preds:
            raise ValueError(f"Step {step}: no actions available for action ensemble.")

        curr_act_preds = np.stack(relevant_preds, axis=0)
        if int(curr_act_preds.shape[0]) == 1:
            final_action = np.asarray(curr_act_preds[0], dtype=np.float32).copy()
        else:
            ref = curr_act_preds[-1]
            dot_product = np.sum(curr_act_preds * ref, axis=1)
            norm_previous_pred = np.linalg.norm(curr_act_preds, axis=1)
            norm_ref = np.linalg.norm(ref)
            cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
            weights = np.exp(float(self.action_ensemble_alpha) * cos_similarity)
            weights = weights / weights.sum()
            final_action = np.sum(weights[:, None] * curr_act_preds, axis=0).astype(np.float32)

        binary_idx = self._bounded_indices(final_action.shape[-1], self.action_binary_indices)
        if binary_idx.size > 0:
            final_action[binary_idx] = (final_action[binary_idx] > self._DEFAULT_BINARY_THRESHOLD).astype(np.float32)
        final_action = np.asarray(final_action[None, :], dtype=np.float32)
        self._normalize_robotwin_eef_quaternions_(final_action, self.passthrough_indices)
        return final_action[0]

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
        if unnorm_key is None or unnorm_key not in norm_stats:
            return next(iter(norm_stats.keys()))
        return str(unnorm_key)

    @classmethod
    def _resolve_image_size(
        cls,
        model_config: dict[str, Any],
        image_size: Optional[Sequence[int]],
    ) -> tuple[int, int]:
        if image_size is not None:
            if len(image_size) != 2:
                raise ValueError(f"`image_size` must have length 2, got {image_size}.")
            return int(image_size[0]), int(image_size[1])

        datasets_cfg = model_config.get("datasets", {})
        if isinstance(datasets_cfg, dict):
            vla_data_cfg = datasets_cfg.get("vla_data", {})
            if isinstance(vla_data_cfg, dict):
                image_resolution = vla_data_cfg.get("image_resolution", None)
                if image_resolution is not None:
                    size = int(image_resolution)
                    if size > 0:
                        return size, size
                cfg_image_size = vla_data_cfg.get("image_size", None)
                if isinstance(cfg_image_size, (list, tuple)) and len(cfg_image_size) == 2:
                    return int(cfg_image_size[0]), int(cfg_image_size[1])

        return cls._DEFAULT_IMAGE_SIZE

    @staticmethod
    def _resolve_replan_steps(replan_steps: Optional[int | str]) -> Optional[int]:
        if replan_steps is None:
            return None
        if isinstance(replan_steps, str):
            stripped = replan_steps.strip()
            if not stripped:
                return None
            resolved = int(stripped)
        else:
            resolved = int(replan_steps)
        if resolved <= 0:
            return None
        return resolved

    @staticmethod
    def _resolve_action_ensemble_enabled(action_ensemble: Optional[bool | str]) -> bool:
        if action_ensemble is None:
            return False
        if isinstance(action_ensemble, bool):
            return bool(action_ensemble)
        stripped = str(action_ensemble).strip().lower()
        if not stripped:
            return False
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off", "none", "null"}:
            return False
        raise ValueError(f"`action_ensemble` must be a boolean-like value, got {action_ensemble!r}.")

    @staticmethod
    def _resolve_action_ensemble_alpha(action_ensemble_alpha: Optional[float | str]) -> float:
        if action_ensemble_alpha is None:
            return 0.0
        if isinstance(action_ensemble_alpha, str):
            stripped = action_ensemble_alpha.strip()
            if not stripped:
                return 0.0
            return float(stripped)
        return float(action_ensemble_alpha)

    @staticmethod
    def _truncate_action_chunk(raw_actions: np.ndarray, replan_steps: Optional[int]) -> np.ndarray:
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"`raw_actions` must have shape [T, D], got {action_chunk.shape}.")
        if replan_steps is None:
            return action_chunk
        return np.asarray(action_chunk[: int(replan_steps)], dtype=np.float32)

def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    replan_steps = usr_args.get("replan_steps", os.getenv("ROBOTWIN_REPLAN_STEPS"))
    action_ensemble = usr_args.get("action_ensemble", os.getenv("ROBOTWIN_ACTION_ENSEMBLE"))
    action_ensemble_alpha = usr_args.get("action_ensemble_alpha", os.getenv("ROBOTWIN_ACTION_ENSEMBLE_ALPHA"))
    action_reorder = usr_args.get("action_reorder", None)

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        replan_steps=replan_steps,
        action_ensemble=action_ensemble,
        action_ensemble_alpha=action_ensemble_alpha,
        action_reorder=action_reorder,
    )


def reset_model(model):
    model.reset_slots()


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    example = model.build_example(str(instruction), observation)
    action = model.step(example, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action, action_type=model.env_action_type)
