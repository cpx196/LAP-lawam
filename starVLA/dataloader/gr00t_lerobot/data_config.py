# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoMaxAspectCrop,
    VideoOffsetCrop,
    VideoResize,
    VideoToTensorUint8,
)
# from gr00t.model.transforms import GR00TTransform

DEFAULT_IMAGE_HW: tuple[int, int] = (256, 256)
_ROBOCASA_TABLETOP_SOURCE_HW: tuple[int, int] = (256, 256)
_ROBOCASA_TABLETOP_OFFSET_TOP_RATIO = 0.17
_ROBOCASA_TABLETOP_OFFSET_HEIGHT_RATIO = 0.66


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self, *, image_hw: tuple[int, int] | None = DEFAULT_IMAGE_HW) -> ModalityTransform:
        pass


def _build_composed_transform(
    transforms: list[ModalityTransform],
    state_keys: list[str],
    action_keys: list[str],
) -> ComposedModalityTransform:
    """Ensure state/action concat is always appended as the last transform."""
    concat_transform = None
    for transform in transforms:
        if isinstance(transform, ConcatTransform):
            concat_transform = transform
            break

    if concat_transform is None:
        transforms.append(
            ConcatTransform(
                state_concat_order=state_keys,
                action_concat_order=action_keys,
            )
        )
    else:
        # Keep concat focused on state/action so existing video key access in dataset code is unchanged.
        concat_transform.video_concat_order = None
        if concat_transform.state_concat_order is None:
            concat_transform.state_concat_order = list(state_keys)
        if concat_transform.action_concat_order is None:
            concat_transform.action_concat_order = list(action_keys)

    return ComposedModalityTransform(transforms=transforms)


def _latent_world_video_transforms(
    video_keys: list[str],
    image_hw: tuple[int, int] | None,
) -> list[ModalityTransform]:
    if image_hw is None:
        image_hw = DEFAULT_IMAGE_HW
    target_h, target_w = int(image_hw[0]), int(image_hw[1])
    return [
        VideoToTensorUint8(apply_to=video_keys),
        VideoMaxAspectCrop(apply_to=video_keys, max_aspect=4.0 / 3.0),
        VideoResize(apply_to=video_keys, height=target_h, width=target_w, interpolation="linear"),
    ]


def _gr1_tabletop_video_transforms(
    video_keys: list[str],
    image_hw: tuple[int, int] | None,
) -> list[ModalityTransform]:
    if image_hw is None:
        image_hw = DEFAULT_IMAGE_HW
    target_h, target_w = int(image_hw[0]), int(image_hw[1])
    source_h, source_w = _ROBOCASA_TABLETOP_SOURCE_HW
    return [
        VideoToTensorUint8(apply_to=video_keys),
        VideoOffsetCrop(
            apply_to=video_keys,
            top=int(source_h * _ROBOCASA_TABLETOP_OFFSET_TOP_RATIO),
            left=0,
            height=int(source_h * _ROBOCASA_TABLETOP_OFFSET_HEIGHT_RATIO),
            width=source_w,
        ),
        VideoResize(apply_to=video_keys, height=target_h, width=target_w, interpolation="linear"),
    ]


###########################################################################################

class DroidDataConfig:
    random_single_non_wrist_view = True
    video_keys = [
        "video.primary_view",
        "video.secondary_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_orientation",
        "state.gripper",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_orientation",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.9),
            # VideoResize(apply_to=self.video_keys, height=256, width=256, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.1,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_orientation": "min_max",
                    "state.gripper": "binary",
                },
                # target_rotations={
                #     "state.eef_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_orientation": "min_max",
                    "action.gripper": "binary",
                },
                # target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class BridgeDataConfig:
    video_keys = [
        "video.primary_view",
        "video.wrist_view",
        # "video.side1",
        # "video.side2",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.eef_orientation_rotvec",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.eef_orientation_rotvec",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "min_max",
                    "state.y": "min_max",
                    "state.z": "min_max",
                    "state.eef_orientation_rotvec": "min_max",
                    "state.gripper": "min_max",
                },
                invert_normalized_keys=["state.gripper"],
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.eef_orientation_rotvec": "min_max",
                    "action.gripper": "min_max",
                },
                invert_normalized_keys=["action.gripper"],
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################

class OxeRT1DataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "min_max",
                    "state.y": "min_max",
                    "state.z": "min_max",
                    "state.rx": "min_max",
                    "state.ry": "min_max",
                    "state.rz": "min_max",
                    "state.rw": "min_max",
                    "state.gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "min_max",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class FractalDataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.eef_orientation_rotvec",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.eef_orientation_rotvec",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "min_max",
                    "state.y": "min_max",
                    "state.z": "min_max",
                    "state.eef_orientation_rotvec": "min_max",
                    "state.gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.eef_orientation_rotvec": "min_max",
                    "action.gripper": "min_max",
                },
                invert_normalized_keys=["action.gripper"],
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class Libero4in1DataConfig:
    """Configuration for LIBERO dataset with Franka robot"""
    video_keys = [
        "video.primary_view",
        "video.wrist_view",
    ]
    
    state_keys = [
        "state.eef_position",
        "state.eef_orientation",
        "state.gripper",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_orientation",
        "action.gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_orientation": "min_max",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_orientation": "min_max",
                    "action.gripper": "binary",
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)

###########################################################################################

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.right_arm",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None) -> ModalityTransform:
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class FourierGr1ArmsWaistJointEefDataConfig:
    video_keys = ["video.ego_view"]
    joint_state_keys = [
        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    eef_state_keys = [
        "state.wrist_r_pos",
        "state.wrist_r_rot6d",
        "state.wrist_l_pos",
        "state.wrist_l_rot6d",
    ]
    state_keys = eef_state_keys + joint_state_keys
    joint_action_keys = [
        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    eef_action_keys = [
        "action.wrist_r_pos",
        "action.wrist_r_rot6d",
        "action.wrist_l_pos",
        "action.wrist_l_rot6d",
    ]
    action_keys = eef_action_keys + joint_action_keys
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_normalization_modes = {key: "min_max" for key in state_keys if key.endswith("pos")}
    action_normalization_modes = {key: "min_max" for key in action_keys if not key.endswith("rot6d")}

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None) -> ModalityTransform:
        transforms = [
            *_gr1_tabletop_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.joint_state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class PandaOmronDataConfig:
    video_keys = [
        "video.robot0_agentview_right",
        "video.wrist_view",
        "video.robot0_agentview_left",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None) -> ModalityTransform:
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.end_effector_position_relative": "min_max",
                    "state.end_effector_rotation_relative": "min_max",
                    "state.gripper_qpos": "min_max",
                    "state.base_position": "min_max",
                    "state.base_rotation": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.end_effector_position": "min_max",
                    "action.end_effector_rotation": "min_max",
                    "action.gripper_close": "binary",
                    "action.base_motion": "min_max",
                    "action.control_mode": "binary",
                },
            ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


###########################################################################################

class SO101Config:
    #input
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.shoulder_pan.pos",
        "state.shoulder_lift.pos",
        "state.elbow_flex.pos",
        "state.wrist_flex.pos",
        "state.wrist_roll.pos",
        "state.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # output
    action_keys = [
        "action.shoulder_pan.pos",
        "action.shoulder_lift.pos",
        "action.elbow_flex.pos",
        "action.wrist_flex.pos",
        "action.wrist_roll.pos",
        "action.gripper.pos",
    ]
    

    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max" for key in self.state_keys
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: "min_max" for key in self.action_keys
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)



###########################################################################################

class SOFollowerDataConfig:
    """LeRobot SO follower joint-position data used by Three_Cubes_1."""

    video_keys = [
        "video.front",
        "video.right",
        "video.wrist",
    ]

    state_keys = [
        "state.shoulder_pan_pos",
        "state.shoulder_lift_pos",
        "state.elbow_flex_pos",
        "state.wrist_flex_pos",
        "state.wrist_roll_pos",
        "state.gripper_pos",
    ]
    action_keys = [
        "action.shoulder_pan_pos",
        "action.shoulder_lift_pos",
        "action.elbow_flex_pos",
        "action.wrist_flex_pos",
        "action.wrist_roll_pos",
        "action.gripper_pos",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class SO101EEFDeltaDataConfig:
    """SO101 absolute EEF state with relative EEF action chunks."""

    video_keys = [
        "video.front",
        "video.right",
        "video.wrist",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_orientation_rotvec",
        "state.gripper",
    ]
    action_keys = [
        "action.eef_delta_position",
        "action.eef_delta_orientation_rotvec",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class AgilexDataConfig:
    """Configuration for Agilex dual-arm robot with rich state/action information"""
    video_keys = [
        "video.cam_high",                # front/high camera
        "video.cam_left_wrist",          # left wrist camera
        "video.cam_right_wrist",         # right wrist camera
    ]
    state_keys = [

        # # Joint velocity
        # "state.left_joint_velocity",     # 6-dim: left arm joint velocities
        # "state.right_joint_velocity",    # 6-dim: right arm joint velocities
        # # Joint effort (torque)
        # "state.left_joint_effort",       # 6-dim: left arm joint efforts
        # "state.right_joint_effort",      # 6-dim: right arm joint efforts
        # Task space (end effector)
        "state.left_joints",       # 6-dim: left arm joints
        "state.left_gripper",            # 1-dim: left gripper
        "state.right_joints",      # 6-dim: right arm joints
        "state.right_gripper",           # 1-dim: right gripper
    ]
    action_keys = [
        # Joint space
        "action.left_joints",            # 6-dim: left arm joints
        "action.left_gripper",           # 1-dim: left gripper
        "action.right_joints",           # 6-dim: right arm joints
        "action.right_gripper",          # 1-dim: right gripper
        # # Joint velocity
        # "action.left_joint_velocity",    # 6-dim: left arm joint velocities
        # "action.right_joint_velocity",   # 6-dim: right arm joint velocities
        # Task space (end effector)
        # "action.delta_left_eef_position",      # 3-dim: left end effector position (x, y, z)
        # "action.delta_left_eef_orientation",   # 4-dim: left end effector orientation (quaternion)
        # "action.delta_left_gripper",           # 1-dim: left gripper
        # "action.delta_right_eef_position",     # 3-dim: right end effector position (x, y, z)
        # "action.delta_right_eef_orientation",  # 3-dim: right end effector orientation (quaternion)
        # "action.delta_right_gripper",          # 1-dim: right gripper
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_joints": "min_max",
                    "state.right_gripper": "binary",
                },
                invert_normalized_keys=[
                    "state.left_gripper",
                    "state.right_gripper",
                ],
                binary_threshold=0.49,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_joints": "min_max",
                    "action.right_gripper": "binary",
                },
                invert_normalized_keys=[
                    "action.left_gripper",
                    "action.right_gripper",
                ],
                binary_threshold=0.49,
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)

###########################################################################################


class RobotwinEEFDataConfig:
    """Configuration for RoboTwin dual-arm end-effector actions and states."""

    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_orientation_quat",
        "state.left_gripper",
        "state.right_eef_position",
        "state.right_eef_orientation_quat",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_orientation_quat",
        "action.left_gripper",
        "action.right_eef_position",
        "action.right_eef_orientation_quat",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_eef_position": "min_max",
                    "state.left_gripper": "min_max",
                    "state.right_eef_position": "min_max",
                    "state.right_gripper": "min_max",
                },
                invert_normalized_keys=[
                    "state.left_gripper",
                    "state.right_gripper",
                ],
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_position": "min_max",
                    "action.left_gripper": "min_max",
                    "action.right_eef_position": "min_max",
                    "action.right_gripper": "min_max",
                },
                invert_normalized_keys=[
                    "action.left_gripper",
                    "action.right_gripper",
                ],
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)

###########################################################################################


class FoldTowelDataConfig:
    """Configuration for the real dual-arm fold-towel teleoperation dataset."""

    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_eef_position",
        "state.left_eef_orientation_rpy",
        "state.left_gripper",
        "state.right_eef_position",
        "state.right_eef_orientation_rpy",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_eef_position",
        "action.left_eef_orientation_rpy",
        "action.left_gripper",
        "action.right_eef_position",
        "action.right_eef_orientation_rpy",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max" for key in self.state_keys
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: "min_max" for key in self.action_keys
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)

###########################################################################################


class AgibotGenieDataConfig:
    """Configuration for AgiBOT humanoid robot with dual arms"""
    video_keys = [
        "video.head",
        "video.wrist_left",
        "video.wrist_right",
    ]
    state_keys = [
        # "state.joint_left",              # 7-dim: left arm joints
        # "state.joint_right",             # 7-dim: right arm joints
        "state.end_position_left",        # 3-dim: left end effector position (x, y, z)
        "state.end_orientation_left_rotvec",     # 3-dim: left end effector orientation (rotvec)
        "state.gripper_left",            # 1-dim: left gripper position
        "state.end_position_right",       # 3-dim: right end effector position (x, y, z)
        "state.end_orientation_right_rotvec",    # 3-dim: right end effector orientation (rotvec)
        "state.gripper_right",           # 1-dim: right gripper position
        # "state.waist",                   # 2-dim: waist position (pitch, lift)
        # "state.head",                    # 2-dim: head position (yaw, patch)
    ]
    action_keys = [
        # "action.joint_left",             # 7-dim: left arm joints
        # "action.joint_right",            # 7-dim: right arm joints
        "action.end_position_left",      # 3-dim: left end effector position (x, y, z)
        "action.end_orientation_left",   # 4-dim: left end effector orientation (quaternion)
        "action.gripper_left",           # 1-dim: left gripper
        "action.end_position_right",     # 3-dim: right end effector position (x, y, z)
        "action.end_orientation_right",  # 3-dim: right end effector orientation (quaternion)
        "action.gripper_right",          # 1-dim: right gripper
        # "action.waist",                  # 2-dim: waist position (pitch, lift)
        # "action.head",                   # 2-dim: head position (yaw, patch)
        # "action.robot_velocity",         # 2-dim: robot base velocity (x_vel, yaw_vel)
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    # "state.joint_left": "min_max",
                    # "state.joint_right": "min_max",
                    "state.end_position_left": "min_max",
                    "state.end_orientation_left_rotvec": "min_max",
                    "state.gripper_left": "min_max",
                    "state.end_position_right": "min_max",
                    "state.end_orientation_right_rotvec": "min_max",
                    "state.gripper_right": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    # "action.joint_left": "min_max",
                    # "action.joint_right": "min_max",
                    "action.end_position_left": "min_max",
                    "action.end_orientation_left": "min_max",
                    "action.gripper_left": "min_max",
                    "action.end_position_right": "min_max",
                    "action.end_orientation_right": "min_max",
                    "action.gripper_right": "min_max",
                    # "action.waist": "min_max",
                    # "action.head": "min_max",
                    # "action.robot_velocity": "min_max",
                },
            ),
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class RoboMINDFranka1RGBDataConfig:
    """Configuration for Franka single-arm robot with 1 RGB camera"""
    video_keys = [
        "video.primary_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_orientation",
        "state.gripper",                 # 1-dim: gripper position
    ]
    action_keys = [
        "action.eef_position",     # 3-dim: end effector position (x, y, z)
        "action.eef_orientation",
        "action.gripper",                # 1-dim: gripper position
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_orientation": "min_max",
                    "state.gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",     # 3-dim: end effector position (x, y, z)
                    "action.eef_orientation": "min_max",
                    "action.gripper": "min_max",
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class RoboMINDFranka3RGBDataConfig:
    """Configuration for RoboMIND Franka single-arm robot with 3 RGB cameras"""
    video_keys = [
        "video.primary_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_orientation",
        "state.gripper",                 # 1-dim: gripper position
    ]
    action_keys = [
        "action.eef_position",     # 3-dim: end effector position (x, y, z)
        "action.eef_orientation",
        "action.gripper",                # 1-dim: gripper position
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_orientation": "min_max",
                    "state.gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_orientation": "min_max",
                    "action.gripper": "min_max",
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class RoboMINDFrankaFR3DualDataConfig:
    """Configuration for RoboMIND dual-arm Franka FR3 robot with 4 RGB cameras"""
    video_keys = [
        "video.camera_front",            # front camera (1280x720)
        "video.camera_top",              # top camera (1280x720)
        # "video.camera_left",             # left-view camera (640x480)
        # "video.camera_right",            # right-view camera (640x480)
    ]
    state_keys = [
        # Task space (end effector)
        "state.left_eef_position",       # 3-dim: left end effector position (x, y, z)
        "state.left_eef_orientation_rotvec",    # 3-dim: left end effector orientation (roll, pitch, yaw)
                # Joint space
        # "state.left_joints",             # 7-dim: left arm joints
        "state.left_gripper",            # 1-dim: left gripper
        # "state.right_joints",            # 7-dim: right arm joints
        "state.right_eef_position",      # 3-dim: right end effector position (x, y, z)
        "state.right_eef_orientation_rotvec",   # 3-dim: right end effector orientation (roll, pitch, yaw)
        "state.right_gripper",           # 1-dim: right gripper
    ]
    action_keys = [
        # Joint space
        "action.left_eef_abs_position", 
        "action.left_eef_abs_orientation_rotvec",
        "action.left_eef_abs_gripper",
        "action.right_eef_abs_position",
        "action.right_eef_abs_orientation_rotvec",
        "action.right_eef_abs_gripper",

    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_eef_position": "min_max",
                    "state.left_eef_orientation_rotvec": "min_max",
                    "state.right_eef_position": "min_max",
                    "state.right_eef_orientation_rotvec": "min_max",
                    # "state.left_joints": "min_max",
                    "state.left_gripper": "min_max",
                    # "state.right_joints": "min_max",
                    "state.right_gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_eef_abs_position": "min_max",
                    "action.left_eef_abs_orientation_rotvec": "min_max",
                    "action.left_eef_abs_gripper": "min_max",
                    "action.right_eef_abs_position": "min_max",
                    "action.right_eef_abs_orientation_rotvec": "min_max",
                    "action.right_eef_abs_gripper": "min_max",
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)


###########################################################################################


class RoboMINDUR1RGBDataConfig:
    """Configuration for RoboMIND UR robot with 1 RGB camera"""
    video_keys = [
        "video.camera_top",              # top-view camera (640x480)
    ]
    state_keys = [
        "state.end_effector_position",   # 3-dim: end effector position (x, y, z)
        "state.eef_orientation_rotvec",# 3-dim: end effector orientation (roll, pitch, yaw)
        # "state.joint_position",          # 6-dim: 6 arm joints (UR robot)
        "state.gripper",                 # 1-dim: gripper position
    ]
    action_keys = [
        "action.eef_position",   
        "action.eef_orientation_rotvec",
        "action.eef_gripper",          # 1-dim: gripper command
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.end_effector_position": "min_max",
                    "state.eef_orientation_rotvec": "min_max",
                    # "state.joint_position": "min_max",
                    "state.gripper": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.eef_orientation_rotvec": "min_max",
                    "action.eef_gripper": "min_max",
                },
            ),
        ]

        return _build_composed_transform(transforms, self.state_keys, self.action_keys)




###########################################################################################

class HumanVideoDataConfig:
    video_keys = [
        "video.primary_view",
    ]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys, 
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, *, image_hw: tuple[int, int] | None = None):
        transforms = [
            *_latent_world_video_transforms(self.video_keys, image_hw),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "min_max",
                    "state.right_gripper": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "min_max",
                    "action.right_gripper": "min_max",
                },
            ),
        ]
        return _build_composed_transform(transforms, self.state_keys, self.action_keys)
###########################################################################################



ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "droid": DroidDataConfig(),
    "bridge": BridgeDataConfig(),
    "oxe_rt1": OxeRT1DataConfig(),
    "SO101": SO101Config(),
    "so_follower": SOFollowerDataConfig(),
    "so101_eef_delta": SO101EEFDeltaDataConfig(),
    "agilex": AgilexDataConfig(),
    "robotwin_joint": AgilexDataConfig(),
    "robotwin_eef": RobotwinEEFDataConfig(),
    "agibot_genie": AgibotGenieDataConfig(),
    "robomind_franka_1rgb": RoboMINDFranka1RGBDataConfig(),
    "robomind_franka_3rgb": RoboMINDFranka3RGBDataConfig(),
    "robomind_franka_fr3_dual": RoboMINDFrankaFR3DualDataConfig(),
    "robomind_ur_1rgb": RoboMINDUR1RGBDataConfig(),
    "human": HumanVideoDataConfig(),
    "gr1": FourierGr1ArmsWaistDataConfig(),
    "gr1_joint_eef": FourierGr1ArmsWaistJointEefDataConfig(),
    "PandaOmron": PandaOmronDataConfig(),
    "fractal": FractalDataConfig(),
    "fold_towel": FoldTowelDataConfig(),
}
