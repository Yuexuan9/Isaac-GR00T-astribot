from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig, ActionConfig, 
    ActionRepresentation, ActionType, ActionFormat
)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

# ========== UR双臂配置 ==========
UR_Config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_qpos", "left_arm_gripper",
            "right_arm_qpos", "right_arm_gripper",
            "left_arm_eef", "right_arm_eef",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(30)),
        modality_keys=[
            "left_arm_qpos", "left_arm_gripper",
            "right_arm_qpos", "right_arm_gripper",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# ========== 青龙配置 ==========
qingloong_Config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_qpos", "right_arm_qpos",
            "left_arm_eef", "right_arm_eef", "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(30)),
        modality_keys=["left_arm_qpos", "right_arm_qpos", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# ========== Astribot 配置 (astribot) ==========
astribot_Config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head",
            "torso",
            "wrist_left",
            "wrist_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "arm_right",
            "gripper_left",
            "gripper_right",
            "head",
            "torso",
            "chassis",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(50)),
        modality_keys=[
            "arm_left",
            "arm_right",
            "gripper_left",
            "gripper_right",
            "head",
            "torso",
            "chassis",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# ========== 注册配置（使用枚举值对应的字符串） ==========
MODALITY_CONFIGS[EmbodimentTag.QINGLOONG.value] = qingloong_Config
MODALITY_CONFIGS[EmbodimentTag.UR_BIMANUAL.value] = UR_Config
MODALITY_CONFIGS[EmbodimentTag.ASTRIBOT.value] = astribot_Config

print(f"✓ Registered: {EmbodimentTag.QINGLOONG.value}, {EmbodimentTag.UR_BIMANUAL.value}, {EmbodimentTag.ASTRIBOT.value}")