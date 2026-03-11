from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


astribot_config = {
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
        delta_indices=list(range(50)),  # 50步动作预测序列
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
            # arm_left: 7维关节角
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # arm_right: 7维关节角
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_left: 1维夹爪
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_right: 1维夹爪
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # head: 2维头部关节
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # torso: 4维躯干关节
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # chassis: 3维底盘
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.task_description",
        ],
    ),
}

register_modality_config(
    astribot_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

