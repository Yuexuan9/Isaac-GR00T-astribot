#!/usr/bin/env python
"""多机器人联合训练启动脚本"""

import os
import sys
import argparse
import yaml
from pathlib import Path


# ==================== 1. 配置注册部分 (原 multi_robot_config.py 内容) ====================
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
# =====================================================================================


from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run

def main():
    parser = argparse.ArgumentParser(description="Multi-robot training launch script")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="/workspace1/checkpoint/GR00T/a800_multi_robot_debug")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--start-from-checkpoint", type=str, default="/workspace1/model/GR00T-N1.6-3B")
    # 新增一个参数用于指定 yaml 文件路径，默认为刚才创建的文件
    parser.add_argument("--dataset-config", type=str, default="/workspace/gr00t/examples/MixData/multi_robot_datasets.yaml")
    args, _ = parser.parse_known_args()
    
    # 读取 YAML 配置
    dataset_config_path = args.dataset_config
    if not os.path.exists(dataset_config_path):
        # 尝试在脚本所在目录查找
        potential_path = os.path.join(os.path.dirname(__file__), args.dataset_config)
        if os.path.exists(potential_path):
            dataset_config_path = potential_path
    
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"无法找到数据集配置文件: {dataset_config_path}")
        
    print(f"Loading dataset config from: {dataset_config_path}") # 增加日志方便通过日志确认加载了哪个文件
    
    with open(dataset_config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    

    config = get_default_config().load_dict({
        "data": {
            "datasets": dataset_cfg['datasets'],
            "num_shards_per_epoch": 100000,
            "shard_size": 1024,
            "episode_sampling_rate": 0.1,
        },
        "model": {
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "state_dropout_prob": 0.3,
            "load_bf16": False,
            "reproject_vision": False,
            "eagle_collator": True,
            "model_name": "nvidia/Eagle-Block2A-2B-v2",
            "backbone_trainable_params_fp32": True,
            "use_relative_action": True,
            # ↓↓↓ 把命令行参数放这里 ↓↓↓
            "color_jitter_params": {
                "brightness": 0.3,
                "contrast": 0.4,
                "saturation": 0.5,
                "hue": 0.08,
            },
        },
        "training": {
            "start_from_checkpoint": args.start_from_checkpoint,
            # ↓↓↓ 这些是你原来想通过命令行传的参数 ↓↓↓
            "global_batch_size": args.global_batch_size,
            "max_steps": args.max_steps,           # --max-steps
            "save_steps": args.save_steps,            # --save-steps  
            "save_total_limit": args.save_total_limit,         # --save-total-limit
            "dataloader_num_workers": args.dataloader_num_workers,   # --dataloader-num-workers
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.05,
            "optim": "adamw_torch",
            "output_dir": args.output_dir,
            "num_gpus": int(os.environ.get("NUM_GPUS", 2)),
            "use_wandb": args.use_wandb,             # --use-wandb
            "gradient_accumulation_steps": 1,
            "wandb_project" : str(os.environ.get("WANDB_PROJECT", "finetune-gr00t-n1d6"))
        },
    })
    
    run(config)

if __name__ == "__main__":
    main()