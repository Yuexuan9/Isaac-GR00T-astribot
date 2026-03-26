#!/usr/bin/env python3
"""
支持自定义配置的数据集统计预计算脚本 (批量模式)

使用方法:
    python precompute_with_custom_config.py --dataset-config multi_robot_datasets.yaml --num-workers 8
    python precompute_with_custom_config.py --dataset-config MixData/multi_robot_datasets.yaml --num-workers 8
"""

import argparse
import sys
import os
import yaml
import concurrent.futures # 引入并发库
from pathlib import Path


# ==================== 1. 配置注册部分 (原 multi_robot_config.py 内容) ====================
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig, ActionConfig, 
    ActionRepresentation, ActionType, ActionFormat
)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

# ========== UR双臂配置==========
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

# ========== 青龙配置（暂未注册：EmbodimentTag 无 QINGLOONG，需要时取消注释并在枚举中新增后注册） ==========
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

print(f"✓ Registered Configs In-Memory: {EmbodimentTag.QINGLOONG.value}, {EmbodimentTag.UR_BIMANUAL.value}, {EmbodimentTag.ASTRIBOT.value}")
# =========================================================

def process_single_dataset(dataset_path: str, embodiment_tag_str: str):
    """
    处理单个数据集的统计计算
    """
    # 必须在函数内部引用，确保使用全局修改过的 MODALITY_CONFIGS
    from gr00t.data.stats import generate_stats, generate_rel_stats
    
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        print(f"❌ 错误: 数据集路径不存在 -> {dataset_path}")
        return

    # Tag 解析
    try:
        if hasattr(EmbodimentTag, embodiment_tag_str.upper()):
            emb_tag = EmbodimentTag[embodiment_tag_str.upper()]
        else:
            emb_tag = EmbodimentTag(embodiment_tag_str.lower())
    except (KeyError, ValueError):
        print(f"❌ 错误: 无效的 embodiment_tag -> {embodiment_tag_str}")
        return
    
    # 简化日志，避免并行时太乱
    print(f"🚀 开始处理: {path_obj.name} (Tag: {emb_tag.value})")
    
    try:
        # 生成基础统计
        generate_stats(path_obj)
        # 生成相对统计
        generate_rel_stats(path_obj, emb_tag)
        print(f"✅ 完成: {path_obj.name}")
    except Exception as e:
        print(f"❌ 处理出错 ({path_obj.name}): {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="批量预计算数据集统计（读取YAML配置）"
    )
    
    parser.add_argument(
        "--dataset-config", 
        type=str, 
        default="multi_robot_datasets.yaml",
        help="包含数据集列表的YAML配置文件路径"
    )
    
    # 新增 workers 参数
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=8,
        help="并发进程数 (默认: 4)"
    )
    
    args = parser.parse_args()
    
    # 检查 ConfigBasePath 环境变量
    if "ConfigBasePath" not in os.environ:
        print("⚠️  Warning: ConfigBasePath 环境变量未设置。统计文件将直接保存到数据集目录下的 meta/ 中。")
    else:
        print(f"ℹ️  ConfigBasePath: {os.environ['ConfigBasePath']}")

    # --- 读取 YAML ---
    config_path = args.dataset_config
    # (保持原有的查找逻辑)
    if not os.path.exists(config_path):
        potential_path = os.path.join(os.path.dirname(__file__), args.dataset_config)
        if os.path.exists(potential_path):
            config_path = potential_path
    
    if not os.path.exists(config_path):
        print(f"❌ 无法找到配置文件: {config_path}")
        sys.exit(1)
        
    print(f"📂 读取配置文件: {config_path}")
    
    with open(config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    
    datasets_list = dataset_cfg.get('datasets', [])
    if not datasets_list:
        print("⚠️ 配置文件中没有找到 'datasets' 列表或列表为空。")
        sys.exit(0)

    # --- 1. 收集所有任务 ---
    all_tasks = []
    for group_idx, item in enumerate(datasets_list):
        tag = item.get('embodiment_tag')
        paths = item.get('dataset_paths', [])
        
        if not tag:
            print(f"⚠️ 第 {group_idx+1} 组缺少 embodiment_tag，跳过。")
            continue
            
        for d_path in paths:
            all_tasks.append((d_path, tag))

    print(f"\n⚡ 准备处理 {len(all_tasks)} 个数据集，使用 {args.num_workers} 个进程并发...")

    # --- 2. 并行执行 ---
    # 使用 ProcessPoolExecutor 绕过 GIL，利用多核 CPU
    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_dataset, path, tag) for path, tag in all_tasks]
            
            # 等待完成，可以在这里加 tqdm 进度条，但简单的 print 也行
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result() # 如果函数内有未捕获异常，这里会抛出
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
    else:
        # 单进程模式 (调试用)
        for path, tag in all_tasks:
            process_single_dataset(path, tag)

    print(f"\n🎉 全部处理完毕!")

if __name__ == "__main__":
    main()
