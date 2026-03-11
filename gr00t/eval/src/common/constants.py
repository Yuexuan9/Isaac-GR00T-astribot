# -*- coding: utf-8 -*-
"""
常量定义 - GR00T LeRobot v2.1 格式

GR00T 模型返回字典格式的 action，需要转换为扁平数组发送给机器人。

格式说明:
- GR00T 输出: {"arm_left": (B,T,7), "arm_right": (B,T,7), ...} 字典
- 机器人输入: [arm_left(7), arm_right(7), ...] 扁平数组 (22或25维)
"""

# ============================================================================
# GR00T Action Keys (v2.1 格式)
# ============================================================================

# GR00T 返回的 action 字典的 key 顺序
# 与 new_embodiment 的 modality_keys 一致
GR00T_ACTION_KEYS = [
    "arm_left",
    "arm_right", 
    "gripper_left",
    "gripper_right",
    "head",
    "torso",
    "chassis",
]

# 不含底盘的 keys
GR00T_ACTION_KEYS_NO_CHASSIS = GR00T_ACTION_KEYS[:6]

# ============================================================================
# 各部件维度配置
# ============================================================================

GR00T_ACTION_DIM_CONFIG = {
    "arm_left": 7,
    "arm_right": 7,
    "gripper_left": 1,
    "gripper_right": 1,
    "head": 2,
    "torso": 4,
    "chassis": 3,
}

# 总维度
LEROBOT_ACTION_DIM_NO_CHASSIS = sum(
    GR00T_ACTION_DIM_CONFIG[k] for k in GR00T_ACTION_KEYS_NO_CHASSIS
)  # = 22

LEROBOT_ACTION_DIM_WITH_CHASSIS = sum(
    GR00T_ACTION_DIM_CONFIG[k] for k in GR00T_ACTION_KEYS
)  # = 25

LEROBOT_ACTION_DIM = LEROBOT_ACTION_DIM_NO_CHASSIS  # 默认

# ============================================================================
# 扁平数组中各部件的索引范围
# ============================================================================

def _compute_index_config():
    """计算各部件在扁平数组中的索引范围"""
    config = {}
    offset = 0
    for key in GR00T_ACTION_KEYS:
        dim = GR00T_ACTION_DIM_CONFIG[key]
        config[key] = (offset, offset + dim)
        offset += dim
    return config

ACTION_INDEX_CONFIG = _compute_index_config()
# 结果: {'arm_left': (0, 7), 'arm_right': (7, 14), 'gripper_left': (14, 15), 
#        'gripper_right': (15, 16), 'head': (16, 18), 'torso': (18, 22), 'chassis': (22, 25)}

# ============================================================================
# 关节索引 (便捷访问)
# ============================================================================

ARM_LEFT_INDICES = list(range(*ACTION_INDEX_CONFIG["arm_left"]))    # [0,1,2,3,4,5,6]
ARM_RIGHT_INDICES = list(range(*ACTION_INDEX_CONFIG["arm_right"]))  # [7,8,9,10,11,12,13]
ARM_INDICES = ARM_LEFT_INDICES + ARM_RIGHT_INDICES                   # [0..13]

GRIPPER_LEFT_INDEX = ACTION_INDEX_CONFIG["gripper_left"][0]          # 14
GRIPPER_RIGHT_INDEX = ACTION_INDEX_CONFIG["gripper_right"][0]        # 15

HEAD_INDICES = list(range(*ACTION_INDEX_CONFIG["head"]))             # [16,17]
TORSO_INDICES = list(range(*ACTION_INDEX_CONFIG["torso"]))           # [18,19,20,21]
CHASSIS_INDICES = list(range(*ACTION_INDEX_CONFIG["chassis"]))       # [22,23,24]

# ============================================================================
# Astribot SDK 部件配置
# ============================================================================

# Astribot waypoint 格式的部件顺序 (与 GR00T 不同!)
ASTRIBOT_NAMES_LIST = [
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
]

ASTRIBOT_NAMES_LIST_WITH_CHASSIS = ASTRIBOT_NAMES_LIST + ["astribot_chassis"]

ASTRIBOT_DOF_CONFIG = {
    "astribot_torso": 4,
    "astribot_arm_left": 7,
    "astribot_gripper_left": 1,
    "astribot_arm_right": 7,
    "astribot_gripper_right": 1,
    "astribot_head": 2,
    "astribot_chassis": 3,
}

# ============================================================================
# gRPC / 网络配置
# ============================================================================

DEFAULT_CONTROL_FREQ = 30.0
DEFAULT_GRPC_PORT = 50051
DEFAULT_GRPC_HOST = "0.0.0.0"

GRPC_MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50MB
GRPC_KEEPALIVE_TIME_MS = 10000
GRPC_KEEPALIVE_TIMEOUT_MS = 5000

# ============================================================================
# 机器人准备位置 (Ready Position)
# ============================================================================

# 22维准备位置 (不含底盘)
# 格式: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
# 数据来源: astribot_catlitter_datasets 所有200个episode第一帧状态的均值
READY_POSITION_22 = [
    # arm_left (7)
    0.154849, -0.022670, -1.421605, 1.660323, -0.346889, 0.115219, 0.126036,
    # arm_right (7)
    -0.161952, -0.022760, 1.418778, 1.660055, 0.343307, 0.115222, -0.123617,
    # gripper_left (1)
    -0.021181,
    # gripper_right (1)
    -0.121321,
    # head (2)
    -0.013063, 0.786349,
    # torso (4)
    0.597646, -1.195333, 0.597043, 0.009469,
]

# 25维准备位置 (含底盘)
READY_POSITION_25 = READY_POSITION_22 + [
    # chassis (3)
    -0.000426, 0.002229, -0.069377,
]
