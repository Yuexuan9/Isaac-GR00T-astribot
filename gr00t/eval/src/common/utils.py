# -*- coding: utf-8 -*-
"""
工具函数 - GR00T LeRobot v2.1 格式

主要功能:
1. GR00T 字典格式 <-> 扁平数组格式 转换
2. 扁平数组格式 <-> Astribot waypoint 格式 转换
3. Action 平滑、限速、过滤
"""

import logging
import sys
from typing import Dict, List, Optional, Union
import numpy as np

from .constants import (
    GR00T_ACTION_KEYS,
    GR00T_ACTION_KEYS_NO_CHASSIS,
    GR00T_ACTION_DIM_CONFIG,
    ACTION_INDEX_CONFIG,
    ARM_INDICES,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger("lerobot_inference")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# GR00T 字典格式 <-> 扁平数组格式 转换
# ============================================================================

def gr00t_action_to_flat(
    gr00t_action: Dict[str, np.ndarray],
    include_chassis: bool = False,
) -> np.ndarray:
    """
    将 GR00T 字典格式的 action 转换为扁平数组格式
    
    GR00T 格式:
        {
            "arm_left": np.ndarray (B, T, 7),
            "arm_right": np.ndarray (B, T, 7),
            "gripper_left": np.ndarray (B, T, 1),
            "gripper_right": np.ndarray (B, T, 1),
            "head": np.ndarray (B, T, 2),
            "torso": np.ndarray (B, T, 4),
            "chassis": np.ndarray (B, T, 3),  # 可选
        }
    
    扁平数组格式 (22或25维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)?]
    
    Args:
        gr00t_action: GR00T 返回的字典格式 action
        include_chassis: 是否包含底盘
        
    Returns:
        np.ndarray: shape (B, T, 22) 或 (B, T, 25)
    """
    keys = GR00T_ACTION_KEYS if include_chassis else GR00T_ACTION_KEYS_NO_CHASSIS
    
    arrays = []
    for key in keys:
        if key in gr00t_action:
            arrays.append(gr00t_action[key])
        else:
            # 如果缺少某个 key，填充零值
            # 从其他 key 推断 shape
            sample = next(iter(gr00t_action.values()))
            B, T = sample.shape[:2]
            dim = GR00T_ACTION_DIM_CONFIG[key]
            arrays.append(np.zeros((B, T, dim), dtype=np.float32))
    
    return np.concatenate(arrays, axis=-1)


def flat_to_gr00t_action(
    flat_action: np.ndarray,
    include_chassis: bool = False,
) -> Dict[str, np.ndarray]:
    """
    将扁平数组格式的 action 转换为 GR00T 字典格式
    
    Args:
        flat_action: shape (..., 22) 或 (..., 25)
        include_chassis: 是否包含底盘
        
    Returns:
        Dict[str, np.ndarray]: GR00T 字典格式
    """
    keys = GR00T_ACTION_KEYS if include_chassis else GR00T_ACTION_KEYS_NO_CHASSIS
    
    result = {}
    for key in keys:
        start, end = ACTION_INDEX_CONFIG[key]
        if end <= flat_action.shape[-1]:
            result[key] = flat_action[..., start:end]
    
    return result


# ============================================================================
# 扁平数组格式 <-> Astribot waypoint 格式 转换
# ============================================================================

def flat_action_to_waypoint(
    action: Union[List[float], np.ndarray],
    include_chassis: bool = False,
) -> List[List[float]]:
    """
    将扁平数组格式的 action 转换为 Astribot waypoint 格式
    
    扁平数组格式 (GR00T LeRobot v2.1):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)?]
    
    Astribot waypoint 格式:
        [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    
    Args:
        action: 扁平数组 (22或25维)
        include_chassis: 是否包含底盘
        
    Returns:
        waypoint: Astribot SDK 格式的 waypoint
    """
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    action_len = len(action)
    if action_len < LEROBOT_ACTION_DIM_NO_CHASSIS:
        raise ValueError(f"Action 长度必须至少为 {LEROBOT_ACTION_DIM_NO_CHASSIS}，当前为 {action_len}")
    
    # 从扁平数组提取各部件
    idx = ACTION_INDEX_CONFIG
    waypoint = [
        action[idx["torso"][0]:idx["torso"][1]],           # torso (4)
        action[idx["arm_left"][0]:idx["arm_left"][1]],     # arm_left (7)
        action[idx["gripper_left"][0]:idx["gripper_left"][1]],  # gripper_left (1)
        action[idx["arm_right"][0]:idx["arm_right"][1]],   # arm_right (7)
        action[idx["gripper_right"][0]:idx["gripper_right"][1]],  # gripper_right (1)
        action[idx["head"][0]:idx["head"][1]],             # head (2)
    ]
    
    if include_chassis and action_len >= LEROBOT_ACTION_DIM_WITH_CHASSIS:
        waypoint.append(action[idx["chassis"][0]:idx["chassis"][1]])  # chassis (3)
    
    return waypoint


def waypoint_to_flat_action(
    waypoint: List[List[float]],
    include_chassis: bool = False,
) -> List[float]:
    """
    将 Astribot waypoint 格式转换为扁平数组格式
    
    Args:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
        include_chassis: 是否包含底盘
        
    Returns:
        action: 扁平数组 (22或25维)
    """
    torso = waypoint[0]           # 4
    arm_left = waypoint[1]        # 7
    gripper_left = waypoint[2]    # 1
    arm_right = waypoint[3]       # 7
    gripper_right = waypoint[4]   # 1
    head = waypoint[5] if len(waypoint) > 5 else [0.0, 0.0]  # 2
    
    # GR00T v2.1 格式: [arm_left, arm_right, gripper_left, gripper_right, head, torso, chassis?]
    action = arm_left + arm_right + gripper_left + gripper_right + head + torso
    
    if include_chassis:
        chassis = waypoint[6] if len(waypoint) > 6 else [0.0, 0.0, 0.0]
        action = action + chassis
    
    return action


# 兼容旧名称
lerobot_action_to_waypoint = flat_action_to_waypoint
waypoint_to_lerobot_action = waypoint_to_flat_action


# ============================================================================
# GR00T 字典格式 -> Astribot waypoint 格式 (直接转换)
# ============================================================================

def gr00t_action_to_waypoint(
    gr00t_action: Dict[str, np.ndarray],
    batch_idx: int = 0,
    time_idx: int = 0,
    include_chassis: bool = False,
) -> List[List[float]]:
    """
    将 GR00T 字典格式的 action 直接转换为 Astribot waypoint 格式
    
    Args:
        gr00t_action: GR00T 返回的字典 {"arm_left": (B,T,7), ...}
        batch_idx: 取哪个 batch
        time_idx: 取哪个时间步
        include_chassis: 是否包含底盘
        
    Returns:
        waypoint: Astribot SDK 格式
    """
    def get_value(key: str) -> List[float]:
        if key in gr00t_action:
            arr = gr00t_action[key]
            # 支持多种 shape
            if arr.ndim == 3:  # (B, T, D)
                return arr[batch_idx, time_idx].tolist()
            elif arr.ndim == 2:  # (T, D) 或 (B, D)
                return arr[time_idx].tolist()
            elif arr.ndim == 1:  # (D,)
                return arr.tolist()
        # 缺失则返回零值
        return [0.0] * GR00T_ACTION_DIM_CONFIG[key]
    
    waypoint = [
        get_value("torso"),          # 4
        get_value("arm_left"),       # 7
        get_value("gripper_left"),   # 1
        get_value("arm_right"),      # 7
        get_value("gripper_right"),  # 1
        get_value("head"),           # 2
    ]
    
    if include_chassis:
        waypoint.append(get_value("chassis"))  # 3
    
    return waypoint


# ============================================================================
# Action 平滑器
# ============================================================================

class ActionSmoother:
    """动作平滑器 - 使用简单移动平均消除抖动"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: List[np.ndarray] = []
    
    def smooth(self, action: Union[List[float], np.ndarray]) -> List[float]:
        action_arr = np.array(action, dtype=np.float64)
        self._history.append(action_arr)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        smoothed = np.mean(self._history, axis=0)
        return smoothed.tolist()
    
    def reset(self):
        self._history = []


# ============================================================================
# 速度限制器
# ============================================================================

class VelocityLimiter:
    """
    速度限制器 - 限制相邻帧之间的最大变化量
    注意：只对手臂关节限制，不对夹爪限制
    """
    
    def __init__(self, max_delta: float = 0.05):
        """
        Args:
            max_delta: 每帧最大角度变化 (弧度)，默认 0.05 rad ≈ 2.9°
        """
        self.max_delta = max_delta
        self._last_action: Optional[np.ndarray] = None
    
    def limit(self, action: Union[List[float], np.ndarray]) -> List[float]:
        action_arr = np.array(action, dtype=np.float64)
        
        if self._last_action is None:
            self._last_action = action_arr.copy()
            return action_arr.tolist()
        
        limited = action_arr.copy()
        for i in ARM_INDICES:
            if i < len(action_arr):
                delta = action_arr[i] - self._last_action[i]
                delta = np.clip(delta, -self.max_delta, self.max_delta)
                limited[i] = self._last_action[i] + delta
        
        self._last_action = limited.copy()
        return limited.tolist()
    
    def reset(self):
        self._last_action = None


# ============================================================================
# Action 过滤 (按部件启用/禁用)
# ============================================================================

def filter_action(
    action: np.ndarray,
    current_state: Optional[np.ndarray] = None,
    enable_head: bool = True,
    enable_torso: bool = True,
    enable_chassis: bool = False,
) -> np.ndarray:
    """
    根据部件启用配置过滤单步 action
    
    禁用的部件使用 current_state 对应值替代 (若无则置零)。
    enable_chassis=False 时截断为 22 维。
    """
    if len(action) < LEROBOT_ACTION_DIM_NO_CHASSIS:
        return action
    
    filtered = action.copy()
    
    head_start, head_end = ACTION_INDEX_CONFIG["head"]
    torso_start, torso_end = ACTION_INDEX_CONFIG["torso"]
    
    if not enable_head:
        if current_state is not None and len(current_state) >= head_end:
            filtered[head_start:head_end] = current_state[head_start:head_end]
        else:
            filtered[head_start:head_end] = 0.0
    
    if not enable_torso:
        if current_state is not None and len(current_state) >= torso_end:
            filtered[torso_start:torso_end] = current_state[torso_start:torso_end]
        else:
            filtered[torso_start:torso_end] = 0.0
    
    if enable_chassis and len(action) >= LEROBOT_ACTION_DIM_WITH_CHASSIS:
        return filtered[:LEROBOT_ACTION_DIM_WITH_CHASSIS]
    else:
        return filtered[:LEROBOT_ACTION_DIM_NO_CHASSIS]


def filter_action_array(
    actions: np.ndarray,
    current_state: Optional[np.ndarray] = None,
    enable_head: bool = True,
    enable_torso: bool = True,
    enable_chassis: bool = False,
) -> np.ndarray:
    """
    对单步 action 或 action chunk 统一执行过滤
    
    Args:
        actions: shape (action_dim,) 单步, 或 (chunk_size, action_dim) chunk
    """
    kwargs = dict(current_state=current_state, enable_head=enable_head,
                  enable_torso=enable_torso, enable_chassis=enable_chassis)
    
    if actions.ndim == 1:
        return filter_action(actions, **kwargs)
    else:
        filtered = [filter_action(actions[i], **kwargs) for i in range(actions.shape[0])]
        return np.stack(filtered, axis=0)
