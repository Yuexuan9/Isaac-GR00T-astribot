# -*- coding: utf-8 -*-
"""
配置管理 - GR00T LeRobot v2.1 格式
"""

from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_HOST,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
)


@dataclass
class ActionConfig:
    """
    Action 配置
    
    GR00T LeRobot v2.1 格式:
    - 22维 (不含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
    - 25维 (含底盘): + chassis(3)
    
    分离三个概念:
    1. state_includes_chassis: 输入 state 是否包含底盘 (22 或 25 维)
    2. enable_*: 执行时哪些部件实际控制
    """
    # ========== 输入配置 ==========
    # 输入 state 是否包含底盘
    state_includes_chassis: bool = False
    
    # ========== 执行配置 ==========
    # 执行 action 时是否控制底盘
    enable_chassis: bool = False
    # 是否启用头部控制
    enable_head: bool = True
    # 是否启用腰部控制
    enable_torso: bool = True
    
    @property
    def state_dim(self) -> int:
        """输入 state 维度"""
        return LEROBOT_ACTION_DIM_WITH_CHASSIS if self.state_includes_chassis else LEROBOT_ACTION_DIM_NO_CHASSIS
    
    @property
    def action_dim(self) -> int:
        """输出 action 维度 (发送给机器人的)"""
        return LEROBOT_ACTION_DIM_WITH_CHASSIS if self.enable_chassis else LEROBOT_ACTION_DIM_NO_CHASSIS


@dataclass
class ServerConfig:
    """gRPC Server 配置"""
    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    max_workers: int = 10
    
    # 推理设备 (默认值，Client 可通过 Configure 覆盖)
    device: str = "cuda"
    
    # 控制频率
    fps: float = DEFAULT_CONTROL_FREQ
    
    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)


@dataclass
class ClientConfig:
    """gRPC Client 配置"""
    server_host: str = "localhost"
    server_port: int = DEFAULT_GRPC_PORT
    timeout: float = 10.0
    
    # 策略配置
    model_path: Optional[str] = None
    device: str = "cuda"
    policy_type: Optional[str] = None
    
    # 控制配置
    control_freq: float = DEFAULT_CONTROL_FREQ
    control_way: str = "direct"  # "direct" or "filter"
    
    # 平滑配置
    smooth_window: int = 0      # 0 = 不平滑
    max_velocity: float = 0.0   # 0 = 不限制
    
    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)
    
    @property
    def server_address(self) -> str:
        return f"{self.server_host}:{self.server_port}"
    
    @property
    def mode(self) -> str:
        return "model" if self.model_path else "none"
