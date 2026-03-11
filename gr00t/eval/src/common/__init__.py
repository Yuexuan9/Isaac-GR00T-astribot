# -*- coding: utf-8 -*-
"""
公共模块 - GR00T LeRobot v2.1 格式
"""

from .constants import (
    # Action keys
    GR00T_ACTION_KEYS,
    GR00T_ACTION_KEYS_NO_CHASSIS,
    GR00T_ACTION_DIM_CONFIG,
    ACTION_INDEX_CONFIG,
    # Dimensions
    LEROBOT_ACTION_DIM,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
    # Indices
    ARM_INDICES,
    ARM_LEFT_INDICES,
    ARM_RIGHT_INDICES,
    GRIPPER_LEFT_INDEX,
    GRIPPER_RIGHT_INDEX,
    HEAD_INDICES,
    TORSO_INDICES,
    CHASSIS_INDICES,
    # Astribot
    ASTRIBOT_NAMES_LIST,
    ASTRIBOT_NAMES_LIST_WITH_CHASSIS,
    ASTRIBOT_DOF_CONFIG,
    # Network
    GRPC_MAX_MESSAGE_LENGTH,
    GRPC_KEEPALIVE_TIME_MS,
    GRPC_KEEPALIVE_TIMEOUT_MS,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_HOST,
    DEFAULT_CONTROL_FREQ,
    # Ready position
    READY_POSITION_22,
    READY_POSITION_25,
)

from .config import ActionConfig, ServerConfig, ClientConfig

from .utils import (
    setup_logging,
    # GR00T dict <-> flat array
    gr00t_action_to_flat,
    flat_to_gr00t_action,
    # Flat array <-> Astribot waypoint
    flat_action_to_waypoint,
    waypoint_to_flat_action,
    # GR00T dict -> Astribot waypoint (direct)
    gr00t_action_to_waypoint,
    # Legacy names
    lerobot_action_to_waypoint,
    waypoint_to_lerobot_action,
    # Smoother & Limiter
    ActionSmoother,
    VelocityLimiter,
    # Filter
    filter_action,
    filter_action_array,
)

__all__ = [
    # Constants
    "GR00T_ACTION_KEYS",
    "GR00T_ACTION_KEYS_NO_CHASSIS",
    "GR00T_ACTION_DIM_CONFIG",
    "ACTION_INDEX_CONFIG",
    "LEROBOT_ACTION_DIM",
    "LEROBOT_ACTION_DIM_NO_CHASSIS",
    "LEROBOT_ACTION_DIM_WITH_CHASSIS",
    "ARM_INDICES",
    "ASTRIBOT_NAMES_LIST",
    "READY_POSITION_22",
    "READY_POSITION_25",
    # Config
    "ActionConfig",
    "ServerConfig", 
    "ClientConfig",
    # Utils
    "setup_logging",
    "gr00t_action_to_flat",
    "flat_to_gr00t_action",
    "flat_action_to_waypoint",
    "waypoint_to_flat_action",
    "gr00t_action_to_waypoint",
    "lerobot_action_to_waypoint",
    "waypoint_to_lerobot_action",
    "ActionSmoother",
    "VelocityLimiter",
    "filter_action",
    "filter_action_array",
]
