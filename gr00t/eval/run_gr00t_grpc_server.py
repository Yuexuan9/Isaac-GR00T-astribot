#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T 推理服务器 (gRPC 版本)

运行环境: Python 3.10+ (gr00t 环境)

将 GR00T 模型封装为 gRPC 服务，与 `lerobot_grpc_inference` 的 Client 兼容。

使用方法:
    # 启动服务器 (使用模型)
    python run_gr00t_grpc_server.py --model-path /path/to/checkpoint \
        --embodiment NEW_EMBODIMENT --device cuda --port 50051

    # 启动服务器 (使用 replay 数据集)
    python run_gr00t_grpc_server.py --dataset-path /path/to/dataset \
        --embodiment NEW_EMBODIMENT --port 50051
"""

import io
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import grpc
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入生成的 protobuf 代码
from src.common.proto_imports import pb2, pb2_grpc

# 导入通用模块
from src.common.config import ActionConfig
from src.common.constants import (
    GR00T_ACTION_DIM_CONFIG as ACTION_DIM_CONFIG,  # 使用别名保持代码一致性
    ACTION_INDEX_CONFIG,
    GRPC_KEEPALIVE_TIME_MS,
    GRPC_KEEPALIVE_TIMEOUT_MS,
    GRPC_MAX_MESSAGE_LENGTH,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
)
from src.common.utils import setup_logging

logger = logging.getLogger("gr00t_inference.server")


# ============================================================================
# GR00T 相关导入
# ============================================================================
HAS_GR00T = False

try:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.replay_policy import ReplayPolicy

    HAS_GR00T = True
except ImportError:
    logger.warning("未找到 gr00t 模块，请确保 gr00t 已安装")


# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_GR00T_PORT = 50051
DEFAULT_CONTROL_FREQ = 30.0
DEFAULT_ACTION_HORIZON = 16

STATE_COMPONENT_ORDER = [
    "arm_left",
    "arm_right",
    "gripper_left",
    "gripper_right",
    "head",
    "torso",
    "chassis",
]


@dataclass
class Gr00tServerConfig:
    """GR00T Server 配置"""

    host: str = "0.0.0.0"
    port: int = DEFAULT_GR00T_PORT
    max_workers: int = 10

    # 模型配置
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    embodiment_tag: str = "NEW_EMBODIMENT"
    device: str = "cuda"
    strict: bool = True

    # 可选配置
    modality_config_path: Optional[str] = None
    execution_horizon: Optional[int] = None
    use_sim_policy_wrapper: bool = False

    # 推理配置
    fps: float = DEFAULT_CONTROL_FREQ

    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)


class Gr00tModelInference:
    """
    GR00T 模型推理器

    封装 `Gr00tPolicy` 或 `ReplayPolicy`，提供统一的推理接口。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "cuda",
        strict: bool = True,
        modality_config_path: Optional[str] = None,
        execution_horizon: Optional[int] = None,
        use_sim_policy_wrapper: bool = False,
    ):
        if not HAS_GR00T:
            raise ImportError("需要安装 gr00t: pip install gr00t")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device
        self.strict = strict
        self.embodiment_tag = EmbodimentTag[embodiment_tag]

        self.policy = None
        self._metadata: Dict[str, Any] = {}
        self._action_horizon = DEFAULT_ACTION_HORIZON
        self._action_dim = LEROBOT_ACTION_DIM_NO_CHASSIS
        self._is_replay = dataset_path is not None
        self._replay_finished = False

        self._load(modality_config_path, execution_horizon, use_sim_policy_wrapper)

    def _load(
        self,
        modality_config_path: Optional[str],
        execution_horizon: Optional[int],
        use_sim_policy_wrapper: bool,
    ):
        """加载模型或 replay 策略"""
        logger.info("加载 GR00T 模型/策略...")
        logger.info("  - Embodiment: %s", self.embodiment_tag)
        logger.info("  - Device: %s", self.device)

        start_time = time.time()

        if self.model_path is not None:
            if self.model_path.startswith("/") and not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

            logger.info("  - 模型路径: %s", self.model_path)

            self.policy = Gr00tPolicy(
                embodiment_tag=self.embodiment_tag,
                model_path=self.model_path,
                device=self.device,
                strict=self.strict,
            )
            self._metadata["type"] = "gr00t_policy"
            self._metadata["model_path"] = self.model_path

        elif self.dataset_path is not None:
            if self.dataset_path.startswith("/") and not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")

            logger.info("  - 数据集路径: %s", self.dataset_path)

            if modality_config_path is None:
                from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

                modality_configs = MODALITY_CONFIGS[self.embodiment_tag.value]
            else:
                with open(modality_config_path, "r", encoding="utf-8") as f:
                    modality_configs = json.load(f)

            self.policy = ReplayPolicy(
                dataset_path=self.dataset_path,
                modality_configs=modality_configs,
                execution_horizon=execution_horizon,
                strict=self.strict,
            )
            self._metadata["type"] = "replay_policy"
            self._metadata["dataset_path"] = self.dataset_path
            if execution_horizon is not None:
                self._metadata["execution_horizon"] = execution_horizon
        else:
            raise ValueError("必须指定 model_path 或 dataset_path")

        if use_sim_policy_wrapper:
            from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

            self.policy = Gr00tSimPolicyWrapper(self.policy)
            self._metadata["sim_wrapper"] = True

        if hasattr(self.policy, "action_horizon"):
            self._action_horizon = int(self.policy.action_horizon)
        if hasattr(self.policy, "action_dim"):
            self._action_dim = int(self.policy.action_dim)

        load_time = time.time() - start_time
        logger.info("模型加载完成，耗时: %.2fs", load_time)
        logger.info("  - Action horizon: %s", self._action_horizon)
        logger.info("  - Action dim: %s", self._action_dim)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def replay_finished(self) -> bool:
        return self._replay_finished

    def reset(self):
        """重置策略状态"""
        self._replay_finished = False
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _flatten_component_action_dict(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """
        将 GR00T 分部件 action dict 转为扁平动作数组，输出 shape 为 `(T, D)`。
        """
        component_arrays: Dict[str, np.ndarray] = {}
        time_steps = None

        for key in STATE_COMPONENT_ORDER:
            raw_value = action_dict.get(key)
            if raw_value is None:
                continue

            arr = np.asarray(raw_value, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 2:
                pass
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]
            else:
                raise ValueError(f"不支持的 action 维度: key={key}, shape={arr.shape}")

            expected_dim = ACTION_DIM_CONFIG[key]
            if arr.shape[-1] != expected_dim:
                raise ValueError(
                    f"Action 维度不匹配: key={key}, got={arr.shape[-1]}, expected={expected_dim}"
                )

            if time_steps is None:
                time_steps = arr.shape[0]
            elif arr.shape[0] != time_steps:
                raise ValueError(
                    f"Action 时间维不一致: key={key}, got={arr.shape[0]}, expected={time_steps}"
                )

            component_arrays[key] = arr

        if not component_arrays:
            raise RuntimeError("模型未返回可识别的 action 组件")

        use_chassis = "chassis" in component_arrays
        action_dim = max(ACTION_INDEX_CONFIG[k][1] for k in component_arrays.keys())
        if not use_chassis:
            action_dim = min(action_dim, LEROBOT_ACTION_DIM_NO_CHASSIS)

        flat_actions = np.zeros((time_steps, action_dim), dtype=np.float32)
        for key, arr in component_arrays.items():
            start_idx, end_idx = ACTION_INDEX_CONFIG[key]
            if end_idx > action_dim:
                continue
            flat_actions[:, start_idx:end_idx] = arr

        return flat_actions

    def _coerce_actions(self, action_payload: Any) -> np.ndarray:
        """
        将各种 GR00T 输出格式统一为 `(T, D)` 的 numpy 数组。
        """
        if isinstance(action_payload, dict):
            if any(key in action_payload for key in STATE_COMPONENT_ORDER):
                return self._flatten_component_action_dict(action_payload)

            nested = action_payload.get("actions", action_payload.get("action"))
            if nested is None:
                raise RuntimeError("模型未返回 action")
            actions = np.asarray(nested, dtype=np.float32)
        else:
            actions = np.asarray(action_payload, dtype=np.float32)

        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[np.newaxis, :]
        elif actions.ndim != 2:
            raise ValueError(f"不支持的 action shape: {actions.shape}")

        return actions

    @staticmethod
    def _extract_terminal(info: Any) -> bool:
        """best-effort 提取 replay 结束标志"""
        if not isinstance(info, dict):
            return False

        for key in ("is_terminal", "terminal", "done", "episode_end", "is_done"):
            value = info.get(key)
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return bool(value)
            if isinstance(value, np.ndarray) and value.size == 1:
                return bool(value.reshape(-1)[0])
        return False

    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        执行推理，返回 shape `(T, D)` 的动作序列。
        """
        if self.policy is None:
            raise RuntimeError("模型未加载")

        try:
            result = self.policy.get_action(observation)
        except StopIteration:
            self._replay_finished = True
            return np.zeros((0, self._action_dim), dtype=np.float32)

        info = None
        action_payload = result
        if isinstance(result, tuple):
            if len(result) >= 1:
                action_payload = result[0]
            if len(result) >= 2:
                info = result[1]

        actions = self._coerce_actions(action_payload)

        if self._extract_terminal(info):
            self._replay_finished = True

        return actions


class Gr00tInferenceServicer(pb2_grpc.LeRobotInferenceServiceServicer):
    """
    GR00T 推理服务 gRPC 实现。

    使用与 LeRobot 相同的 proto 接口，但内部使用 GR00T 模型。
    """

    def __init__(self, config: Gr00tServerConfig):
        self.config = config
        self.is_ready = False
        self.model_inference: Optional[Gr00tModelInference] = None

        self.current_episode = 0
        self.current_frame = 0
        self.model_name = "none"
        self.total_frames = 0

        if config.model_path or config.dataset_path:
            self._load_model()
        else:
            logger.info("Server 以空闲模式启动，等待 Client 配置...")

    def _load_model(self):
        """加载模型"""
        logger.info("加载 GR00T 模型...")

        self.model_inference = Gr00tModelInference(
            model_path=self.config.model_path,
            dataset_path=self.config.dataset_path,
            embodiment_tag=self.config.embodiment_tag,
            device=self.config.device,
            strict=self.config.strict,
            modality_config_path=self.config.modality_config_path,
            execution_horizon=self.config.execution_horizon,
            use_sim_policy_wrapper=self.config.use_sim_policy_wrapper,
        )

        self.model_name = os.path.basename(
            self.config.model_path or self.config.dataset_path or "gr00t"
        )
        self.is_ready = True
        self.current_frame = 0

        try:
            self.total_frames = len(self.model_inference.policy)  # type: ignore[arg-type]
        except Exception:
            self.total_frames = 0

        logger.info("GR00T 模型加载成功")

    @staticmethod
    def _extract_prompt(extra_state: str) -> str:
        """从 extra_state JSON 中提取语言指令"""
        if not extra_state:
            return ""

        try:
            payload = json.loads(extra_state)
        except json.JSONDecodeError:
            return ""

        for key in (
            "prompt",
            "task",
            "instruction",
            "task_description",
            "annotation.human.task_description",
        ):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _build_observation_dict(self, request: "pb2.Observation") -> Dict[str, Any]:
        """
        将 gRPC 请求转换为 GR00T 观测字典格式。

        输出格式:
        - `video`: `{camera_name: np.ndarray(1, 1, H, W, C)}`
        - `state`: `{component: np.ndarray(1, 1, D)}`
        - `language`: `{"annotation.human.task_description": [[text]]}`
        """
        obs_dict = {
            "video": {},
            "state": {},
            "language": {
                "annotation.human.task_description": [[self._extract_prompt(request.extra_state)]]
            },
        }

        if request.joint_positions:
            flat_state = np.asarray(list(request.joint_positions), dtype=np.float32)
            state_dim = int(flat_state.shape[0])

            for key in STATE_COMPONENT_ORDER:
                start_idx, end_idx = ACTION_INDEX_CONFIG[key]
                dim = ACTION_DIM_CONFIG[key]

                if end_idx <= state_dim:
                    component_data = flat_state[start_idx:end_idx]
                else:
                    component_data = np.zeros(dim, dtype=np.float32)

                obs_dict["state"][key] = component_data[np.newaxis, np.newaxis, :]

        for img_data in request.images:
            if not img_data.data:
                continue
            try:
                image = self._decode_image(img_data)
                if image is None:
                    continue
                obs_dict["video"][img_data.camera_name] = image[np.newaxis, np.newaxis, :, :, :]
            except Exception as exc:
                logger.warning("图像解码失败 (%s): %s", img_data.camera_name, exc)

        logger.debug(
            "构建观测: video_keys=%s, state_keys=%s",
            list(obs_dict["video"].keys()),
            list(obs_dict["state"].keys()),
        )
        return obs_dict

    @staticmethod
    def _decode_image(img_data: "pb2.ImageData") -> Optional[np.ndarray]:
        """解码图像数据"""
        from PIL import Image

        encoding = img_data.encoding.lower()
        if encoding in ("jpeg", "jpg", "png"):
            image = Image.open(io.BytesIO(img_data.data)).convert("RGB")
        elif encoding == "raw":
            if img_data.width <= 0 or img_data.height <= 0:
                logger.warning("raw 格式需要指定 width 和 height")
                return None
            image = Image.frombytes("RGB", (img_data.width, img_data.height), img_data.data)
        else:
            logger.warning("不支持的图像编码格式: %s", encoding)
            return None

        return np.asarray(image, dtype=np.uint8)

    def Configure(self, request: "pb2.PolicyConfig", context) -> "pb2.ServiceStatus":
        """
        配置策略。

        受当前 proto 限制，gRPC 动态配置仅支持 `model_path`；`dataset_path`
        仍需通过命令行启动时指定。
        """
        logger.info("收到配置请求: model_path=%s", request.model_path)

        try:
            if not request.model_path:
                return pb2.ServiceStatus(
                    is_ready=self.is_ready,
                    model_name=self.model_name,
                    current_episode=self.current_episode,
                    current_frame=self.current_frame,
                    total_frames=self.total_frames,
                    fps=self.config.fps,
                    message="错误: 当前 proto 仅支持通过 model_path 动态加载模型",
                    mode="dataset" if self.config.dataset_path else ("model" if self.is_ready else "none"),
                )

            self.config.model_path = request.model_path
            self.config.dataset_path = None
            self.config.device = request.device or self.config.device
            self._load_model()
            return self._get_status(f"已加载模型: {request.model_path}")

        except Exception as exc:
            logger.error("配置失败: %s", exc)
            traceback.print_exc()
            return self._get_status(f"配置失败: {exc}")

    def Predict(self, request: "pb2.Observation", context) -> "pb2.Action":
        """单次推理，返回一个 action"""
        if not self.is_ready:
            return pb2.Action(status=pb2.NOT_READY, error_message="服务未就绪，请先加载模型")

        try:
            obs_dict = self._build_observation_dict(request)
            actions = self.model_inference.predict(obs_dict)

            if actions.size == 0 or self.model_inference.replay_finished:
                return pb2.Action(
                    is_terminal=True,
                    status=pb2.EPISODE_END,
                    server_frame_index=self.current_frame,
                )

            first_action = actions[0]
            self.current_frame += 1

            return pb2.Action(
                values=first_action.tolist(),
                is_terminal=False,
                status=pb2.OK,
                server_frame_index=self.current_frame,
            )

        except Exception as exc:
            logger.error("推理错误: %s", exc)
            traceback.print_exc()
            return pb2.Action(status=pb2.ERROR, error_message=str(exc))

    def PredictChunk(self, request: "pb2.Observation", context) -> "pb2.ActionChunk":
        """Chunk 推理，返回完整 action chunk"""
        if not self.is_ready:
            return pb2.ActionChunk(status=pb2.NOT_READY, error_message="服务未就绪，请先加载模型")

        try:
            obs_dict = self._build_observation_dict(request)
            actions = self.model_inference.predict(obs_dict)

            if actions.size == 0 or self.model_inference.replay_finished:
                return pb2.ActionChunk(
                    is_terminal=True,
                    status=pb2.EPISODE_END,
                    server_frame_index=self.current_frame,
                )

            if actions.ndim == 1:
                actions = actions[np.newaxis, :]

            action_steps = [pb2.ActionStep(values=step.tolist()) for step in actions]

            self.current_frame += 1
            logger.debug("返回 action chunk: size=%s, dim=%s", actions.shape[0], actions.shape[1])

            return pb2.ActionChunk(
                actions=action_steps,
                chunk_size=int(actions.shape[0]),
                action_dim=int(actions.shape[1]),
                is_terminal=False,
                status=pb2.OK,
                server_frame_index=self.current_frame,
            )

        except Exception as exc:
            logger.error("Chunk 推理错误: %s", exc)
            traceback.print_exc()
            return pb2.ActionChunk(status=pb2.ERROR, error_message=str(exc))

    def StreamPredict(
        self,
        request_iterator: Iterator["pb2.Observation"],
        context,
    ) -> Iterator["pb2.Action"]:
        """流式推理"""
        logger.info("开始流式推理")

        for obs in request_iterator:
            if not context.is_active():
                break

            action_response = self.Predict(obs, context)
            yield action_response
            if action_response.is_terminal:
                break

        logger.info("流式推理结束")

    def Control(self, request: "pb2.ControlCommand", context) -> "pb2.ServiceStatus":
        """控制命令"""
        cmd_type = request.type
        params = dict(request.params)

        if cmd_type == pb2.CMD_RESET:
            self._reset_state()
            return self._get_status("已重置")

        if cmd_type == pb2.CMD_SET_EPISODE:
            ep = int(params.get("episode", "0"))
            self.current_episode = ep
            self._reset_state()
            return self._get_status(f"切换到 episode {ep}")

        return self._get_status("未知命令")

    def _reset_state(self):
        """重置内部状态"""
        self.current_frame = 0
        if self.model_inference:
            self.model_inference.reset()

    def GetStatus(self, request: "pb2.Empty", context) -> "pb2.ServiceStatus":
        """获取状态"""
        return self._get_status()

    def Reset(self, request: "pb2.Empty", context) -> "pb2.ServiceStatus":
        """重置"""
        self._reset_state()
        return self._get_status("已重置")

    def _get_status(self, message: str = "") -> "pb2.ServiceStatus":
        """构建状态响应"""
        mode = "none"
        if self.config.dataset_path:
            mode = "dataset"
        elif self.is_ready:
            mode = "model"

        return pb2.ServiceStatus(
            is_ready=self.is_ready,
            model_name=self.model_name,
            current_episode=self.current_episode,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            fps=self.config.fps,
            message=message,
            mode=mode,
        )


class Gr00tGrpcServer:
    """GR00T gRPC 服务器"""

    def __init__(self, config: Gr00tServerConfig):
        self.config = config
        self.server = None
        self.servicer = None
        self._stopped = False

    def start(self):
        """启动服务器"""
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.keepalive_time_ms", GRPC_KEEPALIVE_TIME_MS),
                ("grpc.keepalive_timeout_ms", GRPC_KEEPALIVE_TIMEOUT_MS),
            ],
        )

        self.servicer = Gr00tInferenceServicer(self.config)
        pb2_grpc.add_LeRobotInferenceServiceServicer_to_server(self.servicer, self.server)

        address = f"{self.config.host}:{self.config.port}"
        self.server.add_insecure_port(address)
        self.server.start()
        logger.info("GR00T gRPC 服务器已启动: %s", address)
        return self

    def wait_for_termination(self, timeout: Optional[float] = None):
        """等待服务器终止"""
        if self.server:
            self.server.wait_for_termination(timeout)

    def stop(self, grace: float = 5.0):
        """停止服务器"""
        if self.server and not self._stopped:
            logger.info("正在停止服务器...")
            self.server.stop(grace)
            self._stopped = True
            logger.info("服务器已停止")


def run_server(config: Gr00tServerConfig):
    """运行服务器"""
    setup_logging("INFO")

    server = Gr00tGrpcServer(config)

    def signal_handler(signum, frame):
        del frame
        logger.info("收到信号 %s，正在停止...", signum)
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GR00T gRPC 推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用模型推理
  python run_gr00t_grpc_server.py --model-path /path/to/checkpoint \
      --embodiment NEW_EMBODIMENT --device cuda --port 50051

  # 使用 replay 数据集
  python run_gr00t_grpc_server.py --dataset-path /path/to/dataset \
      --embodiment NEW_EMBODIMENT --port 50051
        """,
    )

    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=50051, help="监听端口 (默认: 50051)")
    parser.add_argument("--workers", type=int, default=10, help="工作线程数 (默认: 10)")

    parser.add_argument("--model-path", type=str, default=None, help="GR00T 模型 checkpoint 路径")
    parser.add_argument("--dataset-path", type=str, default=None, help="数据集路径 (用于 replay)")
    parser.add_argument(
        "--embodiment",
        type=str,
        default="NEW_EMBODIMENT",
        help="Embodiment 标签 (默认: NEW_EMBODIMENT)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (默认: cuda)")
    parser.add_argument("--strict", action="store_true", default=True, help="严格验证输入输出")
    parser.add_argument("--no-strict", action="store_false", dest="strict", help="禁用严格验证")

    parser.add_argument("--modality-config", type=str, default=None, help="modality 配置文件路径")
    parser.add_argument("--execution-horizon", type=int, default=None, help="执行 horizon")
    parser.add_argument("--use-sim-wrapper", action="store_true", help="使用 sim policy wrapper")

    parser.add_argument("--fps", type=float, default=30.0, help="目标帧率 (默认: 30)")

    args = parser.parse_args()

    if args.model_path is None and args.dataset_path is None:
        parser.error("必须指定 --model-path 或 --dataset-path")

    config = Gr00tServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        embodiment_tag=args.embodiment,
        device=args.device,
        strict=args.strict,
        modality_config_path=args.modality_config,
        execution_horizon=args.execution_horizon,
        use_sim_policy_wrapper=args.use_sim_wrapper,
        fps=args.fps,
    )
    run_server(config)


if __name__ == "__main__":
    main()
