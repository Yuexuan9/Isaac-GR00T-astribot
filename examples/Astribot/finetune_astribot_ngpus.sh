set -x -e

export NUM_GPUS=4

# ============ 多卡训练 NCCL 配置 ============
export NCCL_P2P_DISABLE=0       # 启用 P2P (GPU 之间直接通信，走 PCIe/NVLink)
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand (无硬件)
export NCCL_SHM_DISABLE=0       # 启用共享内存
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 确保 GPU 顺序一致
export NCCL_DEBUG=INFO          # 查看 NCCL 调试信息

# ============ 启动多卡训练 ============
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /workspace/data/db4_v21/astribot_s1/db4244aed6414f469ed7daad37f746b4 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/Astribot/astribot_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir gr00t/astribot_finetune_${NUM_GPUS}GPU \
  --save-total-limit 5 \
  --save-steps 5000 \
  --max-steps 200000 \
  --global-batch-size 256 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
  # --use-wandb \

