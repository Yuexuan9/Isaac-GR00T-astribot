#!/bin/bash
set -x -e

export NUM_GPUS=2

# ============ WANDB 配置 ============
export WANDB_API_BASE=http://172.16.128.4:8280/api
export WANDB_BASE_URL=http://172.16.128.4:8280
export WANDB_MODE=online
export WANDB_ENTITY=bigmodel
export WANDB_PROJECT="gr00t-n1d6-multi-robot"
export WANDB_API_KEY=local-eaa8aa743c67bfdd959ff16b43093036463f9c11
export WANDB_DISABLE_ARTIFACT=true

# ============ 多卡训练 NCCL 配置 ============
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

# ============ 日志级别 ============
export LOGURU_LEVEL=INFO

# ============ 启动多卡训练 ============
# 注意：所有训练参数都在 launch_multi_robot_train.py 里配置
# 支持 embodiment: qingloong, ur_bimanual, astribot；数据集在 --dataset-config 指定 yaml 中配置
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    /workspace/gr00t/examples/MixData/launch_multi_robot_train.py \
    --global-batch-size 4 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 5 \
    --output-dir "/workspace/checkpoint/GR00T/09_task_2GPU_0324" \
    --use-wandb \
    --start-from-checkpoint "/workspace/gr00t/hfmodel" \
    --dataloader-num-workers 1
