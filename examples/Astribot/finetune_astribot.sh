set -x -e

export NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /workspace/data/db4_v21/astribot_s1/db4244aed6414f469ed7daad37f746b4 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/Astribot/astribot_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir gr00t/astribot_finetune \
  --save-total-limit 5 \
  --save-steps 2000 \
  --max-steps 200000 \
  --no-tune-diffusion-model \
  --global-batch-size 8 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 2
  # --use-wandb \

