# Astribot for NVIDIA Isaac GR00T N1.6 (Stardust Development)

This project implements **fine-tuning** and **gRPC-based remote inference** for the **Astribot robot** using **NVIDIA Isaac GR00T N1.6**, with training data formatted according to **LeRobot v2.1**.

---

# Environment Setup (UV Package Manager)

## 1. Navigate to the GR00T workspace

```bash
cd /workspace/gr00t
```

## 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

## 3. Install the GR00T package in editable mode

```bash
uv pip install -e .
```

---

# Configuration Modifications

All configuration files must align with the **embodiment data dimensions**, including **state**, **action**, and **video indices/fields**.

---

## 1. Modality Configuration

**File path**

```
<dataset_dir>/meta/modality.json
```

**Required modifications**

Update the following fields to match your robot data structure:

- `state` indices  
- `action` indices  
- `video` indices  
- corresponding fields  

---

## 2. G1 Model Configuration

**File path**

```
/workspace/gr00t/examples/Astribot/astribot_config.py
```

**Required modifications**

Ensure the following are consistent with `modality.json`:

- state indices / fields  
- action indices / fields  
- video indices / fields  

**Critical parameter**

```python
action_chunk = 50
```

This value must match the **maximum action horizon** supported by the model.

---

## 3. GR00T Model Hyperparameters

**File path**

```
/workspace/gr00t/gr00t/configs/model/gr00t_n1d6.py
```

Modify **Line 57**:

```python
action_horizon: int = 50  # Set to 50 for Astribot finetuning
```

---

# Training

## Single-GPU Training (RTX 5090)

**Script**

```
finetune_astribot.sh
```

**Important notes**

- The **diffusion model must be frozen** for single-GPU training
- Recommended **maximum batch size: 64** (optimized for RTX 5090)

**Run training**

```bash
bash examples/Astribot/finetune_astribot.sh
```

---

## Multi-GPU Training

**Script**

```
finetune_astribot_ngpus.sh
```

**Important notes**

- No need to freeze the diffusion model
- Maximum global batch size:

```
64 × number_of_GPUs
```

**Run training**

```bash
bash examples/Astribot/finetune_astribot_ngpus.sh
```

---

# gRPC Remote Inference (Server–Client Architecture)

## Server Side (Model Host)

Runs on a **GPU-enabled machine** to serve the fine-tuned **LeRobot policy model**.

```bash
python /workspace/gr00t/gr00t/eval/run_gr00t_grpc_server.py \
  --model-path /workspace/gr00t/gr00t/astribot_finetune/checkpoint-32000 \
  --embodiment NEW_EMBODIMENT \
  --device cuda \
  --port 50051
```

---

## Client Side (Astribot Robot)

Runs directly on the **Astribot robot** for real-time inference.

```bash
cd /lerobot_grpc_inference

python src/client/inference_client.py \
  --server localhost:50051 \
  --enable-camera \
  --control-freq 15 \
  --use-chunk
```

---
