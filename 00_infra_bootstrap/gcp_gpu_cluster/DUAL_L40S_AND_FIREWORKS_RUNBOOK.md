
# SOVEREIGN-PROB-7B: 2x NVIDIA L40S TRAINING + FIREWORKS AI INFERENCE RUNBOOK

**File:** `00_infra_bootstrap/gcp_gpu_cluster/DUAL_L40S_AND_FIREWORKS_RUNBOOK.md`

**Training Hardware:** 2x NVIDIA L40S (2 x 48 GB = 96 GB GDDR6 VRAM, Ada Lovelace, sm_89)

**Live Inference:** Fireworks AI Serverless LoRA ($50 Credit Pool) + Local 4-Bit GGUF Fallback

---

## 1. Why 2x L40S for Training + Fireworks AI for Inference?

Decoupling heavy model training from live inference provides an efficient architecture for SOVEREIGN-PROB-7B.

The proposed architecture uses:

1. Two NVIDIA L40S GPUs for CPT, SFT, and GRPO training.
2. Fireworks AI for hosted model inference.
3. Local 4-bit GGUF as an inference fallback.
4. A hybrid inference router to select the appropriate inference backend.

### 1.1 Why 2x NVIDIA L40S for 7B Post-Training?

Each NVIDIA L40S provides:

- GPU Memory: 48 GB GDDR6.
- Architecture: NVIDIA Ada Lovelace.
- Compute Capability: 8.9 (`sm_89`).
- BF16 Tensor Core support.
- Peak BF16 Tensor Core performance: approximately 362 TFLOPS per GPU, subject to the applicable sparsity and performance specifications.

Combined hardware:

| Specification | Configuration |
|---|---|
| GPU Count | 2 |
| GPU Model | NVIDIA L40S |
| VRAM per GPU | 48 GB |
| Aggregate VRAM | 96 GB |
| Architecture | Ada Lovelace |
| Compute Capability | sm_89 |
| Interconnect | Depends on server configuration |

The Qwen2.5-7B base model requires approximately 14–15 GB for BF16 weights alone, depending on the exact checkpoint representation.

With two GPUs, the training framework can distribute model parameters and training workloads using:

- DistributedDataParallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO
- Hugging Face Accelerate

**Important:** Two GPUs do not automatically form a unified 96 GB memory pool. Model placement and memory distribution must be configured explicitly.

A possible memory configuration for model sharding is:

```python
max_memory = {
    0: "42GiB",
    1: "42GiB",
}
```

The actual memory requirement depends on:

- Model weights.
- Optimizer states.
- Gradient storage.
- Activation memory.
- Sequence length.
- Batch size.
- Gradient accumulation.
- LoRA rank.
- Gradient checkpointing.
- Distributed training strategy.

LoRA configurations can initially use:

```python
lora_config = {
    "r": 64,
    "lora_alpha": 128,
}
```

These are configuration targets, not guarantees of zero OOM risk.

### 1.2 Why Fireworks AI for Live Inference?

Keeping expensive training GPUs running continuously for occasional inference requests is often unnecessary.

The proposed deployment architecture separates training and inference.

**Training phase:**

1. Train the base model using two NVIDIA L40S GPUs.
2. Execute CPT.
3. Execute SFT.
4. Execute GRPO.
5. Evaluate the trained model.
6. Export the LoRA adapter.
7. Export the local GGUF fallback.
8. Package deployment artifacts.

**Inference phase:**

1. Stop the GPU training instance after confirming artifact persistence.
2. Deploy the LoRA adapter using a supported Fireworks AI deployment workflow.
3. Run the Stage 6 inference router on a laptop or small VPS.
4. Route inference requests to Fireworks AI.
5. Use local GGUF inference when the hosted service is unavailable or unsuitable.

The proposed inference triggers include:

- 3-sigma volume spikes.
- Odds movements greater than 4%.
- Other configured market-event triggers.

These thresholds are strategy parameters and must be independently validated.

**Target architecture:**

```text
             TRAINING INFRASTRUCTURE
             ======================

       NVIDIA L40S GPU 0 + GPU 1
                  |
                  v
           Qwen2.5-7B Base
                  |
                  v
             Stage 4A CPT
                  |
                  v
             Stage 4B SFT
                  |
                  v
              Stage 5 GRPO
                  |
                  v
           Model Evaluation
                  |
          +-------+-------+
          |               |
          v               v
      LoRA Export      GGUF Export
          |               |
          v               v
    Fireworks Bundle   Local Fallback


             INFERENCE INFRASTRUCTURE
             ========================

              Market Event
                   |
                   v
          Stage 6 Hybrid Router
                   |
             +-----+-----+
             |           |
             v           v
         Fireworks    Local GGUF
             |           |
             +-----+-----+
                   |
                   v
             Model Output
                   |
                   v
          Validation / Risk Checks
                   |
                   v
             Downstream System
```

---

## 2. Push the Dual-L40S Update to GitHub

Perform these steps from your local Ubuntu WSL2 terminal.

### Step 1: Navigate to the Repository

```bash
cd /mnt/c/Narendra/python_files/projects/Sovereign-Prob-7B

source ~/.venvs/sovereign_env/bin/activate
```

### Step 2: Make the Training Script Executable

```bash
chmod +x \
    00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh
```

### Step 3: Run Repository Audit Tests

```bash
python3 -m unittest discover \
    -s tests \
    -p "test_*.py" \
    -v
```

Verify that all expected tests pass.

### Step 4: Validate Bash Syntax

```bash
bash -n \
    00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh
```

### Step 5: Execute Safe Preflight

```bash
CONFIRM_TRAINING=NO \
bash 00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh
```

This should execute the preflight checks without starting training.

### Step 6: Commit and Push Changes

```bash
git status

git add \
    00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh \
    00_infra_bootstrap/gcp_gpu_cluster/DUAL_L40S_AND_FIREWORKS_RUNBOOK.md

git commit -m "feat: add dual NVIDIA L40S training and Fireworks AI deployment runbook"

git push
```

---

## 3. Execute Training on the 2x NVIDIA L40S Server

### Step 1: Connect to the GPU Server

Connect to your GPU server through SSH.

```bash
ssh USERNAME@GPU_SERVER_IP
```

Replace the placeholders with your actual server credentials.

### Step 2: Start a Persistent tmux Session

```bash
tmux new -s sovereign_l40s
```

This keeps the terminal session alive when your SSH connection disconnects.

To detach without stopping the process:

```bash
Ctrl+B
D
```

To reconnect:

```bash
tmux attach -t sovereign_l40s
```

### Step 3: Verify Both GPUs

```bash
nvidia-smi
```

Check GPU details:

```bash
nvidia-smi \
    --query-gpu=name,memory.total,compute_cap \
    --format=csv
```

Expected hardware:

```text
NVIDIA L40S, 49140 MiB, 8.9
NVIDIA L40S, 49140 MiB, 8.9
```

The exact reported memory can vary slightly.

### Step 4: Clone the Repository

```bash
git clone https://github.com/nukanarendra/sovereign.git

cd sovereign
```

### Step 5: Create the Python Environment

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip setuptools wheel
```

### Step 6: Install CUDA-Compatible PyTorch

For the CUDA 12.4 PyTorch wheel index:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify the installation:

```bash
python -c "
import torch

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"
```

### Step 7: Install Project Dependencies

```bash
pip install \
    polars \
    pyarrow \
    pydantic \
    httpx \
    transformers \
    trl \
    peft \
    accelerate \
    datasets \
    bitsandbytes \
    safetensors \
    sentencepiece \
    protobuf \
    gguf \
    fireworks-ai
```

Install the project's pinned requirements instead if a `requirements.txt` or lockfile exists. Verify compatible package versions before production training.

### Step 8: Pull Git LFS Datasets

```bash
git lfs install

git lfs pull
```

Verify that the required Parquet shards are actual files rather than Git LFS pointers.

### Step 9: Run Safe Preflight

```bash
CONFIRM_TRAINING=NO \
bash 00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh
```

Verify:

- All seven required Parquet shards exist.
- Parquet files can be read.
- Both L40S GPUs are detected.
- BF16 GPU smoke tests pass.
- Python dependencies are available.

**Do not proceed if any preflight check fails.**

### Step 10: Launch Full Training

Only after verifying the hardware, datasets, distributed training configuration, checkpoints, and estimated GPU costs:

```bash
CONFIRM_TRAINING=YES \
bash 00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh
```

The intended execution sequence is:

| Stage | Operation | Steps |
|---|---|---:|
| Stage 4A | Tokenizer Extension | Configuration-dependent |
| Stage 4A | Continued Pretraining (CPT) | 16 |
| Stage 4B | Supervised Fine-Tuning (SFT) | 20 |
| Stage 5 | GRPO Training | 24 |
| Stage 5 | Model Evaluation | Configuration-dependent |
| Stage 5 | Quantization and Export | Configuration-dependent |
| Stage 5 | Fireworks Bundle Packaging | Configuration-dependent |
| Stage 6 | Hybrid Router Verification | Configuration-dependent |

The step counts are trainer arguments, not guarantees of completed epochs or successful model convergence.

---

## 4. Configure Fireworks AI for Live Inference

After successful training and export, the intended artifact structure is:

### 4.1 Fireworks LoRA Bundle

```text
03_llm_training/
└── model_registry/
    └── exports/
        └── fireworks_lora_bundle/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── tokenizer_config.json
            └── fireworks_deployment_spec.json
```

### 4.2 Local GGUF Fallback

```text
03_llm_training/
└── model_registry/
    └── checkpoints/
        └── quantized_gguf_q4_k_m/
            └── sovereign-prob-7b-q4_k_m.gguf
```

These are expected output paths. Confirm that the exporter actually produces these files.

### 4.3 Configure Fireworks AI API Credentials

Create or obtain your Fireworks AI API key.

Store credentials in a local `.env` file:

```bash
FIREWORKS_AI_API_KEY=fw_YOUR_ACTUAL_FIREWORKS_API_KEY
```

Protect the file:

```bash
chmod 600 .env
```

Ensure `.env` is included in `.gitignore`.

For shell usage, load the variable:

```bash
set -a
source .env
set +a
```

Verify that the variable exists without printing the secret:

```bash
if [[ -n "${FIREWORKS_AI_API_KEY:-}" ]]; then
    echo "Fireworks API key is configured."
else
    echo "Fireworks API key is missing."
fi
```

### 4.4 Upload the LoRA Adapter

The upload process must use a Fireworks-supported model customization or deployment workflow.

First, inspect the deployment script:

```bash
python3 -m 03_llm_training.model_registry.fireworks_upload_deployer --help
```

If the module does not support `--help`, inspect its documented command-line interface.

Then execute the configured deployment workflow:

```bash
python3 -m \
    03_llm_training.model_registry.fireworks_upload_deployer
```

Verify:

- API authentication succeeds.
- The base model is supported.
- The LoRA adapter format is accepted.
- The adapter is uploaded successfully.
- The deployment endpoint is available.
- A test inference request succeeds.

Do not assume that packaging a LoRA directory automatically creates a live Fireworks endpoint.

---

## 5. Configure Stage 6 Hybrid Model Router

The hybrid router is responsible for selecting the appropriate inference backend.

**Primary backend:** Fireworks AI.

**Fallback backend:** Local GGUF inference.

Expected architecture:

```text
              Incoming Request
                     |
                     v
              Hybrid Router
                     |
                     v
             Fireworks Health Check
                     |
               +-----+-----+
               |           |
             Healthy     Unavailable
               |           |
               v           v
          Fireworks AI   Local GGUF
               |           |
               +-----+-----+
                     |
                     v
              Response Validation
                     |
                     v
                Final Output
```

### 5.1 Start the Hybrid Router

```bash
python3 -m \
    04_mcp_runtime.inference_router.hybrid_model_router
```

### 5.2 Verify the Router

Confirm that:

1. Fireworks API authentication works.
2. The configured model or deployment identifier is correct.
3. The hosted endpoint responds successfully.
4. Local GGUF inference is operational.
5. Fallback activates when the hosted service fails.
6. Timeouts and API errors are handled.
7. Credit exhaustion is detected through actual billing or usage information.

**Important:** API-key availability alone does not prove that sufficient credits remain. Implement explicit usage and billing monitoring rather than assuming that the router can read the remaining balance.

---

## 6. Final Deployment Checklist

### Training Infrastructure

- [ ] Two NVIDIA L40S GPUs detected.
- [ ] CUDA and PyTorch compatibility verified.
- [ ] BF16 smoke test passed on both GPUs.
- [ ] Distributed training strategy configured.
- [ ] All seven Parquet shards verified.
- [ ] Git LFS datasets downloaded.
- [ ] Checkpoint persistence tested.
- [ ] GPU cost and runtime estimates reviewed.

### Training Pipeline

- [ ] Tokenizer extension completed.
- [ ] CPT training completed.
- [ ] SFT training completed.
- [ ] GRPO training completed.
- [ ] Evaluation completed.
- [ ] Model artifacts validated.

### Model Export

- [ ] LoRA adapter exported.
- [ ] Adapter configuration verified.
- [ ] Tokenizer files exported.
- [ ] GGUF export completed.
- [ ] GGUF model tested locally.

### Fireworks AI

- [ ] API key configured securely.
- [ ] LoRA bundle uploaded.
- [ ] Deployment successfully created.
- [ ] Test inference completed.
- [ ] Usage and credit monitoring configured.

### Stage 6

- [ ] Fireworks inference operational.
- [ ] Local GGUF inference operational.
- [ ] Fallback logic tested.
- [ ] Timeout handling tested.
- [ ] Response validation enabled.
- [ ] Router logs verified.

---

## 7. Final Architecture

The intended SOVEREIGN-PROB-7B deployment architecture:

| Component | Technology | Responsibility |
|---|---|---|
| Training Hardware | 2x NVIDIA L40S | Model training |
| Base Model | Qwen2.5-7B | Foundation model |
| CPT | PyTorch / Transformers | Domain adaptation |
| SFT | Transformers / PEFT | Supervised fine-tuning |
| GRPO | TRL / PEFT | Policy optimization |
| Model Export | Safetensors / GGUF | Artifact packaging |
| Hosted Inference | Fireworks AI | Primary inference |
| Local Inference | GGUF Runtime | Fallback inference |
| Orchestration | Hybrid Model Router | Backend selection |
| Runtime | Laptop / VPS | Continuous operation |

**Final objective:** Train SOVEREIGN-PROB-7B on dedicated dual-L40S infrastructure, persist the resulting model artifacts, deploy hosted inference through Fireworks AI, and maintain a tested local GGUF fallback for inference resilience.
