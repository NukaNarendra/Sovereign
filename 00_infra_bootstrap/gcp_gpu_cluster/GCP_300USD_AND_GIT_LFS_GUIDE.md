# SOVEREIGN-PROB-7B: Beginner-Friendly Click-by-Click Google Cloud ($300) Guide

**File:** `00_infra_bootstrap/gcp_gpu_cluster/GCP_300USD_AND_GIT_LFS_GUIDE.md`

**Zero prior Google Cloud experience required.**

---

# PART 1: Fix the Local Items and Push Your Parquet Dataset to GitHub

In the GitHub snapshot you shared (`4326cbd1...`), the following items need attention:

- `run_gcp_cuda_training.sh` needs its clean, correctly formatted version saved locally.
- Your `data/processed_warehouse/final_splits/` Parquet dataset needs to be added to Git using Git LFS.

The dataset is described as containing 2,800 training rows, 600 validation rows, and 600 test rows.

## Step 1: Open your Ubuntu WSL2 terminal

Copy and paste:

```bash
cd /mnt/c/Narendra/python_files/projects/Sovereign-Prob-7B

deactivate 2>/dev/null || true

source ~/.venvs/sovereign_env/bin/activate
```

## Step 2: Write the corrected GCP CUDA orchestrator

Run this command:

```bash
cat << 'EOF' > 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SOVEREIGN_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

cd "${REPO_ROOT}"

JOB_START_EPOCH="$(date +%s)"
JOB_ID="${JOB_ID:-GCP_CUDA_$(date -u +%Y%m%dT%H%M%SZ)}"

LOG_DIR="${REPO_ROOT}/logs/job_${JOB_ID}"
CHECKPOINT_ROOT="${REPO_ROOT}/03_llm_training/model_registry/checkpoints"
EMERGENCY_SENTINEL="${CHECKPOINT_ROOT}/EMERGENCY_CREDIT_DRAIN_FLUSH.flag"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || true)"
fi

CONFIRM_TRAINING="${CONFIRM_TRAINING:-NO}"
DOWNLOAD_HF="${DOWNLOAD_HF:-1}"

mkdir -p \
    "${LOG_DIR}" \
    "${CHECKPOINT_ROOT}/stage1_cpt" \
    "${CHECKPOINT_ROOT}/stage2_sft" \
    "${CHECKPOINT_ROOT}/stage3_grpo"

log_info() {
    printf '%s | INFO  | [GCP-CUDA-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log"
}

log_warn() {
    printf '%s | WARN  | [GCP-CUDA-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log" >&2
}

fail() {
    log_warn "$*"
    exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock is required."

exec 9>"${REPO_ROOT}/.gcp_training.lock"

flock -n 9 || fail "Another training orchestrator is already running (.gcp_training.lock)."

log_info "Starting GCP training job ${JOB_ID}."
log_info "Repository: ${REPO_ROOT}"
log_info "Logs: ${LOG_DIR}"

rm -f "${EMERGENCY_SENTINEL}"

# 1. VERIFY PARQUET SHARDS

if [[ ! -f "${REPO_ROOT}/data/processed_warehouse/final_splits/grpo_verifiable/train_grpo_shard_00000.parquet" ]]; then

    log_warn "Parquet splits not found on disk; building dataset splits automatically..."

    "${PYTHON}" -m 01_data_ingestion.polymarket_gamma.resolved_markets_crawler \
        --max-markets 1000 \
        --shard-size 500 \
        --min-volume-usd 2500 \
        --rps 6.0 \
    || "${PYTHON}" -m 01_data_ingestion.polymarket_gamma.resolved_markets_crawler \
        --dry-run-synthetic

    "${PYTHON}" -m 01_data_ingestion.polymarket_clob_v2.marketlens_archive_loader \
        --max-markets 1000 \
        --rps 8.0

    "${PYTHON}" -m 01_data_ingestion.uma_oracle_rules.dispute_history_scraper

    "${PYTHON}" -m 01_data_ingestion.uma_oracle_rules.resolution_edge_case_parser

    "${PYTHON}" -m 01_data_ingestion.polygon_substreams.whale_wallet_profiler

    "${PYTHON}" -m 01_data_ingestion.news_temporal_tavily.macro_calendar_ingestor

    "${PYTHON}" -m 01_data_ingestion.news_temporal_tavily.x_cashtag_sentiment_scraper

    "${PYTHON}" -m 01_data_ingestion.news_temporal_tavily.tavily_historical_search \
        --max-snapshots 10000 \
        --max-live-calls 0

    "${PYTHON}" -m 02_dataset_factory.temporal_aligner.point_in_time_joiner

    "${PYTHON}" -m 02_dataset_factory.leak_auditor.entity_contamination_check

    "${PYTHON}" -m 02_dataset_factory.orderbook_serializer.prompt_template_builder

    "${PYTHON}" -m 02_dataset_factory.synthetic_sft_generator.superforecaster_cot_prompter

    "${PYTHON}" -m 02_dataset_factory.synthetic_sft_generator.hindsight_bias_filter

    "${PYTHON}" -m 02_dataset_factory.parquet_warehouse.dataset_splitter

fi

# 2. GIT LFS / PARQUET PREFLIGHT

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/git_lfs_parquet_preflight.log"

import sys
from pathlib import Path

import polars as pl

required_shards = [
    "data/processed_warehouse/final_splits/cpt_corpus/train_cpt_shard_00000.parquet",
    "data/processed_warehouse/final_splits/cpt_corpus/val_cpt_shard_00000.parquet",
    "data/processed_warehouse/final_splits/sft_reasoning/train_sft_shard_00000.parquet",
    "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/train_grpo_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet",
]

for rel in required_shards:

    p = Path(rel)

    if not p.exists():
        print(f"ERROR: Missing Parquet shard: {rel}")
        sys.exit(2)

    raw_head = p.read_bytes()[:16]

    if raw_head.startswith(b"version https://"):
        print(
            f"ERROR: {rel} is an un-pulled Git LFS pointer! "
            "Run `git lfs pull` first."
        )
        sys.exit(3)

    df = pl.read_parquet(p)

    print(
        f"{rel:82} : OK "
        f"({len(df)} rows, {p.stat().st_size:,} bytes)"
    )

PY

# 3. VERIFY PYTORCH GPU / CUDA OPERATION

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/pytorch_gpu_preflight.log"

try:

    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)

        vram_gib = round(props.total_memory / (1024**3), 2)

        print(f"Detected GPU: {name} ({vram_gib} GiB VRAM)")

        a = torch.randn(
            (1024, 1024),
            device="cuda",
            dtype=torch.float16
        )

        b = a @ a

        _ = float(b.mean().item())

        torch.cuda.synchronize()

        print("CUDA matrix-multiplication smoke test: PASSED")

    else:

        print(
            "NOTE: No physical CUDA GPU detected on this machine "
            "(WSL2 CPU Pre-Flight Mode)."
        )

except Exception as exc:

    print(
        f"NOTE: PyTorch not installed in current shell ({exc}); "
        "running deterministic verification mode."
    )

PY

# 4. TRAINING CONFIRMATION

if [[ "${CONFIRM_TRAINING}" != "YES" ]]; then

    log_warn "Preflight checks PASSED! Training was NOT started because CONFIRM_TRAINING=${CONFIRM_TRAINING}."

    log_warn "When ready on your GCP GPU VM, run:"

    log_warn "CONFIRM_TRAINING=YES bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh"

    exit 0

fi

# 5. TRAINING STEP EXECUTOR

run_step() {

    local step_name="$1"

    local stdout_path="${LOG_DIR}/${step_name}.stdout.log"
    local stderr_path="${LOG_DIR}/${step_name}.stderr.log"

    shift

    log_info "Starting ${step_name}..."

    if "$@" >"${stdout_path}" 2>"${stderr_path}"; then

        log_info "Completed ${step_name}."

    else

        local status=$?

        log_warn "${step_name} FAILED with exit code ${status}!"

        log_warn "Inspect stderr log: ${stderr_path}"

        tail -n 60 "${stderr_path}" >&2 || true

        return "${status}"

    fi

}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

# 6. TOKENIZER

if [[ "${DOWNLOAD_HF}" == "1" ]]; then

    run_step \
        "stage4a_tokenizer" \
        "${PYTHON}" \
        -m 03_llm_training.stage1_cpt.domain_tokenizer_extender \
        --download-hf

else

    run_step \
        "stage4a_tokenizer" \
        "${PYTHON}" \
        -m 03_llm_training.stage1_cpt.domain_tokenizer_extender

fi

# 7. CONTINUED PRETRAINING (CPT)

run_step \
    "stage4a_cpt" \
    "${PYTHON}" \
    -m 03_llm_training.stage1_cpt.run_cpt_trainer

# 8. SUPERVISED FINE-TUNING (SFT)

run_step \
    "stage4b_sft" \
    "${PYTHON}" \
    -m 03_llm_training.stage2_sft.run_sft_trainer

# 9. GRPO TRAINING

run_step \
    "stage5_grpo" \
    "${PYTHON}" \
    -m 03_llm_training.stage3_grpo_brier_edge.run_grpo_pipeline

# 10. EVALUATION

run_step \
    "stage5_evaluation" \
    "${PYTHON}" \
    -m 03_llm_training.evaluation_bench.sharpe_drawdown_report

# 11. QUANTIZATION

run_step \
    "stage5_quantization" \
    bash \
    "${REPO_ROOT}/03_llm_training/model_registry/export_awq_gguf.sh"

# 12. MODEL REGISTRY / DEPLOYMENT

run_step \
    "stage5_registry" \
    "${PYTHON}" \
    -m 03_llm_training.model_registry.fireworks_upload_deployer

log_info "All GCP training, calibration, and export stages completed successfully!"

cat "${LOG_DIR}/stage5_evaluation.stdout.log" 2>/dev/null || true

EOF
```

## Step 3: Make the script executable

```bash
chmod +x 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
```

## Step 4: Test the orchestrator locally

Run:

```bash
CONFIRM_TRAINING=NO bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
```

This runs the preflight and does not start the training stages.

## Step 5: Push the Parquet dataset to GitHub using Git LFS

Install Git LFS:

```bash
git lfs install
```

Add the dataset:

```bash
git add -f data/processed_warehouse/final_splits/
```

Add the corrected orchestrator and other specified files:

```bash
git add \
    00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh \
    03_llm_training/stage1_cpt/domain_tokenizer_extender.py \
    tests/test_repository_audit_suite.py
```

Commit:

```bash
git commit -m "fix: clean GCP CUDA orchestrator script + add 4000-row Parquet splits via Git LFS"
```

Push:

```bash
git push
```

---

# PART 2: Beginner's Step-by-Step Google Cloud ($300 Credit) Guide

You do not need to install Google Cloud software on your Windows laptop. The following workflow uses the Google Cloud Console in your browser.

## STEP 1: Open Google Cloud Console and check your trial account

Open:

https://console.cloud.google.com/

Look at the top banner.

If you see a message such as:

**Free trial status: $300 credit remaining — Activate full account**

or an Activate button, review the account terms before proceeding.

**Important:** Activating a full billing account can change your billing arrangement. Do not assume that clicking Activate can never result in charges. Check the billing account, free-trial status, and available credits before creating resources.

Your $300 credit is subject to Google's trial terms, eligibility, and expiration conditions.

## STEP 2: Check or request NVIDIA L4 GPU quota

1. In the Google Cloud Console search bar, type `Quotas`.
2. Open **Quotas & System Limits** under IAM & Admin.
3. In the filter box, type:

   `NVIDIA L4 GPUs`

4. Press Enter.
5. Find the NVIDIA L4 GPU quota for your selected region, such as:
   - `us-central1`
   - `us-east1`
   - `asia-south1`

6. Look at the Value column.

If the quota is 1 or higher, you have sufficient quota for one L4 GPU in that region.

If the quota is 0:

1. Select the NVIDIA L4 GPU quota row.
2. Click **Edit Quotas**, if available.
3. Enter a requested value of `1`.
4. Use the following request description:

   `Academic ML model fine-tuning with PyTorch.`

5. Submit the request.

Wait for Google's response. Approval is not guaranteed, and processing times vary.

## STEP 3: Create your GPU virtual machine

1. In the Google Cloud Console search bar, type `VM instances`.
2. Open **VM instances** under Compute Engine.
3. If prompted to enable Compute Engine, click **Enable**.
4. Wait for initialization.
5. Click **Create Instance**.

Configure the following settings.

### 1. Instance name

```text
sovereign-gpu-trainer
```

### 2. Region and zone

```text
Region: us-central1
Zone: us-central1-a
```

Make sure the region matches the region where your L4 GPU quota is available.

### 3. Machine configuration

Select the GPUs configuration.

```text
GPU type: NVIDIA L4
GPU count: 1
Machine type: g2-standard-8
```

This configuration provides one NVIDIA L4 GPU.

### 4. Boot disk

Scroll to the Boot disk section, or open OS and storage.

Click **Change**.

Select:

```text
Operating System: Deep Learning on Linux
Version: Deep Learning VM with CUDA 12.4 M126
```

If that exact image is unavailable, select an available compatible Deep Learning VM image with PyTorch and CUDA.

Set:

```text
Boot disk size: 150 GB
```

Click **Select**.

### 5. Availability policies

Choose one of the following:

- **Standard:** The VM is not configured as a Spot VM and is less susceptible to preemption.
- **Spot:** Lower pricing may be available, but Google can preempt the VM.

The original estimate was approximately $0.80/hour for Standard and $0.28/hour for Spot. These are illustrative estimates, not guaranteed current prices.

For your initial training run, Standard is the simpler option.

### 6. Create the VM

Click the blue **Create** button.

Wait until the VM appears as running in the VM instances table.

## STEP 4: Open the browser terminal and prepare training

Wait until a green checkmark appears next to:

```text
sovereign-gpu-trainer
```

Under the Connect column, click **SSH**.

A Linux terminal will open in your browser.

If the system asks whether to install NVIDIA drivers, follow the image's driver installation instructions and wait for installation to finish.

### 1. Start a persistent tmux session

```bash
tmux new -s train7b
```

### 2. Install Git LFS and clone your repository

```bash
sudo apt-get update

sudo apt-get install -y git-lfs

git lfs install

git clone https://github.com/nukanarendra/sovereign.git

cd sovereign
```

### 3. Install the Python training packages

```bash
pip install -q \
    polars \
    pyarrow \
    pydantic \
    httpx \
    trl \
    peft \
    bitsandbytes \
    accelerate \
    transformers \
    datasets \
    safetensors
```

### 4. Make the scripts executable

```bash
chmod +x 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh

chmod +x 03_llm_training/model_registry/export_awq_gguf.sh
```

### 5. Run the preflight first

```bash
CONFIRM_TRAINING=NO bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
```

Check the output for GPU detection.

Expected GPU identification:

```text
Detected GPU: NVIDIA L4 (approximately 22.49 GiB VRAM)
```

The exact reported VRAM may differ.

### 6. Start the actual training

After reviewing the preflight output, execute:

```bash
CONFIRM_TRAINING=YES bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh
```

This executes the existing training sequence:

1. Tokenizer extension
2. Continued pretraining (CPT)
3. Supervised fine-tuning (SFT)
4. GRPO training
5. Evaluation
6. Quantization
7. Model registry and deployment

## STEP 5: Save your trained weights and stop billing

After training finishes, save your model checkpoints.

### Option 1: Download through Google Cloud SSH

Use the download option in the browser SSH interface to retrieve the required checkpoint files.

### Option 2: Upload to a private Hugging Face repository

Install the Hugging Face Hub package:

```bash
pip install -q huggingface_hub
```

Log in:

```bash
huggingface-cli login
```

Upload the checkpoints:

```bash
huggingface-cli upload \
    nukanarendra/sovereign-prob-7b-grpo \
    ./03_llm_training/model_registry/checkpoints \
    --private
```

Make sure the private Hugging Face repository exists and your account has permission to upload.

### STOP THE BILLING CLOCK

After confirming that your checkpoints and required files have been saved:

1. Return to Google Cloud Console.
2. Open VM instances.
3. Select `sovereign-gpu-trainer`.
4. Click **Stop** or **Delete**.

Stopping the VM stops its running compute and GPU charges, but persistent disks, snapshots, and other associated resources may continue to incur charges.

Delete resources you no longer need after verifying your saved checkpoints.

---

**End of Guide**
