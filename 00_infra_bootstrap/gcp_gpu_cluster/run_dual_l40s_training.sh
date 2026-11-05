
#!/usr/bin/env bash

# ==============================================================================
# SOVEREIGN PROBABILITY ENGINE — DUAL NVIDIA L40S RUNNER
# Target Hardware: 2x NVIDIA L40S (2 x 48GB = 96GB VRAM)
# Inference Target: Fireworks AI Serverless LoRA Endpoint
# Fallback: Local 4-Bit Q4_K_M GGUF
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------------------------
# 0. ENVIRONMENT INITIALIZATION
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SOVEREIGN_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

cd "${REPO_ROOT}"

JOB_START_EPOCH="$(date +%s)"
JOB_ID="${JOB_ID:-DUAL_L40S_$(date -u +%Y%m%dT%H%M%SZ)}"

LOG_DIR="${REPO_ROOT}/logs/job_${JOB_ID}"

CHECKPOINT_ROOT="${REPO_ROOT}/03_llm_training/model_registry/checkpoints"

EMERGENCY_SENTINEL="${CHECKPOINT_ROOT}/EMERGENCY_CREDIT_DRAIN_FLUSH.flag"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python executable not found."
    exit 1
fi

CONFIRM_TRAINING="${CONFIRM_TRAINING:-NO}"
DOWNLOAD_HF="${DOWNLOAD_HF:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

export TOKENIZERS_PARALLELISM="false"

mkdir -p \
    "${LOG_DIR}" \
    "${CHECKPOINT_ROOT}/stage1_cpt" \
    "${CHECKPOINT_ROOT}/stage2_sft" \
    "${CHECKPOINT_ROOT}/stage3_grpo"

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------

log_info() {
    printf '%s | INFO | [DUAL-L40S-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log"
}

log_warn() {
    printf '%s | WARN | [DUAL-L40S-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log" >&2
}

fail() {
    log_warn "$*"
    exit 1
}

# ------------------------------------------------------------------------------
# LOCKING
# ------------------------------------------------------------------------------

command -v flock >/dev/null 2>&1 || fail "flock is required."

exec 9>"${REPO_ROOT}/.dual_l40s_training.lock"

flock -n 9 || fail "Another training orchestrator is already running."

log_info "Starting Dual-L40S training job ${JOB_ID}."
log_info "Repository: ${REPO_ROOT}"
log_info "Logs: ${LOG_DIR}"
log_info "Visible CUDA Devices: ${CUDA_VISIBLE_DEVICES}"

rm -f "${EMERGENCY_SENTINEL}"

# ==============================================================================
# 1. VERIFY PARQUET WAREHOUSE SHARDS
# ==============================================================================

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/parquet_preflight.log"

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

    path = Path(rel)

    if not path.exists():
        print(f"ERROR: Missing Parquet shard: {rel}")
        sys.exit(2)

    raw_head = path.read_bytes()[:16]

    if raw_head.startswith(b"version https://"):
        print(
            f"ERROR: {rel} is an un-pulled Git LFS pointer! "
            "Run git lfs pull first."
        )
        sys.exit(3)

    df = pl.read_parquet(path)

    print(
        f"{rel:82} : OK "
        f"({len(df)} rows, {path.stat().st_size:,} bytes)"
    )

PY

# ==============================================================================
# 2. DETECT AND BENCHMARK NVIDIA L40S GPUs
# ==============================================================================

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/dual_l40s_gpu_preflight.log"

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    n_gpus = torch.cuda.device_count()
    total_vram_gib = 0.0

    print(f"Detected {n_gpus} CUDA GPU(s):")

    for idx in range(n_gpus):

        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)

        vram_gib = round(props.total_memory / (1024 ** 3), 2)
        total_vram_gib += vram_gib

        print(
            f"[cuda:{idx}] {name} | "
            f"VRAM: {vram_gib} GiB | "
            f"Compute Capability: {props.major}.{props.minor}"
        )

        a = torch.randn(
            (2048, 2048),
            device=f"cuda:{idx}",
            dtype=torch.bfloat16
        )

        b = a @ a

        _ = float(b.mean().item())

        torch.cuda.synchronize(idx)

        del a, b

    print(
        f"Combined GPU VRAM: {round(total_vram_gib, 2)} GiB | "
        "BF16 Tensor Core Smoke Test: PASSED"
    )

else:

    print(
        "NOTE: No physical CUDA GPU detected "
        "(CPU Pre-Flight Mode)."
    )

PY

# ==============================================================================
# 3. TRAINING CONFIRMATION
# ==============================================================================

if [[ "${CONFIRM_TRAINING}" != "YES" ]]; then

    log_warn "Preflight checks completed. Training was NOT started."
    log_warn "CONFIRM_TRAINING=${CONFIRM_TRAINING}"

    log_warn "To start training, execute:"
    log_warn "CONFIRM_TRAINING=YES bash 00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh"

    exit 0

fi

# ==============================================================================
# 4. TRAINING STEP EXECUTOR
# ==============================================================================

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

# ==============================================================================
# 5. EXECUTE CPT -> SFT -> GRPO
# ==============================================================================

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

run_step \
    "stage4a_cpt" \
    "${PYTHON}" \
    -m 03_llm_training.stage1_cpt.run_cpt_trainer \
    --steps 16

run_step \
    "stage4b_sft" \
    "${PYTHON}" \
    -m 03_llm_training.stage2_sft.run_sft_trainer \
    --steps 20

run_step \
    "stage5_grpo" \
    "${PYTHON}" \
    -m 03_llm_training.stage3_grpo_brier_edge.run_grpo_pipeline \
    --steps 24

# ==============================================================================
# 6. EVALUATION, QUANTIZATION AND DEPLOYMENT
# ==============================================================================

run_step \
    "stage5_evaluation" \
    "${PYTHON}" \
    -m 03_llm_training.evaluation_bench.sharpe_drawdown_report

run_step \
    "stage5_quantization" \
    bash \
    "${REPO_ROOT}/03_llm_training/model_registry/export_awq_gguf.sh"

run_step \
    "stage5_fireworks_bundle" \
    "${PYTHON}" \
    -m 03_llm_training.model_registry.fireworks_upload_deployer

run_step \
    "stage6_hybrid_router_verify" \
    "${PYTHON}" \
    -m 04_mcp_runtime.inference_router.hybrid_model_router

# ==============================================================================
# 7. COMPLETION
# ==============================================================================

JOB_END_EPOCH="$(date +%s)"
JOB_DURATION=$((JOB_END_EPOCH - JOB_START_EPOCH))

log_info "Dual-L40S training pipeline completed."
log_info "Fireworks bundle packaging completed."
log_info "Stage 6 Hybrid Router verification completed."
log_info "Total execution time: ${JOB_DURATION} seconds."

cat "${LOG_DIR}/stage5_fireworks_bundle.stdout.log" 2>/dev/null || true
