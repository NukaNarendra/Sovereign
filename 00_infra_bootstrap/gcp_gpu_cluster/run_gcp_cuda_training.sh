#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Sovereign-Prob-7B: GCP CUDA training orchestrator
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SOVEREIGN_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -d "${REPO_ROOT}" ]]; then
    printf 'ERROR: Repository directory does not exist: %s\n' "${REPO_ROOT}" >&2
    exit 1
fi

cd -- "${REPO_ROOT}"

JOB_ID="${JOB_ID:-GCP_CUDA_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${REPO_ROOT}/logs/job_${JOB_ID}"
CHECKPOINT_ROOT="${REPO_ROOT}/03_llm_training/model_registry/checkpoints"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIRM_TRAINING="${CONFIRM_TRAINING:-NO}"
DOWNLOAD_HF="${DOWNLOAD_HF:-1}"
BUILD_DATASET="${BUILD_DATASET:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
SKIP_COMPLETED_STEPS="${SKIP_COMPLETED_STEPS:-1}"

mkdir -p \
    "${LOG_DIR}" \
    "${CHECKPOINT_ROOT}/stage1_cpt" \
    "${CHECKPOINT_ROOT}/stage2_sft" \
    "${CHECKPOINT_ROOT}/stage3_grpo"

log_info() {
    printf '%s | INFO | [GCP-CUDA-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log"
}

log_warn() {
    printf '%s | WARN | [GCP-CUDA-ORCHESTRATOR] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${LOG_DIR}/orchestrator.log" >&2
}

fail() {
    log_warn "$*"
    exit 1
}

# Select a usable Python interpreter.
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 || true)"
fi

[[ -n "${PYTHON}" && -x "${PYTHON}" ]] ||
    fail "Python 3 was not found. Set PYTHON to a valid interpreter."

"${PYTHON}" -c 'import sys; print("Python:", sys.version.split()[0])' \
    | tee -a "${LOG_DIR}/orchestrator.log"

command -v flock >/dev/null 2>&1 || fail "flock is required."

exec 9>"${REPO_ROOT}/.gcp_training.lock"
flock -n 9 || fail "Another training orchestrator is already running (.gcp_training.lock)."

log_info "Starting job ${JOB_ID}."
log_info "Repository: ${REPO_ROOT}"
log_info "Python: ${PYTHON}"
log_info "Logs: ${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

# Stop if an external emergency-stop flag has been created.
EMERGENCY_SENTINEL="${CHECKPOINT_ROOT}/EMERGENCY_CREDIT_DRAIN_FLUSH.flag"

if [[ -e "${EMERGENCY_SENTINEL}" ]]; then
    fail "Emergency-stop flag exists: ${EMERGENCY_SENTINEL}. Resolve it before starting."
fi

# -----------------------------------------------------------------------------
# 1. Dataset preparation
# -----------------------------------------------------------------------------

GRPO_TRAIN_SHARD="${REPO_ROOT}/data/processed_warehouse/final_splits/grpo_verifiable/train_grpo_shard_00000.parquet"

if [[ ! -f "${GRPO_TRAIN_SHARD}" ]]; then
    if [[ "${BUILD_DATASET}" != "1" ]]; then
        fail "Dataset shard is missing and BUILD_DATASET=${BUILD_DATASET}: ${GRPO_TRAIN_SHARD}"
    fi

    log_warn "Dataset splits are missing. Starting dataset preparation."

    if ! "${PYTHON}" -m 01_data_ingestion.polymarket_gamma.resolved_markets_crawler \
        --max-markets 1000 \
        --shard-size 500 \
        --min-volume-usd 2500 \
        --rps 6.0; then

        log_warn "Live market crawl failed; attempting synthetic dry-run fallback."

        "${PYTHON}" -m 01_data_ingestion.polymarket_gamma.resolved_markets_crawler \
            --dry-run-synthetic
    fi

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

# -----------------------------------------------------------------------------
# 2. Parquet and Git LFS preflight
# -----------------------------------------------------------------------------

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/git_lfs_parquet_preflight.log"
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError as exc:
    print(f"ERROR: Polars is required for Parquet validation: {exc}")
    sys.exit(10)

required_shards = [
    "data/processed_warehouse/final_splits/cpt_corpus/train_cpt_shard_00000.parquet",
    "data/processed_warehouse/final_splits/cpt_corpus/val_cpt_shard_00000.parquet",
    "data/processed_warehouse/final_splits/sft_reasoning/train_sft_shard_00000.parquet",
    "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/train_grpo_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet",
    "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet",
]

errors = []

for rel in required_shards:
    path = Path(rel)

    if not path.is_file():
        errors.append(f"Missing Parquet shard: {rel}")
        continue

    try:
        with path.open("rb") as stream:
            raw_head = stream.read(128)

        if raw_head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(f"Git LFS pointer has not been pulled: {rel}")
            continue

        frame = pl.read_parquet(path)

        print(
            f"{rel:82} : OK "
            f"({frame.height:,} rows, {path.stat().st_size:,} bytes)"
        )

    except Exception as exc:
        errors.append(f"Cannot read {rel}: {exc}")

if errors:
    for error in errors:
        print(f"ERROR: {error}")
    sys.exit(2)

print("Parquet preflight: PASSED")
PY

# -----------------------------------------------------------------------------
# 3. PyTorch / CUDA preflight
# -----------------------------------------------------------------------------

"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/pytorch_gpu_preflight.log"
import os
import sys

require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"

try:
    import torch
except ImportError as exc:
    print(f"ERROR: PyTorch is not installed in {sys.executable}: {exc}")
    sys.exit(11 if require_cuda else 0)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    message = "No CUDA GPU detected."

    if require_cuda:
        print(
            f"ERROR: {message} Training requires a CUDA-enabled "
            "PyTorch build and a visible GPU."
        )
        sys.exit(12)

    print(
        f"NOTE: {message} Continuing in explicitly permitted "
        "CPU preflight mode."
    )
    sys.exit(0)

try:
    device_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024 ** 3)

    print(f"Detected GPU: {device_name} ({vram_gib:.2f} GiB VRAM)")

    left = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    right = left @ left

    _ = float(right.mean().item())
    torch.cuda.synchronize()

    print("CUDA matrix multiplication smoke test: PASSED")

except Exception as exc:
    print(f"ERROR: CUDA smoke test failed: {exc}")
    sys.exit(13)
PY

# -----------------------------------------------------------------------------
# 4. Explicit training confirmation
# -----------------------------------------------------------------------------

if [[ "${CONFIRM_TRAINING}" != "YES" ]]; then
    log_warn "Preflight checks passed. Training has NOT been started."
    log_warn "To start on the GCP GPU VM, run:"
    log_warn "CONFIRM_TRAINING=YES bash 00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh"
    exit 0
fi

# -----------------------------------------------------------------------------
# 5. Training step executor with resumable completion markers
# Reuse the same JOB_ID when resuming a failed job.
# -----------------------------------------------------------------------------

run_step() {
    local step_name="$1"
    shift

    local stdout_path="${LOG_DIR}/${step_name}.stdout.log"
    local stderr_path="${LOG_DIR}/${step_name}.stderr.log"
    local marker_path="${LOG_DIR}/${step_name}.completed"
    local status=0

    if [[ "${SKIP_COMPLETED_STEPS}" == "1" && -f "${marker_path}" ]]; then
        log_info "Skipping ${step_name}; completion marker already exists."
        return 0
    fi

    log_info "Starting ${step_name}."
    rm -f -- "${marker_path}"

    if "$@" >"${stdout_path}" 2>"${stderr_path}"; then
        : >"${marker_path}"
        log_info "Completed ${step_name}."
    else
        status=$?

        log_warn "${step_name} failed with exit code ${status}."
        log_warn "Inspect logs: ${stdout_path} and ${stderr_path}"

        tail -n 60 "${stderr_path}" >&2 || true

        return "${status}"
    fi
}

# -----------------------------------------------------------------------------
# 6. Tokenizer
# -----------------------------------------------------------------------------

if [[ "${DOWNLOAD_HF}" == "1" ]]; then
    run_step "stage4a_tokenizer" \
        "${PYTHON}" -m 03_llm_training.stage1_cpt.domain_tokenizer_extender \
        --download-hf
else
    run_step "stage4a_tokenizer" \
        "${PYTHON}" -m 03_llm_training.stage1_cpt.domain_tokenizer_extender
fi

# -----------------------------------------------------------------------------
# 7. Continued pretraining (CPT)
# -----------------------------------------------------------------------------

run_step "stage4a_cpt" \
    "${PYTHON}" -m 03_llm_training.stage1_cpt.run_cpt_trainer

# -----------------------------------------------------------------------------
# 8. Supervised fine-tuning (SFT)
# -----------------------------------------------------------------------------

run_step "stage4b_sft" \
    "${PYTHON}" -m 03_llm_training.stage2_sft.run_sft_trainer

# -----------------------------------------------------------------------------
# 9. GRPO training
# -----------------------------------------------------------------------------

run_step "stage5_grpo" \
    "${PYTHON}" -m 03_llm_training.stage3_grpo_brier_edge.run_grpo_pipeline

# -----------------------------------------------------------------------------
# 10. Evaluation
# -----------------------------------------------------------------------------

run_step "stage5_evaluation" \
    "${PYTHON}" -m 03_llm_training.evaluation_bench.sharpe_drawdown_report

# -----------------------------------------------------------------------------
# 11. GGUF / AWQ export
# -----------------------------------------------------------------------------

EXPORT_SCRIPT="${REPO_ROOT}/03_llm_training/model_registry/export_awq_gguf.sh"

[[ -f "${EXPORT_SCRIPT}" ]] || fail "Export script not found: ${EXPORT_SCRIPT}"

run_step "stage5_quantization" bash "${EXPORT_SCRIPT}"

# -----------------------------------------------------------------------------
# 12. Model registry / deployment
# -----------------------------------------------------------------------------

run_step "stage5_registry" \
    "${PYTHON}" -m 03_llm_training.model_registry.fireworks_upload_deployer

log_info "All configured training, evaluation, export, and registry stages completed."

if [[ -f "${LOG_DIR}/stage5_evaluation.stdout.log" ]]; then
    cat "${LOG_DIR}/stage5_evaluation.stdout.log"
fi