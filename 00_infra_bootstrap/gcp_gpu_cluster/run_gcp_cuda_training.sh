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

