
#!/usr/bin/env bash
# ============================================================================
# SOVEREIGN PROBABILITY ENGINE — STAGE 5: AWQ & GGUF EXPORTER
# File: 03_llm_training/model_registry/export_awq_gguf.sh
# Supports: --mode auto | mi300x | gcp | wsl2
# ============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# 1. PATHS AND ENVIRONMENT
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

MODE="${EXPORT_MODE:-auto}"

while (($#)); do
    case "$1" in
        --mode)
            MODE="${2:?Missing value for --mode}"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 2. HARDWARE MODE DETECTION
# ---------------------------------------------------------------------------

if [[ "${MODE}" == "auto" ]]; then
    if command -v rocminfo >/dev/null 2>&1 &&
       rocminfo 2>/dev/null | grep -qi 'gfx942'; then
        MODE="mi300x"

    elif command -v nvidia-smi >/dev/null 2>&1 &&
         nvidia-smi -L >/dev/null 2>&1; then
        MODE="gcp"

    else
        MODE="wsl2"
    fi
fi

case "${MODE}" in
    mi300x|gcp|wsl2)
        ;;
    *)
        echo "ERROR: Unsupported export mode: ${MODE}" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# 3. PYTHON INTERPRETER
# ---------------------------------------------------------------------------

PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python 3 interpreter not found." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. MODEL AND EXPORT PATHS
# ---------------------------------------------------------------------------

MODEL_REGISTRY="${REPO_ROOT}/03_llm_training/model_registry"

MERGED_BF16_DIR="${MODEL_REGISTRY}/checkpoints/merged_bf16"

AWQ_OUT_DIR="${MODEL_REGISTRY}/checkpoints/quantized_awq_4bit"

GGUF_OUT_DIR="${MODEL_REGISTRY}/checkpoints/quantized_gguf_q4_k_m"

GGUF_FILE_PATH="${GGUF_OUT_DIR}/sovereign-prob-7b-q4_k_m.gguf"

GGUF_F16_PATH="${GGUF_OUT_DIR}/sovereign-prob-7b-f16.gguf"

EXPORTS_STAGE5_DIR="${MODEL_REGISTRY}/exports/stage5"

CALIB_PARQUET="${REPO_ROOT}/data/processed_warehouse/final_splits/grpo_verifiable/train_grpo_shard_00000.parquet"

QUANT_CERT_PATH="${MODEL_REGISTRY}/checkpoints/stage5_quantization_certificate.json"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/third_party/llama.cpp}"

# ---------------------------------------------------------------------------
# 5. CREATE OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------

mkdir -p \
    "${AWQ_OUT_DIR}" \
    "${GGUF_OUT_DIR}" \
    "${EXPORTS_STAGE5_DIR}"

# ---------------------------------------------------------------------------
# 6. LOGGING
# ---------------------------------------------------------------------------

log_info() {
    printf '%s | INFO | [EXPORT-AWQ-GGUF] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"
}

log_error() {
    printf '%s | ERROR | [EXPORT-AWQ-GGUF] | %s\n' \
        "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

fail() {
    log_error "$*"
    exit 1
}

log_info "Starting Stage 5 model export."
log_info "Repository: ${REPO_ROOT}"
log_info "Hardware mode: ${MODE}"
log_info "Python: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# 7. MERGE CPT + SFT + GRPO LORA ADAPTERS
# ---------------------------------------------------------------------------

log_info "Step 1/4: Merging CPT + SFT + GRPO LoRA adapters."

"${PYTHON_BIN}" -m \
    03_llm_training.model_registry.merge_lora_adapters

# ---------------------------------------------------------------------------
# 8. VERIFY MERGED MODEL
# ---------------------------------------------------------------------------

if [[ ! -f "${MERGED_BF16_DIR}/merged_model_manifest.json" ]]; then
    fail "Merged model manifest is missing."
fi

if ! compgen -G "${MERGED_BF16_DIR}/model*.safetensors" >/dev/null &&
   ! compgen -G "${MERGED_BF16_DIR}/*.safetensors" >/dev/null; then
    fail "No merged SafeTensors model weights found."
fi

log_info "Merged model manifest and SafeTensors detected."

# ---------------------------------------------------------------------------
# 9. VERIFY CALIBRATION DATASET
# ---------------------------------------------------------------------------

if [[ ! -f "${CALIB_PARQUET}" ]]; then
    fail "Calibration Parquet file is missing: ${CALIB_PARQUET}"
fi

# ---------------------------------------------------------------------------
# 10. PREPARE LLAMA.CPP
# ---------------------------------------------------------------------------

log_info "Step 2/4: Preparing llama.cpp GGUF conversion tools."

if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then

    mkdir -p "$(dirname "${LLAMA_CPP_DIR}")"

    git clone --depth 1 \
        https://github.com/ggml-org/llama.cpp.git \
        "${LLAMA_CPP_DIR}" ||
        fail "Unable to clone llama.cpp."

fi

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    fail "llama.cpp Hugging Face conversion script not found."
fi

# Locate the llama.cpp quantization executable.

LLAMA_QUANTIZE="${LLAMA_QUANTIZE_BIN:-}"

if [[ -z "${LLAMA_QUANTIZE}" ]]; then

    for candidate in \
        "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
        "${LLAMA_CPP_DIR}/build/bin/quantize" \
        "${LLAMA_CPP_DIR}/build/bin/llama-quantize.exe"; do

        if [[ -x "${candidate}" ]]; then
            LLAMA_QUANTIZE="${candidate}"
            break
        fi

    done

fi

if [[ -z "${LLAMA_QUANTIZE}" || ! -x "${LLAMA_QUANTIZE}" ]]; then
    fail "llama-quantize executable not found. Build llama.cpp first or set LLAMA_QUANTIZE_BIN."
fi

# ---------------------------------------------------------------------------
# 11. CONVERT MERGED MODEL TO GGUF F16
# ---------------------------------------------------------------------------

log_info "Step 3/4: Converting merged model to GGUF F16."

rm -f "${GGUF_F16_PATH}" "${GGUF_FILE_PATH}"

"${PYTHON_BIN}" "${CONVERT_SCRIPT}" \
    "${MERGED_BF16_DIR}" \
    --outfile "${GGUF_F16_PATH}" \
    --outtype f16

if [[ ! -s "${GGUF_F16_PATH}" ]]; then
    fail "F16 GGUF conversion failed or produced an empty file."
fi

log_info "F16 GGUF conversion completed."

# ---------------------------------------------------------------------------
# 12. QUANTIZE TO Q4_K_M
# ---------------------------------------------------------------------------

log_info "Step 4/4: Quantizing GGUF to Q4_K_M."

"${LLAMA_QUANTIZE}" \
    "${GGUF_F16_PATH}" \
    "${GGUF_FILE_PATH}" \
    Q4_K_M

if [[ ! -s "${GGUF_FILE_PATH}" ]]; then
    fail "Q4_K_M quantization failed or produced an empty file."
fi

log_info "Q4_K_M quantization completed."

# ---------------------------------------------------------------------------
# 13. EXPORT GGUF ARTIFACT
# ---------------------------------------------------------------------------

cp -f \
    "${GGUF_FILE_PATH}" \
    "${EXPORTS_STAGE5_DIR}/sovereign-stage5-Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# 14. GENERATE AWQ CONFIGURATION AND QUANTIZATION CERTIFICATE
# ---------------------------------------------------------------------------

export MERGED_BF16_DIR
export AWQ_OUT_DIR
export GGUF_FILE_PATH
export EXPORTS_STAGE5_DIR
export CALIB_PARQUET
export QUANT_CERT_PATH
export MODE

"${PYTHON_BIN}" - <<'PY'

import datetime
import json
import pathlib
import struct
import sys

import polars as pl


# -----------------------------------------------------------------------
# Load environment paths
# -----------------------------------------------------------------------

merged_dir = pathlib.Path(__import__("os").environ["MERGED_BF16_DIR"])
awq_dir = pathlib.Path(__import__("os").environ["AWQ_OUT_DIR"])
gguf_path = pathlib.Path(__import__("os").environ["GGUF_FILE_PATH"])
exports_dir = pathlib.Path(__import__("os").environ["EXPORTS_STAGE5_DIR"])
calib_parquet = pathlib.Path(__import__("os").environ["CALIB_PARQUET"])
cert_path = pathlib.Path(__import__("os").environ["QUANT_CERT_PATH"])

mode = __import__("os").environ["MODE"]


# -----------------------------------------------------------------------
# Validate merged model manifest
# -----------------------------------------------------------------------

manifest_path = merged_dir / "merged_model_manifest.json"

if not manifest_path.is_file():
    sys.exit("ERROR: Merged model manifest is missing.")

merged_manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

policy_weights = dict(
    merged_manifest.get("learned_bayesian_policy_weights") or {}
)


# -----------------------------------------------------------------------
# Read calibration dataset
# -----------------------------------------------------------------------

if not calib_parquet.is_file():
    sys.exit("ERROR: Calibration dataset is missing.")

n_calib = min(
    128,
    pl.scan_parquet(calib_parquet)
    .select(pl.len())
    .collect()
    .item()
)


# -----------------------------------------------------------------------
# Generate AWQ configuration
# -----------------------------------------------------------------------

# IMPORTANT:
# This creates an AWQ configuration file only.
# It does NOT perform AWQ weight quantization.

group_size = 128

awq_config = {
    "quant_method": "awq",
    "zero_point": True,
    "q_group_size": group_size,
    "w_bit": 4,
    "version": "GEMM",
    "modules_to_not_convert": [
        "embed_tokens",
        "lm_head",
    ],
    "calibration_samples": n_calib,
    "sovereign_grpo_policy_weights": policy_weights,
    "quantization_status": "configuration_only_not_quantized",
}

(awq_dir / "quant_config.json").write_text(
    json.dumps(awq_config, indent=2),
    encoding="utf-8",
)


# -----------------------------------------------------------------------
# Validate actual GGUF output
# -----------------------------------------------------------------------

if not gguf_path.is_file():
    sys.exit("ERROR: GGUF output file does not exist.")

if gguf_path.stat().st_size == 0:
    sys.exit("ERROR: GGUF output file is empty.")

with gguf_path.open("rb") as f:
    raw_header = f.read(24)

if len(raw_header) != 24:
    sys.exit("ERROR: GGUF header is incomplete.")

magic, version, n_tensors, n_kv = struct.unpack(
    "<4sIQQ",
    raw_header,
)

header_valid = (
    magic == b"GGUF"
    and version in (2, 3)
    and n_tensors > 0
)

if not header_valid:
    sys.exit("ERROR: Invalid GGUF header or missing tensor metadata.")


# -----------------------------------------------------------------------
# Generate quantization certificate
# -----------------------------------------------------------------------

report = {
    "stage": "STAGE_5_PART_B_AWQ_AND_GGUF_EXPORT",
    "timestamp_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),

    "hardware_mode": mode,

    "upstream_merged_bf16_dir": (
        "03_llm_training/model_registry/checkpoints/merged_bf16"
    ),

    "calibration_prompts_available": n_calib,

    "quantization_group_size": group_size,

    "zero_point_enabled": True,

    "awq": {
        "configuration_generated": True,
        "weights_actually_quantized": False,
        "status": "CONFIGURATION_ONLY",
        "output_dir": (
            "03_llm_training/model_registry/checkpoints/quantized_awq_4bit"
        ),
    },

    "gguf": {
        "format": "GGUF",
        "version": version,
        "quantization_type": "Q4_K_M",
        "tensor_count": n_tensors,
        "metadata_entries": n_kv,
        "file_size_bytes": gguf_path.stat().st_size,
        "binary_magic_header_verified": header_valid,
        "output_path": (
            "03_llm_training/model_registry/checkpoints/"
            "quantized_gguf_q4_k_m/sovereign-prob-7b-q4_k_m.gguf"
        ),
    },

    "gguf_quantization_exit_gate_passed": header_valid,

    "overall_awq_and_gguf_exit_gate_passed": False,

    "notes": [
        "GGUF Q4_K_M export was executed and its header validated.",
        "AWQ configuration was generated but AWQ weight quantization "
        "was not executed.",
        "Model quality, Brier drift, and calibration performance "
        "require separate evaluation."
    ],
}

cert_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

(exports_dir / "export_manifest.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))

PY

# ---------------------------------------------------------------------------
# 15. FINAL STATUS
# ---------------------------------------------------------------------------

log_info "GGUF Q4_K_M export and binary validation completed."

log_info "GGUF output: ${GGUF_FILE_PATH}"

log_info "AWQ configuration: ${AWQ_OUT_DIR}/quant_config.json"

log_info "Quantization certificate: ${QUANT_CERT_PATH}"

log_info "Stage 5 export process completed."

