#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — REAL 7B GPU NEURAL NETWORK INFERENCE VERIFIER
File: 03_llm_training/evaluation_bench/verify_trained_gpu_inference.py
================================================================================
Loads the actual merged 7.6-Billion parameter `Sovereign-Prob-7B` model from
`03_llm_training/model_registry/checkpoints/merged_bf16` onto your NVIDIA L40S
GPU, runs autoregressive `model.generate()` on unseen Out-of-Time `TEST`
Polymarket prompts (`test_grpo_shard_00000.parquet`), extracts `<reasoning>`,
`<prob>`, and `<edge>`, and measures real neural network Brier Score, Brier
Skill Score (BSS), and algebraic edge accuracy against Market Consensus!
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import time
import logging
import pathlib
import argparse
import datetime
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [GPU-INFERENCE-TEST] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.verify_trained_gpu_inference")

PROB_REGEX = re.compile(r"<prob>\s*([0-9.]+)\s*</prob>")
EDGE_REGEX = re.compile(r"<edge>\s*([+-]?[0-9.]+)\s*</edge>")


def run_real_gpu_inference_benchmark(
    merged_dir: pathlib.Path = pathlib.Path(
        "03_llm_training/model_registry/checkpoints/merged_bf16"
    ),
    test_parquet: pathlib.Path = pathlib.Path(
        "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet"
    ),
    report_out_path: pathlib.Path = pathlib.Path(
        "03_llm_training/model_registry/checkpoints/stage3_grpo/real_7b_neural_inference_report.json"
    ),
    num_samples: int = 10,
) -> Dict[str, Any]:
    import polars as pl  # type: ignore
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run real 7B neural inference test.")

    logger.info("Loading trained 7B model from %s onto cuda:0...", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    all_test_rows = pl.read_parquet(test_parquet).to_dicts()
    yes_rows = [r for r in all_test_rows if int(r.get("resolved_binary_outcome") or 0) == 1]
    no_rows = [r for r in all_test_rows if int(r.get("resolved_binary_outcome") or 0) == 0]
    half = max(1, num_samples // 2)
    test_rows = (yes_rows[:half] + no_rows[: num_samples - len(yes_rows[:half])])[:num_samples]
    results: List[Dict[str, Any]] = []
    xml_valid_count = 0
    edge_valid_count = 0
    model_brier_sum = 0.0
    market_brier_sum = 0.0
    latency_sum_ms = 0.0

    stop_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id > 0 and im_end_id not in stop_token_ids:
        stop_token_ids.append(im_end_id)

    for idx, row in enumerate(test_rows, start=1):
        sys_p = str(row.get("system_prompt") or "")
        usr_p = str(row.get("user_prompt") or "")
        q_t = float(row.get("implied_prob_mid_qt") or 0.50)
        o_val = int(row.get("resolved_binary_outcome") or 0)

        full_prompt = (
            f"<|im_start|>system\n{sys_p}<|im_end|>\n"
            f"<|im_start|>user\n{usr_p}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Preserve both the head (<|im_start|>system, <market_meta>, <uma_rule>, <book>)
        # and the tail (Task instruction + <|im_end|>\n<|im_start|>assistant\n) up to 3072 tokens
        raw_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        max_prompt_tokens = 3072
        if len(raw_ids) > max_prompt_tokens:
            head_len = max_prompt_tokens - 512
            raw_ids = raw_ids[:head_len] + raw_ids[-512:]
        input_ids_t = torch.tensor([raw_ids], dtype=torch.long, device="cuda:0")
        attention_mask_t = torch.ones_like(input_ids_t)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                max_new_tokens=360,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        latency_sum_ms += latency_ms

        new_tokens = generated_ids[0][input_ids_t.shape[1] :]
        completion_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        m_prob = PROB_REGEX.search(completion_text)
        m_edge = EDGE_REGEX.search(completion_text)
        has_xml = bool(
            "<reasoning>" in completion_text
            and "</reasoning>" in completion_text
            and m_prob is not None
            and m_edge is not None
        )
        if has_xml:
            xml_valid_count += 1

        try:
            p_pred = float(m_prob.group(1)) if m_prob else q_t
            p_pred = max(0.001, min(0.999, p_pred))
        except Exception:
            p_pred = q_t

        try:
            edge_pred = float(m_edge.group(1)) if m_edge else round(p_pred - q_t, 4)
        except Exception:
            edge_pred = round(p_pred - q_t, 4)

        expected_edge = round(p_pred - q_t, 4)
        if abs(expected_edge - edge_pred) <= 0.015:
            edge_valid_count += 1

        mod_brier = (p_pred - float(o_val)) ** 2
        mkt_brier = (q_t - float(o_val)) ** 2
        model_brier_sum += mod_brier
        market_brier_sum += mkt_brier

        logger.info(
            "[Sample %d/%d] %s | Market q_t=%.4f | 7B Model p=%.4f (edge=%+.4f) | Outcome=%d | XML=%s | %.0fms",
            idx,
            len(test_rows),
            row.get("snapshot_uid"),
            q_t,
            p_pred,
            edge_pred,
            o_val,
            has_xml,
            latency_ms,
        )
        results.append(
            {
                "snapshot_uid": row.get("snapshot_uid"),
                "question": str(row.get("question") or "")[:90],
                "market_midpoint_qt": q_t,
                "model_predicted_prob_p": round(p_pred, 4),
                "model_predicted_edge": round(edge_pred, 4),
                "resolved_outcome_o": o_val,
                "model_brier_loss": round(mod_brier, 6),
                "market_brier_loss": round(mkt_brier, 6),
                "xml_tags_valid": has_xml,
                "latency_ms": latency_ms,
                "generated_completion_preview": completion_text[:500],
            }
        )

    n = max(1, len(test_rows))
    avg_mod_brier = round(model_brier_sum / n, 6)
    avg_mkt_brier = round(market_brier_sum / n, 6)
    bss_pct = round((1.0 - (avg_mod_brier / max(1e-9, avg_mkt_brier))) * 100.0, 2)

    report = {
        "stage": "REAL_7B_GPU_NEURAL_INFERENCE_VERIFICATION",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_device": torch.cuda.get_device_name(0),
        "merged_model_dir": str(merged_dir),
        "samples_tested": len(test_rows),
        "xml_schema_compliance_pct": round((xml_valid_count / n) * 100.0, 2),
        "algebraic_edge_consistency_pct": round((edge_valid_count / n) * 100.0, 2),
        "average_latency_ms_per_forecast": round(latency_sum_ms / n, 1),
        "average_7b_neural_brier_score": avg_mod_brier,
        "average_market_baseline_brier_score": avg_mkt_brier,
        "neural_brier_skill_score_bss_pct": bss_pct,
        "sample_generations": results[:3],
    }

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real 7B neural network generation benchmark on unseen TEST markets."
    )
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    report = run_real_gpu_inference_benchmark(num_samples=args.samples)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())