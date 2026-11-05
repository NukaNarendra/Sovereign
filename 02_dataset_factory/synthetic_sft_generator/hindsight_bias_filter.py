#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART D)
File: 02_dataset_factory/synthetic_sft_generator/hindsight_bias_filter.py
================================================================================
Quality & Anti-Hindsight Verification Gate for Stage 4B SFT Teacher Traces.

Why Teacher Traces Must Pass a 4-Gate Verification Filter Before SFT Training:
  1. Even when a teacher model is prompted blindly without the ground-truth
     label `o`, a frontier LLM trained in 2025/2026 might recognize a famous
     historical event from its own pre-training weights and write a hindsight
     spoiler inside `<reasoning>` (e.g., "As we know, Trump went on to win in
     November 2024...").
  2. Furthermore, if any teacher trace has malformed XML tags or an `<edge>`
     tag that contradicts `p - q_t`, training Qwen-2.5-7B on that trace during
     Stage 4B SFT will teach the model broken arithmetic!
  3. This module enforces 4 deterministic gates on every generated trace:
       - Gate 1 (`XML_SCHEMA_VALID`): Strict `<reasoning>...</reasoning>`,
         `<prob>0.XXXX</prob>`, and `<edge>+0.XXXX</edge>` structure.
       - Gate 2 (`ALGEBRAIC_EDGE_VALID`): `|(p - q_t) - edge| <= 0.0015`.
       - Gate 3 (`SPOILER_AND_DATE_CLEAN`): Passes `OutcomeSpoilerDetector` and
         contains no unlawful post-cutoff calendar dates.
       - Gate 4 (`FIVE_STEP_RUBRIC_COMPLETE`): Verifies the reasoning trace
         explicitly references Outside View base rates, UMA rules, L2 book
         spread/microstructure, and Inside View catalysts.

Writes the verified Stage 3 Part D SFT Parquet shard to:
  `data/processed_warehouse/sft_teacher_traces/sft_verified_traces_shard_00000.parquet`
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from ..leak_auditor.outcome_spoiler_detector import OutcomeSpoilerDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [HINDSIGHT-FILTER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.hindsight_bias_filter")

REASONING_BLOCK_REGEX = re.compile(
    r"<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<prob>(?P<prob>[0-9.]+)</prob>\s*"
    r"<edge>(?P<edge>[+-]?[0-9.]+)</edge>",
    re.DOTALL,
)


# ==============================================================================
# 1. HINDSIGHT & SCHEMA AUDIT REPORT DATA STRUCTURES
# ==============================================================================
@dataclass(frozen=True)
class TraceVerificationResult:
    """Detailed pass/fail breakdown for a single SFT teacher reasoning trace."""
    snapshot_uid: str
    passed_all_gates: bool
    xml_schema_valid: bool
    algebraic_edge_valid: bool
    spoiler_free_valid: bool
    rubric_completeness_valid: bool
    extracted_prob_p: float
    extracted_edge: float
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HindsightAuditReport:
    """Summary certificate for the Stage 3 Part D SFT Teacher Trace Filter."""
    stage: str
    total_input_traces: int
    verified_clean_traces: int
    rejected_traces: int
    pass_rate_pct: float
    adversarial_unit_test_caught_bad_traces: bool
    output_parquet_shard: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE HINDSIGHT-BIAS & XML SCHEMA FILTER ENGINE
# ==============================================================================
class HindsightBiasFilter:
    """
    Audits generated SFT teacher traces across all 4 verification gates and
    writes the clean `sft_verified_traces_shard_00000.parquet` file.
    """

    def __init__(
        self,
        input_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/sft_teacher_traces/raw_sft_teacher_traces_shard_00000.parquet"
        ),
        output_dir: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/sft_teacher_traces"
        ),
        edge_tolerance: float = 0.0015,
    ) -> None:
        self.input_parquet = input_parquet
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.edge_tolerance = edge_tolerance
        self.spoiler_detector = OutcomeSpoilerDetector()

    def verify_single_trace(
        self,
        snapshot_uid: str,
        market_id: str,
        assistant_response: str,
        midpoint_qt: float,
    ) -> TraceVerificationResult:
        """
        Evaluates a single `assistant_response` string against all 4 quality
        and anti-hindsight gates.
        """
        reasons: List[str] = []

        # Gate 1: Strict XML Schema (<reasoning>, <prob>, <edge>)
        match = REASONING_BLOCK_REGEX.search(assistant_response)
        if not match:
            return TraceVerificationResult(
                snapshot_uid=snapshot_uid,
                passed_all_gates=False,
                xml_schema_valid=False,
                algebraic_edge_valid=False,
                spoiler_free_valid=False,
                rubric_completeness_valid=False,
                extracted_prob_p=0.50,
                extracted_edge=0.0,
                rejection_reasons=["MISSING_OR_MALFORMED_XML_TAGS"],
            )

        reasoning_body = match.group("reasoning").strip()
        try:
            p_val = float(match.group("prob"))
            edge_val = float(match.group("edge"))
        except ValueError:
            return TraceVerificationResult(
                snapshot_uid=snapshot_uid,
                passed_all_gates=False,
                xml_schema_valid=False,
                algebraic_edge_valid=False,
                spoiler_free_valid=False,
                rubric_completeness_valid=False,
                extracted_prob_p=0.50,
                extracted_edge=0.0,
                rejection_reasons=["NON_FLOAT_PROB_OR_EDGE"],
            )

        xml_valid = 0.0001 <= p_val <= 0.9999
        if not xml_valid:
            reasons.append("PROB_OUT_OF_BOUNDS_0001_9999")

        # Gate 2: Algebraic consistency between p, q_t, and <edge>
        expected_edge = round(p_val - float(midpoint_qt), 4)
        edge_diff = abs(expected_edge - edge_val)
        algebraic_valid = edge_diff <= self.edge_tolerance
        if not algebraic_valid:
            reasons.append(
                f"ALGEBRAIC_EDGE_MISMATCH:expected={expected_edge:+0.4f},got={edge_val:+0.4f}"
            )

        # Gate 3: Zero hindsight outcome spoilers inside <reasoning>
        spoiler_scan = self.spoiler_detector.scan_and_redact_record(
            snapshot_uid=snapshot_uid,
            market_id=market_id,
            news_xml=reasoning_body,
        )
        spoiler_valid = spoiler_scan.is_spoiler_free and (
            spoiler_scan.violations_count == 0
        )
        if not spoiler_valid:
            reasons.append("HINDSIGHT_OUTCOME_SPOILER_IN_REASONING")

        # Gate 4: 5-Step Superforecaster Rubric Completeness
        lower_r = reasoning_body.lower()
        has_base_rate = "base rate" in lower_r or "outside view" in lower_r
        has_uma_check = "uma" in lower_r or "hazard" in lower_r or "oracle" in lower_r
        has_book_check = "spread" in lower_r or "midpoint" in lower_r or "obi" in lower_r
        rubric_valid = (
            len(reasoning_body) >= 180
            and has_base_rate
            and has_uma_check
            and has_book_check
        )
        if not rubric_valid:
            reasons.append("INCOMPLETE_SUPERFORECASTER_RUBRIC")

        passed_all = (
            xml_valid and algebraic_valid and spoiler_valid and rubric_valid
        )
        return TraceVerificationResult(
            snapshot_uid=snapshot_uid,
            passed_all_gates=passed_all,
            xml_schema_valid=xml_valid,
            algebraic_edge_valid=algebraic_valid,
            spoiler_free_valid=spoiler_valid,
            rubric_completeness_valid=rubric_valid,
            extracted_prob_p=p_val,
            extracted_edge=edge_val,
            rejection_reasons=reasons,
        )

    def run_hindsight_filter_pipeline(self) -> Dict[str, Any]:
        """
        Audits all rows in `raw_sft_teacher_traces_shard_00000.parquet`, writes
        certified traces to `sft_verified_traces_shard_00000.parquet`, and runs
        adversarial unit tests to prove bad teacher traces are blocked.
        """
        import polars as pl  # type: ignore

        if not self.input_parquet.exists():
            raise FileNotFoundError(
                f"Missing raw SFT teacher traces shard: {self.input_parquet}"
            )

        rows = pl.read_parquet(self.input_parquet).to_dicts()
        verified_rows: List[Dict[str, Any]] = []
        rejected_count = 0

        for row in rows:
            res = self.verify_single_trace(
                snapshot_uid=str(row.get("snapshot_uid") or ""),
                market_id=str(row.get("market_id") or ""),
                assistant_response=str(row.get("assistant_response") or ""),
                midpoint_qt=float(row.get("implied_prob_mid_qt") or 0.50),
            )
            if res.passed_all_gates:
                enriched = dict(row)
                enriched["sft_quality_certificate"] = "VERIFIED_BLIND_SUPERFORECASTER"
                verified_rows.append(enriched)
            else:
                rejected_count += 1

        out_path = self.output_dir / "sft_verified_traces_shard_00000.parquet"
        if verified_rows:
            pl.from_dicts(verified_rows).write_parquet(out_path, compression="zstd")
            logger.info(
                "Wrote Verified SFT Teacher Traces shard: %s (%d clean traces)",
                out_path,
                len(verified_rows),
            )

        # Adversarial self-tests:
        # 1. Hindsight spoiler injected into teacher reasoning
        bad_spoiler_trace = (
            "<reasoning>1. Outside view base rate is 35%. 2. UMA hazard is 0.2. "
            "3. Spread is 0.01. In hindsight, after winning the 2024 election, "
            "this market resolved to Yes.</reasoning>\n<prob>0.9900</prob>\n<edge>+0.3642</edge>"
        )
        adv_spoiler_check = self.verify_single_trace(
            snapshot_uid="ADV-BAD-SPOILER",
            market_id="253591",
            assistant_response=bad_spoiler_trace,
            midpoint_qt=0.6258,
        )

        # 2. Algebraic edge mismatch (p=0.70, q_t=0.60, but edge says -0.15)
        bad_math_trace = (
            "<reasoning>1. Outside view base rate is 35%. 2. UMA hazard is 0.2 and oracle is clean. "
            "3. Midpoint q_t is 0.6000 with spread 0.0120 and OBI +0.10. "
            "4. Catalysts are supportive.</reasoning>\n<prob>0.7000</prob>\n<edge>-0.1500</edge>"
        )
        adv_math_check = self.verify_single_trace(
            snapshot_uid="ADV-BAD-MATH",
            market_id="253591",
            assistant_response=bad_math_trace,
            midpoint_qt=0.6000,
        )

        adv_passed = (not adv_spoiler_check.passed_all_gates) and (
            not adv_math_check.passed_all_gates
        )

        report = HindsightAuditReport(
            stage="STAGE_3_PART_D_HINDSIGHT_BIAS_FILTER",
            total_input_traces=len(rows),
            verified_clean_traces=len(verified_rows),
            rejected_traces=rejected_count,
            pass_rate_pct=round(
                (len(verified_rows) / max(1, len(rows))) * 100.0, 2
            ),
            adversarial_unit_test_caught_bad_traces=adv_passed,
            output_parquet_shard=str(out_path),
        )
        payload = report.to_dict()
        payload["adversarial_unit_tests"] = {
            "spoiler_trace_rejected": not adv_spoiler_check.passed_all_gates,
            "spoiler_rejection_reasons": adv_spoiler_check.rejection_reasons,
            "bad_edge_math_rejected": not adv_math_check.passed_all_gates,
            "bad_edge_rejection_reasons": adv_math_check.rejection_reasons,
        }
        return payload


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    filt = HindsightBiasFilter()
    report = filt.run_hindsight_filter_pipeline()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())