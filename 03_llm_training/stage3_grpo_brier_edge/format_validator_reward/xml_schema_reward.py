#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: XML SCHEMA & RUBRIC FORMAT REWARD
File: 03_llm_training/stage3_grpo_brier_edge/format_validator_reward/xml_schema_reward.py
================================================================================
Deterministic XML Schema, 5-Step Superforecaster Rubric, and Algebraic Edge
Subtraction Verifier Reward Function for Stage 5 GRPO (`TRL GRPOTrainer`).

Why a Structural & Algebraic Format Reward Is Essential During GRPO:
  1. During Group Relative Policy Optimization (GRPO), if the policy is only
     rewarded on `<prob>0.XXXX</prob>` Brier accuracy, reinforcement learning
     can cause "CoT Collapse": the model stops writing structured reasoning,
     drops the `<edge>` tag, or outputs unparseable prose that crashes our
     Stage 6 live MCP execution parser.
  2. Even worse, an LLM might output `<prob>0.6500</prob>` when $q_t = 0.6000$,
     but hallucinate `<edge>-0.1200</edge>` (a sign and arithmetic contradiction).
  3. This verifier enforces 4 deterministic structural invariants on every
     rollout completion:
       - **Invariant 1 (Strict Canonical XML Envelope):**
         Must contain exactly 1 `<reasoning>...</reasoning>`, 1 `<prob>0.XXXX</prob>`,
         and 1 `<edge>+0.XXXX</edge>` in canonical order (`+0.25` bonus; `-1.50`
         hard penalty if malformed).
       - **Invariant 2 (Exact 4-Decimal Formatting & Probability Bounds):**
         `p` must lie in `[0.0001, 0.9999]` and `<edge>` must carry an explicit
         `+` or `-` sign (`+0.10` bonus).
       - **Invariant 3 (Algebraic Subtraction Consistency):**
         Verifies that `|(p - q_t) - edge| <= 0.0015` (`+0.20` bonus; `-0.45`
         penalty if the model's subtraction is wrong).
       - **Invariant 4 (5-Step Tetlock Superforecaster Reasoning Rubric):**
         Verifies that `<reasoning>` explicitly analyzes (1) Outside View base
         rates, (2) UMA Optimistic Oracle rules/hazards, (3) L2 CLOB spread/OBI,
         and (4) Inside View news/macro catalysts (`+0.20` bonus).
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import math
import logging
import pathlib
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [XML-FORMAT-REWARD] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.xml_schema_reward")

STRICT_COMPLETION_REGEX = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<prob>(?P<prob>[0-9]+\.[0-9]{4})</prob>\s*"
    r"<edge>(?P<edge>[+-][0-9]+\.[0-9]{4})</edge>\s*$",
    re.DOTALL,
)

RELAXED_COMPLETION_REGEX = re.compile(
    r"<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<prob>\s*(?P<prob>[0-9.]+)\s*</prob>\s*"
    r"<edge>\s*(?P<edge>[+-]?[0-9.]+)\s*</edge>",
    re.DOTALL,
)


# ==============================================================================
# 1. DATA STRUCTURES FOR XML FORMAT & ALGEBRAIC REWARD TELEMETRY
# ==============================================================================
@dataclass(frozen=True)
class XMLFormatRewardEvaluation:
    """
    Granular pass/fail and reward breakdown for a single GRPO rollout completion.
    """
    snapshot_uid: str
    strict_xml_envelope_valid: bool
    four_decimal_precision_valid: bool
    algebraic_edge_math_valid: bool
    five_step_tetlock_rubric_valid: bool
    extracted_prob_p: float
    extracted_edge: float
    expected_algebraic_edge: float
    algebraic_edge_abs_error: float
    rubric_pillars_matched: int
    total_format_reward: float
    violation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XMLFormatBatchAuditReport:
    """
    Summary audit report verifying the XML Format & Algebraic Edge reward
    across the Stage 5 GRPO Parquet warehouse and adversarial unit tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_completions_evaluated: int
    strict_xml_compliance_rate_pct: float
    algebraic_edge_consistency_rate_pct: float
    tetlock_rubric_completeness_rate_pct: float
    average_total_format_reward: float
    malformed_xml_hard_penalty_verified: bool
    bad_subtraction_penalty_verified: bool
    perfect_trace_max_bonus_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE XML SCHEMA & ALGEBRAIC EDGE REWARD ENGINE (TRL GRPO COMPATIBLE)
# ==============================================================================
class XMLFormatValidatorRewardFunction:
    """
    Evaluates structural XML schema compliance, 4-decimal formatting, algebraic
    edge accuracy (`edge == p - q_t`), and 5-step Superforecaster rubric depth.
    """

    def __init__(
        self,
        base_weight: float = 0.35,
        valid_xml_bonus: float = 0.25,
        four_decimal_bonus: float = 0.10,
        algebraic_edge_match_bonus: float = 0.20,
        rubric_completeness_bonus: float = 0.20,
        algebraic_edge_mismatch_penalty: float = -0.45,
        malformed_xml_hard_penalty: float = -1.50,
        edge_tolerance: float = 0.0015,
    ) -> None:
        self.base_weight = float(base_weight)
        self.valid_xml_bonus = float(valid_xml_bonus)
        self.four_decimal_bonus = float(four_decimal_bonus)
        self.algebraic_edge_match_bonus = float(algebraic_edge_match_bonus)
        self.rubric_completeness_bonus = float(rubric_completeness_bonus)
        self.algebraic_edge_mismatch_penalty = float(
            algebraic_edge_mismatch_penalty
        )
        self.malformed_xml_hard_penalty = float(malformed_xml_hard_penalty)
        self.edge_tolerance = float(edge_tolerance)

    @staticmethod
    def _count_tetlock_rubric_pillars(reasoning_text: str) -> int:
        """
        Counts how many of the 4 core quantitative pillars are explicitly
        analyzed inside `<reasoning>...</reasoning>`:
          1. Outside View & Base Rate prior
          2. UMA Optimistic Oracle rule / ambiguity hazard
          3. L2 CLOB Order Book spread / midpoint / OBI
          4. Inside View news wire / macro / social catalysts
        """
        lower = reasoning_text.lower()
        pillars = 0
        if "base rate" in lower or "outside view" in lower:
            pillars += 1
        if "uma" in lower or "oracle" in lower or "hazard" in lower:
            pillars += 1
        if "spread" in lower or "midpoint" in lower or "obi" in lower:
            pillars += 1
        if "wire" in lower or "catalyst" in lower or "inside view" in lower:
            pillars += 1
        return pillars

    def evaluate_single_completion(
        self,
        completion_text: str,
        market_midpoint_qt: float,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> XMLFormatRewardEvaluation:
        """
        Evaluates a single model completion against all 4 structural and
        algebraic invariants.
        """
        clean_text = str(completion_text or "").replace("<|im_end|>", "").strip()
        q_t = round(float(market_midpoint_qt), 4)
        notes: List[str] = []

        # Check tag counts to prevent duplicate tag injection attacks
        if (
            clean_text.count("<reasoning>") != 1
            or clean_text.count("</reasoning>") != 1
            or clean_text.count("<prob>") != 1
            or clean_text.count("</prob>") != 1
            or clean_text.count("<edge>") != 1
            or clean_text.count("</edge>") != 1
        ):
            notes.append("INVALID_XML_TAG_MULTIPLICITY_OR_MISSING_TAGS")
            return XMLFormatRewardEvaluation(
                snapshot_uid=snapshot_uid,
                strict_xml_envelope_valid=False,
                four_decimal_precision_valid=False,
                algebraic_edge_math_valid=False,
                five_step_tetlock_rubric_valid=False,
                extracted_prob_p=0.50,
                extracted_edge=0.0,
                expected_algebraic_edge=round(0.50 - q_t, 4),
                algebraic_edge_abs_error=1.0,
                rubric_pillars_matched=0,
                total_format_reward=round(
                    self.base_weight * self.malformed_xml_hard_penalty, 5
                ),
                violation_notes=notes,
            )

        strict_match = STRICT_COMPLETION_REGEX.match(clean_text)
        relaxed_match = RELAXED_COMPLETION_REGEX.search(clean_text)

        if not relaxed_match:
            notes.append("REGEX_EXTRACTION_FAILED")
            return XMLFormatRewardEvaluation(
                snapshot_uid=snapshot_uid,
                strict_xml_envelope_valid=False,
                four_decimal_precision_valid=False,
                algebraic_edge_math_valid=False,
                five_step_tetlock_rubric_valid=False,
                extracted_prob_p=0.50,
                extracted_edge=0.0,
                expected_algebraic_edge=round(0.50 - q_t, 4),
                algebraic_edge_abs_error=1.0,
                rubric_pillars_matched=0,
                total_format_reward=round(
                    self.base_weight * self.malformed_xml_hard_penalty, 5
                ),
                violation_notes=notes,
            )

        reasoning_body = relaxed_match.group("reasoning").strip()
        prob_str = relaxed_match.group("prob").strip()
        edge_str = relaxed_match.group("edge").strip()

        try:
            p_val = float(prob_str)
            edge_val = float(edge_str)
        except ValueError:
            notes.append("NON_FLOAT_PROB_OR_EDGE")
            return XMLFormatRewardEvaluation(
                snapshot_uid=snapshot_uid,
                strict_xml_envelope_valid=False,
                four_decimal_precision_valid=False,
                algebraic_edge_math_valid=False,
                five_step_tetlock_rubric_valid=False,
                extracted_prob_p=0.50,
                extracted_edge=0.0,
                expected_algebraic_edge=round(0.50 - q_t, 4),
                algebraic_edge_abs_error=1.0,
                rubric_pillars_matched=0,
                total_format_reward=round(
                    self.base_weight * self.malformed_xml_hard_penalty, 5
                ),
                violation_notes=notes,
            )

        # 1. XML Envelope & Probability Bounds
        prob_in_bounds = 0.0001 <= p_val <= 0.9999
        xml_envelope_ok = prob_in_bounds
        raw_score = self.valid_xml_bonus if xml_envelope_ok else -0.75
        if not prob_in_bounds:
            notes.append("PROB_OUT_OF_BOUNDS_0001_9999")

        # 2. Exact 4-Decimal Precision & Signed Edge Check
        four_dec_ok = strict_match is not None
        if four_dec_ok:
            raw_score += self.four_decimal_bonus
        else:
            notes.append("NON_CANONICAL_4_DECIMAL_OR_SIGN_FORMAT")

        # 3. Algebraic Edge Consistency: |(p - q_t) - edge| <= 0.0015
        expected_edge = round(p_val - q_t, 4)
        edge_err = round(abs(expected_edge - edge_val), 6)
        algebraic_ok = edge_err <= self.edge_tolerance
        if algebraic_ok:
            raw_score += self.algebraic_edge_match_bonus
        else:
            raw_score += self.algebraic_edge_mismatch_penalty
            notes.append(
                f"ALGEBRAIC_EDGE_MISMATCH:expected={expected_edge:+0.4f},got={edge_val:+0.4f}"
            )

        # 4. 5-Step Tetlock Superforecaster Rubric Completeness
        pillars_cnt = self._count_tetlock_rubric_pillars(reasoning_body)
        rubric_ok = (pillars_cnt >= 3) and (len(reasoning_body) >= 160)
        if rubric_ok:
            raw_score += self.rubric_completeness_bonus * (pillars_cnt / 4.0)
        else:
            notes.append(f"INCOMPLETE_RUBRIC_PILLARS:{pillars_cnt}/4")

        total_r = round(self.base_weight * raw_score, 5)
        return XMLFormatRewardEvaluation(
            snapshot_uid=snapshot_uid,
            strict_xml_envelope_valid=xml_envelope_ok,
            four_decimal_precision_valid=four_dec_ok,
            algebraic_edge_math_valid=algebraic_ok,
            five_step_tetlock_rubric_valid=rubric_ok,
            extracted_prob_p=round(p_val, 4),
            extracted_edge=round(edge_val, 4),
            expected_algebraic_edge=expected_edge,
            algebraic_edge_abs_error=edge_err,
            rubric_pillars_matched=pillars_cnt,
            total_format_reward=total_r,
            violation_notes=notes,
        )

    # --------------------------------------------------------------------------
    # TRL `GRPOTrainer` Batch Callable Interface
    # --------------------------------------------------------------------------
    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        implied_prob_mid_qt: Sequence[float],
        snapshot_uid: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Batch reward interface invoked by `trl.GRPOTrainer` across `G=8` group
        completions per prompt.
        """
        rewards: List[float] = []
        uids = snapshot_uid or [f"SNAP-{i}" for i in range(len(completions))]
        for idx, comp in enumerate(completions):
            ev = self.evaluate_single_completion(
                completion_text=comp,
                market_midpoint_qt=float(implied_prob_mid_qt[idx]),
                snapshot_uid=str(uids[idx]),
            )
            rewards.append(ev.total_format_reward)
        return rewards

    # --------------------------------------------------------------------------
    # Warehouse Verification & Adversarial Format Test Suite
    # --------------------------------------------------------------------------
    def run_warehouse_verification(
        self,
        grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> XMLFormatBatchAuditReport:
        """
        Evaluates the XML Format & Algebraic Edge reward across the `VAL` split
        and runs 3 adversarial format/math unit tests.
        """
        import polars as pl  # type: ignore

        grpo_rows = pl.read_parquet(grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        evaluations: List[XMLFormatRewardEvaluation] = []
        for g_row in grpo_rows:
            uid = str(g_row["snapshot_uid"])
            q_t = float(g_row["implied_prob_mid_qt"])
            sft_comp = str(sft_by_uid.get(uid, {}).get("assistant_response", ""))
            evaluations.append(
                self.evaluate_single_completion(
                    completion_text=sft_comp,
                    market_midpoint_qt=q_t,
                    snapshot_uid=uid,
                )
            )

        n_ev = max(1, len(evaluations))
        xml_pct = round(
            (sum(1 for e in evaluations if e.strict_xml_envelope_valid) / n_ev)
            * 100.0,
            2,
        )
        alg_pct = round(
            (sum(1 for e in evaluations if e.algebraic_edge_math_valid) / n_ev)
            * 100.0,
            2,
        )
        rub_pct = round(
            (sum(1 for e in evaluations if e.five_step_tetlock_rubric_valid) / n_ev)
            * 100.0,
            2,
        )
        avg_rew = round(
            sum(e.total_format_reward for e in evaluations) / n_ev, 5
        )

        # Unit Test 1: Malformed XML missing </reasoning> and <edge> -> hard penalty
        malformed_ev = self.evaluate_single_completion(
            completion_text="I think the probability is <prob>0.6500</prob>",
            market_midpoint_qt=0.6000,
        )

        # Unit Test 2: Broken subtraction (p=0.6800, q_t=0.6000, but edge=-0.0500) -> penalized
        bad_math_ev = self.evaluate_single_completion(
            completion_text=(
                "<reasoning>1. Outside view base rate is 35.0% in this macro domain. "
                "2. UMA oracle hazard is 0.20 with deterministic settlement rules. "
                "3. Midpoint q_t = 0.6000 and spread s_t = 0.0120 with OBI +0.10. "
                "4. Inside view pre-cutoff wire catalyst is strongly bullish.</reasoning>\n"
                "<prob>0.6800</prob>\n<edge>-0.0500</edge>"
            ),
            market_midpoint_qt=0.6000,
        )

        # Unit Test 3: Perfect 5-step Superforecaster trace -> maximum format bonus (+0.2625)
        perfect_ev = self.evaluate_single_completion(
            completion_text=(
                "<reasoning>1. Outside view base rate is 35.0% in this macro domain. "
                "2. UMA oracle hazard is 0.20 with deterministic settlement rules. "
                "3. Midpoint q_t = 0.6000 and spread s_t = 0.0120 with OBI +0.10. "
                "4. Inside view pre-cutoff wire catalyst is strongly bullish.</reasoning>\n"
                "<prob>0.6800</prob>\n<edge>+0.0800</edge>"
            ),
            market_midpoint_qt=0.6000,
        )

        return XMLFormatBatchAuditReport(
            stage="STAGE_5_PART_A_XML_FORMAT_VALIDATOR_REWARD",
            parquet_shard_evaluated=str(grpo_parquet),
            total_completions_evaluated=len(evaluations),
            strict_xml_compliance_rate_pct=xml_pct,
            algebraic_edge_consistency_rate_pct=alg_pct,
            tetlock_rubric_completeness_rate_pct=rub_pct,
            average_total_format_reward=avg_rew,
            malformed_xml_hard_penalty_verified=(
                malformed_ev.total_format_reward < -0.50
            ),
            bad_subtraction_penalty_verified=(
                bad_math_ev.total_format_reward < perfect_ev.total_format_reward
                and not bad_math_ev.algebraic_edge_math_valid
            ),
            perfect_trace_max_bonus_verified=(
                perfect_ev.strict_xml_envelope_valid
                and perfect_ev.algebraic_edge_math_valid
                and perfect_ev.total_format_reward > 0.25
            ),
            sample_evaluations=[e.to_dict() for e in evaluations[:2]],
        )


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    reward_fn = XMLFormatValidatorRewardFunction()
    report = reward_fn.run_warehouse_verification()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())