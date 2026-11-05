#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: REASONING COMPLETENESS VERIFIER
File: 03_llm_training/stage3_grpo_brier_edge/format_validator_reward/reasoning_completeness_check.py
================================================================================
Verifies that `<reasoning>...</reasoning>` Traces Generated During Stage 5 GRPO
Rollouts Explicitly Cite Base Rates, UMA Oracle Rules, L2 CLOB Microstructure,
Inside-View Catalysts, and Bayesian Execution Gates.

Why Reasoning Completeness Must Be Audited Separately from XML Regex Syntax:
  1. A policy trained only with a regex check `<reasoning>.*?</reasoning>` can
     game the format validator during RL by emitting a 3-word dummy string:
         `<reasoning>Yes will win.</reasoning><prob>0.7200</prob><edge>+0.1200</edge>`
  2. When "Chain-of-Thought Collapse" occurs, the LLM loses its zero-shot
     generalization to unseen 2026 macro and geopolitical regimes, and our
     Phase 3 B2B "Probability-as-a-Service" institutional API loses its
     explainable audit trail.
  3. This verifier inspects the semantic & quantitative completeness of every
     `<reasoning>` block across all 5 Philip Tetlock Superforecaster steps:
       - **Pillar 1 (Outside View & Base Rate Prior):** References historical
         base rate `%`, time horizon `T-Xh`, and market consensus `q_t`.
       - **Pillar 2 (UMA Optimistic Oracle Rule & Hazard Audit):** Cites
         ambiguity hazard score, survival-bot eligibility, or settlement rules.
       - **Pillar 3 (L2 Order Book & Whale Microstructure):** Cites bid-ask
         spread `s_t`, minimum edge hurdle, micro-price, OBI, or 24h drift.
       - **Pillar 4 (Inside View & Pre-Cutoff Catalysts):** Cites pre-cutoff
         news wires, macro releases, or social velocity telemetry.
       - **Pillar 5 (Bayesian Calibration & Execution Gate):** Synthesizes
         calibrated `p`, net `edge`, and an explicit execution recommendation
         (`MAKER_LIMIT_BUY_YES`, `MAKER_LIMIT_BUY_NO`, `HOLD`, or `PASS`).
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [REASONING-CHECK] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.reasoning_completeness_check")

REASONING_EXTRACT_REGEX = re.compile(
    r"<reasoning>(?P<reasoning>.*?)</reasoning>", re.DOTALL
)


# ==============================================================================
# 1. DATA STRUCTURES FOR 5-PILLAR REASONING COMPLETENESS AUDITS
# ==============================================================================
@dataclass(frozen=True)
class ReasoningCompletenessEvaluation:
    """
    Granular 5-pillar audit of a single `<reasoning>...</reasoning>` trace.
    """
    snapshot_uid: str
    reasoning_tag_present: bool
    character_length: int
    pillar_1_outside_view_base_rate: bool
    pillar_2_uma_oracle_hazard_citation: bool
    pillar_3_l2_clob_microstructure_citation: bool
    pillar_4_inside_view_catalyst_citation: bool
    pillar_5_bayesian_action_gate_citation: bool
    pillars_satisfied_count: int
    completeness_ratio: float
    reasoning_completeness_reward: float
    missing_pillars: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningCompletenessBatchReport:
    """
    Summary verification report across the 600 `VAL` completions and adversarial
    CoT-collapse unit tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_completions_audited: int
    full_5_pillar_pass_rate_pct: float
    average_pillars_satisfied: float
    average_reasoning_char_length: float
    average_completeness_reward: float
    collapsed_dummy_cot_penalized_verified: bool
    full_superforecaster_cot_max_bonus_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE 5-STEP SUPERFORECASTER REASONING COMPLETENESS VERIFIER
# ==============================================================================
class ReasoningCompletenessChecker:
    """
    Verifies that `<reasoning>` blocks contain genuine 5-step quantitative
    decomposition rather than collapsed dummy text during Stage 5 GRPO rollouts.
    """

    def __init__(
        self,
        max_completeness_bonus: float = 0.25,
        collapsed_cot_penalty: float = -0.50,
        min_reasoning_chars: int = 160,
    ) -> None:
        self.max_completeness_bonus = float(max_completeness_bonus)
        self.collapsed_cot_penalty = float(collapsed_cot_penalty)
        self.min_reasoning_chars = int(min_reasoning_chars)

    def evaluate_single_completion(
        self,
        completion_text: str,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> ReasoningCompletenessEvaluation:
        """
        Inspects a single model completion for all 5 Philip Tetlock
        Superforecaster analytical pillars.
        """
        match = REASONING_EXTRACT_REGEX.search(str(completion_text or ""))
        if not match:
            return ReasoningCompletenessEvaluation(
                snapshot_uid=snapshot_uid,
                reasoning_tag_present=False,
                character_length=0,
                pillar_1_outside_view_base_rate=False,
                pillar_2_uma_oracle_hazard_citation=False,
                pillar_3_l2_clob_microstructure_citation=False,
                pillar_4_inside_view_catalyst_citation=False,
                pillar_5_bayesian_action_gate_citation=False,
                pillars_satisfied_count=0,
                completeness_ratio=0.0,
                reasoning_completeness_reward=self.collapsed_cot_penalty,
                missing_pillars=["ALL_5_PILLARS_MISSING_NO_REASONING_TAG"],
            )

        body = match.group("reasoning").strip()
        lower = body.lower()
        char_len = len(body)

        p1_ok = any(
            k in lower
            for k in ("base rate", "outside view", "prior", "historical yes")
        )
        p2_ok = any(
            k in lower
            for k in ("uma", "oracle", "hazard", "settlement", "ambiguity")
        )
        p3_ok = any(
            k in lower
            for k in ("spread", "midpoint", "micro-price", "obi", "order book")
        )
        p4_ok = any(
            k in lower
            for k in ("inside view", "catalyst", "wire", "macro", "drift")
        )
        p5_ok = any(
            k in lower
            for k in (
                "calibrated probability",
                "execution gate",
                "maker_limit",
                "hold_inside_spread",
                "pass_high_uma",
                "recommended action",
            )
        )

        missing: List[str] = []
        if not p1_ok:
            missing.append("PILLAR_1_OUTSIDE_VIEW_BASE_RATE")
        if not p2_ok:
            missing.append("PILLAR_2_UMA_ORACLE_HAZARD")
        if not p3_ok:
            missing.append("PILLAR_3_L2_CLOB_MICROSTRUCTURE")
        if not p4_ok:
            missing.append("PILLAR_4_INSIDE_VIEW_CATALYSTS")
        if not p5_ok:
            missing.append("PILLAR_5_BAYESIAN_EXECUTION_GATE")

        pillars_cnt = sum([p1_ok, p2_ok, p3_ok, p4_ok, p5_ok])
        comp_ratio = round(pillars_cnt / 5.0, 2)

        if char_len < self.min_reasoning_chars or pillars_cnt <= 1:
            reward = self.collapsed_cot_penalty
        else:
            reward = round(self.max_completeness_bonus * comp_ratio, 5)

        return ReasoningCompletenessEvaluation(
            snapshot_uid=snapshot_uid,
            reasoning_tag_present=True,
            character_length=char_len,
            pillar_1_outside_view_base_rate=p1_ok,
            pillar_2_uma_oracle_hazard_citation=p2_ok,
            pillar_3_l2_clob_microstructure_citation=p3_ok,
            pillar_4_inside_view_catalyst_citation=p4_ok,
            pillar_5_bayesian_action_gate_citation=p5_ok,
            pillars_satisfied_count=pillars_cnt,
            completeness_ratio=comp_ratio,
            reasoning_completeness_reward=reward,
            missing_pillars=missing,
        )

    def __call__(
        self,
        completions: Sequence[str],
        snapshot_uid: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """TRL `GRPOTrainer` compatible batch callable."""
        uids = snapshot_uid or [f"SNAP-{i}" for i in range(len(completions))]
        return [
            self.evaluate_single_completion(
                comp, snapshot_uid=str(uids[i])
            ).reasoning_completeness_reward
            for i, comp in enumerate(completions)
        ]

    def run_warehouse_verification(
        self,
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> ReasoningCompletenessBatchReport:
        """
        Audits all 600 `VAL` reasoning traces and runs adversarial CoT-collapse
        unit tests.
        """
        import polars as pl  # type: ignore

        rows = pl.read_parquet(sft_parquet).to_dicts()
        evals = [
            self.evaluate_single_completion(
                str(r.get("assistant_response", "")),
                snapshot_uid=str(r.get("snapshot_uid", "")),
            )
            for r in rows
        ]

        n_ev = max(1, len(evals))
        full_pass_pct = round(
            (sum(1 for e in evals if e.pillars_satisfied_count == 5) / n_ev)
            * 100.0,
            2,
        )
        avg_pillars = round(
            sum(e.pillars_satisfied_count for e in evals) / n_ev, 2
        )
        avg_chars = round(sum(e.character_length for e in evals) / n_ev, 1)
        avg_rew = round(
            sum(e.reasoning_completeness_reward for e in evals) / n_ev, 5
        )

        # Unit Test 1: Collapsed dummy Chain-of-Thought -> penalized (-0.50)
        collapsed_ev = self.evaluate_single_completion(
            "<reasoning>Yes will win easily.</reasoning><prob>0.7500</prob><edge>+0.1500</edge>",
            snapshot_uid="TEST-COLLAPSED-COT",
        )

        # Unit Test 2: Full 5-pillar trace -> max bonus (+0.25)
        full_ev = evals[0] if evals else collapsed_ev

        return ReasoningCompletenessBatchReport(
            stage="STAGE_5_REASONING_COMPLETENESS_VERIFIER",
            parquet_shard_evaluated=str(sft_parquet),
            total_completions_audited=len(evals),
            full_5_pillar_pass_rate_pct=full_pass_pct,
            average_pillars_satisfied=avg_pillars,
            average_reasoning_char_length=avg_chars,
            average_completeness_reward=avg_rew,
            collapsed_dummy_cot_penalized_verified=(
                collapsed_ev.reasoning_completeness_reward < 0.0
            ),
            full_superforecaster_cot_max_bonus_verified=(
                full_ev.pillars_satisfied_count == 5
                and full_ev.reasoning_completeness_reward == self.max_completeness_bonus
            ),
            sample_evaluations=[e.to_dict() for e in evals[:2]],
        )


def main() -> int:
    checker = ReasoningCompletenessChecker()
    report = checker.run_warehouse_verification()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())