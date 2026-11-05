#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: FALSE-EDGE OVERCONFIDENCE PENALTY
File: 03_llm_training/stage3_grpo_brier_edge/clob_spread_edge_reward/overconfidence_penalty.py
================================================================================
Penalizes False-Edge Overconfidence and Reckless Bid-Ask Spread Crossing During
Group Relative Policy Optimization (GRPO).

Why an Explicit Overconfidence Penalty Is Needed for the $5.00 Survival Bot:
  1. Suppose a Polymarket contract sits at midpoint $q_t = 0.5500$ with a wide
     bid-ask spread $s_t = 0.0400$ (`min_required_edge = 0.0250`).
  2. During GRPO exploration, if one rollout hallucinates an aggressive forecast
     $p = 0.8500$ ($\text{edge} = +0.3000$) when the true outcome is $o = 0$,
     that trade would have caused our Fractional Kelly sizer to allocate the
     maximum $15\%$ survival bankroll cap ($0.75) into a total loss!
  3. Even when an aggressive forecast is only slightly wrong or inside a noisy
     high-entropy zone without supporting order-book imbalance, crossing a wide
     CLOB spread destroys expected value via execution friction.
  4. This module computes a dedicated **Quadratic Overconfidence & Friction
     Penalty**:
       - **Case 1 (Wrong-Direction Active Trade):**
         When $|p - q_t| \ge \text{min\_edge}$ AND the trade loses money
         ($(p - q_t)(o - q_t) < 0$), applies a convex penalty proportional to
         both the excess edge $(|p - q_t| - \text{min\_edge})$ and the squared
         excess error over the market consensus:
             $$P_{\text{overconf}} = -\lambda_{\text{oc}} \cdot (1 + |p - q_t|) \cdot \max\left(0,\; (p - o)^2 - (q_t - o)^2\right)$$
       - **Case 2 (Churning Inside the Bid-Ask Spread):**
         When $0.5 \cdot \text{min\_edge} \le |p - q_t| < \text{min\_edge}$ on a
         wide-spread contract ($s_t \ge 0.025$), applies a small spread-friction
         warning nudge so the policy learns clean binary regime separation:
         either hold tightly near $q_t$ or express high-conviction alpha.
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from ..brier_calibration_reward.brier_score_fn import BrierCalibrationRewardFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [OVERCONF-PENALTY] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.overconfidence_penalty")


# ==============================================================================
# 1. OVERCONFIDENCE PENALTY EVALUATION & AUDIT REPORT SCHEMAS
# ==============================================================================
@dataclass(frozen=True)
class OverconfidencePenaltyEvaluation:
    """
    Granular telemetry for a single forecast's false-edge overconfidence and
    bid-ask spread friction evaluation.
    """
    snapshot_uid: str
    predicted_prob_p: float
    market_midpoint_qt: float
    clob_spread_st: float
    min_required_edge: float
    signed_edge: float
    abs_edge: float
    resolved_outcome_o: int
    model_worse_than_market_brier: bool
    triggered_wrong_way_trade: bool
    excess_brier_error_vs_market: float
    spread_friction_drag: float
    total_overconfidence_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverconfidenceBatchAuditReport:
    """
    Summary verification report for `OverconfidencePenaltyEngine` across the
    Stage 5 `VAL` split and adversarial overconfidence unit tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_snapshots_evaluated: int
    wrong_way_active_trades_penalized_count: int
    disciplined_forecasts_zero_penalty_count: int
    average_overconfidence_penalty: float
    reckless_false_edge_heavily_penalized_verified: bool
    calibrated_winning_edge_unpenalized_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE FALSE-EDGE OVERCONFIDENCE PENALTY ENGINE
# ==============================================================================
class OverconfidencePenaltyEngine:
    """
    Evaluates and penalizes false-edge overconfidence where the LLM policy
    deviates aggressively from the CLOB midpoint `q_t` in the wrong direction
    of the resolved outcome `o`.
    """

    def __init__(
        self,
        overconfidence_lambda: float = 1.25,
        wrong_trade_base_penalty: float = -0.35,
        spread_churn_friction_coeff: float = 0.50,
    ) -> None:
        self.overconfidence_lambda = float(overconfidence_lambda)
        self.wrong_trade_base_penalty = float(wrong_trade_base_penalty)
        self.spread_churn_friction_coeff = float(spread_churn_friction_coeff)

    def evaluate_single_forecast(
        self,
        predicted_prob_p: float,
        resolved_outcome_o: int,
        market_midpoint_qt: float,
        clob_spread_st: float,
        min_required_edge: float,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> OverconfidencePenaltyEvaluation:
        """
        Computes the false-edge overconfidence penalty for a single `(p, q_t, s_t, o)`
        tuple.
        """
        o_val = 1 if int(resolved_outcome_o) == 1 else 0
        p_val = max(0.001, min(0.999, float(predicted_prob_p)))
        q_t = max(0.001, min(0.999, float(market_midpoint_qt)))
        s_t = max(0.001, float(clob_spread_st))
        min_edge = max(0.005, float(min_required_edge))

        signed_edge = round(p_val - q_t, 4)
        abs_edge = abs(signed_edge)

        model_brier = (p_val - float(o_val)) ** 2
        market_brier = (q_t - float(o_val)) ** 2
        excess_brier = max(0.0, model_brier - market_brier)
        worse_than_mkt = model_brier > (market_brier + 1e-6)

        # Check if an active trade was triggered (|p - q_t| >= min_edge) in the losing direction
        buys_yes_and_loses = (signed_edge >= min_edge) and (o_val == 0)
        buys_no_and_loses = (signed_edge <= -min_edge) and (o_val == 1)
        wrong_way_trade = buys_yes_and_loses or buys_no_and_loses

        penalty = 0.0
        friction_drag = 0.0

        if wrong_way_trade:
            # Base wrong-trade spread crossing penalty + convex excess Brier term
            friction_drag = -round((s_t / 2.0) * self.spread_churn_friction_coeff, 5)
            convex_term = (
                -self.overconfidence_lambda * (1.0 + abs_edge) * excess_brier
            )
            penalty = (
                self.wrong_trade_base_penalty * (1.0 + abs_edge)
                + convex_term
                + friction_drag
            )
        elif worse_than_mkt and abs_edge >= (0.5 * min_edge):
            # Mild friction penalty for moving away from truth inside/near the spread
            penalty = -0.50 * self.overconfidence_lambda * excess_brier

        return OverconfidencePenaltyEvaluation(
            snapshot_uid=snapshot_uid,
            predicted_prob_p=round(p_val, 4),
            market_midpoint_qt=round(q_t, 4),
            clob_spread_st=round(s_t, 4),
            min_required_edge=round(min_edge, 4),
            signed_edge=signed_edge,
            abs_edge=round(abs_edge, 4),
            resolved_outcome_o=o_val,
            model_worse_than_market_brier=worse_than_mkt,
            triggered_wrong_way_trade=wrong_way_trade,
            excess_brier_error_vs_market=round(excess_brier, 6),
            spread_friction_drag=friction_drag,
            total_overconfidence_penalty=round(penalty, 5),
        )

    def __call__(
        self,
        completions: Sequence[str],
        resolved_binary_outcome: Sequence[int],
        implied_prob_mid_qt: Sequence[float],
        estimated_spread_st: Sequence[float],
        min_required_edge_to_trade: Sequence[float],
        snapshot_uid: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """TRL `GRPOTrainer` batch callable interface."""
        penalties: List[float] = []
        uids = snapshot_uid or [f"SNAP-{i}" for i in range(len(completions))]
        for idx, comp in enumerate(completions):
            p_ext, ok = BrierCalibrationRewardFunction.extract_probability_from_completion(
                comp
            )
            p_val = p_ext if (ok and p_ext is not None) else 0.50
            ev = self.evaluate_single_forecast(
                predicted_prob_p=p_val,
                resolved_outcome_o=int(resolved_binary_outcome[idx]),
                market_midpoint_qt=float(implied_prob_mid_qt[idx]),
                clob_spread_st=float(estimated_spread_st[idx]),
                min_required_edge=float(min_required_edge_to_trade[idx]),
                snapshot_uid=str(uids[idx]),
            )
            penalties.append(ev.total_overconfidence_penalty)
        return penalties

    def run_warehouse_verification(
        self,
        grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> OverconfidenceBatchAuditReport:
        """
        Audits the False-Edge Overconfidence Penalty across all 600 `VAL`
        snapshots and verifies adversarial unit tests.
        """
        import polars as pl  # type: ignore

        grpo_rows = pl.read_parquet(grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        evals: List[OverconfidencePenaltyEvaluation] = []
        for row in grpo_rows:
            uid = str(row["snapshot_uid"])
            q_t = float(row["implied_prob_mid_qt"])
            p_sft = float(
                sft_by_uid.get(uid, {}).get("teacher_predicted_prob_p", q_t)
            )
            evals.append(
                self.evaluate_single_forecast(
                    predicted_prob_p=p_sft,
                    resolved_outcome_o=int(row["resolved_binary_outcome"]),
                    market_midpoint_qt=q_t,
                    clob_spread_st=float(row["estimated_spread_st"]),
                    min_required_edge=float(row["min_required_edge_to_trade"]),
                    snapshot_uid=uid,
                )
            )

        wrong_trades = sum(1 for e in evals if e.triggered_wrong_way_trade)
        zero_pen = sum(1 for e in evals if abs(e.total_overconfidence_penalty) <= 1e-6)
        avg_pen = round(
            sum(e.total_overconfidence_penalty for e in evals) / max(1, len(evals)),
            5,
        )

        # Unit Test 1: Reckless false edge (q_t=0.55, p=0.90, o=0) -> heavy penalty (< -0.80)
        reckless = self.evaluate_single_forecast(
            predicted_prob_p=0.90,
            resolved_outcome_o=0,
            market_midpoint_qt=0.55,
            clob_spread_st=0.02,
            min_required_edge=0.015,
        )

        # Unit Test 2: Winning calibrated edge (q_t=0.55, p=0.75, o=1) -> 0.0 penalty
        winning = self.evaluate_single_forecast(
            predicted_prob_p=0.75,
            resolved_outcome_o=1,
            market_midpoint_qt=0.55,
            clob_spread_st=0.02,
            min_required_edge=0.015,
        )

        return OverconfidenceBatchAuditReport(
            stage="STAGE_5_OVERCONFIDENCE_PENALTY_VERIFIER",
            parquet_shard_evaluated=str(grpo_parquet),
            total_snapshots_evaluated=len(evals),
            wrong_way_active_trades_penalized_count=wrong_trades,
            disciplined_forecasts_zero_penalty_count=zero_pen,
            average_overconfidence_penalty=avg_pen,
            reckless_false_edge_heavily_penalized_verified=(
                reckless.total_overconfidence_penalty < -0.80
            ),
            calibrated_winning_edge_unpenalized_verified=(
                abs(winning.total_overconfidence_penalty) <= 1e-6
            ),
            sample_evaluations=[e.to_dict() for e in evals[:2]],
        )


def main() -> int:
    engine = OverconfidencePenaltyEngine()
    report = engine.run_warehouse_verification()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())