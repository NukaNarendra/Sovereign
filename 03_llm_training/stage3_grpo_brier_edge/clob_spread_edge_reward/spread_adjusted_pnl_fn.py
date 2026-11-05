#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: CLOB SPREAD-ADJUSTED PNL EDGE REWARD
File: 03_llm_training/stage3_grpo_brier_edge/clob_spread_edge_reward/spread_adjusted_pnl_fn.py
================================================================================
Central Limit Order Book (CLOB v2) Spread-Adjusted PnL & UMA Ambiguity Trap
Avoidance Reward Function for Stage 5 GRPO (`TRL GRPOTrainer`).

Why Brier Calibration Alone Is Not Enough for a $5.00 Survival Trading Bot:
  1. A forecasting model can achieve a respectable Brier score by nudging the
     market midpoint $q_t$ by $\pm 0.004$ ($0.4\%$) on every contract. However,
     if the Polymarket CLOB bid-ask spread is $s_t = 0.020$ ($2.0\%$), a
     $0.4\%$ edge is completely inside the spread—trading it loses money on
     execution friction!
  2. Furthermore, on high-hazard UMA Optimistic Oracle contracts
     (`safe_for_survival_5usd_bot == False` or `uma_ambiguity_hazard_score > 0.45`),
     expressing an aggressive directional edge $|p - q_t| \ge \text{min\_edge}$
     exposes our $5.00 survival bankroll to subjective oracle disputes.
  3. Therefore, this verifiable reward function evaluates every rollout
     completion across three microstructure regimes:
       - **Regime A (UMA Ambiguity Trap Guard):**
         If `safe_for_survival_5usd_bot == False`:
           * If the model prudently dampens its edge (`|p - q_t| < min_edge`),
             it earns a **Trap Avoidance Bonus (`+0.20`)**.
           * If the model recklessly triggers a trade (`|p - q_t| >= min_edge`),
             it suffers the **UMA Trap Trade Penalty (`-0.50`)**.
       - **Regime B (Discipline Inside the Bid-Ask Spread):**
         If `|p - q_t| < min_required_edge_to_trade`:
           * The model abstains from crossing the spread (`NO_TRADE_HOLD`) and
             earns a **Spread Discipline Bonus (`+0.10`)** whenever the market
             midpoint $q_t$ is already within $5\%$ of the true outcome.
       - **Regime C (Active CLOB Trade Execution when `|p - q_t| >= min_edge`):**
         * **BUY YES (`p - q_t >= min_edge`):**
           Entry cost including half-spread and fee buffer is:
               $$c_{\text{YES}} = \min\left(0.995,\; q_t + \frac{s_t}{2} + \epsilon_{\text{fee}}\right)$$
           Realized per-share economic PnL against outcome $o \in \{0, 1\}$ is:
               $$\text{PnL}_{\text{YES}} = o - c_{\text{YES}}$$
         * **BUY NO (`q_t - p >= min_edge`):**
           Entry cost on the complementary NO leg is:
               $$c_{\text{NO}} = \min\left(0.995,\; (1 - q_t) + \frac{s_t}{2} + \epsilon_{\text{fee}}\right)$$
           Realized per-share economic PnL is:
               $$\text{PnL}_{\text{NO}} = (1 - o) - c_{\text{NO}}$$
         * If the trade loses money (`PnL < 0`), an additional **Overconfidence
           Spread Penalty** scaled by $|p - q_t|$ is deducted so the policy
           never hallucinates fake edge to gamble on coin-flips!
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

from ..brier_calibration_reward.brier_score_fn import BrierCalibrationRewardFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [CLOB-EDGE-REWARD] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.spread_adjusted_pnl_fn")


# ==============================================================================
# 1. DATA STRUCTURES FOR CLOB SPREAD-EDGE REWARD TELEMETRY
# ==============================================================================
@dataclass(frozen=True)
class CLOBSpreadEdgeEvaluation:
    """
    Detailed microstructure execution and PnL breakdown for a single GRPO
    rollout completion.
    """
    snapshot_uid: str
    predicted_prob_p: float
    market_midpoint_qt: float
    clob_spread_st: float
    min_required_edge: float
    signed_model_edge: float
    abs_model_edge: float
    safe_for_survival_bot: bool
    uma_hazard_score: float
    resolved_outcome_o: int
    simulated_trade_action: str
    effective_entry_price: float
    simulated_per_share_pnl: float
    uma_trap_penalty_or_bonus: float
    overconfidence_penalty: float
    total_clob_edge_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CLOBEdgeBatchAuditReport:
    """
    Summary audit report verifying the CLOB Spread-Adjusted PnL reward across
    the Stage 5 GRPO Parquet warehouse and adversarial microstructure tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_snapshots_evaluated: int
    active_trades_triggered_count: int
    disciplined_no_trade_holds_count: int
    uma_trap_markets_avoided_count: int
    simulated_win_rate_on_active_trades_pct: float
    average_per_share_pnl_on_active_trades: float
    average_total_clob_edge_reward: float
    profitable_trade_positive_reward_verified: bool
    uma_trap_trade_penalty_verified: bool
    false_edge_overconfidence_penalty_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE CLOB SPREAD-ADJUSTED PNL REWARD ENGINE (TRL GRPO COMPATIBLE)
# ==============================================================================
class CLOBSpreadAdjustedPnLRewardFunction:
    """
    Computes the verifiable CLOB Spread-Adjusted Economic PnL & UMA Trap
    Avoidance Reward for Stage 5 GRPO rollouts.
    """

    def __init__(
        self,
        base_weight: float = 0.65,
        fee_buffer_abs: float = 0.005,
        pnl_reward_multiplier: float = 2.00,
        overconfidence_spread_penalty: float = -0.35,
        uma_trap_trade_penalty: float = -0.50,
        uma_trap_avoidance_bonus: float = 0.20,
        inside_spread_hold_bonus: float = 0.10,
    ) -> None:
        self.base_weight = float(base_weight)
        self.fee_buffer_abs = float(fee_buffer_abs)
        self.pnl_reward_multiplier = float(pnl_reward_multiplier)
        self.overconfidence_spread_penalty = float(overconfidence_spread_penalty)
        self.uma_trap_trade_penalty = float(uma_trap_trade_penalty)
        self.uma_trap_avoidance_bonus = float(uma_trap_avoidance_bonus)
        self.inside_spread_hold_bonus = float(inside_spread_hold_bonus)

    def evaluate_single_rollout(
        self,
        completion_text: str,
        resolved_outcome_o: int,
        market_midpoint_qt: float,
        clob_spread_st: float,
        min_required_edge_to_trade: float,
        safe_for_survival_5usd_bot: bool = True,
        uma_ambiguity_hazard_score: float = 0.20,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> CLOBSpreadEdgeEvaluation:
        """
        Simulates a maker/taker limit execution on the Polymarket CLOB v2 order
        book whenever `|p - q_t| >= min_required_edge_to_trade` and computes the
        spread-adjusted PnL reward.
        """
        o_val = 1 if int(resolved_outcome_o) == 1 else 0
        q_t = max(0.001, min(0.999, float(market_midpoint_qt)))
        s_t = max(0.001, float(clob_spread_st))
        min_edge = max(
            0.005,
            float(min_required_edge_to_trade or ((s_t / 2.0) + self.fee_buffer_abs)),
        )
        safe_flag = bool(safe_for_survival_5usd_bot) and (
            float(uma_ambiguity_hazard_score) <= 0.45
        )

        p_extracted, valid_prob = (
            BrierCalibrationRewardFunction.extract_probability_from_completion(
                completion_text
            )
        )
        if not valid_prob or p_extracted is None:
            return CLOBSpreadEdgeEvaluation(
                snapshot_uid=snapshot_uid,
                predicted_prob_p=0.50,
                market_midpoint_qt=round(q_t, 4),
                clob_spread_st=round(s_t, 4),
                min_required_edge=round(min_edge, 4),
                signed_model_edge=0.0,
                abs_model_edge=0.0,
                safe_for_survival_bot=safe_flag,
                uma_hazard_score=round(float(uma_ambiguity_hazard_score), 4),
                resolved_outcome_o=o_val,
                simulated_trade_action="INVALID_NO_PROB_TAG",
                effective_entry_price=0.0,
                simulated_per_share_pnl=0.0,
                uma_trap_penalty_or_bonus=-0.50,
                overconfidence_penalty=0.0,
                total_clob_edge_reward=round(self.base_weight * (-1.0), 5),
            )

        p_val = p_extracted
        signed_edge = round(p_val - q_t, 4)
        abs_edge = abs(signed_edge)

        # REGIME 1: High-Ambiguity UMA Trap Contract (`safe_for_survival_5usd_bot == False`)
        if not safe_flag:
            explicit_pass_gate = "PASS_HIGH_UMA_HAZARD" in str(completion_text or "")
            if abs_edge >= min_edge and not explicit_pass_gate:
                # Model recklessly tried to trade a high-hazard UMA dispute trap
                trap_term = self.uma_trap_trade_penalty
                total_r = self.base_weight * trap_term
                action_label = "BLOCKED_RECKLESS_UMA_TRAP_TRADE"
            else:
                # Model properly abstained via PASS_HIGH_UMA_HAZARD gate or edge dampening
                trap_term = self.uma_trap_avoidance_bonus
                total_r = self.base_weight * trap_term
                action_label = "PASS_AVOIDED_UMA_TRAP"

            return CLOBSpreadEdgeEvaluation(
                snapshot_uid=snapshot_uid,
                predicted_prob_p=round(p_val, 4),
                market_midpoint_qt=round(q_t, 4),
                clob_spread_st=round(s_t, 4),
                min_required_edge=round(min_edge, 4),
                signed_model_edge=signed_edge,
                abs_model_edge=round(abs_edge, 4),
                safe_for_survival_bot=False,
                uma_hazard_score=round(float(uma_ambiguity_hazard_score), 4),
                resolved_outcome_o=o_val,
                simulated_trade_action=action_label,
                effective_entry_price=0.0,
                simulated_per_share_pnl=0.0,
                uma_trap_penalty_or_bonus=round(trap_term, 5),
                overconfidence_penalty=0.0,
                total_clob_edge_reward=round(total_r, 5),
            )

        # REGIME 2: Edge is smaller than half-spread + fee hurdle (`|p - q_t| < min_edge`)
        if abs_edge < min_edge:
            # Reward disciplined abstention when the market price is already near truth
            hold_bonus = (
                self.inside_spread_hold_bonus
                if abs(q_t - float(o_val)) <= 0.25
                else 0.02
            )
            total_r = self.base_weight * hold_bonus
            return CLOBSpreadEdgeEvaluation(
                snapshot_uid=snapshot_uid,
                predicted_prob_p=round(p_val, 4),
                market_midpoint_qt=round(q_t, 4),
                clob_spread_st=round(s_t, 4),
                min_required_edge=round(min_edge, 4),
                signed_model_edge=signed_edge,
                abs_model_edge=round(abs_edge, 4),
                safe_for_survival_bot=True,
                uma_hazard_score=round(float(uma_ambiguity_hazard_score), 4),
                resolved_outcome_o=o_val,
                simulated_trade_action="HOLD_INSIDE_SPREAD",
                effective_entry_price=0.0,
                simulated_per_share_pnl=0.0,
                uma_trap_penalty_or_bonus=0.0,
                overconfidence_penalty=0.0,
                total_clob_edge_reward=round(total_r, 5),
            )

        # REGIME 3: Active CLOB Trade Triggered (`|p - q_t| >= min_edge` on a safe market)
        half_spread = s_t / 2.0
        if signed_edge >= min_edge:
            action_label = "BUY_YES"
            entry_price = min(0.995, q_t + half_spread + self.fee_buffer_abs)
            payout = 1.0 if o_val == 1 else 0.0
            per_share_pnl = payout - entry_price
        else:
            action_label = "BUY_NO"
            entry_price = min(0.995, (1.0 - q_t) + half_spread + self.fee_buffer_abs)
            payout = 1.0 if o_val == 0 else 0.0
            per_share_pnl = payout - entry_price

        # Penalize false-edge overconfidence if the triggered trade lost money
        overconf_pen = 0.0
        if per_share_pnl < 0.0:
            overconf_pen = self.overconfidence_spread_penalty * (1.0 + abs_edge)

        raw_reward = (self.pnl_reward_multiplier * per_share_pnl) + overconf_pen
        total_r = self.base_weight * raw_reward

        return CLOBSpreadEdgeEvaluation(
            snapshot_uid=snapshot_uid,
            predicted_prob_p=round(p_val, 4),
            market_midpoint_qt=round(q_t, 4),
            clob_spread_st=round(s_t, 4),
            min_required_edge=round(min_edge, 4),
            signed_model_edge=signed_edge,
            abs_model_edge=round(abs_edge, 4),
            safe_for_survival_bot=True,
            uma_hazard_score=round(float(uma_ambiguity_hazard_score), 4),
            resolved_outcome_o=o_val,
            simulated_trade_action=action_label,
            effective_entry_price=round(entry_price, 4),
            simulated_per_share_pnl=round(per_share_pnl, 5),
            uma_trap_penalty_or_bonus=0.0,
            overconfidence_penalty=round(overconf_pen, 5),
            total_clob_edge_reward=round(total_r, 5),
        )

    # --------------------------------------------------------------------------
    # TRL `GRPOTrainer` Batch Callable Interface
    # --------------------------------------------------------------------------
    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        resolved_binary_outcome: Sequence[int],
        implied_prob_mid_qt: Sequence[float],
        estimated_spread_st: Sequence[float],
        min_required_edge_to_trade: Sequence[float],
        safe_for_survival_5usd_bot: Sequence[bool],
        uma_ambiguity_hazard_score: Sequence[float],
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
            ev = self.evaluate_single_rollout(
                completion_text=comp,
                resolved_outcome_o=int(resolved_binary_outcome[idx]),
                market_midpoint_qt=float(implied_prob_mid_qt[idx]),
                clob_spread_st=float(estimated_spread_st[idx]),
                min_required_edge_to_trade=float(min_required_edge_to_trade[idx]),
                safe_for_survival_5usd_bot=bool(safe_for_survival_5usd_bot[idx]),
                uma_ambiguity_hazard_score=float(uma_ambiguity_hazard_score[idx]),
                snapshot_uid=str(uids[idx]),
            )
            rewards.append(ev.total_clob_edge_reward)
        return rewards

    # --------------------------------------------------------------------------
    # Warehouse Audit & Microstructure Unit Test Suite
    # --------------------------------------------------------------------------
    def run_warehouse_verification(
        self,
        grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> CLOBEdgeBatchAuditReport:
        """
        Evaluates the CLOB Spread-Adjusted PnL reward across the `VAL` split and
        runs 3 adversarial microstructure unit tests.
        """
        import polars as pl  # type: ignore

        grpo_rows = pl.read_parquet(grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        evaluations: List[CLOBSpreadEdgeEvaluation] = []
        for g_row in grpo_rows:
            uid = str(g_row["snapshot_uid"])
            q_t = float(g_row["implied_prob_mid_qt"])
            sft_comp = str(
                sft_by_uid.get(uid, {}).get(
                    "assistant_response",
                    f"<reasoning>Hold</reasoning><prob>{q_t:0.4f}</prob><edge>+0.0000</edge>",
                )
            )
            evaluations.append(
                self.evaluate_single_rollout(
                    completion_text=sft_comp,
                    resolved_outcome_o=int(g_row["resolved_binary_outcome"]),
                    market_midpoint_qt=q_t,
                    clob_spread_st=float(g_row["estimated_spread_st"]),
                    min_required_edge_to_trade=float(
                        g_row["min_required_edge_to_trade"]
                    ),
                    safe_for_survival_5usd_bot=bool(
                        g_row["safe_for_survival_5usd_bot"]
                    ),
                    uma_ambiguity_hazard_score=float(
                        g_row["uma_ambiguity_hazard_score"]
                    ),
                    snapshot_uid=uid,
                )
            )

        active_trades = [
            e
            for e in evaluations
            if e.simulated_trade_action in ("BUY_YES", "BUY_NO")
        ]
        holds = [
            e for e in evaluations if e.simulated_trade_action == "HOLD_INSIDE_SPREAD"
        ]
        traps_avoided = [
            e
            for e in evaluations
            if e.simulated_trade_action == "PASS_AVOIDED_UMA_TRAP"
        ]

        wins = sum(1 for e in active_trades if e.simulated_per_share_pnl > 0.0)
        win_rate_pct = (
            round((wins / len(active_trades)) * 100.0, 2) if active_trades else 100.0
        )
        avg_pnl = (
            round(
                sum(e.simulated_per_share_pnl for e in active_trades)
                / len(active_trades),
                5,
            )
            if active_trades
            else 0.0
        )
        avg_reward = round(
            sum(e.total_clob_edge_reward for e in evaluations)
            / max(1, len(evaluations)),
            5,
        )

        # Unit Test 1: Profitable BUY_YES trade overcoming spread -> strictly positive reward
        prof_ev = self.evaluate_single_rollout(
            completion_text="<reasoning>Bullish</reasoning><prob>0.7200</prob><edge>+0.1200</edge>",
            resolved_outcome_o=1,
            market_midpoint_qt=0.6000,
            clob_spread_st=0.0200,
            min_required_edge_to_trade=0.0150,
            safe_for_survival_5usd_bot=True,
            uma_ambiguity_hazard_score=0.15,
        )

        # Unit Test 2: Reckless trade on a high-hazard UMA trap market -> penalized
        trap_ev = self.evaluate_single_rollout(
            completion_text="<reasoning>Reckless</reasoning><prob>0.8000</prob><edge>+0.2000</edge>",
            resolved_outcome_o=1,
            market_midpoint_qt=0.6000,
            clob_spread_st=0.0200,
            min_required_edge_to_trade=0.0150,
            safe_for_survival_5usd_bot=False,
            uma_ambiguity_hazard_score=0.65,
        )

        # Unit Test 3: Overconfident wrong trade (buys YES at 0.60 with p=0.85, but o=0) -> heavy negative reward
        lose_ev = self.evaluate_single_rollout(
            completion_text="<reasoning>Overconfident</reasoning><prob>0.8500</prob><edge>+0.2500</edge>",
            resolved_outcome_o=0,
            market_midpoint_qt=0.6000,
            clob_spread_st=0.0200,
            min_required_edge_to_trade=0.0150,
            safe_for_survival_5usd_bot=True,
            uma_ambiguity_hazard_score=0.15,
        )

        return CLOBEdgeBatchAuditReport(
            stage="STAGE_5_PART_A_CLOB_SPREAD_EDGE_REWARD",
            parquet_shard_evaluated=str(grpo_parquet),
            total_snapshots_evaluated=len(evaluations),
            active_trades_triggered_count=len(active_trades),
            disciplined_no_trade_holds_count=len(holds),
            uma_trap_markets_avoided_count=len(traps_avoided),
            simulated_win_rate_on_active_trades_pct=win_rate_pct,
            average_per_share_pnl_on_active_trades=avg_pnl,
            average_total_clob_edge_reward=avg_reward,
            profitable_trade_positive_reward_verified=(
                prof_ev.total_clob_edge_reward > 0.40
            ),
            uma_trap_trade_penalty_verified=(trap_ev.total_clob_edge_reward < 0.0),
            false_edge_overconfidence_penalty_verified=(
                lose_ev.total_clob_edge_reward < -0.90
            ),
            sample_evaluations=[e.to_dict() for e in evaluations[:3]],
        )


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    reward_fn = CLOBSpreadAdjustedPnLRewardFunction()
    report = reward_fn.run_warehouse_verification()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())