#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: GRPO BRIER CALIBRATION REWARD
File: 03_llm_training/stage3_grpo_brier_edge/brier_calibration_reward/brier_score_fn.py
================================================================================
Strictly Proper Brier Score + Market-Relative Brier Skill Score (BSS) + Clamped
Log-Loss Verifiable Reward Function for Stage 5 GRPO (`TRL GRPOTrainer`).

Mathematical Foundation (Why Strictly Proper Scoring Rules Force True Calibration):
  1. Let $o \in \{0, 1\}$ denote the verifiable ground-truth binary resolution
     of a Polymarket contract, and let $p \in [0.0001, 0.9999]$ denote the
     probability emitted by Qwen-2.5-7B inside `<prob>0.XXXX</prob>`.
  2. The raw Brier loss is $\mathcal{L}_{\text{Brier}}(p, o) = (p - o)^2 \in [0, 1]$.
     We transform this into a zero-centered, bounded proper reward:
         $$R_{\text{proper}}(p, o) = 1.0 - 4.0 \cdot (p - o)^2 \in [-3.0, +1.0]$$
     - Why this scaling? A lazy model that hedges at $p = 0.5000$ receives
       $R_{\text{proper}}(0.50, o) = 1.0 - 4.0(0.25) = 0.0000$.
     - A well-calibrated forecast moving toward the true outcome $o$ earns up to
       $+1.0000$, while overconfident wrong predictions ($|p - o| \to 1$) are
       heavily penalized down to $-3.0000$.
  3. Proof of Strict Propriety:
     Suppose the true conditional probability of the event is $p^*$. The
     expected reward maximized by the LLM policy is:
         $$\mathbb{E}_{o \sim \text{Bernoulli}(p^*)}[R_{\text{proper}}(p, o)]
           = p^*(1 - 4(p - 1)^2) + (1 - p^*)(1 - 4p^2)$$
     Differentiating with respect to $p$ and setting to zero:
         $$\frac{\partial \mathbb{E}[R]}{\partial p}
           = -8p^*(p - 1) - 8(1 - p^*)p = 8(p^* - p) = 0 \iff p = p^*$$
     Because the second derivative is $-8 < 0$, the unique global maximum is
     achieved if and only if the LLM reports its true Bayesian belief $p = p^*$!
  4. Market-Relative Brier Skill Bonus ($\Delta \text{Brier}$):
     To prevent the policy from merely copying the CLOB midpoint $q_t$, we add
     a verifiable Brier Skill bonus proportional to how much the model beats
     the market consensus Brier loss $(q_t - o)^2$:
         $$R_{\text{skill}}(p, q_t, o) = \beta_{\text{skill}} \cdot \left((q_t - o)^2 - (p - o)^2\right)$$
     Note that $(q_t - o)^2$ is constant with respect to the model's action $p$,
     so adding $R_{\text{skill}}$ preserves the strict propriety of the reward!
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
    format="%(asctime)s | %(levelname)-8s | [BRIER-REWARD] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.brier_score_fn")

PROB_TAG_REGEX = re.compile(r"<prob>\s*([0-9]+(?:\.[0-9]+)?)\s*</prob>", re.DOTALL)


# ==============================================================================
# 1. DATA STRUCTURES FOR BRIER REWARD TELEMETRY & VERIFICATION
# ==============================================================================
@dataclass(frozen=True)
class BrierRewardEvaluation:
    """
    Detailed decomposition of the Brier Calibration Reward for a single rollout
    completion generated during Stage 5 GRPO training.
    """
    snapshot_uid: str
    predicted_prob_p: float
    market_midpoint_qt: float
    resolved_outcome_o: int
    prob_tag_extracted: bool
    model_brier_loss: float
    market_brier_loss: float
    brier_improvement_over_market: float
    proper_brier_reward_component: float
    skill_bonus_component: float
    clamped_log_loss_component: float
    total_brier_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrierBatchAuditReport:
    """
    Summary audit report verifying the Brier reward function across the Stage 5
    GRPO Parquet warehouse and mathematical propriety unit tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_snapshots_evaluated: int
    average_market_baseline_brier_loss: float
    average_sft_policy_brier_loss: float
    brier_skill_score_vs_market: float
    average_total_brier_reward: float
    strict_propriety_derivative_verified: bool
    lazy_50pct_hedge_zero_reward_verified: bool
    overconfident_wrong_penalty_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE BRIER CALIBRATION REWARD ENGINE (TRL GRPO COMPATIBLE)
# ==============================================================================
class BrierCalibrationRewardFunction:
    """
    Computes the verifiable Brier Calibration Reward for Stage 5 GRPO rollouts.
    Implements `__call__(prompts, completions, **kwargs)` for direct plug-and-play
    compatibility with HuggingFace `trl.GRPOTrainer`.
    """

    def __init__(
        self,
        base_weight: float = 1.00,
        skill_bonus_multiplier: float = 1.50,
        log_loss_regularizer_weight: float = 0.10,
        epsilon_clamp: float = 0.001,
        missing_prob_penalty: float = -2.00,
    ) -> None:
        self.base_weight = float(base_weight)
        self.skill_bonus_multiplier = float(skill_bonus_multiplier)
        self.log_loss_regularizer_weight = float(log_loss_regularizer_weight)
        self.epsilon_clamp = float(epsilon_clamp)
        self.missing_prob_penalty = float(missing_prob_penalty)

    @staticmethod
    def extract_probability_from_completion(
        completion_text: str,
    ) -> Tuple[Optional[float], bool]:
        """
        Extracts the floating-point probability `p` inside `<prob>0.XXXX</prob>`.
        Returns `(parsed_prob, is_valid)` where `parsed_prob` is in `[0.0001, 0.9999]`.
        """
        match = PROB_TAG_REGEX.search(str(completion_text or ""))
        if not match:
            return None, False
        try:
            p_raw = float(match.group(1))
            if math.isnan(p_raw) or math.isinf(p_raw):
                return None, False
            if not (0.0001 <= p_raw <= 0.9999):
                return None, False
            return round(p_raw, 6), True
        except ValueError:
            return None, False

    def evaluate_single_forecast(
        self,
        completion_text: str,
        resolved_outcome_o: int,
        market_midpoint_qt: float,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> BrierRewardEvaluation:
        """
        Computes the exact Brier Calibration Reward components for a single
        model completion against ground-truth outcome `o in {0, 1}` and market
        consensus midpoint `q_t`.
        """
        o_val = 1 if int(resolved_outcome_o) == 1 else 0
        q_t = max(0.001, min(0.999, float(market_midpoint_qt)))
        market_brier = (q_t - float(o_val)) ** 2

        p_extracted, valid_prob = self.extract_probability_from_completion(
            completion_text
        )
        if not valid_prob or p_extracted is None:
            return BrierRewardEvaluation(
                snapshot_uid=snapshot_uid,
                predicted_prob_p=0.50,
                market_midpoint_qt=round(q_t, 4),
                resolved_outcome_o=o_val,
                prob_tag_extracted=False,
                model_brier_loss=0.25,
                market_brier_loss=round(market_brier, 6),
                brier_improvement_over_market=round(market_brier - 0.25, 6),
                proper_brier_reward_component=self.missing_prob_penalty,
                skill_bonus_component=0.0,
                clamped_log_loss_component=0.0,
                total_brier_reward=self.missing_prob_penalty,
            )

        p_val = p_extracted
        model_brier = (p_val - float(o_val)) ** 2
        brier_improvement = market_brier - model_brier

        # 1. Zero-centered Proper Brier Score: 1.0 - 4.0 * (p - o)^2 in [-3.0, +1.0]
        proper_brier_reward = 1.0 - (4.0 * model_brier)

        # 2. Market-Relative Brier Skill Bonus: +skill_mult * ((q_t - o)^2 - (p - o)^2)
        skill_bonus = self.skill_bonus_multiplier * brier_improvement

        # 3. Clamped Log-Loss Tail Regularizer (centered at p=0.50 -> log(0.5) = -0.693147)
        p_clamped = max(self.epsilon_clamp, min(1.0 - self.epsilon_clamp, p_val))
        prob_of_true_class = p_clamped if o_val == 1 else (1.0 - p_clamped)
        log_score_centered = math.log(prob_of_true_class) - math.log(0.50)
        log_reg_component = self.log_loss_regularizer_weight * log_score_centered

        raw_total = self.base_weight * (
            proper_brier_reward + skill_bonus + log_reg_component
        )

        return BrierRewardEvaluation(
            snapshot_uid=snapshot_uid,
            predicted_prob_p=round(p_val, 4),
            market_midpoint_qt=round(q_t, 4),
            resolved_outcome_o=o_val,
            prob_tag_extracted=True,
            model_brier_loss=round(model_brier, 6),
            market_brier_loss=round(market_brier, 6),
            brier_improvement_over_market=round(brier_improvement, 6),
            proper_brier_reward_component=round(proper_brier_reward, 5),
            skill_bonus_component=round(skill_bonus, 5),
            clamped_log_loss_component=round(log_reg_component, 5),
            total_brier_reward=round(raw_total, 5),
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
        snapshot_uid: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Batch reward interface invoked by `trl.GRPOTrainer` across `G=8` group
        completions per prompt. Returns a list of scalar float rewards.
        """
        rewards: List[float] = []
        uids = snapshot_uid or [f"SNAP-{i}" for i in range(len(completions))]
        for idx, comp in enumerate(completions):
            o_val = int(resolved_binary_outcome[idx])
            q_t = float(implied_prob_mid_qt[idx])
            uid = str(uids[idx])
            ev = self.evaluate_single_forecast(
                completion_text=comp,
                resolved_outcome_o=o_val,
                market_midpoint_qt=q_t,
                snapshot_uid=uid,
            )
            rewards.append(ev.total_brier_reward)
        return rewards

    # --------------------------------------------------------------------------
    # Mathematical Propriety & Warehouse Verification Suite
    # --------------------------------------------------------------------------
    def run_warehouse_verification(
        self,
        grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> BrierBatchAuditReport:
        """
        Verifies the Brier Calibration Reward across the held-out `VAL` split
        and executes 3 mathematical unit tests:
          1. Lazy 50% hedging (`p = 0.50, q_t = 0.50`) yields `0.0` proper reward.
          2. Overconfident wrong forecast (`p = 0.99, o = 0`) yields `< -2.80`.
          3. Expected reward under `p* = 0.70` is strictly maximized at `p = 0.70`.
        """
        import polars as pl  # type: ignore

        if not grpo_parquet.exists() or not sft_parquet.exists():
            raise FileNotFoundError(
                f"Missing GRPO or SFT Parquet shards at {grpo_parquet} / {sft_parquet}"
            )

        grpo_rows = pl.read_parquet(grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        evaluations: List[BrierRewardEvaluation] = []
        for g_row in grpo_rows:
            uid = str(g_row["snapshot_uid"])
            o_val = int(g_row["resolved_binary_outcome"])
            q_t = float(g_row["implied_prob_mid_qt"])
            sft_comp = str(
                sft_by_uid.get(uid, {}).get(
                    "assistant_response",
                    f"<reasoning>Baseline</reasoning><prob>{q_t:0.4f}</prob><edge>+0.0000</edge>",
                )
            )
            evaluations.append(
                self.evaluate_single_forecast(
                    completion_text=sft_comp,
                    resolved_outcome_o=o_val,
                    market_midpoint_qt=q_t,
                    snapshot_uid=uid,
                )
            )

        n_ev = max(1, len(evaluations))
        avg_mkt_brier = round(
            sum(e.market_brier_loss for e in evaluations) / n_ev, 6
        )
        avg_mod_brier = round(
            sum(e.model_brier_loss for e in evaluations) / n_ev, 6
        )
        bss = round(1.0 - (avg_mod_brier / max(1e-6, avg_mkt_brier)), 4)
        avg_reward = round(
            sum(e.total_brier_reward for e in evaluations) / n_ev, 5
        )

        # Unit Test 1: Lazy 50% hedge at q_t = 0.50 -> 0.0000 reward
        lazy_ev = self.evaluate_single_forecast(
            completion_text="<reasoning>Hedge</reasoning><prob>0.5000</prob><edge>+0.0000</edge>",
            resolved_outcome_o=1,
            market_midpoint_qt=0.5000,
        )
        lazy_zero_ok = abs(lazy_ev.total_brier_reward) <= 1e-6

        # Unit Test 2: Overconfident wrong prediction (p=0.99 when o=0) -> severe penalty
        wrong_ev = self.evaluate_single_forecast(
            completion_text="<reasoning>Wrong</reasoning><prob>0.9900</prob><edge>+0.4900</edge>",
            resolved_outcome_o=0,
            market_midpoint_qt=0.5000,
        )
        wrong_penalty_ok = wrong_ev.total_brier_reward < -3.0

        # Unit Test 3: Strict Propriety Verification (True belief p* = 0.70)
        p_star = 0.70
        expected_at_true = (
            p_star
            * self.evaluate_single_forecast(
                "<prob>0.7000</prob>", 1, 0.55
            ).proper_brier_reward_component
            + (1.0 - p_star)
            * self.evaluate_single_forecast(
                "<prob>0.7000</prob>", 0, 0.55
            ).proper_brier_reward_component
        )
        expected_at_hedged = (
            p_star
            * self.evaluate_single_forecast(
                "<prob>0.5000</prob>", 1, 0.55
            ).proper_brier_reward_component
            + (1.0 - p_star)
            * self.evaluate_single_forecast(
                "<prob>0.5000</prob>", 0, 0.55
            ).proper_brier_reward_component
        )
        expected_at_overconf = (
            p_star
            * self.evaluate_single_forecast(
                "<prob>0.9000</prob>", 1, 0.55
            ).proper_brier_reward_component
            + (1.0 - p_star)
            * self.evaluate_single_forecast(
                "<prob>0.9000</prob>", 0, 0.55
            ).proper_brier_reward_component
        )
        propriety_ok = (expected_at_true > expected_at_hedged) and (
            expected_at_true > expected_at_overconf
        )

        return BrierBatchAuditReport(
            stage="STAGE_5_PART_A_BRIER_CALIBRATION_REWARD",
            parquet_shard_evaluated=str(grpo_parquet),
            total_snapshots_evaluated=len(evaluations),
            average_market_baseline_brier_loss=avg_mkt_brier,
            average_sft_policy_brier_loss=avg_mod_brier,
            brier_skill_score_vs_market=bss,
            average_total_brier_reward=avg_reward,
            strict_propriety_derivative_verified=propriety_ok,
            lazy_50pct_hedge_zero_reward_verified=lazy_zero_ok,
            overconfident_wrong_penalty_verified=wrong_penalty_ok,
            sample_evaluations=[e.to_dict() for e in evaluations[:3]],
        )


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Stage 5 Brier Calibration Reward Verifier"
    )
    parser.add_argument(
        "--grpo-shard",
        type=pathlib.Path,
        default=pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
    )
    parser.add_argument(
        "--sft-shard",
        type=pathlib.Path,
        default=pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    )
    args = parser.parse_args()

    reward_fn = BrierCalibrationRewardFunction()
    report = reward_fn.run_warehouse_verification(
        grpo_parquet=args.grpo_shard, sft_parquet=args.sft_shard
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())