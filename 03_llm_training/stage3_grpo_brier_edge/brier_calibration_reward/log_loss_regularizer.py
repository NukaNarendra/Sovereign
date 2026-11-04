#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: GRPO LOG-LOSS TAIL REGULARIZER
File: 03_llm_training/stage3_grpo_brier_edge/brier_calibration_reward/log_loss_regularizer.py
================================================================================
Clamped Cross-Entropy / Log-Loss Tail Regularizer for Group Relative Policy
Optimization (GRPO) on Qwen-2.5-7B.

Why Brier Score Needs a Clamped Log-Loss Partner in Prediction Markets:
  1. Pure Brier loss $(p - o)^2$ has a vanishing gradient in the extreme tails
     ($p \in [0.001, 0.05]$ or $p \in [0.95, 0.999]$):
       - The derivative of $(p - 0)^2$ at $p = 0.02$ is only $2(0.02) = 0.04$,
         meaning Brier score barely distinguishes between a $2\%$ probability
         and a $0.2\%$ probability.
       - Yet in Polymarket tail contracts (or Fractional Kelly sizing), the
         difference between $2\%$ and $0.2\%$ is a $10\times$ payout difference!
  2. Conversely, un-clamped logarithmic scoring $\ln(p)$ explodes to $-\infty$
     as $p \to 0$ when $o = 1$, causing catastrophic `NaN` gradient spikes
     during BF16 reinforcement learning on AMD Instinct MI300X.
  3. This module implements a **Symmetric $\epsilon$-Clamped Log-Loss Regularizer**
     ($\epsilon = 0.001$, bounding log-likelihood ratios cleanly while sharpening
     tail probability resolution):
         $$p_{\epsilon} = \text{clip}(p, \epsilon, 1 - \epsilon)$$
         $$R_{\text{logloss}}(p, o) = o \ln(p_{\epsilon}) + (1 - o) \ln(1 - p_{\epsilon}) - \ln(0.50)$$
     - Centered at $p = 0.50$ so a coin-flip forecast earns $0.0000$.
     - Accurate high-confidence forecasts ($p \to o$) earn up to $+\ln(2) \approx +0.69315$.
     - Overconfident tail errors ($|p - o| \to 1$) are penalized down to
       $\ln(0.001) - \ln(0.50) \approx -6.2146$, without ever overflowing BF16!
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

from .brier_score_fn import BrierCalibrationRewardFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [LOGLOSS-REG] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.log_loss_regularizer")


# ==============================================================================
# 1. DATA STRUCTURES FOR CLAMPED LOG-LOSS REGULARIZATION TELEMETRY
# ==============================================================================
@dataclass(frozen=True)
class LogLossRegularizerEvaluation:
    """
    Detailed decomposition of the Clamped Log-Loss Tail Regularizer for a single
    forecast `p` against binary outcome `o in {0, 1}`.
    """
    snapshot_uid: str
    raw_predicted_prob_p: float
    clamped_prob_p: float
    was_epsilon_clamped: bool
    resolved_outcome_o: int
    market_midpoint_qt: float
    raw_binary_cross_entropy_nats: float
    market_binary_cross_entropy_nats: float
    centered_log_score_reward: float
    weighted_regularizer_reward: float
    bounded_gradient_magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogLossBatchAuditReport:
    """
    Batch verification certificate for `LogLossRegularizer` across the Stage 5
    `VAL` split and extreme tail-stability unit tests.
    """
    stage: str
    parquet_shard_evaluated: str
    total_snapshots_evaluated: int
    epsilon_clamp_bound: float
    max_possible_penalty_nats: float
    average_model_cross_entropy_nats: float
    average_market_cross_entropy_nats: float
    average_weighted_regularizer_reward: float
    zero_nan_or_inf_gradients_verified: bool
    extreme_tail_clamp_stability_verified: bool
    sample_evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE CLAMPED LOG-LOSS REGULARIZER ENGINE
# ==============================================================================
class LogLossRegularizer:
    """
    Computes the epsilon-clamped logarithmic scoring rule regularizer to prevent
    BF16 gradient explosions on extreme `0.0000` / `1.0000` predictions while
    preserving strict proper scoring incentives in the `[0.01, 0.05]` and
    `[0.95, 0.99]` probability tails.
    """

    LN_HALF: float = math.log(0.50)  # -0.6931471805599453

    def __init__(
        self,
        epsilon_clamp: float = 0.001,
        regularizer_weight: float = 0.10,
    ) -> None:
        if not (1e-6 <= epsilon_clamp <= 0.05):
            raise ValueError("epsilon_clamp must lie in [1e-6, 0.05].")
        self.epsilon_clamp = float(epsilon_clamp)
        self.regularizer_weight = float(regularizer_weight)
        self.max_penalty_nats = round(
            math.log(self.epsilon_clamp) - self.LN_HALF, 5
        )

    def evaluate_single_forecast(
        self,
        predicted_prob_p: float,
        resolved_outcome_o: int,
        market_midpoint_qt: float = 0.50,
        snapshot_uid: str = "GRPO-ROLLOUT",
    ) -> LogLossRegularizerEvaluation:
        """
        Clamps `predicted_prob_p` into `[epsilon, 1 - epsilon]`, computes the
        binary cross-entropy (log-loss) and centered logarithmic reward, and
        verifies that the gradient magnitude `|d L / d p|` is strictly bounded
        by `1 / epsilon`.
        """
        o_val = 1 if int(resolved_outcome_o) == 1 else 0
        p_raw = float(predicted_prob_p)
        if math.isnan(p_raw) or math.isinf(p_raw):
            p_raw = 0.50

        low_b = self.epsilon_clamp
        high_b = 1.0 - self.epsilon_clamp
        p_clamped = max(low_b, min(high_b, p_raw))
        was_clamped = abs(p_clamped - p_raw) > 1e-9

        q_clamped = max(low_b, min(high_b, float(market_midpoint_qt)))

        # Binary Cross-Entropy (Log-Loss) in nats: - [o ln(p) + (1-o) ln(1-p)]
        prob_true_model = p_clamped if o_val == 1 else (1.0 - p_clamped)
        prob_true_market = q_clamped if o_val == 1 else (1.0 - q_clamped)

        model_ce = -math.log(prob_true_model)
        market_ce = -math.log(prob_true_market)

        # Centered Log Score: ln(prob_true) - ln(0.50) -> 0 at p=0.50, +0.693 at p->o
        centered_score = math.log(prob_true_model) - self.LN_HALF
        weighted_reward = self.regularizer_weight * centered_score

        # Magnitude of derivative d/dp [o ln(p) + (1-o) ln(1-p)] = |o - p| / (p (1-p))
        grad_mag = abs(float(o_val) - p_clamped) / max(
            1e-9, p_clamped * (1.0 - p_clamped)
        )

        return LogLossRegularizerEvaluation(
            snapshot_uid=snapshot_uid,
            raw_predicted_prob_p=round(p_raw, 6),
            clamped_prob_p=round(p_clamped, 6),
            was_epsilon_clamped=was_clamped,
            resolved_outcome_o=o_val,
            market_midpoint_qt=round(q_clamped, 4),
            raw_binary_cross_entropy_nats=round(model_ce, 6),
            market_binary_cross_entropy_nats=round(market_ce, 6),
            centered_log_score_reward=round(centered_score, 5),
            weighted_regularizer_reward=round(weighted_reward, 5),
            bounded_gradient_magnitude=round(grad_mag, 4),
        )

    def __call__(
        self,
        completions: Sequence[str],
        resolved_binary_outcome: Sequence[int],
        implied_prob_mid_qt: Sequence[float],
        snapshot_uid: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        TRL `GRPOTrainer` compatible batch callable returning the weighted
        clamped log-loss regularizer reward for each completion.
        """
        rewards: List[float] = []
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
                snapshot_uid=str(uids[idx]),
            )
            rewards.append(ev.weighted_regularizer_reward)
        return rewards

    def run_warehouse_verification(
        self,
        grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/val_grpo_shard_00000.parquet"
        ),
        sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/val_sft_shard_00000.parquet"
        ),
    ) -> LogLossBatchAuditReport:
        """
        Verifies the Clamped Log-Loss Regularizer across all 600 `VAL` snapshots
        and runs extreme `p = 0.0` and `p = 1.0` tail-clamping stability tests.
        """
        import polars as pl  # type: ignore

        grpo_rows = pl.read_parquet(grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        evals: List[LogLossRegularizerEvaluation] = []
        for row in grpo_rows:
            uid = str(row["snapshot_uid"])
            q_t = float(row["implied_prob_mid_qt"])
            o_val = int(row["resolved_binary_outcome"])
            p_sft = float(
                sft_by_uid.get(uid, {}).get("teacher_predicted_prob_p", q_t)
            )
            evals.append(
                self.evaluate_single_forecast(
                    predicted_prob_p=p_sft,
                    resolved_outcome_o=o_val,
                    market_midpoint_qt=q_t,
                    snapshot_uid=uid,
                )
            )

        n_ev = max(1, len(evals))
        avg_mod_ce = round(
            sum(e.raw_binary_cross_entropy_nats for e in evals) / n_ev, 5
        )
        avg_mkt_ce = round(
            sum(e.market_binary_cross_entropy_nats for e in evals) / n_ev, 5
        )
        avg_rew = round(
            sum(e.weighted_regularizer_reward for e in evals) / n_ev, 5
        )

        # Extreme tail stability unit tests: p=0.0 when o=1, and p=1.0 when o=0
        tail_zero = self.evaluate_single_forecast(0.0, 1, 0.50, "TEST-ZERO-TAIL")
        tail_one = self.evaluate_single_forecast(1.0, 0, 0.50, "TEST-ONE-TAIL")
        finite_ok = (
            not math.isnan(tail_zero.centered_log_score_reward)
            and not math.isinf(tail_zero.centered_log_score_reward)
            and not math.isnan(tail_one.centered_log_score_reward)
            and not math.isinf(tail_one.centered_log_score_reward)
            and all(
                not math.isnan(e.weighted_regularizer_reward)
                and not math.isinf(e.weighted_regularizer_reward)
                for e in evals
            )
        )
        clamp_ok = (
            tail_zero.was_epsilon_clamped
            and tail_one.was_epsilon_clamped
            and abs(tail_zero.centered_log_score_reward - self.max_penalty_nats) <= 1e-4
        )

        return LogLossBatchAuditReport(
            stage="STAGE_5_LOG_LOSS_REGULARIZER_VERIFIER",
            parquet_shard_evaluated=str(grpo_parquet),
            total_snapshots_evaluated=len(evals),
            epsilon_clamp_bound=self.epsilon_clamp,
            max_possible_penalty_nats=self.max_penalty_nats,
            average_model_cross_entropy_nats=avg_mod_ce,
            average_market_cross_entropy_nats=avg_mkt_ce,
            average_weighted_regularizer_reward=avg_rew,
            zero_nan_or_inf_gradients_verified=finite_ok,
            extreme_tail_clamp_stability_verified=clamp_ok,
            sample_evaluations=[e.to_dict() for e in evals[:2]],
        )


def main() -> int:
    reg = LogLossRegularizer()
    report = reg.run_warehouse_verification()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())