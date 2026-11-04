#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART A)
File: 02_dataset_factory/temporal_aligner/horizon_stratifier.py
================================================================================
Multi-Horizon Stratification & Inverse-Propensity Curriculum Weighting Engine.

Why Horizon & Uncertainty Stratification Is Crucial for Stage 5 GRPO Training:
  1. In prediction markets, 1 hour before resolution (`T-1h`), 90%+ of contracts
     have already converged to `0.98` or `0.02`. If we train Qwen-2.5-7B on an
     unstratified dataset dominated by late-stage snapshots, the model learns a
     degenerate shortcut: simply copy the market midpoint (`p = q_t`)!
  2. Conversely, at `T-168h` (7 days out) and `T-72h` (3 days out), market
     midpoints sit in the high-entropy zone (`0.25 <= q_t <= 0.75`) where
     informed UMA rule parsing and news synthesis generate massive Brier Skill
     Score (BSS) alpha.
  3. This module assigns every `(market_id, horizon_hours_before_close)` snapshot:
       - A canonical `horizon_regime_label` (`LONG_7D_PLUS`, `MEDIUM_3D`,
         `SHORT_24H`, `IMMINENT_6H_OR_LESS`).
       - An `entropy_uncertainty_zone` (`HIGH_UNCERTAINTY_CORE`,
         `MODERATE_SLOPE`, `TAIL_CONVERGENCE`).
       - A quantitative `grpo_curriculum_weight` in `[0.25, 2.50]` that upweights
         high-entropy, medium-to-long horizon examples and inverse-weights
         over-represented categories (such as routine sports games) so macro,
         geopolitics, politics, and crypto receive balanced gradient updates.
================================================================================
"""

from __future__ import annotations

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
    format="%(asctime)s | %(levelname)-8s | [HORIZON-STRATIFIER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.horizon_stratifier")


# ==============================================================================
# 1. HORIZON TIER & STRATIFIED SNAPSHOT DATA STRUCTURES
# ==============================================================================
@dataclass(frozen=True)
class HorizonTierSpec:
    """
    Defines the quantitative properties and base GRPO curriculum multiplier for
    each time-to-expiry horizon tier.
    """
    tier_name: str
    min_hours_exclusive: int
    max_hours_inclusive: int
    base_curriculum_multiplier: float
    target_portfolio_holding_period: str


HORIZON_TIERS: Tuple[HorizonTierSpec, ...] = (
    HorizonTierSpec(
        tier_name="HORIZON_30D_MACRO_STRUCTURAL",
        min_hours_exclusive=240,
        max_hours_inclusive=99999,
        base_curriculum_multiplier=1.35,
        target_portfolio_holding_period="MULTI_WEEK_THESIS",
    ),
    HorizonTierSpec(
        tier_name="HORIZON_7D_SWING_CATALYST",
        min_hours_exclusive=96,
        max_hours_inclusive=240,
        base_curriculum_multiplier=1.40,
        target_portfolio_holding_period="WEEKLY_CATALYST",
    ),
    HorizonTierSpec(
        tier_name="HORIZON_72H_PRE_EVENT",
        min_hours_exclusive=36,
        max_hours_inclusive=96,
        base_curriculum_multiplier=1.30,
        target_portfolio_holding_period="THREE_DAY_EVENT_WINDOW",
    ),
    HorizonTierSpec(
        tier_name="HORIZON_24H_ACTIVE_BOOK",
        min_hours_exclusive=12,
        max_hours_inclusive=36,
        base_curriculum_multiplier=1.15,
        target_portfolio_holding_period="INTRADAY_LIQUID_CLOB",
    ),
    HorizonTierSpec(
        tier_name="HORIZON_6H_TERMINAL_CONVERGENCE",
        min_hours_exclusive=-1,
        max_hours_inclusive=12,
        base_curriculum_multiplier=0.75,
        target_portfolio_holding_period="TERMINAL_SETTLEMENT_ARBITRAGE",
    ),
)


@dataclass(frozen=True)
class StratifiedHorizonRow:
    """
    Enriches a single `(market_id, horizon_hours_before_close)` snapshot with
    entropy, probability bucket, domain inverse-propensity, and GRPO curriculum
    sampling weights.
    """
    market_id: str
    horizon_hours_before_close: int
    horizon_tier_name: str
    implied_prob_mid_qt: float
    probability_decile_bucket: str
    shannon_binary_entropy_bits: float
    entropy_uncertainty_zone: str
    domain_inverse_propensity_weight: float
    grpo_curriculum_weight: float
    include_in_grpo_rl_split: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. MULTI-HORIZON STRATIFIER & CURRICULUM WEIGHTING ENGINE
# ==============================================================================
class HorizonStratifier:
    """
    Computes binary Shannon entropy, probability decile buckets, domain balance
    factors, and GRPO curriculum weights across all snapshots in the lake.
    """

    def __init__(
        self,
        min_allowed_prob_qt: float = 0.015,
        max_allowed_prob_qt: float = 0.985,
    ) -> None:
        self.min_allowed_prob_qt = min_allowed_prob_qt
        self.max_allowed_prob_qt = max_allowed_prob_qt

    @staticmethod
    def resolve_horizon_tier(horizon_hours: int) -> HorizonTierSpec:
        """Maps `horizon_hours_before_close` to its canonical `HorizonTierSpec`."""
        h = int(horizon_hours)
        for spec in HORIZON_TIERS:
            if spec.min_hours_exclusive < h <= spec.max_hours_inclusive:
                return spec
        return HORIZON_TIERS[-1]

    @staticmethod
    def compute_binary_shannon_entropy(q_t: float) -> float:
        """
        Computes binary Shannon entropy in bits (`[0.0, 1.0]`):
            H(q) = -q * log2(q) - (1 - q) * log2(1 - q)
        Maximum uncertainty (`1.0` bit) occurs at `q_t = 0.50`.
        """
        p = max(1e-6, min(1.0 - 1e-6, float(q_t)))
        entropy = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
        return round(max(0.0, min(1.0, entropy)), 5)

    @staticmethod
    def classify_uncertainty_zone(q_t: float) -> Tuple[str, str]:
        """
        Returns `(probability_decile_bucket, entropy_uncertainty_zone)` for a
        given market implied probability `q_t`.
        """
        p = max(0.0, min(1.0, float(q_t)))
        decile_idx = min(9, int(p * 10.0))
        low_b = decile_idx * 10
        high_b = (decile_idx + 1) * 10
        bucket_str = f"PROB_{low_b:02d}_{high_b:02d}_PCT"

        if 0.25 <= p <= 0.75:
            zone = "HIGH_UNCERTAINTY_CORE"
        elif 0.08 <= p < 0.25 or 0.75 < p <= 0.92:
            zone = "MODERATE_SLOPE_TAIL"
        else:
            zone = "EXTREME_TAIL_CONVERGENCE"

        return bucket_str, zone

    def compute_domain_propensity_weights(
        self, snapshots: Sequence[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Computes smoothed inverse-frequency weights per `canonical_category` so
        high-frequency categories (like daily sports) do not drown out high-alpha
        Macro/Fed, Geopolitics, and Crypto contracts during GRPO updates.
        """
        counts: Dict[str, int] = {}
        for row in snapshots:
            cat = str(row.get("canonical_category") or "OTHER")
            counts[cat] = counts.get(cat, 0) + 1

        if not counts:
            return {"OTHER": 1.0}

        total = float(len(snapshots))
        num_cats = float(len(counts))
        target_per_cat = total / max(1.0, num_cats)

        weights: Dict[str, float] = {}
        for cat, cnt in counts.items():
            # Square-root dampened inverse propensity clamped to [0.60, 1.75]
            raw_ratio = math.sqrt(target_per_cat / max(1.0, float(cnt)))
            # Boost Tier-1 institutional domains slightly
            if cat in (
                "MACRO_ECONOMICS_FED",
                "GEOPOLITICS_CONFLICT",
                "CRYPTO_DIGITAL_ASSETS",
                "US_GLOBAL_POLITICS",
            ):
                raw_ratio *= 1.15
            weights[cat] = round(max(0.60, min(1.75, raw_ratio)), 4)

        return weights

    def stratify_snapshots(
        self, snapshots: Sequence[Dict[str, Any]]
    ) -> Tuple[List[StratifiedHorizonRow], Dict[str, Any]]:
        """
        Evaluates a batch of CLOB horizon snapshots and computes their exact
        `StratifiedHorizonRow` metadata and aggregate distribution telemetry.
        """
        domain_weights = self.compute_domain_propensity_weights(snapshots)
        stratified_rows: List[StratifiedHorizonRow] = []

        tier_counts: Dict[str, int] = {}
        zone_counts: Dict[str, int] = {}
        eligible_grpo_count = 0

        for row in snapshots:
            mkt_id = str(row.get("market_id") or "")
            horizon_h = int(row.get("horizon_hours_before_close") or 24)
            q_t = float(row.get("implied_prob_mid_qt") or 0.50)
            cat = str(row.get("canonical_category") or "OTHER")

            tier_spec = self.resolve_horizon_tier(horizon_h)
            entropy_bits = self.compute_binary_shannon_entropy(q_t)
            bucket_str, zone_str = self.classify_uncertainty_zone(q_t)
            dom_w = domain_weights.get(cat, 1.0)

            # Combine horizon multiplier, Shannon entropy (0.5 + 0.8 * H), and domain weight
            entropy_factor = 0.55 + (0.75 * entropy_bits)
            curriculum_w = round(
                max(
                    0.25,
                    min(2.50, tier_spec.base_curriculum_multiplier * entropy_factor * dom_w),
                ),
                4,
            )

            # Exclude already-settled extreme tail snapshots (<1.5% or >98.5%) from RL rollouts
            include_grpo = self.min_allowed_prob_qt <= q_t <= self.max_allowed_prob_qt
            if include_grpo:
                eligible_grpo_count += 1

            tier_counts[tier_spec.tier_name] = tier_counts.get(tier_spec.tier_name, 0) + 1
            zone_counts[zone_str] = zone_counts.get(zone_str, 0) + 1

            stratified_rows.append(
                StratifiedHorizonRow(
                    market_id=mkt_id,
                    horizon_hours_before_close=horizon_h,
                    horizon_tier_name=tier_spec.tier_name,
                    implied_prob_mid_qt=q_t,
                    probability_decile_bucket=bucket_str,
                    shannon_binary_entropy_bits=entropy_bits,
                    entropy_uncertainty_zone=zone_str,
                    domain_inverse_propensity_weight=dom_w,
                    grpo_curriculum_weight=curriculum_w,
                    include_in_grpo_rl_split=include_grpo,
                )
            )

        summary = {
            "stage": "STAGE_3_PART_A_HORIZON_STRATIFIER",
            "total_snapshots_stratified": len(stratified_rows),
            "grpo_eligible_snapshots": eligible_grpo_count,
            "horizon_tier_distribution": tier_counts,
            "uncertainty_zone_distribution": zone_counts,
            "domain_propensity_weights": domain_weights,
        }
        return stratified_rows, summary


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Stage 3 Part A: Horizon Stratifier"
    )
    parser.add_argument(
        "--clob-shard",
        type=pathlib.Path,
        default=pathlib.Path(
            "data/raw_lake/polymarket_clob_v2/clob_horizon_snapshots_shard_00000.parquet"
        ),
    )
    args = parser.parse_args()

    if not args.clob_shard.exists():
        raise FileNotFoundError(f"Missing CLOB shard at {args.clob_shard}")

    import polars as pl  # type: ignore

    snapshots = pl.read_parquet(args.clob_shard).to_dicts()
    stratifier = HorizonStratifier()
    _, summary = stratifier.stratify_snapshots(snapshots)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())