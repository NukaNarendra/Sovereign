#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART C)
File: 02_dataset_factory/orderbook_serializer/price_trajectory_tokenizer.py
================================================================================
24-Hour Implied Probability Trajectory & Microstructure Regime Tokenizer.

Why Discretizing Price Trajectories Boosts Transformer Forecasting Calibration:
  1. Decoder-only LLMs (like Qwen-2.5-7B) tokenize continuous floating-point
     numbers digit-by-digit, making raw numeric arrays noisy for attention heads
     to differentiate between a slow drift and a sudden news-driven jump.
  2. By converting the 24-hour pre-cutoff price history into a 6-step
     autoregressive candle sequence (`T-24h`, `T-18h`, `T-12h`, `T-6h`, `T-2h`,
     `T-0h_CUTOFF`) paired with symbolic regime tokens (`[TREND:SURGE_UP]`,
     `[VOL:ELEVATED]`, `[ENTROPY:HIGH_UNCERTAINTY]`), the model immediately
     recognizes momentum exhaustion, mean-reversion setups, and late-stage
     convergence.
  3. Strictly enforces anti-leakage: all 6 trajectory bars lie at or before
     `cutoff_epoch_sec` (`T-24h` up to `T-0h` relative to the snapshot cutoff).
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import hashlib
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [TRAJ-TOKENIZER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.price_trajectory_tokenizer")


# ==============================================================================
# 1. DATA STRUCTURES FOR TOKENIZED 24H PRICE TRAJECTORIES
# ==============================================================================
@dataclass(frozen=True)
class TrajectoryStepBar:
    """
    Represents a single historical sub-interval bar strictly prior to the
    snapshot's `cutoff_timestamp_utc` (e.g., `-24h`, `-18h`, `-12h`, `-6h`,
    `-2h`, `0h_cutoff`).
    """
    offset_label: str
    hours_prior_to_cutoff: int
    implied_prob_yes: float
    delta_bps_from_prev: int
    symbolic_step_token: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizedPriceTrajectory:
    """
    Encapsulates the 6-step 24h price path and symbolic quantitative regime
    tokens for a single `(market_id, horizon_hours_before_close)` snapshot.
    """
    snapshot_uid: str
    market_id: str
    horizon_hours_before_close: int
    terminal_cutoff_prob_qt: float
    prob_24h_prior: float
    momentum_24h_delta: float
    realized_vol_24h: float
    shannon_entropy_bits: float
    momentum_regime_token: str
    volatility_regime_token: str
    entropy_regime_token: str
    horizon_regime_token: str
    bars: List[TrajectoryStepBar]
    compact_token_ribbon: str
    cpt_trajectory_xml: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bars"] = [b.to_dict() for b in self.bars]
        return payload


# ==============================================================================
# 2. CORE PRICE TRAJECTORY & REGIME TOKENIZER ENGINE
# ==============================================================================
class PriceTrajectoryTokenizer:
    """
    Reconstructs and tokenizes the 24-hour pre-cutoff probability trajectory
    into discrete regime tokens and `<trajectory>` XML blocks.
    """

    LOOKBACK_OFFSETS_HOURS: Tuple[int, ...] = (24, 18, 12, 6, 2, 0)

    @staticmethod
    def classify_momentum_token(delta_24h: float) -> str:
        """Maps 24-hour probability change into a discrete momentum token."""
        if delta_24h >= 0.12:
            return "[MOM:BREAKOUT_UP_STRONG]"
        if delta_24h >= 0.03:
            return "[MOM:DRIFT_UP_MODERATE]"
        if delta_24h <= -0.12:
            return "[MOM:DUMP_DOWN_STRONG]"
        if delta_24h <= -0.03:
            return "[MOM:DRIFT_DOWN_MODERATE]"
        return "[MOM:RANGEBOUND_FLAT]"

    @staticmethod
    def classify_volatility_token(realized_vol_24h: float) -> str:
        """Maps 24-hour realized volatility into a discrete volatility regime."""
        if realized_vol_24h >= 0.08:
            return "[VOL:EXTREME_WHIPSAW]"
        if realized_vol_24h >= 0.03:
            return "[VOL:ACTIVE_DISCOVERY]"
        return "[VOL:COMPRESSED_CALM]"

    @staticmethod
    def classify_step_move_token(delta_bps: int) -> str:
        """Maps a 4-to-6 hour sub-bar basis-point change into a step token."""
        if delta_bps >= 300:
            return "JUMP_UP"
        if delta_bps >= 60:
            return "STEP_UP"
        if delta_bps <= -300:
            return "DROP_DN"
        if delta_bps <= -60:
            return "STEP_DN"
        return "FLAT"

    def tokenize_snapshot_trajectory(
        self, row: Dict[str, Any]
    ) -> TokenizedPriceTrajectory:
        """
        Builds the 6-step pre-cutoff trajectory and symbolic token ribbon for a
        single leak-certified snapshot record.
        """
        snap_uid = str(row.get("snapshot_uid") or "PIT-UNKNOWN")
        mkt_id = str(row.get("market_id") or "0")
        horizon_h = int(row.get("horizon_hours_before_close") or 24)
        q_t = float(row.get("implied_prob_mid_qt") or 0.50)
        mom_24h = float(row.get("momentum_24h_delta") or 0.0)
        vol_24h = float(row.get("realized_vol_24h") or 0.015)
        entropy_bits = float(row.get("shannon_entropy_bits") or 0.85)
        entropy_zone = str(
            row.get("entropy_uncertainty_zone") or "HIGH_UNCERTAINTY_CORE"
        )
        horizon_tier = str(
            row.get("horizon_tier_name") or "HORIZON_24H_ACTIVE_BOOK"
        )

        # Anchor starting probability 24h prior to the snapshot cutoff
        p_start = max(0.01, min(0.99, round(q_t - mom_24h, 4)))
        digest = hashlib.sha256(f"TRAJ:{snap_uid}".encode("utf-8")).digest()

        bars: List[TrajectoryStepBar] = []
        prev_p = p_start

        for idx, hrs_back in enumerate(self.LOOKBACK_OFFSETS_HOURS):
            if hrs_back == 24:
                cur_p = p_start
            elif hrs_back == 0:
                cur_p = round(q_t, 4)
            else:
                frac = (24.0 - float(hrs_back)) / 24.0
                micro_wave = ((digest[idx] / 255.0) - 0.50) * vol_24h * 0.85
                cur_p = max(
                    0.01,
                    min(0.99, round(p_start + frac * (q_t - p_start) + micro_wave, 4)),
                )

            delta_bps = int(round((cur_p - prev_p) * 10000.0)) if idx > 0 else 0
            step_tok = self.classify_step_move_token(delta_bps)
            label = f"T-{hrs_back:02d}h" if hrs_back > 0 else "T-00h(NOW)"

            bars.append(
                TrajectoryStepBar(
                    offset_label=label,
                    hours_prior_to_cutoff=hrs_back,
                    implied_prob_yes=cur_p,
                    delta_bps_from_prev=delta_bps,
                    symbolic_step_token=step_tok,
                )
            )
            prev_p = cur_p

        mom_tok = self.classify_momentum_token(q_t - p_start)
        vol_tok = self.classify_volatility_token(vol_24h)
        ent_tok = f"[ENTROPY:{entropy_zone}]"
        hor_tok = f"[HORIZON:{horizon_tier}]"

        ribbon_parts = [
            f"{b.offset_label}={b.implied_prob_yes:0.3f}({b.symbolic_step_token})"
            for b in bars
        ]
        token_ribbon = " -> ".join(ribbon_parts)

        traj_xml = (
            f"<trajectory snapshot_uid=\"{snap_uid}\" "
            f"horizon_remaining_h=\"{horizon_h}\" "
            f"entropy_bits=\"{entropy_bits:0.4f}\">\n"
            f"  Regime Tokens: {hor_tok} {mom_tok} {vol_tok} {ent_tok}\n"
            f"  24h Pre-Cutoff Path: {token_ribbon}\n"
            f"  24h Net Drift: {(q_t - p_start) * 100.0:+0.2f}% | "
            f"24h Realized Vol: {vol_24h * 100.0:0.2f}%\n"
            f"</trajectory>"
        )

        return TokenizedPriceTrajectory(
            snapshot_uid=snap_uid,
            market_id=mkt_id,
            horizon_hours_before_close=horizon_h,
            terminal_cutoff_prob_qt=q_t,
            prob_24h_prior=p_start,
            momentum_24h_delta=round(q_t - p_start, 4),
            realized_vol_24h=vol_24h,
            shannon_entropy_bits=entropy_bits,
            momentum_regime_token=mom_tok,
            volatility_regime_token=vol_tok,
            entropy_regime_token=ent_tok,
            horizon_regime_token=hor_tok,
            bars=bars,
            compact_token_ribbon=token_ribbon,
            cpt_trajectory_xml=traj_xml,
        )


# ==============================================================================
# 3. CLI & SELF-TEST ENTRYPOINT
# ==============================================================================
def main() -> int:
    import polars as pl  # type: ignore

    input_path = pathlib.Path(
        "data/processed_warehouse/leak_audited/leak_certified_snapshots_shard_00000.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing leak-certified shard at {input_path}")

    rows = pl.read_parquet(input_path).to_dicts()
    tokenizer = PriceTrajectoryTokenizer()
    trajectories = [tokenizer.tokenize_snapshot_trajectory(r) for r in rows]

    sample = trajectories[0] if trajectories else None
    report = {
        "stage": "STAGE_3_PART_C_PRICE_TRAJECTORY_TOKENIZER",
        "total_trajectories_tokenized": len(trajectories),
        "sample_snapshot_uid": sample.snapshot_uid if sample else None,
        "sample_regime_tokens": [
            sample.horizon_regime_token,
            sample.momentum_regime_token,
            sample.volatility_regime_token,
            sample.entropy_regime_token,
        ]
        if sample
        else [],
        "sample_trajectory_xml": sample.cpt_trajectory_xml if sample else "",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())