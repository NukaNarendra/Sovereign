#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: PORTFOLIO CORRELATION CAP
File: 05_execution_risk_engine/fractional_kelly_sizer/portfolio_correlation_cap.py
================================================================================
Caps cumulative bankroll exposure across correlated market clusters
(`correlation_cluster_key`, e.g., `CLUSTER_MACRO_FED_RATES` or `CLUSTER_CRYPTO_BTC`)
at `30.0%` (`0.30`) of deployable NAV, and total portfolio exposure across all
open positions at `70.0%` (`0.70`).
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [CORRELATION-CAP] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.portfolio_correlation_cap")


@dataclass(frozen=True)
class CorrelationCapResult:
    """
    Result of checking and clamping a proposed Kelly stake against existing
    cluster and portfolio exposure limits.
    """
    market_id: str
    correlation_cluster_key: str
    deployable_bankroll_usd: float
    proposed_stake_usd: float
    existing_cluster_exposure_usd: float
    max_cluster_allowance_usd: float
    approved_clamped_stake_usd: float
    was_clamped_by_cluster_cap: bool
    cluster_headroom_remaining_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioCorrelationCap:
    """
    Tracks active USD exposure per `correlation_cluster_key` and prevents
    correlated macro/election/crypto bets from exceeding 30% of bankroll.
    """

    def __init__(
        self,
        max_cluster_fraction: float = 0.30,
        max_total_portfolio_fraction: float = 0.70,
    ) -> None:
        self.max_cluster_fraction = float(max_cluster_fraction)
        self.max_total_portfolio_fraction = float(max_total_portfolio_fraction)
        self._cluster_exposure_usd: Dict[str, float] = {}
        self._market_exposure_usd: Dict[str, float] = {}

    def record_filled_exposure(
        self, market_id: str, cluster_key: str, stake_usd: float
    ) -> None:
        self._market_exposure_usd[market_id] = round(
            self._market_exposure_usd.get(market_id, 0.0) + float(stake_usd), 4
        )
        self._cluster_exposure_usd[cluster_key] = round(
            self._cluster_exposure_usd.get(cluster_key, 0.0) + float(stake_usd), 4
        )

    def apply_correlation_cap(
        self,
        market_id: str,
        correlation_cluster_key: str,
        proposed_stake_usd: float,
        deployable_bankroll_usd: float,
    ) -> CorrelationCapResult:
        # Block duplicate bet on the exact same market_id if already open
        if self._market_exposure_usd.get(market_id, 0.0) > 0.0:
            return CorrelationCapResult(
                market_id=market_id,
                correlation_cluster_key=correlation_cluster_key,
                deployable_bankroll_usd=deployable_bankroll_usd,
                proposed_stake_usd=proposed_stake_usd,
                existing_cluster_exposure_usd=self._cluster_exposure_usd.get(
                    correlation_cluster_key, 0.0
                ),
                max_cluster_allowance_usd=round(
                    deployable_bankroll_usd * self.max_cluster_fraction, 4
                ),
                approved_clamped_stake_usd=0.0,
                was_clamped_by_cluster_cap=True,
                cluster_headroom_remaining_usd=0.0,
            )

        max_cluster_usd = round(
            deployable_bankroll_usd * self.max_cluster_fraction, 4
        )
        max_total_usd = round(
            deployable_bankroll_usd * self.max_total_portfolio_fraction, 4
        )

        cur_cluster_usd = self._cluster_exposure_usd.get(
            correlation_cluster_key, 0.0
        )
        cur_total_usd = sum(self._cluster_exposure_usd.values())

        avail_cluster = max(0.0, max_cluster_usd - cur_cluster_usd)
        avail_total = max(0.0, max_total_usd - cur_total_usd)
        allowed_usd = round(
            min(float(proposed_stake_usd), avail_cluster, avail_total), 4
        )

        was_clamped = allowed_usd < (float(proposed_stake_usd) - 1e-4)
        rem_headroom = round(max(0.0, avail_cluster - allowed_usd), 4)

        return CorrelationCapResult(
            market_id=market_id,
            correlation_cluster_key=correlation_cluster_key,
            deployable_bankroll_usd=deployable_bankroll_usd,
            proposed_stake_usd=round(float(proposed_stake_usd), 4),
            existing_cluster_exposure_usd=round(cur_cluster_usd, 4),
            max_cluster_allowance_usd=max_cluster_usd,
            approved_clamped_stake_usd=allowed_usd,
            was_clamped_by_cluster_cap=was_clamped,
            cluster_headroom_remaining_usd=rem_headroom,
        )


def main() -> int:
    cap = PortfolioCorrelationCap(max_cluster_fraction=0.30)
    # 1st trade in CLUSTER_MACRO_FED_RATES ($0.6975 on $4.65 deployable bankroll)
    r1 = cap.apply_correlation_cap("253591", "CLUSTER_MACRO_FED_RATES", 0.6975, 4.65)
    cap.record_filled_exposure("253591", "CLUSTER_MACRO_FED_RATES", r1.approved_clamped_stake_usd)

    # 2nd trade in same cluster ($0.6975) -> fits within $1.395 (30% of $4.65)
    r2 = cap.apply_correlation_cap("253592", "CLUSTER_MACRO_FED_RATES", 0.6975, 4.65)
    cap.record_filled_exposure("253592", "CLUSTER_MACRO_FED_RATES", r2.approved_clamped_stake_usd)

    # 3rd trade in same cluster ($0.6975) -> MUST BE CLAMPED TO $0.00!
    r3 = cap.apply_correlation_cap("253593", "CLUSTER_MACRO_FED_RATES", 0.6975, 4.65)
    print(
        json.dumps(
            {
                "stage": "STAGE_7_PORTFOLIO_CORRELATION_CAP",
                "first_trade_approved_usd": r1.approved_clamped_stake_usd,
                "second_trade_approved_usd": r2.approved_clamped_stake_usd,
                "third_correlated_trade_clamped_to_zero": r3.approved_clamped_stake_usd == 0.0,
                "third_trade_detail": r3.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())