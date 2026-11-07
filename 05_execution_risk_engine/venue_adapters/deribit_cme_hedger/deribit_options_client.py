#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: DERIBIT OPTIONS HEDGER CLIENT
File: 05_execution_risk_engine/venue_adapters/deribit_cme_hedger/deribit_options_client.py
================================================================================
Constructs vertical call/put spread (digital-equivalent) and delta hedges on
Deribit (`https://www.deribit.com/api/v2`) when `Sovereign-Prob-7B` detects a
mispricing between Polymarket crypto/macro binary odds and options implied
probabilities.
================================================================================
"""

from __future__ import annotations

import sys
import json
import hashlib
import logging
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [DERIBIT-HEDGER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.deribit_options_client")


@dataclass(frozen=True)
class DeribitOptionHedgeOrder:
    """Digital-equivalent vertical call/put spread hedge order on Deribit."""
    hedge_id: str
    underlying_asset: str
    long_strike_instrument: str
    short_strike_instrument: str
    polymarket_calibrated_prob_p: float
    deribit_nd2_implied_prob: float
    probability_divergence_edge: float
    hedge_direction: str
    contracts_size: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeribitOptionsClient:
    """
    Constructs vertical option spread hedges replicating binary digital payoffs
    when Polymarket and Deribit Black-Scholes $N(d_2)$ probabilities diverge.
    """

    def build_digital_spread_hedge(
        self,
        underlying: str,
        strike_usd: int,
        expiry_code: str,
        polymarket_prob_p: float,
        deribit_nd2_prob: float,
        notional_contracts: float = 0.10,
    ) -> DeribitOptionHedgeOrder:
        edge = round(float(polymarket_prob_p) - float(deribit_nd2_prob), 4)
        step = 1000 if underlying.upper() == "BTC" else 100
        long_inst = f"{underlying.upper()}-{expiry_code}-{strike_usd}-C"
        short_inst = f"{underlying.upper()}-{expiry_code}-{strike_usd + step}-C"
        direction = (
            "BUY_DERIBIT_CALL_SPREAD_SELL_POLYMARKET_YES"
            if edge < 0.0
            else "SELL_DERIBIT_CALL_SPREAD_BUY_POLYMARKET_YES"
        )
        hid = hashlib.sha256(f"{long_inst}:{edge}".encode("utf-8")).hexdigest()[:12]

        return DeribitOptionHedgeOrder(
            hedge_id=f"DRB-{hid.upper()}",
            underlying_asset=underlying.upper(),
            long_strike_instrument=long_inst,
            short_strike_instrument=short_inst,
            polymarket_calibrated_prob_p=round(float(polymarket_prob_p), 4),
            deribit_nd2_implied_prob=round(float(deribit_nd2_prob), 4),
            probability_divergence_edge=edge,
            hedge_direction=direction,
            contracts_size=round(float(notional_contracts), 4),
            timestamp_utc=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        )


def main() -> int:
    client = DeribitOptionsClient()
    order = client.build_digital_spread_hedge(
        underlying="BTC",
        strike_usd=100000,
        expiry_code="26JUN26",
        polymarket_prob_p=0.6450,
        deribit_nd2_prob=0.5620,
        notional_contracts=0.10,
    )
    print(json.dumps(order.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())