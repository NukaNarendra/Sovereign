#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: IMPLIED VOLATILITY SURFACE MAPPER
File: 05_execution_risk_engine/venue_adapters/deribit_cme_hedger/implied_vol_surface_mapper.py
================================================================================
Converts calibrated binary event probabilities $p$ from `Sovereign-Prob-7B`
into Black-Scholes risk-neutral digital probabilities $N(d_2)$ and implied
volatility skew signals:
    $$d_2 = \frac{\ln(S_0 / K) + \left(r - \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad P_{\text{RN}}(S_T \ge K) = \Phi(d_2)$$
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [IV-SURFACE-MAPPER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.implied_vol_surface_mapper")


@dataclass(frozen=True)
class ImpliedVolSkewSignal:
    """
    Maps a binary strike probability to Black-Scholes $N(d_2)$ and detects
    cross-market volatility skew arbitrage.
    """
    underlying_asset: str
    spot_price_usd: float
    strike_price_usd: float
    time_to_expiry_years: float
    options_implied_vol_annualized: float
    black_scholes_d2: float
    options_nd2_digital_prob: float
    sovereign_7b_calibrated_prob: float
    cross_venue_prob_divergence: float
    actionable_vol_arb_signal: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImpliedVolSurfaceMapper:
    r"""
    Computes exact Black-Scholes $d_2$ and $\Phi(d_2)$ risk-neutral digital
    probabilities and compares them against `Sovereign-Prob-7B` forecasts.
    """

    @staticmethod
    def standard_normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

    def evaluate_strike_divergence(
        self,
        underlying: str,
        spot_price: float,
        strike_price: float,
        days_to_expiry: float,
        implied_vol: float,
        sovereign_7b_prob: float,
        risk_free_rate: float = 0.045,
    ) -> ImpliedVolSkewSignal:
        t_yrs = max(1.0 / 365.0, float(days_to_expiry) / 365.0)
        sigma = max(0.05, float(implied_vol))
        d2 = (
            math.log(float(spot_price) / float(strike_price))
            + (float(risk_free_rate) - 0.5 * (sigma ** 2)) * t_yrs
        ) / (sigma * math.sqrt(t_yrs))

        nd2_prob = round(self.standard_normal_cdf(d2), 4)
        p_llm = round(float(sovereign_7b_prob), 4)
        div = round(p_llm - nd2_prob, 4)

        return ImpliedVolSkewSignal(
            underlying_asset=underlying.upper(),
            spot_price_usd=round(float(spot_price), 2),
            strike_price_usd=round(float(strike_price), 2),
            time_to_expiry_years=round(t_yrs, 4),
            options_implied_vol_annualized=round(sigma, 4),
            black_scholes_d2=round(d2, 4),
            options_nd2_digital_prob=nd2_prob,
            sovereign_7b_calibrated_prob=p_llm,
            cross_venue_prob_divergence=div,
            actionable_vol_arb_signal=(abs(div) >= 0.050),
        )


def main() -> int:
    mapper = ImpliedVolSurfaceMapper()
    sig = mapper.evaluate_strike_divergence(
        underlying="BTC",
        spot_price=96500.0,
        strike_price=100000.0,
        days_to_expiry=30.0,
        implied_vol=0.58,
        sovereign_7b_prob=0.5420,
    )
    print(json.dumps(sig.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())