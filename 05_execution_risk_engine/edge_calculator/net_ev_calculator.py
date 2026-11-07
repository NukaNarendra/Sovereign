#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: NET EXPECTED VALUE (EV) CALCULATOR
File: 05_execution_risk_engine/edge_calculator/net_ev_calculator.py
================================================================================
Computes Expected Value (EV) per conditional share and return on capital net of:
  1. Level-2 Bid-Ask Spread & Maker/Taker Execution Price (`c_entry`)
  2. Polymarket CLOB v2 Exchange Fee (`fee_bps`)
  3. Polygon Mainnet (Chain ID 137) Gas / Relayer Amortization (`gas_drag_usd`)
  4. Virtual Inference Compute Burn (`$0.00020` per forecast)

Mathematical Formulation:
  - Given calibrated model probability $p \in (0, 1)$ and CLOB midpoint $q_t$:
    * If $p > q_t$, candidate side is `BUY_YES` with true win probability $p_{\text{win}} = p$.
      Effective maker-limit entry price is $c_{\text{YES}} = q_t + \alpha_{\text{slip}} \frac{s_t}{2}$.
    * If $p < q_t$, candidate side is `BUY_NO` with true win probability $p_{\text{win}} = 1 - p$.
      Effective maker-limit entry price is $c_{\text{NO}} = (1 - q_t) + \alpha_{\text{slip}} \frac{s_t}{2}$.
  - Net per-share Expected Value after fee rate $f_{\text{fee}}$ is:
      $$\text{EV}_{\text{share}} = p_{\text{win}} \cdot (1.0 - f_{\text{fee}}) - c_{\text{entry}}$$
  - Net Return on Invested Capital (ROIC) is:
      $$\text{ROIC}_{\text{net}} = \frac{\text{EV}_{\text{share}}}{c_{\text{entry}}}$$
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
    format="%(asctime)s | %(levelname)-8s | [NET-EV-CALC] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.net_ev_calculator")


@dataclass(frozen=True)
class NetEVEvaluation:
    """
    Deterministic Expected Value decomposition for a candidate prediction-market
    trade on Polymarket CLOB v2.
    """
    snapshot_uid: str
    market_id: str
    candidate_side: str
    model_prob_yes_p: float
    market_midpoint_yes_qt: float
    clob_spread_st: float
    raw_signed_edge: float
    raw_abs_edge: float
    win_probability_on_side: float
    effective_entry_price_usd: float
    exchange_fee_drag_per_share: float
    gas_and_compute_drag_per_share: float
    total_all_in_cost_per_share: float
    net_ev_per_share_usd: float
    net_expected_roic_pct: float
    is_positive_net_ev: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NetExpectedValueCalculator:
    """
    Computes exact per-share Expected Value and Return on Invested Capital
    (ROIC) net of spread, exchange fees, gas, and virtual LLM inference cost.
    """

    def __init__(
        self,
        maker_spread_capture_ratio: float = 0.35,
        exchange_fee_bps: float = 0.0,
        polygon_relayer_gas_usd: float = 0.0005,
        virtual_inference_burn_usd: float = 0.0002,
        reference_order_shares: float = 2.0,
    ) -> None:
        self.maker_spread_capture_ratio = float(maker_spread_capture_ratio)
        self.exchange_fee_bps = float(exchange_fee_bps)
        self.polygon_relayer_gas_usd = float(polygon_relayer_gas_usd)
        self.virtual_inference_burn_usd = float(virtual_inference_burn_usd)
        self.reference_order_shares = max(0.5, float(reference_order_shares))

    def evaluate_net_ev(
        self,
        model_prob_yes_p: float,
        market_midpoint_yes_qt: float,
        clob_spread_st: float,
        snapshot_uid: str = "SNAP-EV",
        market_id: str = "253591",
        is_post_only_maker: bool = True,
    ) -> NetEVEvaluation:
        """
        Evaluates whether buying YES or buying NO has positive net Expected Value
        after accounting for bid-ask spread, fees, and gas/compute overhead.
        """
        p_yes = max(0.0001, min(0.9999, float(model_prob_yes_p)))
        q_yes = max(0.0010, min(0.9990, float(market_midpoint_yes_qt)))
        s_t = max(0.0010, float(clob_spread_st))

        raw_signed = round(p_yes - q_yes, 6)
        raw_abs = abs(raw_signed)

        # Post-only maker orders rest inside the spread; taker orders cross the full half-spread
        slip_mult = self.maker_spread_capture_ratio if is_post_only_maker else 1.00
        half_spread_drag = (s_t / 2.0) * slip_mult

        if raw_signed >= 0.0:
            side = "BUY_YES"
            win_prob = p_yes
            entry_price = min(0.9950, q_yes + half_spread_drag)
        else:
            side = "BUY_NO"
            win_prob = 1.0 - p_yes
            entry_price = min(0.9950, (1.0 - q_yes) + half_spread_drag)

        fee_rate = max(0.0, self.exchange_fee_bps) / 10_000.0
        fee_drag = round(win_prob * fee_rate, 6)
        fixed_overhead_per_share = round(
            (self.polygon_relayer_gas_usd + self.virtual_inference_burn_usd)
            / self.reference_order_shares,
            6,
        )

        all_in_cost = round(entry_price + fee_drag + fixed_overhead_per_share, 6)
        net_ev = round(win_prob - all_in_cost, 6)
        roic_pct = round((net_ev / max(0.01, all_in_cost)) * 100.0, 2)

        return NetEVEvaluation(
            snapshot_uid=snapshot_uid,
            market_id=market_id,
            candidate_side=side,
            model_prob_yes_p=round(p_yes, 4),
            market_midpoint_yes_qt=round(q_yes, 4),
            clob_spread_st=round(s_t, 4),
            raw_signed_edge=round(raw_signed, 4),
            raw_abs_edge=round(raw_abs, 4),
            win_probability_on_side=round(win_prob, 4),
            effective_entry_price_usd=round(entry_price, 4),
            exchange_fee_drag_per_share=fee_drag,
            gas_and_compute_drag_per_share=fixed_overhead_per_share,
            total_all_in_cost_per_share=round(all_in_cost, 4),
            net_ev_per_share_usd=round(net_ev, 4),
            net_expected_roic_pct=roic_pct,
            is_positive_net_ev=(net_ev > 0.0),
        )


def main() -> int:
    calc = NetExpectedValueCalculator()
    bullish = calc.evaluate_net_ev(
        model_prob_yes_p=0.6950,
        market_midpoint_yes_qt=0.6000,
        clob_spread_st=0.0140,
        snapshot_uid="TEST-BULLISH-YES",
    )
    bearish = calc.evaluate_net_ev(
        model_prob_yes_p=0.0050,
        market_midpoint_yes_qt=0.3147,
        clob_spread_st=0.0162,
        snapshot_uid="PIT-558938-T168H",
    )
    noisy = calc.evaluate_net_ev(
        model_prob_yes_p=0.5020,
        market_midpoint_yes_qt=0.5000,
        clob_spread_st=0.0300,
        snapshot_uid="TEST-NOISY-INSIDE-SPREAD",
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_7_NET_EV_CALCULATOR",
                "bullish_yes_positive_ev": bullish.is_positive_net_ev,
                "bearish_no_positive_ev": bearish.is_positive_net_ev,
                "noisy_inside_spread_rejected": not noisy.is_positive_net_ev,
                "samples": [bullish.to_dict(), bearish.to_dict(), noisy.to_dict()],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())