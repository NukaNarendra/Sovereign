#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: MAKER VS TAKER ORDER SELECTOR
File: 05_execution_risk_engine/slippage_simulator/maker_taker_selector.py
================================================================================
Selects between:
  1. `POST_ONLY_GTC_MAKER` (Default for $5.00 survival bot): Places a resting
     limit order 1 tick inside the best ask/bid so we earn/save half the spread
     and pay 0% taker slippage.
  2. `IMMEDIATE_OR_CANCEL_TAKER` (`FAK`/`IOC`): Used only when a high-urgency
     3-sigma breaking news catalyst (`urgency_score >= 4.5`) provides massive
     net edge (`>= 8.0%`) that outweighs crossing the top-of-book ask.
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
    format="%(asctime)s | %(levelname)-8s | [MAKER-TAKER-SEL] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.maker_taker_selector")


@dataclass(frozen=True)
class OrderExecutionTypeDecision:
    """
    Execution order type (`GTC` Post-Only Maker vs `FAK` IOC Taker) and limit
    price placement decision.
    """
    market_id: str
    side: str
    order_time_in_force: str
    post_only_flag: bool
    execution_style_label: str
    target_limit_price: float
    stale_cancel_timeout_sec: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MakerTakerSelector:
    """
    Determines optimal order placement tactics on Polymarket CLOB v2 based on
    net edge, VWAP slippage, and anomaly urgency score.
    """

    def __init__(
        self,
        taker_urgency_threshold: float = 4.5,
        taker_min_net_edge: float = 0.080,
        default_stale_timeout_sec: int = 180,
    ) -> None:
        self.taker_urgency_threshold = float(taker_urgency_threshold)
        self.taker_min_net_edge = float(taker_min_net_edge)
        self.default_stale_timeout_sec = int(default_stale_timeout_sec)

    def select_execution_tactic(
        self,
        market_id: str,
        side: str,
        best_bid: float,
        best_ask: float,
        net_ev_per_share: float,
        vwap_slippage_bps: float,
        urgency_score: float = 2.0,
        tick_size: float = 0.01,
    ) -> OrderExecutionTypeDecision:
        if (
            float(urgency_score) >= self.taker_urgency_threshold
            and float(net_ev_per_share) >= self.taker_min_net_edge
            and float(vwap_slippage_bps) <= 25.0
        ):
            return OrderExecutionTypeDecision(
                market_id=market_id,
                side=side,
                order_time_in_force="FAK",
                post_only_flag=False,
                execution_style_label="IMMEDIATE_OR_CANCEL_TAKER_SWEEP",
                target_limit_price=round(float(best_ask), 4),
                stale_cancel_timeout_sec=5,
                rationale="HIGH_URGENCY_CATALYST_EXCEEDS_8PCT_NET_EDGE",
            )

        # Default survival bot tactic: rest a POST_ONLY limit order 1 tick below best_ask
        maker_price = round(
            max(float(best_bid), float(best_ask) - float(tick_size)), 4
        )
        return OrderExecutionTypeDecision(
            market_id=market_id,
            side=side,
            order_time_in_force="GTC",
            post_only_flag=True,
            execution_style_label="POST_ONLY_LIMIT_MAKER",
            target_limit_price=maker_price,
            stale_cancel_timeout_sec=self.default_stale_timeout_sec,
            rationale="SURVIVAL_SPREAD_PRESERVATION_ZERO_TAKER_SLIPPAGE",
        )


def main() -> int:
    selector = MakerTakerSelector()
    maker_dec = selector.select_execution_tactic(
        market_id="558938",
        side="BUY_NO",
        best_bid=0.6800,
        best_ask=0.6900,
        net_ev_per_share=0.0650,
        vwap_slippage_bps=0.0,
        urgency_score=2.8,
    )
    taker_dec = selector.select_execution_tactic(
        market_id="253591",
        side="BUY_YES",
        best_bid=0.6000,
        best_ask=0.6100,
        net_ev_per_share=0.1450,
        vwap_slippage_bps=5.0,
        urgency_score=5.2,
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_7_MAKER_TAKER_SELECTOR",
                "standard_anomaly_uses_post_only_maker": maker_dec.post_only_flag,
                "extreme_catalyst_uses_ioc_taker": not taker_dec.post_only_flag,
                "maker_decision": maker_dec.to_dict(),
                "taker_decision": taker_dec.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())