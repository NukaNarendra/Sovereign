#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: PRE-TRADE VWAP WALK-THE-BOOK SIMULATOR
File: 05_execution_risk_engine/slippage_simulator/vwap_walk_the_book.py
================================================================================
Simulates filling a proposed Fractional Kelly order across resting Level-2 CLOB
ask/bid levels before order submission, blocking any market order whose VWAP
price impact would erode net Expected Value below the safety hurdle.
================================================================================
"""

from __future__ import annotations

import sys
import json
import importlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [VWAP-SIMULATOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.vwap_walk_the_book")


@dataclass(frozen=True)
class PreTradeVWAPSimulation:
    """
    Pre-submission Level-2 order-book sweep simulation report.
    """
    market_id: str
    side: str
    target_notional_usd: float
    best_top_of_book_price: float
    simulated_vwap_price: float
    slippage_bps: float
    levels_consumed: int
    is_fully_fillable: bool
    net_edge_after_slippage: float
    passes_zero_slippage_erosion_gate: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VWAPWalkTheBookSimulator:
    """
    Walks the live 5-level L2 ladder to verify whether an order of size
    `target_notional_usd` eats through multiple price levels.
    """

    def __init__(self, max_allowed_taker_slippage_bps: float = 50.0) -> None:
        self.max_allowed_taker_slippage_bps = float(max_allowed_taker_slippage_bps)
        clob_mod = importlib.import_module(
            "01_data_ingestion.polymarket_clob_v2.clob_v2_rest_client"
        )
        extr_mod = importlib.import_module(
            "01_data_ingestion.polymarket_clob_v2.spread_depth_extractor"
        )
        self.OrderBookLevel = clob_mod.OrderBookLevel
        self.Extractor = extr_mod.SpreadDepthExtractor

    def simulate_order_impact(
        self,
        market_id: str,
        side: str,
        win_probability: float,
        target_notional_usd: float,
        ladder_levels: Sequence[Dict[str, float]],
    ) -> PreTradeVWAPSimulation:
        typed_levels = [
            self.OrderBookLevel(
                price=float(lvl["price"]), size=float(lvl["size"])
            )
            for lvl in ladder_levels
        ]
        walk_res = self.Extractor.simulate_walk_the_book(
            levels=typed_levels,
            target_notional_usd=float(target_notional_usd),
            side=side,
        )
        net_edge_after_slip = round(
            float(win_probability) - walk_res.vwap_execution_price, 4
        )
        gate_ok = (
            walk_res.is_fully_fillable
            and walk_res.slippage_bps <= self.max_allowed_taker_slippage_bps
            and net_edge_after_slip >= 0.025
        )
        return PreTradeVWAPSimulation(
            market_id=market_id,
            side=side,
            target_notional_usd=walk_res.target_notional_usd,
            best_top_of_book_price=walk_res.reference_best_price,
            simulated_vwap_price=walk_res.vwap_execution_price,
            slippage_bps=walk_res.slippage_bps,
            levels_consumed=walk_res.levels_consumed,
            is_fully_fillable=walk_res.is_fully_fillable,
            net_edge_after_slippage=net_edge_after_slip,
            passes_zero_slippage_erosion_gate=gate_ok,
        )


def main() -> int:
    sim = VWAPWalkTheBookSimulator()
    ladder = [
        {"price": 0.6900, "size": 250.0},
        {"price": 0.7000, "size": 600.0},
        {"price": 0.7200, "size": 1500.0},
    ]
    res_5usd = sim.simulate_order_impact(
        market_id="558938",
        side="BUY_NO",
        win_probability=0.9950,
        target_notional_usd=0.69,
        ladder_levels=ladder,
    )
    print(json.dumps(res_5usd.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())