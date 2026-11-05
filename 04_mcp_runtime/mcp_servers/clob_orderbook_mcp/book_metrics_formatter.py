#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: CLOB BOOK METRICS FORMATTER (MCP)
File: 04_mcp_runtime/mcp_servers/clob_orderbook_mcp/book_metrics_formatter.py
================================================================================
Formats live Polymarket CLOB v2 Level-2 order books into the exact `<book>` and
`<trajectory>` XML/Markdown schema learned by Qwen-2.5-7B during Stage 4 & 5
training (`02_dataset_factory/orderbook_serializer/l2_book_to_markdown.py`).
================================================================================
"""

from __future__ import annotations

import sys
import json
import importlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [BOOK-MCP-FMT] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.book_metrics_formatter")


@dataclass(frozen=True)
class LiveFormattedBookPayload:
    """
    Identical schema to Stage 3/4/5 `<book>` training prompts, enriched with
    live $5.00 survival VWAP and minimum edge hurdle telemetry.
    """
    market_id: str
    snapshot_uid: str
    midpoint_qt: float
    best_bid: float
    best_ask: float
    spread_st: float
    micro_price: float
    order_book_imbalance: float
    min_required_edge_to_trade: float
    survival_5usd_vwap_ask: float
    liquidity_regime_token: str
    cpt_book_xml_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveBookMetricsFormatter:
    """
    Wraps `L2BookToMarkdownSerializer` from `02_dataset_factory` so that live
    MCP tool responses match the training distribution with 0% schema drift.
    """

    def __init__(self) -> None:
        l2_mod = importlib.import_module(
            "02_dataset_factory.orderbook_serializer.l2_book_to_markdown"
        )
        self._serializer = l2_mod.L2BookToMarkdownSerializer()

    def format_live_market_state(
        self, market_row: Dict[str, Any]
    ) -> LiveFormattedBookPayload:
        serialized = self._serializer.serialize_snapshot_book(market_row)
        return LiveFormattedBookPayload(
            market_id=serialized.market_id,
            snapshot_uid=serialized.snapshot_uid,
            midpoint_qt=serialized.midpoint_qt,
            best_bid=serialized.best_bid_price,
            best_ask=serialized.best_ask_price,
            spread_st=serialized.spread_abs_st,
            micro_price=serialized.micro_price,
            order_book_imbalance=serialized.order_book_imbalance,
            min_required_edge_to_trade=serialized.min_required_edge_to_trade,
            survival_5usd_vwap_ask=serialized.survival_5usd_vwap_ask,
            liquidity_regime_token=serialized.liquidity_regime_token,
            cpt_book_xml_markdown=serialized.cpt_book_xml_markdown,
        )


def main() -> int:
    formatter = LiveBookMetricsFormatter()
    sample = formatter.format_live_market_state(
        {
            "snapshot_uid": "LIVE-MCP-253591",
            "market_id": "253591",
            "horizon_hours_before_close": 24,
            "implied_prob_mid_qt": 0.6120,
            "estimated_best_bid": 0.6060,
            "estimated_best_ask": 0.6180,
            "estimated_spread_st": 0.0120,
            "market_volume_usd": 450000.0,
            "momentum_24h_delta": 0.045,
            "collateral_era": "APRIL_2026_PUSD_V2",
        }
    )
    print(json.dumps(sample.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())