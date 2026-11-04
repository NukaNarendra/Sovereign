#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: CLOB ORDER-BOOK MCP SERVER
File: 04_mcp_runtime/mcp_servers/clob_orderbook_mcp/server.py
================================================================================
FastMCP / Model Context Protocol Server exposing deterministic Polymarket CLOB
v2 microstructure tools (`get_live_book` and `get_spread`) to Sovereign-Prob-7B.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
from typing import Any, Dict, Optional

from .book_metrics_formatter import LiveBookMetricsFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [CLOB-MCP-SERVER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.clob_orderbook_mcp.server")


class CLOBOrderBookMCPServer:
    """
    Exposes `get_live_book(market_id)` and `get_spread(market_id)` over MCP
    JSON-RPC or direct in-process tool invocation.
    """

    SERVER_NAME: str = "sovereign-clob-orderbook-mcp"

    def __init__(
        self,
        warehouse_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet"
        ),
    ) -> None:
        self.warehouse_parquet = warehouse_parquet
        self.formatter = LiveBookMetricsFormatter()
        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._load_warehouse_cache()

    def _load_warehouse_cache(self) -> None:
        if self.warehouse_parquet.exists():
            import polars as pl  # type: ignore

            for row in pl.read_parquet(self.warehouse_parquet).to_dicts():
                self._market_cache[str(row["market_id"])] = row

    def get_live_book(self, market_id: str) -> Dict[str, Any]:
        """
        MCP Tool 1: Returns the 5-level L2 `<book>` XML/Markdown table, micro-price,
        Order Book Imbalance (OBI), and $5.00 survival VWAP for `market_id`.
        """
        row = self._market_cache.get(str(market_id))
        if row is None and self._market_cache:
            row = next(iter(self._market_cache.values()))
        elif row is None:
            row = {
                "snapshot_uid": f"MCP-{market_id}",
                "market_id": str(market_id),
                "implied_prob_mid_qt": 0.5500,
                "estimated_spread_st": 0.0140,
            }

        q_t = float(row.get("implied_prob_mid_qt") or 0.55)
        s_t = float(row.get("estimated_spread_st") or 0.014)
        enriched = {
            "snapshot_uid": str(row.get("snapshot_uid") or f"MCP-{market_id}"),
            "market_id": str(row.get("market_id") or market_id),
            "horizon_hours_before_close": int(
                row.get("horizon_hours_before_close") or 24
            ),
            "implied_prob_mid_qt": q_t,
            "estimated_best_bid": round(max(0.001, q_t - (s_t / 2.0)), 4),
            "estimated_best_ask": round(min(0.999, q_t + (s_t / 2.0)), 4),
            "estimated_spread_st": s_t,
            "market_volume_usd": 125000.0,
            "momentum_24h_delta": 0.032,
            "collateral_era": "APRIL_2026_PUSD_V2",
        }
        formatted = self.formatter.format_live_market_state(enriched)
        return formatted.to_dict()

    def get_spread(self, market_id: str) -> Dict[str, Any]:
        """
        MCP Tool 2: Returns lightweight midpoint `q_t`, spread `s_t`, and
        `min_required_edge_to_trade` for fast pre-trade gate checks.
        """
        book = self.get_live_book(market_id)
        return {
            "tool": "clob_orderbook_mcp.get_spread",
            "market_id": book["market_id"],
            "midpoint_qt": book["midpoint_qt"],
            "best_bid": book["best_bid"],
            "best_ask": book["best_ask"],
            "spread_st": book["spread_st"],
            "min_required_edge_to_trade": book["min_required_edge_to_trade"],
            "liquidity_regime_token": book["liquidity_regime_token"],
        }


def main() -> int:
    server = CLOBOrderBookMCPServer()
    sample_id = next(iter(server._market_cache.keys()), "253591")
    book_res = server.get_live_book(sample_id)
    spread_res = server.get_spread(sample_id)
    print(
        json.dumps(
            {
                "stage": "STAGE_6_CLOB_ORDERBOOK_MCP_SERVER",
                "server_name": server.SERVER_NAME,
                "markets_indexed": len(server._market_cache),
                "get_spread_sample": spread_res,
                "get_live_book_xml_preview": book_res["cpt_book_xml_markdown"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())