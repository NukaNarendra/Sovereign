#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: MODEL CONTEXT PROTOCOL (MCP) SERVERS
File: 04_mcp_runtime/mcp_servers/__init__.py
================================================================================
Root package initializer for the three deterministic Model Context Protocol
(MCP) tool servers exposed to Sovereign-Prob-7B at runtime:
  1. `clob_orderbook_mcp`     : Live Level-2 bid-ask spread, liquidity depth,
                                VWAP slippage, and time-to-expiry tools.
  2. `tavily_live_search_mcp` : Real-time breaking news verification & Tier-1
                                wire credibility ranking ($120 Tavily plan).
  3. `uma_rule_checker_mcp`   : UMA Optimistic Oracle v3 ancillary resolution
                                rule parser and ambiguity trap scanner.
================================================================================
"""

from typing import Final

MCP_SERVER_MODULES: Final[tuple[str, ...]] = (
    "clob_orderbook_mcp",
    "tavily_live_search_mcp",
    "uma_rule_checker_mcp",
)

__all__ = ["MCP_SERVER_MODULES"]