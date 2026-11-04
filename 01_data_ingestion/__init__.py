#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION & COLLECTION LAYER
File: 01_data_ingestion/__init__.py
================================================================================
Root package initializer for the raw Polymarket & World-State data collectors:
  - polymarket_gamma      : Fetches 100,000+ resolved & active market metadata
  - polymarket_clob_v2    : Pulls 1-min L2 order book depth, spreads, & pUSD fills
  - uma_oracle_rules      : Extracts ancillary resolution fine print & disputes
  - polygon_substreams    : On-chain Chain ID 137 smart-contract event listeners
  - news_temporal_tavily  : Tavily API connectors ($120 free tier) for news wires
================================================================================
"""

from typing import Final

__version__: Final[str] = "1.0.0-sovereign-stage2"
__stage__: Final[str] = "STAGE_2_DATA_INGESTION"
__author__: Final[str] = "Sovereign-Prob-7B Quantitative Architecture"

SUBMODULES: Final[tuple[str, ...]] = (
    "polymarket_gamma",
    "polymarket_clob_v2",
    "uma_oracle_rules",
    "polygon_substreams",
    "news_temporal_tavily",
)

__all__ = [
    "__version__",
    "__stage__",
    "__author__",
    "SUBMODULES",
]