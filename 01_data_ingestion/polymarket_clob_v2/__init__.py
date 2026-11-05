#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART B)
File: 01_data_ingestion/polymarket_clob_v2/__init__.py
================================================================================
Public interface for Polymarket's Central Limit Order Book (CLOB v2) ingestion
layer (`https://clob.polymarket.com`).

Core Capabilities Exposed by This Subpackage:
  1. `clob_v2_rest_client`         : Async HTTP/2 client for live & historical
                                     CLOB order books (`/book`), midpoints,
                                     spreads, and 1-min `/prices-history` bars.
  2. `spread_depth_extractor`      : Quantitative microstructure analyzer that
                                     computes bid-ask spreads, 1%/5% book depth,
                                     micro-price, Order Book Imbalance (OBI),
                                     and VWAP walk-the-book slippage curves.
  3. `pusd_migration_normalizer`   : Unifies legacy bridged USDC.e fills, April
                                     2026 CLOB v2 pUSD wrapper fills, and
                                     NegRisk multi-outcome adapter conversions
                                     into a single canonical USD price basis.
  4. `marketlens_archive_loader`   : Reads Stage 2 Part A Gamma Parquet shards,
                                     backfills point-in-time historical odds &
                                     microstructure trajectories at T-30d, T-7d,
                                     T-24h, and T-1h before market resolution,
                                     and writes GRPO-ready CLOB Parquet shards.
================================================================================
"""

from .clob_v2_rest_client import (
    OrderBookLevel,
    RawClobOrderBook,
    PriceHistoryPoint,
    ClobV2ClientConfig,
    ClobV2AsyncRestClient,
)
from .spread_depth_extractor import (
    WalkTheBookResult,
    OrderBookMicrostructureMetrics,
    SpreadDepthExtractor,
)
from .pusd_migration_normalizer import (
    SettlementRegime,
    NormalizedFillRecord,
    PusdMigrationNormalizer,
)
from .marketlens_archive_loader import (
    HorizonSnapshotConfig,
    MarketLensArchiveLoader,
)

__all__ = [
    "OrderBookLevel",
    "RawClobOrderBook",
    "PriceHistoryPoint",
    "ClobV2ClientConfig",
    "ClobV2AsyncRestClient",
    "WalkTheBookResult",
    "OrderBookMicrostructureMetrics",
    "SpreadDepthExtractor",
    "SettlementRegime",
    "NormalizedFillRecord",
    "PusdMigrationNormalizer",
    "HorizonSnapshotConfig",
    "MarketLensArchiveLoader",
]