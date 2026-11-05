#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART A)
File: 01_data_ingestion/polymarket_gamma/__init__.py
================================================================================
Exposes the public interfaces for crawling, validating, and categorizing
resolved and active prediction markets from `https://gamma-api.polymarket.com`.
================================================================================
"""

from .gamma_schema import (
    MarketResolutionStatus,
    CollateralEra,
    GammaTagModel,
    ClobTokenPair,
    GammaMarketRecord,
    GammaEventRecord,
    CrawlerCheckpointState,
)
from .tag_category_taxonomy import (
    CanonicalDomain,
    DomainTaxonomyClassification,
    PolymarketCategoryTaxonomyEngine,
)
from .gamma_rest_client import (
    GammaClientConfig,
    TokenBucketRateLimiter,
    GammaAsyncRestClient,
)
from .resolved_markets_crawler import (
    CrawlerConfig,
    ResolvedMarketsParquetCrawler,
)

__all__ = [
    "MarketResolutionStatus",
    "CollateralEra",
    "GammaTagModel",
    "ClobTokenPair",
    "GammaMarketRecord",
    "GammaEventRecord",
    "CrawlerCheckpointState",
    "CanonicalDomain",
    "DomainTaxonomyClassification",
    "PolymarketCategoryTaxonomyEngine",
    "GammaClientConfig",
    "TokenBucketRateLimiter",
    "GammaAsyncRestClient",
    "CrawlerConfig",
    "ResolvedMarketsParquetCrawler",
]