#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART C)
File: 01_data_ingestion/polygon_substreams/__init__.py
================================================================================
Public package interface for Polygon PoS Mainnet (Chain ID 137) RPC failover
pooling, raw `OrderFilled` log decoding, and institutional Sharp Whale wallet
profiling.
================================================================================
"""

from typing import Any

__all__ = [
    "POLYGON_EVENT_TOPICS",
    "RpcEndpointHealth",
    "PolygonRpcFallbackPool",
    "WhaleArchetype",
    "WhaleTradeObservation",
    "SharpWhaleWalletProfile",
    "WhaleWalletProfiler",
]


def __getattr__(name: str) -> Any:
    if name in ("POLYGON_EVENT_TOPICS", "RpcEndpointHealth", "PolygonRpcFallbackPool"):
        from . import rpc_fallback_pool
        return getattr(rpc_fallback_pool, name)
    if name in (
        "WhaleArchetype",
        "WhaleTradeObservation",
        "SharpWhaleWalletProfile",
        "WhaleWalletProfiler",
    ):
        from . import whale_wallet_profiler
        return getattr(whale_wallet_profiler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")