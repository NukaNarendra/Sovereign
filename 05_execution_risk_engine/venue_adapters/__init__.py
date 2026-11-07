#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: MULTI-VENUE EXECUTION ADAPTERS
File: 05_execution_risk_engine/venue_adapters/__init__.py
================================================================================
Root package initializer for the unified multi-venue execution bridges:
  1. `base_venue_interface.py`     : Abstract base class for unified order placement.
  2. `polymarket_clob_executor`    : Phase 1 & 2 Polygon EIP-712 CLOB v2 executor.
  3. `kalshi_fix_executor`         : Phase 2 US regulated event-contract bridge.
  4. `deribit_cme_hedger`          : Phase 2 BTC/ETH options delta/vol cross-hedger.
================================================================================
"""

from .base_venue_interface import (
    VenueOrderRequest,
    VenueOrderReceipt,
    BaseExecutionVenueAdapter,
)

__all__ = [
    "VenueOrderRequest",
    "VenueOrderReceipt",
    "BaseExecutionVenueAdapter",
]