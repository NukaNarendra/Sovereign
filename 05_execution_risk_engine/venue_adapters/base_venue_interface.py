#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: BASE VENUE EXECUTION INTERFACE
File: 05_execution_risk_engine/venue_adapters/base_venue_interface.py
================================================================================
Abstract Base Class (`BaseExecutionVenueAdapter`) and canonical data contracts
for placing, tracking, and canceling orders across Polymarket CLOB v2 (Polygon),
Kalshi FIX/REST (US Regulated Event Contracts), and Deribit Options.
================================================================================
"""

from __future__ import annotations

import abc
import sys
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VenueOrderRequest:
    """Unified venue-agnostic order request produced by the Risk Engine."""
    client_order_id: str
    venue_name: str
    market_id: str
    instrument_token_id: str
    side: str
    limit_price: float
    quantity_shares: float
    notional_usd: float
    time_in_force: str
    post_only: bool
    neg_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VenueOrderReceipt:
    """Unified venue-agnostic execution receipt returned after order placement."""
    venue_order_id: str
    client_order_id: str
    venue_name: str
    market_id: str
    status: str
    side: str
    limit_price: float
    filled_shares: float
    remaining_shares: float
    notional_usd: float
    tx_or_signature_hash: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseExecutionVenueAdapter(abc.ABC):
    """
    Abstract base interface implemented by `PolymarketCLOBExecutor`,
    `KalshiRestFixClient`, and `DeribitOptionsClient`.
    """

    @property
    @abc.abstractmethod
    def venue_name(self) -> str:
        """Returns the canonical venue identifier."""

    @abc.abstractmethod
    async def place_limit_order(
        self, request: VenueOrderRequest
    ) -> VenueOrderReceipt:
        """Signs and submits a limit or post-only order to the venue."""

    @abc.abstractmethod
    async def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        """Cancels an open resting order by its venue order ID."""

    @abc.abstractmethod
    async def cancel_all_open_orders(self) -> List[str]:
        """Emergency Reaper Watchdog hook: cancels all resting orders immediately."""