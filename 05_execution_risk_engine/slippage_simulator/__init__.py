#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: L2 ORDER-BOOK SLIPPAGE SIMULATOR
File: 05_execution_risk_engine/slippage_simulator/__init__.py
================================================================================
Lazy-loaded package initializer for `vwap_walk_the_book.py` and
`maker_taker_selector.py`.
================================================================================
"""

from typing import Any

__all__ = [
    "PreTradeVWAPSimulation",
    "VWAPWalkTheBookSimulator",
    "OrderExecutionTypeDecision",
    "MakerTakerSelector",
]


def __getattr__(name: str) -> Any:
    if name in ("PreTradeVWAPSimulation", "VWAPWalkTheBookSimulator"):
        from . import vwap_walk_the_book
        return getattr(vwap_walk_the_book, name)
    if name in ("OrderExecutionTypeDecision", "MakerTakerSelector"):
        from . import maker_taker_selector
        return getattr(maker_taker_selector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")