#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `polygon_wallet_auditor`."""

from typing import Any

__all__ = [
    "OnChainWalletBalanceSnapshot",
    "OnChainBalancePoller",
    "OpenPositionValuation",
    "MarkToMarketPortfolioValuation",
    "MarkToMarketValuator",
]


def __getattr__(name: str) -> Any:
    if name in ("OnChainWalletBalanceSnapshot", "OnChainBalancePoller"):
        from . import onchain_balance_poller
        return getattr(onchain_balance_poller, name)
    if name in (
        "OpenPositionValuation",
        "MarkToMarketPortfolioValuation",
        "MarkToMarketValuator",
    ):
        from . import mark_to_market_valuator
        return getattr(mark_to_market_valuator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")