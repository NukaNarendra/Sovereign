#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `kalshi_fix_executor`."""

from typing import Any

__all__ = [
    "KalshiOrderReceipt",
    "KalshiRestFixClient",
    "CrossVenueContractPair",
    "KalshiContractTickerMapper",
]


def __getattr__(name: str) -> Any:
    if name in ("KalshiOrderReceipt", "KalshiRestFixClient"):
        from . import kalshi_rest_fix_client
        return getattr(kalshi_rest_fix_client, name)
    if name in ("CrossVenueContractPair", "KalshiContractTickerMapper"):
        from . import contract_ticker_mapper
        return getattr(contract_ticker_mapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")