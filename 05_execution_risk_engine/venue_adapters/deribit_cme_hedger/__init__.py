#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `deribit_cme_hedger`."""

from typing import Any

__all__ = [
    "DeribitOptionHedgeOrder",
    "DeribitOptionsClient",
    "ImpliedVolSkewSignal",
    "ImpliedVolSurfaceMapper",
]


def __getattr__(name: str) -> Any:
    if name in ("DeribitOptionHedgeOrder", "DeribitOptionsClient"):
        from . import deribit_options_client
        return getattr(deribit_options_client, name)
    if name in ("ImpliedVolSkewSignal", "ImpliedVolSurfaceMapper"):
        from . import implied_vol_surface_mapper
        return getattr(implied_vol_surface_mapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")