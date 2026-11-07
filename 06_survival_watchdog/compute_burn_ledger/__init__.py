#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `compute_burn_ledger`."""

from typing import Any

__all__ = [
    "BurnRateAccrualReport",
    "BurnRateAccumulator",
    "InvoiceSweepReceipt",
    "InvoiceSweeper",
]


def __getattr__(name: str) -> Any:
    if name in ("BurnRateAccrualReport", "BurnRateAccumulator"):
        from . import burn_rate_accumulator
        return getattr(burn_rate_accumulator, name)
    if name in ("InvoiceSweepReceipt", "InvoiceSweeper"):
        from . import invoice_sweeper
        return getattr(invoice_sweeper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")