#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: 0.25x FRACTIONAL KELLY POSITION SIZER
File: 05_execution_risk_engine/fractional_kelly_sizer/__init__.py
================================================================================
Lazy-loaded package initializer for `kelly_criterion.py`,
`portfolio_correlation_cap.py`, and `micro_bankroll_rounder.py`.
================================================================================
"""

from typing import Any

__all__ = [
    "KellySizingResult",
    "FractionalKellyCriterion",
    "CorrelationCapResult",
    "PortfolioCorrelationCap",
    "RoundedMicroOrderSpec",
    "MicroBankrollRounder",
]


def __getattr__(name: str) -> Any:
    if name in ("KellySizingResult", "FractionalKellyCriterion"):
        from . import kelly_criterion
        return getattr(kelly_criterion, name)
    if name in ("CorrelationCapResult", "PortfolioCorrelationCap"):
        from . import portfolio_correlation_cap
        return getattr(portfolio_correlation_cap, name)
    if name in ("RoundedMicroOrderSpec", "MicroBankrollRounder"):
        from . import micro_bankroll_rounder
        return getattr(micro_bankroll_rounder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")