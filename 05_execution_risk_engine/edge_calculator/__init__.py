#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: EDGE & EXPECTED VALUE CALCULATOR
File: 05_execution_risk_engine/edge_calculator/__init__.py
================================================================================
Lazy-loaded package initializer for `net_ev_calculator.py` and
`minimum_hurdle_gate.py`.
================================================================================
"""

from typing import Any

__all__ = [
    "NetEVEvaluation",
    "NetExpectedValueCalculator",
    "HurdleGateDecision",
    "MinimumHurdleGate",
]


def __getattr__(name: str) -> Any:
    if name in ("NetEVEvaluation", "NetExpectedValueCalculator"):
        from . import net_ev_calculator
        return getattr(net_ev_calculator, name)
    if name in ("HurdleGateDecision", "MinimumHurdleGate"):
        from . import minimum_hurdle_gate
        return getattr(minimum_hurdle_gate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")