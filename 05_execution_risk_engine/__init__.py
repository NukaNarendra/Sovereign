#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: DETERMINISTIC EXECUTION & RISK ENGINE
File: 05_execution_risk_engine/__init__.py
================================================================================
Root package initializer for the Deterministic Math & Order Routing Layer
(`05_execution_risk_engine/`).

Core Architectural Principle:
  The 7B LLM (`Sovereign-Prob-7B`) NEVER decides dollar bet sizes or submits
  orders directly. It outputs only a calibrated probability `p`. All Expected
  Value (EV) calculations, 0.25x Fractional Kelly sizing, correlation caps,
  VWAP slippage checks, and EIP-712 cryptographic order signing are executed
  deterministically in Python by this module.
================================================================================
"""

from typing import Final

__version__: Final[str] = "1.0.0-sovereign-stage7-exec"
__stage__: Final[str] = "STAGE_7_EXECUTION_RISK_ENGINE"

SUBMODULES: Final[tuple[str, ...]] = (
    "edge_calculator",
    "fractional_kelly_sizer",
    "slippage_simulator",
    "venue_adapters",
)

__all__ = [
    "__version__",
    "__stage__",
    "SUBMODULES",
]