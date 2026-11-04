#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: OUT-OF-TIME EVALUATION BENCHMARK SUITE
File: 03_llm_training/evaluation_bench/__init__.py
================================================================================
Public package interface for the Out-of-Time Calibration, Historical CLOB
Walk-Forward Backtesting, and Institutional Sharpe/Drawdown Reporting Suite:

  1. `expected_calibration_error.py` : Computes 10-bin Expected Calibration Error
                                       (ECE), Maximum Calibration Error (MCE),
                                       Murphy Brier Decomposition, and SVG
                                       reliability diagrams vs market midpoints.
  2. `historical_clob_backtester.py` : Replays the trained Stage 5 GRPO model on
                                       out-of-time `TEST` Polymarket contracts
                                       with 0.25x Fractional Kelly sizing.
  3. `sharpe_drawdown_report.py`     : Generates institutional tear-sheet metrics
                                       (Annualized Sharpe, Sortino, Calmar, Win
                                       Rate, Brier Skill Score, and Max Drawdown)
                                       to certify Stage 5 exit gate readiness.

Uses PEP 562 lazy loading (`__getattr__`) so `python3 -m` executions never emit
a `<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "ExpectedCalibrationErrorEngine",
    "HistoricalCLOBBacktester",
    "SharpeDrawdownReporter",
]


def __getattr__(name: str) -> Any:
    if name == "ExpectedCalibrationErrorEngine":
        from .expected_calibration_error import ExpectedCalibrationErrorEngine
        return ExpectedCalibrationErrorEngine
    if name == "HistoricalCLOBBacktester":
        from .historical_clob_backtester import HistoricalCLOBBacktester
        return HistoricalCLOBBacktester
    if name == "SharpeDrawdownReporter":
        from .sharpe_drawdown_report import SharpeDrawdownReporter
        return SharpeDrawdownReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")