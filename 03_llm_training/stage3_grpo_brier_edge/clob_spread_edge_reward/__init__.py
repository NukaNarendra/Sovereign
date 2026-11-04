#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: CLOB SPREAD-ADJUSTED PNL EDGE REWARD
File: 03_llm_training/stage3_grpo_brier_edge/clob_spread_edge_reward/__init__.py
================================================================================
Lazy-loaded package initializer for the Polymarket Central Limit Order Book
(CLOB v2) Spread-Adjusted PnL & UMA Trap Avoidance Reward Function.
================================================================================
"""

from typing import Any

__all__ = [
    "CLOBSpreadEdgeEvaluation",
    "CLOBEdgeBatchAuditReport",
    "CLOBSpreadAdjustedPnLRewardFunction",
    "OverconfidencePenaltyEvaluation",
    "OverconfidenceBatchAuditReport",
    "OverconfidencePenaltyEngine",
]


def __getattr__(name: str) -> Any:
    if name in (
        "CLOBSpreadEdgeEvaluation",
        "CLOBEdgeBatchAuditReport",
        "CLOBSpreadAdjustedPnLRewardFunction",
    ):
        from . import spread_adjusted_pnl_fn
        return getattr(spread_adjusted_pnl_fn, name)
    if name in (
        "OverconfidencePenaltyEvaluation",
        "OverconfidenceBatchAuditReport",
        "OverconfidencePenaltyEngine",
    ):
        from . import overconfidence_penalty
        return getattr(overconfidence_penalty, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")