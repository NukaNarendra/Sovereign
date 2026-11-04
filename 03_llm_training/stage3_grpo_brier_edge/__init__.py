#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: GRPO VERIFIABLE REINFORCEMENT LEARNING
File: 03_llm_training/stage3_grpo_brier_edge/__init__.py
================================================================================
Public interface for Stage 5 Group Relative Policy Optimization (GRPO) with
Verifiable Quantitative Rewards (RLVR) on AMD Instinct MI300X (192GB HBM3):

  1. `brier_calibration_reward` : Strictly proper Brier Score reward
                                  `1.0 - 4.0 * (p - o)^2` + Brier Skill Score
                                  (BSS) bonus relative to the CLOB midpoint `q_t`
                                  + clamped Log-Loss tail regularizer.
  2. `clob_spread_edge_reward`  : Spread-adjusted Central Limit Order Book (CLOB)
                                  economic PnL reward that rewards trading only
                                  when `|p - q_t| >= min_required_edge_to_trade`
                                  and penalizes trading on high-hazard UMA traps.
  3. `format_validator_reward`  : Deterministic XML schema (`<reasoning>`,
                                  `<prob>`, `<edge>`), 5-step Tetlock rubric, and
                                  algebraic subtraction (`edge == p - q_t`)
                                  verifier.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 03_llm_training.stage3_grpo_brier_edge.<submodule>` never triggers a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "BrierRewardEvaluation",
    "BrierCalibrationRewardFunction",
    "CLOBSpreadEdgeEvaluation",
    "CLOBSpreadAdjustedPnLRewardFunction",
    "XMLFormatRewardEvaluation",
    "XMLFormatValidatorRewardFunction",
]


def __getattr__(name: str) -> Any:
    if name in ("BrierRewardEvaluation", "BrierCalibrationRewardFunction"):
        from .brier_calibration_reward import brier_score_fn
        return getattr(brier_score_fn, name)
    if name in (
        "CLOBSpreadEdgeEvaluation",
        "CLOBSpreadAdjustedPnLRewardFunction",
    ):
        from .clob_spread_edge_reward import spread_adjusted_pnl_fn
        return getattr(spread_adjusted_pnl_fn, name)
    if name in ("XMLFormatRewardEvaluation", "XMLFormatValidatorRewardFunction"):
        from .format_validator_reward import xml_schema_reward
        return getattr(xml_schema_reward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")