#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: GRPO BRIER CALIBRATION REWARD
File: 03_llm_training/stage3_grpo_brier_edge/brier_calibration_reward/__init__.py
================================================================================
Lazy-loaded package initializer for the Strictly Proper Brier Score + Brier
Skill Score (BSS) + Clamped Log-Loss Verifiable Reward module.
================================================================================
"""

from typing import Any

__all__ = [
    "BrierRewardEvaluation",
    "BrierBatchAuditReport",
    "BrierCalibrationRewardFunction",
    "LogLossRegularizerEvaluation",
    "LogLossBatchAuditReport",
    "LogLossRegularizer",
]


def __getattr__(name: str) -> Any:
    if name in (
        "BrierRewardEvaluation",
        "BrierBatchAuditReport",
        "BrierCalibrationRewardFunction",
    ):
        from . import brier_score_fn
        return getattr(brier_score_fn, name)
    if name in (
        "LogLossRegularizerEvaluation",
        "LogLossBatchAuditReport",
        "LogLossRegularizer",
    ):
        from . import log_loss_regularizer
        return getattr(log_loss_regularizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")