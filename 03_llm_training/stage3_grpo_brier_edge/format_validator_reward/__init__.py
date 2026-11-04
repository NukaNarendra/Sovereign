#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: XML SCHEMA & RUBRIC FORMAT REWARD
File: 03_llm_training/stage3_grpo_brier_edge/format_validator_reward/__init__.py
================================================================================
Lazy-loaded package initializer for the Stage 5 GRPO XML Schema, 5-Step Tetlock
Superforecaster Rubric, and Algebraic Edge Consistency Reward Function.
================================================================================
"""

from typing import Any

__all__ = [
    "XMLFormatRewardEvaluation",
    "XMLFormatBatchAuditReport",
    "XMLFormatValidatorRewardFunction",
    "ReasoningCompletenessEvaluation",
    "ReasoningCompletenessBatchReport",
    "ReasoningCompletenessChecker",
]


def __getattr__(name: str) -> Any:
    if name in (
        "XMLFormatRewardEvaluation",
        "XMLFormatBatchAuditReport",
        "XMLFormatValidatorRewardFunction",
    ):
        from . import xml_schema_reward
        return getattr(xml_schema_reward, name)
    if name in (
        "ReasoningCompletenessEvaluation",
        "ReasoningCompletenessBatchReport",
        "ReasoningCompletenessChecker",
    ):
        from . import reasoning_completeness_check
        return getattr(reasoning_completeness_check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")