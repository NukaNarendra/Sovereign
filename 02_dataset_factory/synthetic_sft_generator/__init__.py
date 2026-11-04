#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART D)
File: 02_dataset_factory/synthetic_sft_generator/__init__.py
================================================================================
Public interface for the Blind Superforecaster Chain-of-Thought (CoT) Teacher
Trace Generator and Hindsight-Bias Auditor:
  1. `merge_gateway_client`        : Connects to Merge Gateway ($10/month free
                                     student multi-LLM credit pool) with strict
                                     quota tracking, SHA-256 response caching,
                                     and zero-cost deterministic fallback.
  2. `superforecaster_cot_prompter`: Generates 5-step Philip Tetlock style
                                     (Outside-View Base Rate -> UMA Clause Audit
                                     -> L2 CLOB & Whale Microstructure -> Inside-
                                     View News Catalysts -> Calibrated Bayesian
                                     Update) teacher completions strictly blind
                                     to the final outcome `o`.
  3. `hindsight_bias_filter`       : Audits teacher reasoning traces for XML
                                     tag compliance (`<reasoning>`, `<prob>`,
                                     `<edge>`), algebraic edge consistency
                                     (`edge == p - q_t`), and zero post-cutoff
                                     knowledge leakage from teacher weights.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 02_dataset_factory.synthetic_sft_generator.<module>` never emits a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "MergeGatewayQuotaState",
    "TeacherCompletionResult",
    "MergeGatewayTeacherClient",
    "SuperforecasterTraceRecord",
    "SuperforecasterCoTPrompter",
    "HindsightAuditReport",
    "HindsightBiasFilter",
]


def __getattr__(name: str) -> Any:
    if name in (
        "MergeGatewayQuotaState",
        "TeacherCompletionResult",
        "MergeGatewayTeacherClient",
    ):
        from . import merge_gateway_client
        return getattr(merge_gateway_client, name)
    if name in ("SuperforecasterTraceRecord", "SuperforecasterCoTPrompter"):
        from . import superforecaster_cot_prompter
        return getattr(superforecaster_cot_prompter, name)
    if name in ("HindsightAuditReport", "HindsightBiasFilter"):
        from . import hindsight_bias_filter
        return getattr(hindsight_bias_filter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")