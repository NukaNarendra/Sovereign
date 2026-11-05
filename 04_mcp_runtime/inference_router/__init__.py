#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: HYBRID INFERENCE ROUTER
File: 04_mcp_runtime/inference_router/__init__.py
================================================================================
Lazy-loaded package initializer for the Stage 6 Hybrid Model Router
(`hybrid_model_router.py`) and Virtual Survival Cost Accountant
(`virtual_cost_accountant.py`).
================================================================================
"""

from typing import Any

__all__ = [
    "InferenceCallReceipt",
    "VirtualCostLedgerState",
    "InferenceCostMeter",
    "ParsedForecastOutput",
    "LLMResponseParser",
    "RoutedForecastResponse",
    "HybridRouterAuditReport",
    "HybridModelRouter",
]


def __getattr__(name: str) -> Any:
    if name in (
        "InferenceCallReceipt",
        "VirtualCostLedgerState",
        "InferenceCostMeter",
    ):
        from . import inference_cost_meter
        return getattr(inference_cost_meter, name)
    if name in ("ParsedForecastOutput", "LLMResponseParser"):
        from . import response_parser
        return getattr(response_parser, name)
    if name in (
        "RoutedForecastResponse",
        "HybridRouterAuditReport",
        "HybridModelRouter",
    ):
        from . import hybrid_model_router
        return getattr(hybrid_model_router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")