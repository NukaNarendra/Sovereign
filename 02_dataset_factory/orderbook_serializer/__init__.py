#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART C)
File: 02_dataset_factory/orderbook_serializer/__init__.py
================================================================================
Public interface for the CLOB Microstructure Serializer, 24h Price Trajectory
Tokenizer, and Unified Forecasting Prompt Builder:
  1. `l2_book_to_markdown`        : Converts numeric CLOB midpoint, bid-ask
                                    spread, volume, and depth metrics into a
                                    5-level Level-2 `<book>` markdown/XML table
                                    with $5.00 survival vs $1,000 institutional
                                    VWAP slippage annotations.
  2. `price_trajectory_tokenizer` : Compresses 24-hour implied probability paths,
                                    realized volatility, and Shannon entropy into
                                    discrete regime tokens inside `<trajectory>`.
  3. `prompt_template_builder`    : Assembles the complete, leak-certified XML
                                    prompt (`<market_meta>`, `<uma_rule>`,
                                    `<book>`, `<trajectory>`, `<whale_flow>`,
                                    `<macro_calendar>`, `<social_pulse>`,
                                    `<news_context>`) used identically in Stage 4
                                    CPT/SFT, Stage 5 GRPO, and Stage 6 Live MCP.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 02_dataset_factory.orderbook_serializer.<module>` never triggers a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "L2LadderLevel",
    "SerializedOrderBookBlock",
    "L2BookToMarkdownSerializer",
    "TrajectoryStepBar",
    "TokenizedPriceTrajectory",
    "PriceTrajectoryTokenizer",
    "UnifiedForecastingPrompt",
    "PromptTemplateBuilder",
]


def __getattr__(name: str) -> Any:
    if name in (
        "L2LadderLevel",
        "SerializedOrderBookBlock",
        "L2BookToMarkdownSerializer",
    ):
        from . import l2_book_to_markdown
        return getattr(l2_book_to_markdown, name)
    if name in (
        "TrajectoryStepBar",
        "TokenizedPriceTrajectory",
        "PriceTrajectoryTokenizer",
    ):
        from . import price_trajectory_tokenizer
        return getattr(price_trajectory_tokenizer, name)
    if name in ("UnifiedForecastingPrompt", "PromptTemplateBuilder"):
        from . import prompt_template_builder
        return getattr(prompt_template_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")