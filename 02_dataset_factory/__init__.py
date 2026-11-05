#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY & ANTI-LEAKAGE ENGINE
File: 02_dataset_factory/__init__.py
================================================================================
Root package initializer for the Stage 3 Anti-Leakage & Point-in-Time Alignment
Dataset Factory (`02_dataset_factory/`).

Architectural Subpackages:
  1. `temporal_aligner`        : Performs strict point-in-time ("as-of") joins
                                 across Stage 2 Parquet shards (`t_feature <= t_cutoff`),
                                 corrects UTC/ET/Polygon block clock skew, and
                                 stratifies sampling horizons (`T-168h` to `T-6h`).
  2. `leak_auditor`            : Scans every aligned text field for post-cutoff
                                 future dates, outcome spoilers ("after winning",
                                 "resolved to Yes"), and train/test entity overlap.
  3. `orderbook_serializer`    : Converts numeric CLOB microstructure, 24h price
                                 trajectories, UMA rules, and news wires into
                                 compact XML/Markdown prompts for Qwen-2.5-7B.
  4. `synthetic_sft_generator` : Uses Merge Gateway ($10/mo free student credit)
                                 or deterministic Tetlock superforecaster synthesis
                                 to build blind Chain-of-Thought SFT traces.
  5. `parquet_warehouse`       : Enforces strict chronological train/val/test
                                 splits and writes CPT, SFT, and GRPO manifests.
================================================================================
"""

from typing import Final

__version__: Final[str] = "1.0.0-sovereign-stage3"
__stage__: Final[str] = "STAGE_3_DATASET_FACTORY"
__author__: Final[str] = "Sovereign-Prob-7B Quantitative Architecture"

SUBMODULES: Final[tuple[str, ...]] = (
    "temporal_aligner",
    "leak_auditor",
    "orderbook_serializer",
    "synthetic_sft_generator",
    "parquet_warehouse",
)

__all__ = [
    "__version__",
    "__stage__",
    "__author__",
    "SUBMODULES",
]