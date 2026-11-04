#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART A)
File: 02_dataset_factory/temporal_aligner/__init__.py
================================================================================
Public interface for the Point-in-Time Temporal Alignment, Clock-Skew Correction,
and Multi-Horizon Stratification Engine.

Uses PEP 562 lazy attribute loading (`__getattr__`) so executing individual
modules via `python3 -m 02_dataset_factory.temporal_aligner.<module>` never
triggers `<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "ClockSkewConfig",
    "NormalizedTimestampResult",
    "ClockSkewCorrector",
    "HorizonTierSpec",
    "StratifiedHorizonRow",
    "HorizonStratifier",
    "PointInTimeJoinConfig",
    "PointInTimeAlignedRecord",
    "PointInTimeJoiner",
]


def __getattr__(name: str) -> Any:
    if name in ("ClockSkewConfig", "NormalizedTimestampResult", "ClockSkewCorrector"):
        from . import clock_skew_corrector
        return getattr(clock_skew_corrector, name)
    if name in ("HorizonTierSpec", "StratifiedHorizonRow", "HorizonStratifier"):
        from . import horizon_stratifier
        return getattr(horizon_stratifier, name)
    if name in (
        "PointInTimeJoinConfig",
        "PointInTimeAlignedRecord",
        "PointInTimeJoiner",
    ):
        from . import point_in_time_joiner
        return getattr(point_in_time_joiner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")