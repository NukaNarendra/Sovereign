#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: WEBSOCKET ANOMALY FILTER
File: 04_mcp_runtime/websocket_anomaly_filter/__init__.py
================================================================================
Blueprint-exact lazy-loaded package interface for the Zero-Cost Polymarket CLOB
v2 WebSocket Anomaly Filter:
  - `clob_ws_listener.py`       : Streams wss://ws-subscriptions-clob.polymarket.com/ws/market
  - `zscore_volume_detector.py` : Triggers wake signal when 5-min volume > 3 sigma above mean
  - `odds_velocity_trigger.py`  : Detects sudden >4% implied probability jumps in order book
  - `wake_queue_dispatcher.py`  : Pushes anomalous token IDs to the LLM inference router
================================================================================
"""

from typing import Any

__all__ = [
    "VolumeObservation",
    "VolumeSpikeTrigger",
    "ZScoreVolumeDetector",
    "OddsJumpTrigger",
    "OddsVelocityTrigger",
    "CLOBWSListener",
    "WakeQueueEnvelope",
    "WakeQueueDispatchReport",
    "WakeQueueDispatcher",
]


def __getattr__(name: str) -> Any:
    if name in ("VolumeObservation", "VolumeSpikeTrigger", "ZScoreVolumeDetector"):
        from . import zscore_volume_detector
        return getattr(zscore_volume_detector, name)
    if name in ("OddsJumpTrigger", "OddsVelocityTrigger"):
        from . import odds_velocity_trigger
        return getattr(odds_velocity_trigger, name)
    if name == "CLOBWSListener":
        from . import clob_ws_listener
        return getattr(clob_ws_listener, name)
    if name in ("WakeQueueEnvelope", "WakeQueueDispatchReport", "WakeQueueDispatcher"):
        from . import wake_queue_dispatcher
        return getattr(wake_queue_dispatcher, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")