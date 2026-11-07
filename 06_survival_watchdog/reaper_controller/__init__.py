#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `reaper_controller`."""

from typing import Any

__all__ = [
    "EmergencyLiquidationReport",
    "EmergencyLiquidator",
    "MitosisCloneReceipt",
    "MitosisCloneSpawner",
    "PostMortemForensicReport",
    "PostMortemForensicsEngine",
    "WatchdogCycleVerdict",
    "ReaperWatchdogDaemon",
]


def __getattr__(name: str) -> Any:
    if name in ("EmergencyLiquidationReport", "EmergencyLiquidator"):
        from . import emergency_liquidator
        return getattr(emergency_liquidator, name)
    if name in ("MitosisCloneReceipt", "MitosisCloneSpawner"):
        from . import mitosis_clone_spawner
        return getattr(mitosis_clone_spawner, name)
    if name in ("PostMortemForensicReport", "PostMortemForensicsEngine"):
        from . import post_mortem_forensics
        return getattr(post_mortem_forensics, name)
    if name in ("WatchdogCycleVerdict", "ReaperWatchdogDaemon"):
        from . import watchdog_daemon
        return getattr(watchdog_daemon, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")