#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: THE "EARN OR DIE" SURVIVAL WATCHDOG
File: 06_survival_watchdog/__init__.py
================================================================================
Root package initializer for the isolated autonomous supervisor (`06_survival_watchdog/`):
  1. `polygon_wallet_auditor` : Polls Polygon Chain ID 137 for liquid pUSD/USDC
                                and marks open CTF positions to best-bid value.
  2. `compute_burn_ledger`    : Accrues hourly container hosting + per-token
                                inference burn and sweeps profits to pay invoices.
  3. `auto_settlement_Sweep`  : Scans resolved UMA/Polymarket positions and sends
                                `redeemPositions()` transactions on Polygon.
  4. `reaper_controller`      : Terminates the agent container (`docker stop & rm`)
                                if Net Liquid NAV <= $0.00, or spawns a $5.00
                                child clone via mitosis when NAV hits $50.00 (10x).
================================================================================
"""

from typing import Final

__version__: Final[str] = "1.0.0-sovereign-stage8-watchdog"
__stage__: Final[str] = "STAGE_8_SURVIVAL_WATCHDOG"

SUBMODULES: Final[tuple[str, ...]] = (
    "polygon_wallet_auditor",
    "compute_burn_ledger",
    "auto_settlement_Sweep",
    "reaper_controller",
)

__all__ = [
    "__version__",
    "__stage__",
    "SUBMODULES",
]