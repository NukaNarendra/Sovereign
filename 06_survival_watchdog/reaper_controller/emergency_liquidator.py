#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: EMERGENCY CLOB LIQUIDATOR
File: 06_survival_watchdog/reaper_controller/emergency_liquidator.py
================================================================================
Cancels all open resting Polymarket CLOB v2 orders (`OrderLifecycleManager`)
immediately when Net Liquid NAV approaches or breaches the critical survival
threshold (`<= $0.25` warning or `<= $0.00` Reaper termination).
================================================================================
"""

from __future__ import annotations

import sys
import json
import importlib
import logging
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [EMERGENCY-LIQUIDATOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.emergency_liquidator")


@dataclass(frozen=True)
class EmergencyLiquidationReport:
    """Audit report generated when the Reaper liquidates resting CLOB quotes."""
    trigger_reason: str
    net_liquid_nav_usd: float
    open_orders_canceled_count: int
    canceled_order_ids: List[str]
    all_resting_risk_cleared: bool
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmergencyLiquidator:
    """
    Invokes `OrderLifecycleManager.emergency_cancel_all_open_orders()` to pull
    all resting CLOB quotes before container termination.
    """

    def __init__(self) -> None:
        olm_mod = importlib.import_module(
            "05_execution_risk_engine.venue_adapters.polymarket_clob_executor.order_lifecycle_manager"
        )
        self.lifecycle_mgr = olm_mod.OrderLifecycleManager()

    def cancel_all_open_orders(
        self, net_liquid_nav_usd: float, reason: str = "CRITICAL_NAV_THRESHOLD"
    ) -> EmergencyLiquidationReport:
        canceled = self.lifecycle_mgr.emergency_cancel_all_open_orders()
        return EmergencyLiquidationReport(
            trigger_reason=reason,
            net_liquid_nav_usd=round(float(net_liquid_nav_usd), 4),
            open_orders_canceled_count=len(canceled),
            canceled_order_ids=canceled,
            all_resting_risk_cleared=True,
            timestamp_utc=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        )


def main() -> int:
    liquidator = EmergencyLiquidator()
    rep = liquidator.cancel_all_open_orders(
        net_liquid_nav_usd=0.0, reason="SIMULATED_DEATH_FLOOR_TEST"
    )
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())