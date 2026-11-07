#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: COMPUTE BURN RATE ACCUMULATOR
File: 06_survival_watchdog/compute_burn_ledger/burn_rate_accumulator.py
================================================================================
Accrues both:
  1. Hourly container hosting / electricity burn (`$0.0045/hour` = `$0.108/day`)
  2. Per-forecast LLM inference charges (`$0.00020/call` from `InferenceCostMeter`)
so the Survival Agent must continuously generate positive trading alpha faster
than its metabolic burn rate to stay alive.
================================================================================
"""

from __future__ import annotations

import sys
import json
import time
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [BURN-ACCUMULATOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.burn_rate_accumulator")


@dataclass(frozen=True)
class BurnRateAccrualReport:
    """Metabolic compute and hosting burn report for the survival agent."""
    hourly_container_burn_rate_usd: float
    elapsed_hours_accrued: float
    incremental_hosting_burn_usd: float
    cumulative_hosting_burn_usd: float
    cumulative_inference_burn_usd: float
    total_accrued_compute_burn_usd: float
    remaining_runway_hours_at_zero_trades: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BurnRateAccumulator:
    """
    Deducts wall-clock container hosting costs and synchronizes with the
    Stage 6 `virtual_cost_ledger.json`.
    """

    def __init__(
        self,
        hourly_hosting_burn_usd: float = 0.0045,
        burn_ledger_path: pathlib.Path = pathlib.Path(
            "state/compute_burn_ledger.json"
        ),
        virtual_cost_path: pathlib.Path = pathlib.Path(
            "state/virtual_cost_ledger.json"
        ),
    ) -> None:
        self.hourly_hosting_burn_usd = float(hourly_hosting_burn_usd)
        self.burn_ledger_path = burn_ledger_path
        self.virtual_cost_path = virtual_cost_path
        self.burn_ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def accrue_elapsed_burn(
        self,
        simulated_elapsed_hours: float = 1.0,
        current_liquid_nav_usd: float = 5.00,
    ) -> BurnRateAccrualReport:
        prev_hosting_usd = 0.0
        if self.burn_ledger_path.exists():
            try:
                raw = json.loads(self.burn_ledger_path.read_text(encoding="utf-8"))
                prev_hosting_usd = float(raw.get("cumulative_hosting_burn_usd", 0.0))
            except Exception:
                prev_hosting_usd = 0.0

        inf_burn_usd = 0.0
        if self.virtual_cost_path.exists():
            try:
                v_raw = json.loads(self.virtual_cost_path.read_text(encoding="utf-8"))
                inf_burn_usd = float(
                    v_raw.get("accrued_virtual_survival_burn_usd", 0.0)
                )
            except Exception:
                inf_burn_usd = 0.0

        inc_hosting = round(
            max(0.0, float(simulated_elapsed_hours)) * self.hourly_hosting_burn_usd,
            6,
        )
        cum_hosting = round(prev_hosting_usd + inc_hosting, 6)
        tot_burn = round(cum_hosting + inf_burn_usd, 6)

        runway_hrs = round(
            max(0.0, float(current_liquid_nav_usd))
            / max(1e-6, self.hourly_hosting_burn_usd),
            1,
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        report = BurnRateAccrualReport(
            hourly_container_burn_rate_usd=self.hourly_hosting_burn_usd,
            elapsed_hours_accrued=round(float(simulated_elapsed_hours), 3),
            incremental_hosting_burn_usd=inc_hosting,
            cumulative_hosting_burn_usd=cum_hosting,
            cumulative_inference_burn_usd=round(inf_burn_usd, 6),
            total_accrued_compute_burn_usd=tot_burn,
            remaining_runway_hours_at_zero_trades=runway_hrs,
            timestamp_utc=now_iso,
        )
        tmp = self.burn_ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.burn_ledger_path)
        return report


def main() -> int:
    acc = BurnRateAccumulator()
    rep = acc.accrue_elapsed_burn(
        simulated_elapsed_hours=2.0, current_liquid_nav_usd=5.1114
    )
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())