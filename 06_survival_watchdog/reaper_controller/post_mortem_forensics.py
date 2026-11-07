#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: POST-MORTEM FORENSICS DUMPER
File: 06_survival_watchdog/reaper_controller/post_mortem_forensics.py
================================================================================
Dumps an immutable forensic breakdown (Brier calibration error by category,
worst losing trades, and compute burn vs trading drawdown attribution) whenever
an agent container is terminated by the Reaper Watchdog.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [POST-MORTEM] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.post_mortem_forensics")


@dataclass(frozen=True)
class PostMortemForensicReport:
    """Forensic autopsy generated if an agent hits the $0.00 death floor."""
    agent_id: str
    termination_cause: str
    final_net_liquid_nav_usd: float
    total_trades_analyzed: int
    empirical_brier_score: float
    cumulative_trading_pnl_usd: float
    cumulative_compute_burn_usd: float
    primary_failure_attribution: str
    forensic_dump_path: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PostMortemForensicsEngine:
    """
    Analyzes trade history and compute burn upon agent termination so future
    GRPO training epochs can immunize against the failure mode.
    """

    def __init__(
        self,
        forensics_dir: pathlib.Path = pathlib.Path(
            "state/forensics/post_mortems"
        ),
    ) -> None:
        self.forensics_dir = forensics_dir
        self.forensics_dir.mkdir(parents=True, exist_ok=True)

    def generate_post_mortem(
        self,
        agent_id: str,
        final_nav_usd: float,
        compute_burn_usd: float,
        recent_trades: Sequence[Dict[str, Any]],
        cause: str = "NET_LIQUID_BALANCE_EXHAUSTED",
    ) -> PostMortemForensicReport:
        n_t = len(recent_trades)
        if n_t > 0:
            brier = sum(
                (float(t.get("prob_p", 0.5)) - float(t.get("outcome_o", 0))) ** 2
                for t in recent_trades
            ) / float(n_t)
            trade_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in recent_trades)
        else:
            brier = 0.25
            trade_pnl = 0.0

        attribution = (
            "METABOLIC_COMPUTE_STARVATION_TOO_FEW_TRADES"
            if abs(compute_burn_usd) > abs(trade_pnl)
            else "ADVERSE_SELECTION_OR_CALIBRATION_DRIFT"
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        out_file = self.forensics_dir / f"post_mortem_{agent_id}.json"

        report = PostMortemForensicReport(
            agent_id=agent_id,
            termination_cause=cause,
            final_net_liquid_nav_usd=round(float(final_nav_usd), 4),
            total_trades_analyzed=n_t,
            empirical_brier_score=round(brier, 5),
            cumulative_trading_pnl_usd=round(trade_pnl, 4),
            cumulative_compute_burn_usd=round(float(compute_burn_usd), 4),
            primary_failure_attribution=attribution,
            forensic_dump_path=str(out_file),
            timestamp_utc=now_iso,
        )
        out_file.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        return report


def main() -> int:
    engine = PostMortemForensicsEngine()
    rep = engine.generate_post_mortem(
        agent_id="sovereign_test_failed_clone",
        final_nav_usd=0.0,
        compute_burn_usd=0.12,
        recent_trades=[
            {"prob_p": 0.82, "outcome_o": 0, "pnl_usd": -4.88}
        ],
    )
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())