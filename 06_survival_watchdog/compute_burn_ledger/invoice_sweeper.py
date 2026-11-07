#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: COMPUTE INVOICE SWEEPER
File: 06_survival_watchdog/compute_burn_ledger/invoice_sweeper.py
================================================================================
Ear-marks a configurable fraction (`20%`) of realized Polymarket trading profits
into an autonomous **Compute & API Treasury Reserve** to pay for Fireworks AI
inference, Tavily search renewals, and container hosting once student credits
expire.
================================================================================
"""

from __future__ import annotations

import sys
import json
import hashlib
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [INVOICE-SWEEPER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.invoice_sweeper")


@dataclass(frozen=True)
class InvoiceSweepReceipt:
    """Receipt recording realized trading profit ear-marked to pay compute bills."""
    sweep_id: str
    realized_trade_profit_usd: float
    accrued_compute_debt_usd: float
    swept_to_compute_treasury_usd: float
    net_retained_in_trading_bankroll_usd: float
    compute_debt_fully_settled: bool
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InvoiceSweeper:
    """
    Sweeps realized trading profits to settle accrued compute burn so the
    agent is 100% self-funding.
    """

    def __init__(
        self,
        max_profit_sweep_ratio: float = 0.20,
        treasury_ledger_path: pathlib.Path = pathlib.Path(
            "state/compute_treasury_sweeps.json"
        ),
    ) -> None:
        self.max_profit_sweep_ratio = float(max_profit_sweep_ratio)
        self.treasury_ledger_path = treasury_ledger_path
        self.treasury_ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def sweep_realized_profit(
        self, realized_profit_usd: float, accrued_compute_debt_usd: float
    ) -> InvoiceSweepReceipt:
        prof = max(0.0, float(realized_profit_usd))
        debt = max(0.0, float(accrued_compute_debt_usd))

        max_earmark = prof * self.max_profit_sweep_ratio
        swept = round(min(debt, max_earmark), 6)
        retained = round(prof - swept, 6)
        settled = (debt - swept) <= 1e-6

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sid = hashlib.sha256(f"{now_iso}:{prof}:{swept}".encode("utf-8")).hexdigest()[:10]

        receipt = InvoiceSweepReceipt(
            sweep_id=f"SWP-{sid.upper()}",
            realized_trade_profit_usd=round(prof, 4),
            accrued_compute_debt_usd=round(debt, 6),
            swept_to_compute_treasury_usd=swept,
            net_retained_in_trading_bankroll_usd=retained,
            compute_debt_fully_settled=settled,
            timestamp_utc=now_iso,
        )
        self.treasury_ledger_path.write_text(
            json.dumps(receipt.to_dict(), indent=2), encoding="utf-8"
        )
        return receipt


def main() -> int:
    sweeper = InvoiceSweeper()
    rcpt = sweeper.sweep_realized_profit(
        realized_profit_usd=0.3132, accrued_compute_debt_usd=0.0114
    )
    print(json.dumps(rcpt.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())