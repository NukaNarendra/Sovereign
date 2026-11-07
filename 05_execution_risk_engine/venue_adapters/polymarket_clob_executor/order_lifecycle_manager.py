#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: CLOB ORDER LIFECYCLE MANAGER
File: 05_execution_risk_engine/venue_adapters/polymarket_clob_executor/order_lifecycle_manager.py
================================================================================
Tracks resting `POST_ONLY` limit orders, partial fills, and automatic stale-quote
cancellations (`stale_timeout_sec = 180` seconds) so resting orders never get
adversely picked off when breaking news shifts the order book.
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
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [ORDER-LIFECYCLE] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.order_lifecycle_manager")


@dataclass
class ActiveOrderState:
    """State record for a resting or partially filled Polymarket CLOB order."""
    venue_order_id: str
    client_order_id: str
    market_id: str
    token_id: str
    side: str
    limit_price: float
    total_shares: float
    filled_shares: float
    notional_usd: float
    submitted_epoch_sec: float
    stale_timeout_sec: int
    status: str = "OPEN_RESTING"

    @property
    def remaining_shares(self) -> float:
        return max(0.0, round(self.total_shares - self.filled_shares, 4))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["remaining_shares"] = self.remaining_shares
        return payload


class OrderLifecycleManager:
    """
    Manages the lifecycle of resting limit orders, sweeps stale quotes past
    `stale_timeout_sec`, and provides an emergency `cancel_all` hook for the
    Stage 8 Reaper Watchdog.
    """

    def __init__(
        self,
        ledger_path: pathlib.Path = pathlib.Path(
            "state/open_clob_orders_ledger.json"
        ),
    ) -> None:
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._orders: Dict[str, ActiveOrderState] = {}

    def register_new_order(self, order: ActiveOrderState) -> None:
        self._orders[order.venue_order_id] = order
        self._persist()

    def record_partial_or_full_fill(
        self, venue_order_id: str, filled_delta_shares: float
    ) -> ActiveOrderState:
        ord_state = self._orders[venue_order_id]
        ord_state.filled_shares = min(
            ord_state.total_shares,
            round(ord_state.filled_shares + float(filled_delta_shares), 4),
        )
        if ord_state.remaining_shares <= 1e-4:
            ord_state.status = "FILLED_COMPLETE"
        else:
            ord_state.status = "PARTIALLY_FILLED"
        self._persist()
        return ord_state

    def sweep_and_cancel_stale_orders(
        self, current_epoch_sec: float
    ) -> List[str]:
        canceled_ids: List[str] = []
        for oid, st in self._orders.items():
            if st.status in ("OPEN_RESTING", "PARTIALLY_FILLED"):
                age_sec = float(current_epoch_sec) - st.submitted_epoch_sec
                if age_sec >= float(st.stale_timeout_sec):
                    st.status = "CANCELED_STALE_TIMEOUT"
                    canceled_ids.append(oid)
        if canceled_ids:
            self._persist()
        return canceled_ids

    def emergency_cancel_all_open_orders(self) -> List[str]:
        canceled: List[str] = []
        for oid, st in self._orders.items():
            if st.status in ("OPEN_RESTING", "PARTIALLY_FILLED"):
                st.status = "CANCELED_EMERGENCY_REAPER"
                canceled.append(oid)
        self._persist()
        return canceled

    def _persist(self) -> None:
        tmp = self.ledger_path.with_suffix(".json.tmp")
        serialized = {k: v.to_dict() for k, v in self._orders.items()}
        tmp.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        tmp.replace(self.ledger_path)


def main() -> int:
    mgr = OrderLifecycleManager()
    t0 = 1776000000.0
    mgr.register_new_order(
        ActiveOrderState(
            venue_order_id="PM-ORD-101",
            client_order_id="CLI-101",
            market_id="558938",
            token_id="100998877",
            side="BUY_NO",
            limit_price=0.68,
            total_shares=1.01,
            filled_shares=0.0,
            notional_usd=0.6868,
            submitted_epoch_sec=t0,
            stale_timeout_sec=180,
        )
    )
    stale_canceled = mgr.sweep_and_cancel_stale_orders(current_epoch_sec=t0 + 200.0)
    print(
        json.dumps(
            {
                "stage": "STAGE_7_ORDER_LIFECYCLE_MANAGER",
                "stale_orders_canceled": stale_canceled,
                "order_state": mgr._orders["PM-ORD-101"].to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())