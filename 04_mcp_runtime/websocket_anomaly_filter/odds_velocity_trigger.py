#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: >4% ODDS VELOCITY TRIGGER
File: 04_mcp_runtime/websocket_anomaly_filter/odds_velocity_trigger.py
================================================================================
Detects sudden $\ge 4.0\%$ ($400\text{ bps}$) implied probability jumps or
Order Book Imbalance ($\text{OBI} \ge 0.65$) shocks in the Polymarket CLOB v2
order book:
    $$|\Delta q_{5\text{m}}(t)| = |q_t - q_{t - \Delta t}| \ge 0.040$$
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [ODDS-VELOCITY] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.odds_velocity_trigger")


@dataclass(frozen=True)
class OddsJumpTrigger:
    """Wake signal emitted when CLOB midpoint shifts by >= 4.0% or OBI shocks."""
    snapshot_uid: str
    market_id: str
    condition_id: str
    timestamp_epoch_sec: float
    previous_midpoint_qt: float
    current_midpoint_qt: float
    signed_prob_delta: float
    abs_prob_delta: float
    order_book_imbalance: float
    trigger_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OddsVelocityTrigger:
    """
    Monitors rolling CLOB midpoint trajectories per `market_id` and fires an
    `OddsJumpTrigger` when `|q_t - q_prev| >= 0.040` or `|OBI| >= 0.65`.
    """

    def __init__(
        self,
        prob_jump_threshold: float = 0.040,
        obi_shock_threshold: float = 0.65,
        window_size: int = 12,
    ) -> None:
        self.prob_jump_threshold = float(prob_jump_threshold)
        self.obi_shock_threshold = float(obi_shock_threshold)
        self.window_size = max(3, int(window_size))
        self._midpoints: Dict[str, Deque[Tuple[float, float]]] = {}
        self.total_evaluated: int = 0
        self.total_triggered: int = 0

    def seed_market_midpoint(
        self, market_id: str, timestamp_sec: float, midpoint_qt: float
    ) -> None:
        win = self._midpoints.setdefault(
            market_id, deque(maxlen=self.window_size)
        )
        win.append((float(timestamp_sec), float(midpoint_qt)))

    def evaluate_midpoint_tick(
        self,
        snapshot_uid: str,
        market_id: str,
        condition_id: str,
        timestamp_epoch_sec: float,
        midpoint_qt: float,
        order_book_imbalance: float = 0.0,
        safe_for_survival_5usd_bot: bool = True,
    ) -> Optional[OddsJumpTrigger]:
        """
        Checks if `midpoint_qt` jumped by >= 4.0% relative to the rolling window
        anchor or experienced an OBI shock.
        """
        self.total_evaluated += 1
        if not safe_for_survival_5usd_bot:
            return None

        win = self._midpoints.setdefault(
            market_id, deque(maxlen=self.window_size)
        )
        ref_q = win[0][1] if win else float(midpoint_qt)
        signed_dq = round(float(midpoint_qt) - ref_q, 4)
        abs_dq = round(abs(signed_dq), 4)

        win.append((float(timestamp_epoch_sec), float(midpoint_qt)))

        trig_type: Optional[str] = None
        if abs_dq >= self.prob_jump_threshold:
            trig_type = f"ODDS_VELOCITY_JUMP_{signed_dq * 100.0:+0.2f}PCT"
        elif abs(order_book_imbalance) >= self.obi_shock_threshold and abs_dq >= 0.015:
            trig_type = f"OBI_LIQUIDITY_SHOCK_{order_book_imbalance:+0.2f}"

        if trig_type is not None:
            self.total_triggered += 1
            return OddsJumpTrigger(
                snapshot_uid=snapshot_uid,
                market_id=market_id,
                condition_id=condition_id,
                timestamp_epoch_sec=timestamp_epoch_sec,
                previous_midpoint_qt=round(ref_q, 4),
                current_midpoint_qt=round(float(midpoint_qt), 4),
                signed_prob_delta=signed_dq,
                abs_prob_delta=abs_dq,
                order_book_imbalance=round(float(order_book_imbalance), 4),
                trigger_type=trig_type,
            )
        return None


def main() -> int:
    trigger_engine = OddsVelocityTrigger()
    trigger_engine.seed_market_midpoint("253591", 1776000000.0, 0.5400)

    quiet = trigger_engine.evaluate_midpoint_tick(
        snapshot_uid="QUIET-ODDS-1",
        market_id="253591",
        condition_id="0x" + "11" * 32,
        timestamp_epoch_sec=1776000060.0,
        midpoint_qt=0.5480,
        order_book_imbalance=0.10,
    )
    jump = trigger_engine.evaluate_midpoint_tick(
        snapshot_uid="JUMP-ODDS-2",
        market_id="253591",
        condition_id="0x" + "11" * 32,
        timestamp_epoch_sec=1776000120.0,
        midpoint_qt=0.6050,
        order_book_imbalance=0.68,
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_6_ODDS_VELOCITY_TRIGGER",
                "quiet_move_suppressed": quiet is None,
                "four_pct_jump_triggered": jump is not None,
                "sample_jump": jump.to_dict() if jump else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())