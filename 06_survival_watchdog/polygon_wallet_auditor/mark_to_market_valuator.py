#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: MARK-TO-MARKET POSITION VALUATOR
File: 06_survival_watchdog/polygon_wallet_auditor/mark_to_market_valuator.py
================================================================================
Values open Polymarket Conditional Token Framework (CTF) outcome positions at
their **Conservative Immediate Liquidation Value** (`best_bid` minus taker exit
slippage), never at optimistic midpoints or last-traded prices:
    $$V_{\text{liquidation}} = \sum_{i=1}^{K} Q_i \cdot \max\left(0,\; \text{Bid}_{\text{best}, i} - \epsilon_{\text{exit}}\right)$$
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [MTM-VALUATOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.mark_to_market_valuator")


@dataclass(frozen=True)
class OpenPositionValuation:
    """Liquidation-value mark for a single open CTF outcome share position."""
    market_id: str
    token_id: str
    side: str
    shares_held: float
    entry_price_usd: float
    cost_basis_usd: float
    current_best_bid_usd: float
    liquidation_value_usd: float
    unrealized_pnl_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarkToMarketPortfolioValuation:
    """
    Complete Net Asset Value (NAV) calculation combining liquid pUSD/USDC cash,
    conservative best-bid CTF liquidation value, and accrued compute burn.
    """
    agent_id: str
    liquid_cash_usd: float
    open_positions_cost_basis_usd: float
    open_positions_liquidation_value_usd: float
    unrealized_pnl_usd: float
    gross_portfolio_value_usd: float
    accrued_compute_burn_usd: float
    net_liquid_nav_usd: float
    survival_death_floor_usd: float
    mitosis_clone_trigger_usd: float
    is_alive_above_death_floor: bool
    is_ready_for_mitosis_clone: bool
    positions: List[OpenPositionValuation]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["positions"] = [p.to_dict() for p in self.positions]
        return payload


class MarkToMarketValuator:
    """
    Computes conservative best-bid liquidation value across all open CTF
    positions and updates `survival_nav_state.json`.
    """

    def __init__(
        self,
        nav_state_path: pathlib.Path = pathlib.Path(
            "06_survival_watchdog/polygon_wallet_auditor/survival_nav_state.json"
        ),
        runtime_state_path: pathlib.Path = pathlib.Path(
            "state/survival_nav_state.json"
        ),
        exit_slippage_buffer: float = 0.002,
    ) -> None:
        self.nav_state_path = nav_state_path
        self.runtime_state_path = runtime_state_path
        self.exit_slippage_buffer = float(exit_slippage_buffer)
        self.nav_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)

    def value_portfolio(
        self,
        liquid_cash_usd: float,
        accrued_compute_burn_usd: float,
        open_positions: Optional[Sequence[Dict[str, Any]]] = None,
        persist_to_disk: bool = True,
    ) -> MarkToMarketPortfolioValuation:
        pos_list = list(open_positions or [])
        valuations: List[OpenPositionValuation] = []
        cost_sum = 0.0
        liq_sum = 0.0

        for pos in pos_list:
            shares = float(pos.get("shares_held") or 0.0)
            entry_p = float(pos.get("entry_price_usd") or 0.50)
            best_bid = float(pos.get("current_best_bid_usd") or entry_p)
            net_bid = max(0.0, best_bid - self.exit_slippage_buffer)

            cost_usd = round(shares * entry_p, 4)
            liq_usd = round(shares * net_bid, 4)
            unr_pnl = round(liq_usd - cost_usd, 4)

            cost_sum += cost_usd
            liq_sum += liq_usd

            valuations.append(
                OpenPositionValuation(
                    market_id=str(pos.get("market_id") or "0"),
                    token_id=str(pos.get("token_id") or "0"),
                    side=str(pos.get("side") or "BUY_YES"),
                    shares_held=round(shares, 4),
                    entry_price_usd=round(entry_p, 4),
                    cost_basis_usd=cost_usd,
                    current_best_bid_usd=round(best_bid, 4),
                    liquidation_value_usd=liq_usd,
                    unrealized_pnl_usd=unr_pnl,
                )
            )

        gross_val = round(float(liquid_cash_usd) + liq_sum, 4)
        net_nav = round(gross_val - float(accrued_compute_burn_usd), 4)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        report = MarkToMarketPortfolioValuation(
            agent_id="sovereign_survival_agent",
            liquid_cash_usd=round(float(liquid_cash_usd), 4),
            open_positions_cost_basis_usd=round(cost_sum, 4),
            open_positions_liquidation_value_usd=round(liq_sum, 4),
            unrealized_pnl_usd=round(liq_sum - cost_sum, 4),
            gross_portfolio_value_usd=gross_val,
            accrued_compute_burn_usd=round(float(accrued_compute_burn_usd), 4),
            net_liquid_nav_usd=net_nav,
            survival_death_floor_usd=0.00,
            mitosis_clone_trigger_usd=50.00,
            is_alive_above_death_floor=(net_nav > 0.00),
            is_ready_for_mitosis_clone=(net_nav >= 50.00),
            positions=valuations,
            timestamp_utc=now_iso,
        )

        if persist_to_disk:
            state_payload = {
                "agent_id": report.agent_id,
                "initial_seed_usd": 5.00,
                "liquid_pusd_balance": report.liquid_cash_usd,
                "unrealized_ctf_value_usd": report.open_positions_liquidation_value_usd,
                "accrued_compute_burn_usd": report.accrued_compute_burn_usd,
                "net_liquid_nav_usd": report.net_liquid_nav_usd,
                "survival_death_floor_usd": 0.00,
                "mitosis_clone_trigger_usd": 50.00,
                "status": (
                    "MITOSIS_READY_10X"
                    if report.is_ready_for_mitosis_clone
                    else (
                        "ALIVE_EARNING"
                        if report.is_alive_above_death_floor
                        else "REAPER_DEATH_FLOOR_BREACHED"
                    )
                ),
                "last_audit_timestamp_utc": now_iso,
            }
            for target_p in (self.nav_state_path, self.runtime_state_path):
                tmp_p = target_p.with_suffix(".json.tmp")
                tmp_p.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
                tmp_p.replace(target_p)

        return report


def main() -> int:
    valuator = MarkToMarketValuator()
    res = valuator.value_portfolio(
        liquid_cash_usd=4.31,
        accrued_compute_burn_usd=0.0024,
        open_positions=[
            {
                "market_id": "558938",
                "token_id": "100998877",
                "side": "BUY_NO",
                "shares_held": 1.02,
                "entry_price_usd": 0.6800,
                "current_best_bid_usd": 0.7900,
            }
        ],
    )
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())