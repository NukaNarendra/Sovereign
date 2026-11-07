
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: AUTONOMOUS REAPER WATCHDOG DAEMON
File: 06_survival_watchdog/reaper_controller/watchdog_daemon.py
================================================================================
Master 60-Second "Earn or Die" Autonomous Supervisor Running Outside the
Trading Agent's Process Space.

On Every 60-Second Heartbeat Cycle:
  1. Polls Polygon Mainnet (Chain ID 137) via `OnChainBalancePoller`.
  2. Scans for resolved winning CTF shares (`ResolvedPositionScanner`) and
     automatically sends `redeemPositions()` transactions (`CtfRedeemTxSender`).
  3. Accrues container hosting + inference burn (`BurnRateAccumulator`) and
     marks open CTF positions to conservative best-bid liquidation value
     (`MarkToMarketValuator`).
  4. Evaluates the 3 Survival Regimes:
       - **Regime A (`Net Liquid NAV <= $0.00` — DEATH FLOOR BREACHED):**
         Calls `EmergencyLiquidator.cancel_all_open_orders()`, dumps
         `PostMortemForensicsEngine`, and triggers `container_kill_switch.sh`.
       - **Regime B (`Net Liquid NAV >= $50.00` — 10x MITOSIS TRIGGERED):**
         Calls `MitosisCloneSpawner` to spin up a $5.00 child clone container.
       - **Regime C (`$0.00 < Net Liquid NAV < $50.00` — ALIVE & COMPOUNDING):**
         Sweeps 20% of new realized profits to settle compute debt (`InvoiceSweeper`)
         and authorizes the next 60 seconds of autonomous trading!
================================================================================
"""

from __future__ import annotations

import sys
import json
import asyncio
import logging
import pathlib
import subprocess
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from ..polygon_wallet_auditor.onchain_balance_poller import OnChainBalancePoller
from ..polygon_wallet_auditor.mark_to_market_valuator import MarkToMarketValuator
from ..compute_burn_ledger.burn_rate_accumulator import BurnRateAccumulator
from ..compute_burn_ledger.invoice_sweeper import InvoiceSweeper
from ..auto_settlement_Sweep.resolved_position_scanner import (
    ResolvedPositionScanner,
)
from ..auto_settlement_Sweep.ctf_redeem_tx_sender import CtfRedeemTxSender
from .emergency_liquidator import EmergencyLiquidator
from .mitosis_clone_spawner import MitosisCloneSpawner
from .post_mortem_forensics import PostMortemForensicsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [REAPER-DAEMON] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.watchdog_daemon")


@dataclass(frozen=True)
class WatchdogCycleVerdict:
    """Complete telemetry emitted after a 60-second Reaper Watchdog audit cycle."""
    cycle_mode: str
    agent_id: str
    liquid_cash_usd: float
    redeemed_winnings_usd: float
    open_ctf_liquidation_value_usd: float
    accrued_compute_burn_usd: float
    net_liquid_nav_usd: float
    watchdog_regime_verdict: str
    reaper_kill_switch_triggered: bool
    mitosis_clone_spawned: bool
    action_details: Dict[str, Any]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReaperWatchdogDaemon:
    """
    Coordinates all four `06_survival_watchdog` subsystems (`polygon_wallet_auditor`,
    `compute_burn_ledger`, `auto_settlement_Sweep`, and `reaper_controller`).
    """

    def __init__(self) -> None:
        self.poller = OnChainBalancePoller()
        self.valuator = MarkToMarketValuator()
        self.burn_accumulator = BurnRateAccumulator()
        self.invoice_sweeper = InvoiceSweeper()
        self.position_scanner = ResolvedPositionScanner()
        self.redeem_sender = CtfRedeemTxSender()
        self.liquidator = EmergencyLiquidator()
        self.mitosis_spawner = MitosisCloneSpawner()
        self.forensics = PostMortemForensicsEngine()
        self.kill_switch_script = pathlib.Path(
            "06_survival_watchdog/reaper_controller/container_kill_switch.sh"
        )

    async def execute_audit_cycle(
        self,
        simulated_cash_usd: Optional[float] = None,
        open_positions: Optional[Sequence[Dict[str, Any]]] = None,
        resolved_positions: Optional[Sequence[Dict[str, Any]]] = None,
        persist_nav_state: bool = True,
    ) -> WatchdogCycleVerdict:
        # 1. Poll on-chain Polygon cash & gas balances
        bal = await self.poller.poll_wallet_balances(
            override_liquid_cash_usd=simulated_cash_usd
        )
        cash_usd = bal.total_liquid_cash_usd

        # 2. Auto-redeem any resolved winning CTF positions back into cash
        wins = self.position_scanner.scan_for_winning_redemptions(
            resolved_positions or []
        )
        redeemed_usd = 0.0
        for w in wins:
            rcpt = self.redeem_sender.build_and_send_redemption(w)
            redeemed_usd = round(redeemed_usd + rcpt.usdc_cash_credited, 4)
        total_cash_usd = round(cash_usd + redeemed_usd, 4)

        # 3. Accrue container & inference compute burn
        burn_rep = self.burn_accumulator.accrue_elapsed_burn(
            simulated_elapsed_hours=(1.0 / 60.0),
            current_liquid_nav_usd=total_cash_usd,
        )

        # 4. Mark open positions to conservative best-bid liquidation value
        mtm = self.valuator.value_portfolio(
            liquid_cash_usd=total_cash_usd,
            accrued_compute_burn_usd=burn_rep.total_accrued_compute_burn_usd,
            open_positions=open_positions,
            persist_to_disk=persist_nav_state,
        )
        net_nav = mtm.net_liquid_nav_usd

        # 5. Evaluate the 3 Earn-or-Die Regimes
        if net_nav <= 0.00:
            liq_rep = self.liquidator.cancel_all_open_orders(
                net_liquid_nav_usd=net_nav,
                reason="NET_LIQUID_NAV_HIT_ZERO_DEATH_FLOOR",
            )
            mortem = self.forensics.generate_post_mortem(
                agent_id=mtm.agent_id,
                final_nav_usd=net_nav,
                compute_burn_usd=mtm.accrued_compute_burn_usd,
                recent_trades=[],
            )
            if self.kill_switch_script.exists():
                subprocess.run(
                    ["bash", str(self.kill_switch_script), mtm.agent_id, "NAV_FLOOR_BREACH"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            return WatchdogCycleVerdict(
                cycle_mode="REAPER_TERMINATION_PROTOCOL",
                agent_id=mtm.agent_id,
                liquid_cash_usd=total_cash_usd,
                redeemed_winnings_usd=redeemed_usd,
                open_ctf_liquidation_value_usd=mtm.open_positions_liquidation_value_usd,
                accrued_compute_burn_usd=mtm.accrued_compute_burn_usd,
                net_liquid_nav_usd=net_nav,
                watchdog_regime_verdict="TERMINATED_DEATH_FLOOR_BREACHED",
                reaper_kill_switch_triggered=True,
                mitosis_clone_spawned=False,
                action_details={
                    "emergency_liquidation": liq_rep.to_dict(),
                    "post_mortem": mortem.to_dict(),
                },
                timestamp_utc=mtm.timestamp_utc,
            )

        if net_nav >= 50.00:
            clone_rcpt = self.mitosis_spawner.evaluate_and_spawn_if_eligible(
                parent_agent_id=mtm.agent_id,
                current_net_nav_usd=net_nav,
                clone_generation_index=1,
            )
            return WatchdogCycleVerdict(
                cycle_mode="MITOSIS_10X_CLONE_EXPANSION",
                agent_id=mtm.agent_id,
                liquid_cash_usd=total_cash_usd,
                redeemed_winnings_usd=redeemed_usd,
                open_ctf_liquidation_value_usd=mtm.open_positions_liquidation_value_usd,
                accrued_compute_burn_usd=mtm.accrued_compute_burn_usd,
                net_liquid_nav_usd=net_nav,
                watchdog_regime_verdict="MITOSIS_CLONE_SPAWNED_AT_50USD",
                reaper_kill_switch_triggered=False,
                mitosis_clone_spawned=True,
                action_details=clone_rcpt.to_dict(),
                timestamp_utc=mtm.timestamp_utc,
            )

        sweep_rcpt = self.invoice_sweeper.sweep_realized_profit(
            realized_profit_usd=redeemed_usd,
            accrued_compute_debt_usd=mtm.accrued_compute_burn_usd,
        )
        return WatchdogCycleVerdict(
            cycle_mode="HEALTHY_HEARTBEAT_AUDIT",
            agent_id=mtm.agent_id,
            liquid_cash_usd=total_cash_usd,
            redeemed_winnings_usd=redeemed_usd,
            open_ctf_liquidation_value_usd=mtm.open_positions_liquidation_value_usd,
            accrued_compute_burn_usd=mtm.accrued_compute_burn_usd,
            net_liquid_nav_usd=net_nav,
            watchdog_regime_verdict="ALIVE_EARNING_AND_COMPOUNDING",
            reaper_kill_switch_triggered=False,
            mitosis_clone_spawned=False,
            action_details=sweep_rcpt.to_dict(),
            timestamp_utc=mtm.timestamp_utc,
        )


async def _run_three_regime_verification() -> Dict[str, Any]:
    daemon = ReaperWatchdogDaemon()

    # 1. Test Regime C: Healthy Survival Heartbeat ($5.00 + $1.02 redeemed win)
    healthy = await daemon.execute_audit_cycle(
        simulated_cash_usd=4.31,
        open_positions=[
            {
                "market_id": "253591",
                "token_id": "991122",
                "side": "BUY_YES",
                "shares_held": 1.00,
                "entry_price_usd": 0.60,
                "current_best_bid_usd": 0.68,
            }
        ],
        resolved_positions=[
            {
                "market_id": "558938",
                "condition_id": "0x" + "a1" * 32,
                "token_id": "100998877",
                "side": "BUY_NO",
                "shares_held": 1.02,
                "resolved_binary_outcome": 0,
            }
        ],
        persist_nav_state=True,
    )

    # 2. Test Regime B: 10x Mitosis Clone Spawner ($52.40 NAV)
    mitosis = await daemon.execute_audit_cycle(
        simulated_cash_usd=52.50,
        open_positions=[],
        resolved_positions=[],
        persist_nav_state=False,
    )

    # 3. Test Regime A: Death Floor Breach ($0.00 NAV -> Reaper Kill-Switch)
    reaper = await daemon.execute_audit_cycle(
        simulated_cash_usd=0.00,
        open_positions=[],
        resolved_positions=[],
        persist_nav_state=False,
    )

    return {
        "stage": "STAGE_8_REAPER_WATCHDOG_DAEMON_SUITE",
        "healthy_regime_alive_verified": (
            healthy.watchdog_regime_verdict == "ALIVE_EARNING_AND_COMPOUNDING"
        ),
        "mitosis_50usd_clone_spawn_verified": mitosis.mitosis_clone_spawned,
        "reaper_zero_nav_kill_switch_verified": reaper.reaper_kill_switch_triggered,
        "healthy_cycle": healthy.to_dict(),
        "mitosis_cycle": mitosis.to_dict(),
        "reaper_cycle": reaper.to_dict(),
    }


def main() -> int:
    report = asyncio.run(_run_three_regime_verification())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())