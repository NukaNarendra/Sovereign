#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: RESOLVED WINNING POSITION SCANNER
File: 06_survival_watchdog/auto_settlement_Sweep/resolved_position_scanner.py
================================================================================
Detects UMA Optimistic Oracle v3 resolved Polymarket contracts where the $5.00
survival bot holds winning ERC-1155 Conditional Token Framework (`CTF`) shares,
queuing them for immediate on-chain redemption back into liquid pUSD/USDC cash.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [RESOLVED-SCANNER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.resolved_position_scanner")


@dataclass(frozen=True)
class RedeemableWinningPosition:
    """Winning CTF share position ready for `redeemPositions()` on Polygon."""
    market_id: str
    condition_id: str
    token_id: str
    winning_side: str
    winning_index_set: int
    shares_held: float
    expected_payout_usd: float
    collateral_token_address: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResolvedPositionScanner:
    """
    Scans the bot's held CTF positions against resolved market outcomes (`o in {0, 1}`)
    and identifies winning positions (`indexSet = 1` for YES, `indexSet = 2` for NO).
    """

    USDC_E_ADDRESS: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    def scan_for_winning_redemptions(
        self, held_positions: Sequence[Dict[str, Any]]
    ) -> List[RedeemableWinningPosition]:
        redeemable: List[RedeemableWinningPosition] = []
        for pos in held_positions:
            outcome = pos.get("resolved_binary_outcome")
            if outcome not in (0, 1):
                continue

            side = str(pos.get("side") or "BUY_YES").upper()
            shares = float(pos.get("shares_held") or 0.0)
            if shares <= 0.0:
                continue

            won_yes = ("YES" in side) and (int(outcome) == 1)
            won_no = ("NO" in side) and (int(outcome) == 0)

            if won_yes or won_no:
                idx_set = 1 if won_yes else 2
                redeemable.append(
                    RedeemableWinningPosition(
                        market_id=str(pos.get("market_id") or ""),
                        condition_id=str(
                            pos.get("condition_id") or ("0x" + "11" * 32)
                        ),
                        token_id=str(pos.get("token_id") or ""),
                        winning_side="YES" if won_yes else "NO",
                        winning_index_set=idx_set,
                        shares_held=round(shares, 4),
                        expected_payout_usd=round(shares * 1.00, 4),
                        collateral_token_address=self.USDC_E_ADDRESS,
                    )
                )
        return redeemable


def main() -> int:
    scanner = ResolvedPositionScanner()
    wins = scanner.scan_for_winning_redemptions(
        [
            {
                "market_id": "558938",
                "condition_id": "0x" + "a1" * 32,
                "token_id": "100998877",
                "side": "BUY_NO",
                "shares_held": 1.02,
                "resolved_binary_outcome": 0,
            }
        ]
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_8_RESOLVED_POSITION_SCANNER",
                "winning_positions_found": len(wins),
                "sample_redeemable": wins[0].to_dict() if wins else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())