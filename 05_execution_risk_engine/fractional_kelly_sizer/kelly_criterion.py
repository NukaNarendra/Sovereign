#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: 0.25x FRACTIONAL KELLY CRITERION
File: 05_execution_risk_engine/fractional_kelly_sizer/kelly_criterion.py
================================================================================
Computes the optimal bankroll allocation fraction $f^*$ for binary prediction
market contracts with a `0.25x` ("Quarter-Kelly") variance dampener and a hard
`15.0%` (`0.15`) single-market exposure ceiling:

Mathematical Derivation for Binary Payoff Shares ($1.00 Payout):
  - When purchasing a binary share (`YES` or `NO`) at all-in cost $c \in (0, 1)$
    with true calibrated win probability $p_{\text{win}} > c$:
      * Net profit on win: $b = \frac{1.0 - c}{c}$ per dollar risked.
      * Full Kelly fraction of bankroll:
        $$f_{\text{full}}^* = p_{\text{win}} - \frac{1 - p_{\text{win}}}{b}
                            = \frac{p_{\text{win}} - c}{1.0 - c}$$
      * Quarter-Kelly dampened fraction ($c_{\text{kelly}} = 0.25$):
        $$f^* = \min\left(f_{\max},\; 0.25 \cdot \frac{p_{\text{win}} - c}{1.0 - c}\right)$$
  - Why Quarter-Kelly ($0.25\times$)?
    Full Kelly has a $1/3$ probability of halving the bankroll before doubling it.
    Quarter-Kelly captures $75\%$ of the long-run logarithmic growth rate while
    reducing bankroll variance by $93.75\%$—preventing our $5.00 survival bot
    from ever hitting the Reaper Watchdog death floor!
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [KELLY-SIZER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.kelly_criterion")


@dataclass(frozen=True)
class KellySizingResult:
    """
    Deterministic Quarter-Kelly sizing output for a single trade opportunity.
    """
    snapshot_uid: str
    market_id: str
    side: str
    win_probability: float
    all_in_entry_cost: float
    full_kelly_fraction: float
    dampened_quarter_kelly_fraction: float
    capped_bankroll_fraction: float
    liquid_bankroll_usd: float
    gas_reserve_locked_usd: float
    deployable_bankroll_usd: float
    recommended_stake_usd: float
    expected_log_growth_rate_bps: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FractionalKellyCriterion:
    """
    Calculates `f* = 0.25 * (p_win - c) / (1 - c)` with a 15% single-market cap
    and a mandatory `$0.35` POL gas reserve deduction on $5.00 bankrolls.
    """

    def __init__(
        self,
        kelly_dampener_c: float = 0.25,
        max_single_market_fraction: float = 0.15,
        gas_reserve_usd: float = 0.35,
    ) -> None:
        self.kelly_dampener_c = float(kelly_dampener_c)
        self.max_single_market_fraction = float(max_single_market_fraction)
        self.gas_reserve_usd = float(gas_reserve_usd)

    def compute_kelly_allocation(
        self,
        win_probability: float,
        all_in_entry_cost: float,
        liquid_bankroll_usd: float = 5.00,
        side: str = "BUY_YES",
        snapshot_uid: str = "SNAP-KELLY",
        market_id: str = "253591",
    ) -> KellySizingResult:
        p_win = max(0.0001, min(0.9999, float(win_probability)))
        c_entry = max(0.0010, min(0.9950, float(all_in_entry_cost)))
        deployable_usd = max(
            0.0, round(float(liquid_bankroll_usd) - self.gas_reserve_usd, 4)
        )

        if p_win <= c_entry or deployable_usd <= 0.0:
            return KellySizingResult(
                snapshot_uid=snapshot_uid,
                market_id=market_id,
                side="HOLD_ZERO_KELLY",
                win_probability=round(p_win, 4),
                all_in_entry_cost=round(c_entry, 4),
                full_kelly_fraction=0.0,
                dampened_quarter_kelly_fraction=0.0,
                capped_bankroll_fraction=0.0,
                liquid_bankroll_usd=round(float(liquid_bankroll_usd), 4),
                gas_reserve_locked_usd=self.gas_reserve_usd,
                deployable_bankroll_usd=deployable_usd,
                recommended_stake_usd=0.0,
                expected_log_growth_rate_bps=0.0,
            )

        full_kelly = (p_win - c_entry) / max(0.005, 1.0 - c_entry)
        quarter_kelly = self.kelly_dampener_c * full_kelly
        capped_frac = min(self.max_single_market_fraction, max(0.0, quarter_kelly))
        stake_usd = round(deployable_usd * capped_frac, 4)

        # Expected log bankroll growth E[ln(1 + f * R)]
        win_mult = (1.0 - c_entry) / c_entry
        g_rate = p_win * math.log(1.0 + capped_frac * win_mult) + (
            1.0 - p_win
        ) * math.log(max(1e-6, 1.0 - capped_frac))

        return KellySizingResult(
            snapshot_uid=snapshot_uid,
            market_id=market_id,
            side=side,
            win_probability=round(p_win, 4),
            all_in_entry_cost=round(c_entry, 4),
            full_kelly_fraction=round(full_kelly, 4),
            dampened_quarter_kelly_fraction=round(quarter_kelly, 4),
            capped_bankroll_fraction=round(capped_frac, 4),
            liquid_bankroll_usd=round(float(liquid_bankroll_usd), 4),
            gas_reserve_locked_usd=self.gas_reserve_usd,
            deployable_bankroll_usd=deployable_usd,
            recommended_stake_usd=stake_usd,
            expected_log_growth_rate_bps=round(g_rate * 10_000.0, 2),
        )


def main() -> int:
    sizer = FractionalKellyCriterion()
    res = sizer.compute_kelly_allocation(
        win_probability=0.9950,
        all_in_entry_cost=0.6881,
        liquid_bankroll_usd=5.00,
        side="BUY_NO",
        snapshot_uid="PIT-558938-T168H",
        market_id="558938",
    )
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())