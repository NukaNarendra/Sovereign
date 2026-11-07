#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: MICRO-BANKROLL TICK & LOT ROUNDER
File: 05_execution_risk_engine/fractional_kelly_sizer/micro_bankroll_rounder.py
================================================================================
Quantizes continuous Fractional Kelly dollar stakes and limit prices to exact
Polymarket CLOB v2 contract specifications (`tick_size` in `{0.01, 0.001}` and
6-decimal USDC.e / pUSD atomic integer amounts) on $5.00 survival bankrolls.
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
    format="%(asctime)s | %(levelname)-8s | [MICRO-ROUNDER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.micro_bankroll_rounder")

ATOMIC_DECIMALS: int = 6
ATOMIC_SCALE: int = 10 ** ATOMIC_DECIMALS


@dataclass(frozen=True)
class RoundedMicroOrderSpec:
    """
    Tick-aligned and 6-decimal atomic-unit aligned order parameters ready for
    EIP-712 cryptographic signing on Polygon Chain ID 137.
    """
    market_id: str
    token_id: str
    side: str
    tick_size: float
    rounded_limit_price: float
    rounded_share_quantity: float
    exact_notional_usd: float
    maker_amount_atomic_6dec: int
    taker_amount_atomic_6dec: int
    is_valid_executable_lot: bool
    lot_adjustment_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MicroBankrollRounder:
    """
    Rounds limit prices to exact `tick_size` increments (`floor` for BUY orders
    so we never overpay the limit hurdle) and computes exact 6-decimal integer
    `makerAmount` and `takerAmount` values for `CTFExchange`.
    """

    def __init__(self, min_survival_notional_usd: float = 0.25) -> None:
        self.min_survival_notional_usd = float(min_survival_notional_usd)

    def round_order_for_clob(
        self,
        market_id: str,
        token_id: str,
        side: str,
        raw_limit_price: float,
        target_stake_usd: float,
        tick_size: float = 0.01,
    ) -> RoundedMicroOrderSpec:
        tick = max(0.0001, float(tick_size))
        # Floor limit price to nearest tick so BUY orders never cross above EV ceiling
        ticks_count = math.floor((float(raw_limit_price) + 1e-9) / tick)
        rounded_price = round(max(tick, min(1.0 - tick, ticks_count * tick)), 4)

        if float(target_stake_usd) < self.min_survival_notional_usd:
            return RoundedMicroOrderSpec(
                market_id=market_id,
                token_id=token_id,
                side=side,
                tick_size=tick,
                rounded_limit_price=rounded_price,
                rounded_share_quantity=0.0,
                exact_notional_usd=0.0,
                maker_amount_atomic_6dec=0,
                taker_amount_atomic_6dec=0,
                is_valid_executable_lot=False,
                lot_adjustment_note="STAKE_BELOW_MIN_SURVIVAL_NOTIONAL",
            )

        raw_shares = float(target_stake_usd) / rounded_price
        # Round shares to 2 decimal places so price * shares has clean 6-decimal representation
        rounded_shares = round(math.floor(raw_shares * 100.0) / 100.0, 2)
        exact_usd = round(rounded_shares * rounded_price, 6)

        # On Polymarket CTF Exchange, for a BUY order:
        # maker gives collateral (USDC/pUSD = exact_usd * 1e6) and receives conditional tokens (shares * 1e6)
        maker_atomic = int(round(exact_usd * ATOMIC_SCALE))
        taker_atomic = int(round(rounded_shares * ATOMIC_SCALE))

        valid_lot = (rounded_shares > 0.0) and (exact_usd >= self.min_survival_notional_usd)
        return RoundedMicroOrderSpec(
            market_id=market_id,
            token_id=token_id,
            side=side,
            tick_size=tick,
            rounded_limit_price=rounded_price,
            rounded_share_quantity=rounded_shares,
            exact_notional_usd=exact_usd,
            maker_amount_atomic_6dec=maker_atomic,
            taker_amount_atomic_6dec=taker_atomic,
            is_valid_executable_lot=valid_lot,
            lot_adjustment_note="TICK_AND_6DEC_ATOMIC_ALIGNED",
        )


def main() -> int:
    rounder = MicroBankrollRounder()
    spec = rounder.round_order_for_clob(
        market_id="558938",
        token_id="100998877",
        side="BUY_NO",
        raw_limit_price=0.6881,
        target_stake_usd=0.6975,
        tick_size=0.01,
    )
    print(json.dumps(spec.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())