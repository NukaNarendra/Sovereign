#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART B)
File: 01_data_ingestion/polymarket_clob_v2/pusd_migration_normalizer.py
================================================================================
Unifies Historical & Live Polymarket Trades Across All Three Polygon (Chain 137)
Collateral & Exchange Regimes:
  1. `LEGACY_CTF_USDC_E`    : Pre-April 2026 trades settled against Bridged
                              USDC.e (`0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`)
                              on CTF Exchange v1 (`0x4bFb41d5...`).
  2. `APRIL_2026_PUSD_V2`   : Post-April 2026 CLOB v2 trades settled against
                              wrapped native USDC / pUSD collateral on CTF
                              Exchange v2 (`0xC5d563A3...`).
  3. `NEG_RISK_MULTI_LEG`   : Mutually-exclusive multi-candidate markets routed
                              through the NegRisk Adapter (`0xd91E80cF...`) where
                              `1 NO share` in candidate A can be converted into
                              `YES shares` of sibling candidates.

Why This Matters for Training & Execution:
  - Raw on-chain OrderFilled events and historical archives mix 6-decimal token
    amounts, complementary `BUY_NO` vs `SELL_YES` fills, and NegRisk synthetic
    conversions. This normalizer maps every fill into a single canonical
    `YES-equivalent probability in [0.0, 1.0]` and `6-decimal USD notional`.
================================================================================
"""

from __future__ import annotations

import json
import logging
import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("sovereign_prob_engine.pusd_migration_normalizer")

# Official Polygon PoS Mainnet (Chain ID 137) Contract Addresses
POLYGON_CONTRACTS: Dict[str, str] = {
    "USDC_E_LEGACY": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
    "NATIVE_USDC": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
    "CTF_CONDITIONAL_TOKENS": "0x4d97dcd97ec945f40cf65f87097ace5ea0476045",
    "CTF_EXCHANGE_V1": "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "CTF_EXCHANGE_CLOB_V2": "0xc5d563a36ae78145c45a50134d48a1215220f80a",
    "NEG_RISK_ADAPTER": "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296",
}

COLLATERAL_DECIMALS: int = 6
DECIMAL_SCALE: float = float(10 ** COLLATERAL_DECIMALS)
APRIL_2026_EPOCH_UTC: int = int(
    datetime.datetime(2026, 4, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
)


# ==============================================================================
# 1. SETTLEMENT REGIME ENUMERATION & NORMALIZED FILL RECORD
# ==============================================================================
class SettlementRegime(str, Enum):
    """Identifies the Polygon smart-contract settlement path for a trade/book."""
    LEGACY_CTF_USDC_E = "LEGACY_CTF_USDC_E"
    APRIL_2026_PUSD_V2 = "APRIL_2026_PUSD_V2"
    NEG_RISK_MULTI_LEG = "NEG_RISK_MULTI_LEG"


@dataclass(frozen=True)
class NormalizedFillRecord:
    """
    Canonical representation of an on-chain or CLOB trade mapped strictly to its
    implied `YES` probability (`canonical_yes_price`) and USD volume.
    """
    market_id: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    timestamp_sec: int
    settlement_regime: SettlementRegime
    raw_traded_token_id: str
    is_yes_leg_direct: bool
    raw_execution_price: float
    canonical_yes_price: float
    canonical_no_price: float
    share_quantity: float
    notional_usd: float
    fee_usd: float
    verifying_exchange_contract: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["settlement_regime"] = self.settlement_regime.value
        return payload


# ==============================================================================
# 2. PUSD / USDC.E & NEGRISK MIGRATION NORMALIZER
# ==============================================================================
class PusdMigrationNormalizer:
    """
    Normalizes raw Polygon `OrderFilled` events, CLOB trade prints, and Gamma
    collateral tags into a unified 6-decimal USD and YES-probability basis.
    """

    def __init__(self, depeg_haircut_bps: float = 0.0) -> None:
        self.depeg_haircut_bps = depeg_haircut_bps

    def detect_regime(
        self,
        timestamp_sec: int,
        neg_risk_flag: bool = False,
        exchange_address: Optional[str] = None,
    ) -> SettlementRegime:
        """
        Determines which Polygon smart-contract regime governs a market snapshot
        or fill based on its contract address, `neg_risk` flag, and timestamp.
        """
        if neg_risk_flag:
            return SettlementRegime.NEG_RISK_MULTI_LEG

        if exchange_address:
            clean_addr = exchange_address.strip().lower()
            if clean_addr == POLYGON_CONTRACTS["NEG_RISK_ADAPTER"]:
                return SettlementRegime.NEG_RISK_MULTI_LEG
            if clean_addr == POLYGON_CONTRACTS["CTF_EXCHANGE_CLOB_V2"]:
                return SettlementRegime.APRIL_2026_PUSD_V2
            if clean_addr == POLYGON_CONTRACTS["CTF_EXCHANGE_V1"]:
                return SettlementRegime.LEGACY_CTF_USDC_E

        if timestamp_sec >= APRIL_2026_EPOCH_UTC:
            return SettlementRegime.APRIL_2026_PUSD_V2
        return SettlementRegime.LEGACY_CTF_USDC_E

    def get_verifying_contract_for_regime(self, regime: SettlementRegime) -> str:
        """
        Returns the exact checksummed EIP-712 verifying contract address on
        Polygon Chain ID 137 required for order validation.
        """
        if regime in (SettlementRegime.APRIL_2026_PUSD_V2, SettlementRegime.NEG_RISK_MULTI_LEG):
            return "0xC5d563A36AE78145C45a50134d48A1215220f80a"
        return "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    @staticmethod
    def raw_atomic_units_to_float(raw_amount: int | float | str) -> float:
        """
        Converts 6-decimal Polygon USDC.e / pUSD / CTF atomic integer units
        (e.g. `5000000` -> `5.00`) into standard floats. If the input is
        already a decimal float below `100,000` with fractional digits, preserves it.
        """
        val = float(raw_amount)
        if isinstance(raw_amount, int) or (val >= 1_000_000.0 and val.is_integer()):
            return round(val / DECIMAL_SCALE, 6)
        return round(val, 6)

    def normalize_trade_fill(
        self,
        market_id: str,
        condition_id: str,
        yes_token_id: str,
        no_token_id: str,
        traded_token_id: str,
        raw_price: float,
        raw_size_shares: float,
        timestamp_sec: int,
        neg_risk: bool = False,
        fee_rate_bps: float = 0.0,
        exchange_address: Optional[str] = None,
    ) -> NormalizedFillRecord:
        """
        Converts any fill (whether executed on the `YES` token leg or the
        complementary `NO` token leg) into canonical `YES` implied probability:
          - If `traded_token_id == yes_token_id`: `p_yes = raw_price`
          - If `traded_token_id == no_token_id` : `p_yes = 1.0 - raw_price`
        """
        regime = self.detect_regime(
            timestamp_sec=timestamp_sec,
            neg_risk_flag=neg_risk,
            exchange_address=exchange_address,
        )
        clamped_price = max(0.001, min(0.999, float(raw_price)))
        is_yes_leg = str(traded_token_id).strip() == str(yes_token_id).strip()

        if is_yes_leg:
            canonical_yes = round(clamped_price, 6)
            canonical_no = round(1.0 - clamped_price, 6)
        else:
            canonical_no = round(clamped_price, 6)
            canonical_yes = round(1.0 - clamped_price, 6)

        shares = max(0.0, float(raw_size_shares))
        gross_notional_usd = shares * clamped_price
        haircut_multiplier = 1.0 - (self.depeg_haircut_bps / 10_000.0)
        net_notional_usd = round(gross_notional_usd * haircut_multiplier, 6)
        fee_usd = round(net_notional_usd * (max(0.0, fee_rate_bps) / 10_000.0), 6)

        return NormalizedFillRecord(
            market_id=str(market_id),
            condition_id=str(condition_id),
            yes_token_id=str(yes_token_id),
            no_token_id=str(no_token_id),
            timestamp_sec=int(timestamp_sec),
            settlement_regime=regime,
            raw_traded_token_id=str(traded_token_id),
            is_yes_leg_direct=is_yes_leg,
            raw_execution_price=round(clamped_price, 6),
            canonical_yes_price=canonical_yes,
            canonical_no_price=canonical_no,
            share_quantity=round(shares, 4),
            notional_usd=net_notional_usd,
            fee_usd=fee_usd,
            verifying_exchange_contract=self.get_verifying_contract_for_regime(regime),
        )


if __name__ == "__main__":
    # Self-test: Normalize a complementary NO-leg fill under April 2026 pUSD v2
    normalizer = PusdMigrationNormalizer()
    sample_fill = normalizer.normalize_trade_fill(
        market_id="2169995",
        condition_id="0x" + "c3" * 32,
        yes_token_id="10001",
        no_token_id="10002",
        traded_token_id="10002",  # Traded on the NO token at $0.72 -> implies YES = $0.28
        raw_price=0.72,
        raw_size_shares=150.0,
        timestamp_sec=1776500000,
        neg_risk=False,
        fee_rate_bps=0.0,
    )
    print(json.dumps(sample_fill.to_dict(), indent=2))