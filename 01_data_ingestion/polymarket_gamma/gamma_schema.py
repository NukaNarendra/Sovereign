#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART A)
File: 01_data_ingestion/polymarket_gamma/gamma_schema.py
================================================================================
Strict Pydantic v2 data schemas and deterministic outcome-resolution validators
for Polymarket's Gamma API (`https://gamma-api.polymarket.com`).

Why This Module Is Critical for GRPO Brier Score Training:
  1. Polymarket's raw Gamma API returns fields like `outcomes`, `outcomePrices`,
     and `clobTokenIds` as stringified JSON arrays (e.g. `'["Yes", "No"]'`) on
     some endpoints and native lists on others.
  2. To compute the verifiable Brier Score reward `R = -(p - o)^2` in Stage 5,
     we must deterministically verify that a resolved market settled cleanly to
     a binary ground-truth outcome `o in {0, 1}` (ignoring 50/50 voided markets
     or unresolved/ambiguous multi-outcome scalar markets unless decomposed).
  3. Tracks the collateral contract era (`LEGACY_USDC_E` vs `APRIL_2026_PUSD_V2`)
     so downstream order-book normalizers join the right Polygon subgraph events.
================================================================================
"""

from __future__ import annotations

import re
import json
import math
import logging
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

logger = logging.getLogger("sovereign_prob_engine.gamma_schema")

# Regular expression for validating 32-byte hex Condition IDs and Question IDs
HEX_32BYTE_REGEX = re.compile(r"^0x[a-fA-F0-9]{64}$")
# Regular expression for validating 20-byte EVM contract addresses on Polygon
EVM_ADDRESS_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")


# ==============================================================================
# 1. ENUMERATIONS FOR MARKET RESOLUTION & COLLATERAL REGIMES
# ==============================================================================
class MarketResolutionStatus(str, Enum):
    """
    Canonical lifecycle state of a Polymarket binary condition.
    Only markets with `RESOLVED_BINARY_VALID` enter Stage 5 GRPO Brier RL.
    """
    ACTIVE_TRADING = "ACTIVE_TRADING"
    CLOSED_PENDING_UMA = "CLOSED_PENDING_UMA"
    DISPUTED_ON_UMA = "DISPUTED_ON_UMA"
    RESOLVED_BINARY_VALID = "RESOLVED_BINARY_VALID"
    RESOLVED_VOID_50_50 = "RESOLVED_VOID_50_50"
    INVALID_OR_NON_BINARY = "INVALID_OR_NON_BINARY"


class CollateralEra(str, Enum):
    """
    Identifies whether a market settled under legacy bridged USDC.e or the
    April 2026 Polymarket CLOB v2 / pUSD architecture on Polygon Chain ID 137.
    """
    LEGACY_USDC_E = "LEGACY_USDC_E"
    APRIL_2026_PUSD_V2 = "APRIL_2026_PUSD_V2"
    NEG_RISK_ADAPTER = "NEG_RISK_ADAPTER"


# ==============================================================================
# 2. SUB-MODELS FOR TAGS & CLOB OUTCOME TOKEN PAIRS
# ==============================================================================
class GammaTagModel(BaseModel):
    """Represents a topic tag attached to a Polymarket event or market."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    tag_id: str = Field(default="unknown", alias="id")
    label: str = Field(default="Uncategorized")
    slug: str = Field(default="uncategorized")

    @field_validator("tag_id", mode="before")
    @classmethod
    def _coerce_id_to_str(cls, val: Any) -> str:
        return str(val) if val is not None else "unknown"


class ClobTokenPair(BaseModel):
    """
    Stores the ERC-1155 Conditional Token Framework (CTF) token IDs for the
    binary `YES` (index 0) and `NO` (index 1) legs on the Polymarket CLOB.
    """
    model_config = ConfigDict(extra="ignore", frozen=True)

    yes_token_id: str = Field(..., description="CLOB ERC-1155 token ID for YES outcome")
    no_token_id: str = Field(..., description="CLOB ERC-1155 token ID for NO outcome")


# ==============================================================================
# 3. CORE GAMMA MARKET RECORD SCHEMA (WITH BRIER GROUND-TRUTH DERIVATION)
# ==============================================================================
class GammaMarketRecord(BaseModel):
    """
    Normalized, strictly typed representation of a single Polymarket binary
    market, ready for Apache Parquet serialization and Stage 3 temporal alignment.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # --- Primary Identifiers ---
    market_id: str = Field(..., alias="id")
    condition_id: str = Field(default="", alias="conditionId")
    question_id: str = Field(default="", alias="questionID")
    slug: str = Field(default="")

    # --- Question & UMA Ancillary Resolution Text ---
    question: str = Field(..., min_length=3)
    description: str = Field(default="", description="Full UMA resolution rules and source of truth")
    resolution_source: str = Field(default="", alias="resolutionSource")

    # --- Temporal Lifecycle Bounds (Normalized to UTC) ---
    created_at_utc: Optional[datetime.datetime] = Field(default=None, alias="createdAt")
    start_date_utc: Optional[datetime.datetime] = Field(default=None, alias="startDate")
    end_date_utc: Optional[datetime.datetime] = Field(default=None, alias="endDate")
    closed_time_utc: Optional[datetime.datetime] = Field(default=None, alias="closedTime")

    # --- Market State Flags ---
    active: bool = Field(default=False)
    closed: bool = Field(default=False)
    archived: bool = Field(default=False)
    neg_risk: bool = Field(default=False, alias="negRisk")
    neg_risk_market_id: Optional[str] = Field(default=None, alias="negRiskMarketID")

    # --- Microstructure & Liquidity Parameters ---
    volume_usd: float = Field(default=0.0, alias="volumeNum")
    liquidity_usd: float = Field(default=0.0, alias="liquidityNum")
    order_price_min_tick_size: float = Field(default=0.01, alias="orderPriceMinTickSize")
    order_min_size: float = Field(default=5.0, alias="orderMinSize")
    maker_base_fee_bps: float = Field(default=0.0, alias="makerBaseFee")
    taker_base_fee_bps: float = Field(default=0.0, alias="takerBaseFee")

    # --- Outcomes, Prices, and CLOB Token IDs ---
    outcomes: List[str] = Field(default_factory=list)
    outcome_prices: List[float] = Field(default_factory=list, alias="outcomePrices")
    clob_token_ids: List[str] = Field(default_factory=list, alias="clobTokenIds")
    tags: List[GammaTagModel] = Field(default_factory=list)

    # --- Derived Quantitative Fields for GRPO Training ---
    canonical_category: str = Field(default="OTHER")
    collateral_era: CollateralEra = Field(default=CollateralEra.LEGACY_USDC_E)
    resolution_status: MarketResolutionStatus = Field(
        default=MarketResolutionStatus.ACTIVE_TRADING
    )
    resolved_binary_outcome: Optional[int] = Field(
        default=None,
        description="Ground truth o in {0, 1} where 1 = YES won, 0 = NO won, None = unresolved/void",
    )

    # --------------------------------------------------------------------------
    # Field Validators (Handling Polymarket's Stringified JSON Quirks)
    # --------------------------------------------------------------------------
    @field_validator("market_id", "condition_id", "question_id", mode="before")
    @classmethod
    def _coerce_ids_to_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("outcomes", "clob_token_ids", mode="before")
    @classmethod
    def _parse_json_string_list(cls, v: Any) -> List[str]:
        """Handles both `'["Yes", "No"]'` JSON strings and native Python lists."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v]
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            except json.JSONDecodeError:
                return [s.strip() for s in stripped.split(",") if s.strip()]
        return []

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def _parse_outcome_prices(cls, v: Any) -> List[float]:
        """Parses stringified float arrays like `'["0.999", "0.001"]'` into floats."""
        raw_list: List[Any] = []
        if isinstance(v, list):
            raw_list = v
        elif isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v.strip())
                if isinstance(parsed, list):
                    raw_list = parsed
            except json.JSONDecodeError:
                raw_list = v.split(",")

        floats: List[float] = []
        for item in raw_list:
            try:
                val = float(item)
                floats.append(0.0 if math.isnan(val) or math.isinf(val) else val)
            except (ValueError, TypeError):
                floats.append(0.0)
        return floats

    @field_validator(
        "created_at_utc",
        "start_date_utc",
        "end_date_utc",
        "closed_time_utc",
        mode="before",
    )
    @classmethod
    def _parse_iso_datetime_utc(cls, v: Any) -> Optional[datetime.datetime]:
        """Normalizes any ISO-8601 or epoch timestamp strictly to timezone-aware UTC."""
        if v is None or v == "":
            return None
        if isinstance(v, datetime.datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=datetime.timezone.utc)
            return v.astimezone(datetime.timezone.utc)
        if isinstance(v, (int, float)):
            return datetime.datetime.fromtimestamp(float(v), tz=datetime.timezone.utc)
        if isinstance(v, str):
            cleaned = v.strip().replace("Z", "+00:00")
            try:
                dt = datetime.datetime.fromisoformat(cleaned)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.astimezone(datetime.timezone.utc)
            except ValueError:
                # Fallback for space-separated dates without full ISO format
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        dt = datetime.datetime.strptime(v.strip()[:19], fmt)
                        return dt.replace(tzinfo=datetime.timezone.utc)
                    except ValueError:
                        continue
        return None

    # --------------------------------------------------------------------------
    # Post-Initialization Model Validator: Ground-Truth Outcome & Era Deduction
    # --------------------------------------------------------------------------
    @model_validator(mode="after")
    def _derive_resolution_and_collateral_era(self) -> "GammaMarketRecord":
        """
        Determines:
          1. Whether the market is a clean binary `Yes`/`No` market.
          2. If `closed=True`, whether `outcome_prices` settled to `[1.0, 0.0]`
             (YES won -> `o = 1`) or `[0.0, 1.0]` (NO won -> `o = 0`), or `[0.5, 0.5]`
             (Voided/50-50 split -> excluded from Brier training).
          3. Which Polygon collateral regime applies based on creation date and
             `neg_risk` flags.
        """
        # 1. Determine Collateral Era (April 1, 2026 cutoff for CLOB v2 pUSD)
        april_2026_cutoff = datetime.datetime(2026, 4, 1, tzinfo=datetime.timezone.utc)
        ref_dt = self.created_at_utc or self.start_date_utc
        if self.neg_risk:
            self.collateral_era = CollateralEra.NEG_RISK_ADAPTER
        elif ref_dt is not None and ref_dt >= april_2026_cutoff:
            self.collateral_era = CollateralEra.APRIL_2026_PUSD_V2
        else:
            self.collateral_era = CollateralEra.LEGACY_USDC_E

        # 2. Check Binary Structure (2 outcomes + 2 CLOB token IDs)
        if len(self.outcomes) != 2 or len(self.clob_token_ids) != 2:
            self.resolution_status = MarketResolutionStatus.INVALID_OR_NON_BINARY
            self.resolved_binary_outcome = None
            return self

        # 3. Derive Ground-Truth Binary Outcome o in {0, 1} if market is closed
        if not self.closed:
            self.resolution_status = MarketResolutionStatus.ACTIVE_TRADING
            self.resolved_binary_outcome = None
            return self

        if len(self.outcome_prices) == 2:
            p_yes, p_no = self.outcome_prices[0], self.outcome_prices[1]
            # Threshold >= 0.985 accounts for final redemption rounding in Gamma API
            if p_yes >= 0.985 and p_no <= 0.015:
                self.resolution_status = MarketResolutionStatus.RESOLVED_BINARY_VALID
                self.resolved_binary_outcome = 1
            elif p_no >= 0.985 and p_yes <= 0.015:
                self.resolution_status = MarketResolutionStatus.RESOLVED_BINARY_VALID
                self.resolved_binary_outcome = 0
            elif 0.48 <= p_yes <= 0.52 and 0.48 <= p_no <= 0.52:
                self.resolution_status = MarketResolutionStatus.RESOLVED_VOID_50_50
                self.resolved_binary_outcome = None
            else:
                self.resolution_status = MarketResolutionStatus.CLOSED_PENDING_UMA
                self.resolved_binary_outcome = None
        else:
            self.resolution_status = MarketResolutionStatus.CLOSED_PENDING_UMA
            self.resolved_binary_outcome = None

        return self

    def get_clob_token_pair(self) -> Optional[ClobTokenPair]:
        """Returns the typed `(yes_token_id, no_token_id)` pair if valid binary."""
        if len(self.clob_token_ids) == 2:
            return ClobTokenPair(
                yes_token_id=self.clob_token_ids[0],
                no_token_id=self.clob_token_ids[1],
            )
        return None

    def is_eligible_for_grpo_brier_training(self, min_volume_usd: float = 1000.0) -> bool:
        """
        Gate check used by Stage 2 & Stage 3 to ensure only liquid, cleanly
        resolved binary markets with non-empty UMA rules enter the training set.
        """
        return (
            self.resolution_status == MarketResolutionStatus.RESOLVED_BINARY_VALID
            and self.resolved_binary_outcome in (0, 1)
            and self.volume_usd >= min_volume_usd
            and len(self.condition_id) > 0
            and len(self.description.strip()) >= 15
        )

    def to_parquet_dict(self) -> Dict[str, Any]:
        """Flattens the record into a columnar dictionary ready for Polars/PyArrow."""
        token_pair = self.get_clob_token_pair()
        return {
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "slug": self.slug,
            "question": self.question,
            "description": self.description,
            "resolution_source": self.resolution_source,
            "created_at_utc": self.created_at_utc.isoformat() if self.created_at_utc else None,
            "start_date_utc": self.start_date_utc.isoformat() if self.start_date_utc else None,
            "end_date_utc": self.end_date_utc.isoformat() if self.end_date_utc else None,
            "closed_time_utc": self.closed_time_utc.isoformat() if self.closed_time_utc else None,
            "active": self.active,
            "closed": self.closed,
            "neg_risk": self.neg_risk,
            "volume_usd": float(self.volume_usd),
            "liquidity_usd": float(self.liquidity_usd),
            "tick_size": float(self.order_price_min_tick_size),
            "min_order_size": float(self.order_min_size),
            "yes_token_id": token_pair.yes_token_id if token_pair else None,
            "no_token_id": token_pair.no_token_id if token_pair else None,
            "outcome_yes_label": self.outcomes[0] if len(self.outcomes) >= 1 else "Yes",
            "outcome_no_label": self.outcomes[1] if len(self.outcomes) >= 2 else "No",
            "final_yes_price": self.outcome_prices[0] if len(self.outcome_prices) >= 1 else None,
            "final_no_price": self.outcome_prices[1] if len(self.outcome_prices) >= 2 else None,
            "tags_json": json.dumps([t.label for t in self.tags]),
            "canonical_category": self.canonical_category,
            "collateral_era": self.collateral_era.value,
            "resolution_status": self.resolution_status.value,
            "resolved_binary_outcome": self.resolved_binary_outcome,
        }


# ==============================================================================
# 4. GAMMA EVENT GROUP & CRAWLER CHECKPOINT STATE MODELS
# ==============================================================================
class GammaEventRecord(BaseModel):
    """
    Represents a top-level Polymarket Event (which may bundle multiple sibling
    binary markets, e.g., 'Fed Interest Rate Decision — May 2026').
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    event_id: str = Field(..., alias="id")
    ticker: str = Field(default="")
    slug: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    closed: bool = Field(default=False)
    volume_usd: float = Field(default=0.0, alias="volume")
    tags: List[GammaTagModel] = Field(default_factory=list)
    markets: List[GammaMarketRecord] = Field(default_factory=list)


class CrawlerCheckpointState(BaseModel):
    """Tracks pagination offset and statistics for fault-tolerant crawling."""
    last_completed_offset: int = 0
    total_raw_fetched: int = 0
    total_valid_binary_resolved: int = 0
    shards_written: int = 0
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


if __name__ == "__main__":
    # Self-test of GammaMarketRecord parsing and Brier ground-truth resolution
    sample_raw = {
        "id": "512948",
        "conditionId": "0x" + "a1" * 32,
        "questionID": "0x" + "b2" * 32,
        "slug": "will-bitcoin-exceed-100k-before-july-2026",
        "question": "Will Bitcoin exceed $100,000 before July 1, 2026?",
        "description": "This market resolves to Yes if the Binance BTC/USDT 1-minute candle high reaches or exceeds $100,000.00 before July 1, 2026 23:59:59 UTC.",
        "createdAt": "2026-04-05T12:00:00Z",
        "endDate": "2026-07-01T23:59:59Z",
        "closedTime": "2026-05-14T08:30:00Z",
        "closed": True,
        "volumeNum": 1452890.50,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["1.0", "0.0"]',
        "clobTokenIds": '["9918273645192837465", "1129384756102938475"]',
        "tags": [{"id": "21", "label": "Crypto", "slug": "crypto"}],
    }
    record = GammaMarketRecord.model_validate(sample_raw)
    print(json.dumps(record.to_parquet_dict(), indent=2))