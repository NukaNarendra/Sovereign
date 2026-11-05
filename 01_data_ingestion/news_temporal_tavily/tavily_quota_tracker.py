#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART D)
File: 01_data_ingestion/news_temporal_tavily/tavily_quota_tracker.py
================================================================================
Quota & Budget Governor for the Tavily Search API ($120 Free Project Plan).

Why This Module Is Mandatory for Zero-Cost Bootstrap:
  1. Your Student Offer provides 4 months free on the Tavily Project Plan
     ($30/month value x 4 months = $120 total, granting ~4,000 credits/month).
  2. In `search_depth="advanced"`, Tavily charges 2 credits per call; in
     `search_depth="basic"`, it charges 1 credit per call.
  3. If a naive loop fires 10,000 unthrottled requests during Stage 2 dataset
     backfilling, it will burn the entire month's quota in 45 minutes and leave
     0 credits for Stage 6 live trading (`tavily_live_search_mcp`).
  4. This module enforces two isolated daily reservoirs:
       - `HISTORICAL_INGESTION` : Capped at 100 credits/day (reserving capacity)
       - `LIVE_MCP_SURVIVAL_BOT`: Capped at 30 credits/day (triggered only when
                                  `websocket_anomaly_filter` detects a 3-sigma
                                  CLOB volume or odds spike)
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import logging
import pathlib
import argparse
import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [TAVILY-QUOTA] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.tavily_quota_tracker")


# ==============================================================================
# 1. USAGE MODES & PERSISTENT QUOTA LEDGER DATA STRUCTURES
# ==============================================================================
class TavilyUsageMode(str, Enum):
    """Separates offline dataset backfilling from live $5 survival bot queries."""
    HISTORICAL_INGESTION = "HISTORICAL_INGESTION"
    LIVE_MCP_SURVIVAL_BOT = "LIVE_MCP_SURVIVAL_BOT"


@dataclass
class TavilyQuotaState:
    """
    Tracks cumulative and daily search credit consumption across the 4-month
    ($120 total value) Tavily Project Plan.
    """
    plan_name: str = "Tavily Project Plan (4-Month Student Perk)"
    total_plan_value_usd: float = 120.00
    monthly_credit_limit: int = 4000
    credit_cost_usd_per_unit: float = 0.0075  # $30.00 / 4,000 credits
    historical_daily_cap_credits: int = 100
    live_mcp_daily_cap_credits: int = 30
    current_month_key: str = ""
    current_day_key: str = ""
    monthly_credits_used: int = 0
    daily_historical_credits_used: int = 0
    daily_live_mcp_credits_used: int = 0
    total_lifetime_credits_used: int = 0
    total_cache_hits_saved: int = 0
    last_updated_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE TAVILY QUOTA TRACKER & DETERMINISTIC QUERY CACHE
# ==============================================================================
class TavilyQuotaTracker:
    """
    Manages rate limits, daily credit reservoirs, and local SHA-256 response
    caching so identical market queries never consume duplicate Tavily credits.
    """

    def __init__(
        self,
        ledger_path: pathlib.Path = pathlib.Path(
            "data/raw_lake/news_temporal_tavily/tavily_quota_ledger.json"
        ),
        cache_dir: pathlib.Path = pathlib.Path(
            "data/raw_lake/news_temporal_tavily/query_cache"
        ),
        historical_daily_cap: int = 100,
        live_mcp_daily_cap: int = 30,
    ) -> None:
        self.ledger_path = ledger_path
        self.cache_dir = cache_dir
        self.historical_daily_cap = historical_daily_cap
        self.live_mcp_daily_cap = live_mcp_daily_cap
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_init_state()
        self._roll_date_windows_if_needed()

    def _load_or_init_state(self) -> TavilyQuotaState:
        """Loads persistent quota state from JSON or initializes a fresh ledger."""
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        month_key = now_dt.strftime("%Y-%m")
        day_key = now_dt.strftime("%Y-%m-%d")

        if self.ledger_path.exists():
            try:
                raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                state = TavilyQuotaState(**raw)
                return state
            except Exception as exc:
                logger.warning("Corrupt Tavily ledger (%s); re-initializing.", exc)

        state = TavilyQuotaState(
            historical_daily_cap_credits=self.historical_daily_cap,
            live_mcp_daily_cap_credits=self.live_mcp_daily_cap,
            current_month_key=month_key,
            current_day_key=day_key,
            last_updated_utc=now_dt.isoformat(),
        )
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[TavilyQuotaState] = None) -> None:
        """Atomically writes the quota state to disk."""
        target = state or self.state
        target.last_updated_utc = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        tmp_path = self.ledger_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(target.to_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(self.ledger_path)

    # --------------------------------------------------------------------------
    # Calendar Window Rollover & Budget Reservation Gate
    # --------------------------------------------------------------------------
    def _roll_date_windows_if_needed(self) -> None:
        """Resets daily or monthly counters when UTC date/month boundaries cross."""
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        month_key = now_dt.strftime("%Y-%m")
        day_key = now_dt.strftime("%Y-%m-%d")
        changed = False

        if self.state.current_month_key != month_key:
            logger.info(
                "New UTC billing month (%s -> %s): resetting monthly_credits_used.",
                self.state.current_month_key,
                month_key,
            )
            self.state.current_month_key = month_key
            self.state.monthly_credits_used = 0
            changed = True

        if self.state.current_day_key != day_key:
            logger.info(
                "New UTC day (%s -> %s): resetting daily Tavily reservoirs.",
                self.state.current_day_key,
                day_key,
            )
            self.state.current_day_key = day_key
            self.state.daily_historical_credits_used = 0
            self.state.daily_live_mcp_credits_used = 0
            changed = True

        if changed:
            self._save_state()

    def can_consume_credits(
        self,
        mode: TavilyUsageMode,
        search_depth: str = "basic",
    ) -> Tuple[bool, int, str]:
        """
        Verifies whether `credits_needed` (1 for `basic`, 2 for `advanced`) is
        permitted under both the monthly 4,000-credit cap and the mode's daily cap.
        Returns `(allowed, credits_cost, reason)`.
        """
        self._roll_date_windows_if_needed()
        cost = 2 if str(search_depth).lower() == "advanced" else 1

        if (self.state.monthly_credits_used + cost) > self.state.monthly_credit_limit:
            return False, cost, "MONTHLY_4000_CREDIT_CAP_REACHED"

        if mode == TavilyUsageMode.HISTORICAL_INGESTION:
            if (
                self.state.daily_historical_credits_used + cost
                > self.state.historical_daily_cap_credits
            ):
                return False, cost, "DAILY_HISTORICAL_INGESTION_CAP_REACHED"
        elif mode == TavilyUsageMode.LIVE_MCP_SURVIVAL_BOT:
            if (
                self.state.daily_live_mcp_credits_used + cost
                > self.state.live_mcp_daily_cap_credits
            ):
                return False, cost, "DAILY_LIVE_MCP_SURVIVAL_CAP_REACHED"

        return True, cost, "QUOTA_AVAILABLE"

    def record_credit_consumption(
        self,
        mode: TavilyUsageMode,
        search_depth: str = "basic",
    ) -> Dict[str, Any]:
        """Deducts consumed credits and updates the persistent JSON ledger."""
        allowed, cost, reason = self.can_consume_credits(mode, search_depth)
        if not allowed:
            raise RuntimeError(f"Tavily quota gate blocked consumption: {reason}")

        self.state.monthly_credits_used += cost
        self.state.total_lifetime_credits_used += cost
        if mode == TavilyUsageMode.HISTORICAL_INGESTION:
            self.state.daily_historical_credits_used += cost
        else:
            self.state.daily_live_mcp_credits_used += cost

        self._save_state()
        return self.get_quota_summary()

    # --------------------------------------------------------------------------
    # Content-Addressable SHA-256 Query Cache (Saves Credits on Re-Runs)
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_query_cache_key(
        query: str,
        cutoff_date_utc: str,
        topic: str = "news",
    ) -> str:
        raw = f"{query.strip().lower()}|{cutoff_date_utc.strip()}|{topic.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_response(
        self, query: str, cutoff_date_utc: str, topic: str = "news"
    ) -> Optional[List[Dict[str, Any]]]:
        key = self.compute_query_cache_key(query, cutoff_date_utc, topic)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                self.state.total_cache_hits_saved += 1
                self._save_state()
                return payload
            except Exception:
                return None
        return None

    def save_response_to_cache(
        self,
        query: str,
        cutoff_date_utc: str,
        results: List[Dict[str, Any]],
        topic: str = "news",
    ) -> None:
        key = self.compute_query_cache_key(query, cutoff_date_utc, topic)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def get_quota_summary(self) -> Dict[str, Any]:
        """Returns an auditable summary of remaining Tavily credits and USD value."""
        self._roll_date_windows_if_needed()
        remaining_month = max(
            0, self.state.monthly_credit_limit - self.state.monthly_credits_used
        )
        usd_consumed_month = round(
            self.state.monthly_credits_used * self.state.credit_cost_usd_per_unit, 4
        )
        return {
            "stage": "STAGE_2_PART_D_TAVILY_QUOTA_TRACKER",
            "plan_name": self.state.plan_name,
            "current_month_key": self.state.current_month_key,
            "current_day_key": self.state.current_day_key,
            "monthly_credits_used": self.state.monthly_credits_used,
            "monthly_credits_remaining": remaining_month,
            "monthly_virtual_usd_consumed": usd_consumed_month,
            "daily_historical_used": f"{self.state.daily_historical_credits_used}/{self.state.historical_daily_cap_credits}",
            "daily_live_mcp_used": f"{self.state.daily_live_mcp_credits_used}/{self.state.live_mcp_daily_cap_credits}",
            "total_cache_hits_saved": self.state.total_cache_hits_saved,
            "ledger_path": str(self.ledger_path),
        }


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Tavily $120 Free Plan Quota Tracker"
    )
    parser.add_argument(
        "--simulate-calls",
        type=int,
        default=0,
        help="Simulate N basic search calls to test quota accounting.",
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live_mcp"],
        default="historical",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracker = TavilyQuotaTracker()
    usage_mode = (
        TavilyUsageMode.HISTORICAL_INGESTION
        if args.mode == "historical"
        else TavilyUsageMode.LIVE_MCP_SURVIVAL_BOT
    )
    for _ in range(args.simulate_calls):
        tracker.record_credit_consumption(mode=usage_mode, search_depth="basic")

    print(json.dumps(tracker.get_quota_summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())