#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART B)
File: 01_data_ingestion/polymarket_clob_v2/clob_v2_rest_client.py
================================================================================
High-Throughput Asynchronous Client for Polymarket's Central Limit Order Book
(`https://clob.polymarket.com`).

Why This Module Is Essential for GRPO Edge Training & $5 Survival Execution:
  1. In Stage 5 (`03_llm_training/stage3_grpo_brier_edge`), our reward function
     requires both the true outcome `o` AND the historical market implied
     probability `q_t` and spread `s` at timestamp `t`:
         R_total = -(p - o)^2 + lambda * I(|p - q_t| >= s) * PnL(p, q_t, o)
  2. Polymarket's `/prices-history` endpoint provides historical probability
     trajectories (`t`, `p`) for any ERC-1155 `yes_token_id` extracted in
     Stage 2 Part A (`gamma_resolved_shard_*.parquet`).
  3. Polymarket's `/book` and `/books` endpoints provide full Level-2 (`bids`
     and `asks`) liquidity ladders so our bot never places a $1.25 Kelly bet
     into an illiquid order book with a 12-cent spread.
================================================================================
"""

from __future__ import annotations

import sys
import time
import json
import math
import random
import asyncio
import logging
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [CLOB-V2-CLIENT] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.clob_v2_rest_client")


# ==============================================================================
# 1. PYDANTIC SCHEMAS FOR LEVEL-2 ORDER BOOKS & HISTORICAL PRICE TRAJECTORIES
# ==============================================================================
class OrderBookLevel(BaseModel):
    """
    Represents a single price level on the Polymarket CLOB (`bids` or `asks`).
    Prices are expressed in `[0.001, 0.999]` USD per conditional share; sizes
    are expressed in total conditional shares resting at that limit price.
    """
    model_config = ConfigDict(extra="ignore", frozen=True)

    price: float = Field(..., ge=0.0, le=1.0, description="Limit price in pUSD/USDC")
    size: float = Field(..., ge=0.0, description="Number of conditional outcome shares")

    @field_validator("price", "size", mode="before")
    @classmethod
    def _coerce_numeric(cls, val: Any) -> float:
        try:
            parsed = float(val)
            if math.isnan(parsed) or math.isinf(parsed) or parsed < 0.0:
                return 0.0
            return parsed
        except (ValueError, TypeError):
            return 0.0


class RawClobOrderBook(BaseModel):
    """
    Normalized Level-2 Order Book snapshot returned by `GET /book?token_id=...`.
    Automatically sorts `bids` in descending price order (best bid at index 0)
    and `asks` in ascending price order (best ask at index 0).
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    market_condition_id: str = Field(default="", alias="market")
    asset_token_id: str = Field(..., alias="asset_id")
    book_hash: str = Field(default="", alias="hash")
    timestamp_ms: int = Field(default=0, alias="timestamp")
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    min_order_size: float = Field(default=5.0)
    tick_size: float = Field(default=0.01)

    @field_validator("timestamp_ms", mode="before")
    @classmethod
    def _parse_timestamp_ms(cls, val: Any) -> int:
        if val is None or val == "":
            return int(time.time() * 1000)
        try:
            num = int(float(val))
            # Convert seconds to milliseconds if timestamp is a 10-digit epoch
            return num * 1000 if num < 10_000_000_000 else num
        except (ValueError, TypeError):
            return int(time.time() * 1000)

    def sort_book_levels(self) -> "RawClobOrderBook":
        """
        Ensures strict canonical L2 ordering:
          - `bids`: Highest price first (`best_bid = bids[0].price`)
          - `asks`: Lowest price first (`best_ask = asks[0].price`)
        """
        self.bids = sorted(
            [b for b in self.bids if b.size > 0.0 and 0.0 < b.price < 1.0],
            key=lambda x: x.price,
            reverse=True,
        )
        self.asks = sorted(
            [a for a in self.asks if a.size > 0.0 and 0.0 < a.price < 1.0],
            key=lambda x: x.price,
            reverse=False,
        )
        return self


class PriceHistoryPoint(BaseModel):
    """
    Represents a single historical probability tick from `GET /prices-history`:
      - `timestamp_sec`: UNIX epoch timestamp in seconds UTC (`t`)
      - `price`        : Implied probability in `[0.0, 1.0]` (`p`)
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)

    timestamp_sec: int = Field(..., alias="t")
    price: float = Field(..., alias="p")

    @field_validator("timestamp_sec", mode="before")
    @classmethod
    def _coerce_ts(cls, v: Any) -> int:
        val = int(float(v))
        return val // 1000 if val > 10_000_000_000 else val

    @field_validator("price", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> float:
        p = float(v)
        return max(0.0, min(1.0, p))


# ==============================================================================
# 2. CLIENT CONFIGURATION & ASYNC TOKEN-BUCKET RATE LIMITER
# ==============================================================================
@dataclass(frozen=True)
class ClobV2ClientConfig:
    """Configuration for `ClobV2AsyncRestClient`."""
    base_url: str = "https://clob.polymarket.com"
    requests_per_second: float = 8.0
    max_burst_capacity: int = 16
    request_timeout_sec: float = 20.0
    max_retries: int = 4
    backoff_base_sec: float = 1.0
    user_agent: str = "SovereignProbEngine/1.0 (CLOB-v2-Microstructure-Ingestor)"


class ClobRateLimiter:
    """Smooths concurrent CLOB API calls to prevent Cloudflare 429 throttling."""

    def __init__(self, rps: float, burst: int) -> None:
        self.rps = max(0.2, float(rps))
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._last_ts = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    float(self.burst),
                    self._tokens + (now - self._last_ts) * self.rps,
                )
                self._last_ts = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                sleep_for = (1.0 - self._tokens) / self.rps
            await asyncio.sleep(sleep_for)


# ==============================================================================
# 3. ASYNCHRONOUS POLYMARKET CLOB V2 REST CLIENT
# ==============================================================================
class ClobV2AsyncRestClient:
    """
    Production async client for fetching Level-2 order books, midpoints,
    spreads, and historical 1-min/5-min/1-hour price trajectories from
    `https://clob.polymarket.com`.
    """

    def __init__(self, config: Optional[ClobV2ClientConfig] = None) -> None:
        self.config = config or ClobV2ClientConfig()
        self._limiter = ClobRateLimiter(
            rps=self.config.requests_per_second,
            burst=self.config.max_burst_capacity,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self.total_requests: int = 0
        self.total_retries: int = 0

    async def __aenter__(self) -> "ClobV2AsyncRestClient":
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is None:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.config.user_agent,
                },
                timeout=httpx.Timeout(self.config.request_timeout_sec),
                limits=limits,
                follow_redirects=True,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
    ) -> Any:
        """Executes a rate-limited HTTP request against `clob.polymarket.com`."""
        if self._client is None:
            await self.open()
        assert self._client is not None

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            await self._limiter.acquire()
            self.total_requests += 1
            try:
                resp = await self._client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json_body,
                )
                if resp.status_code == 200:
                    return resp.json()

                # Handle 404/400 gracefully when querying very old resolved tokens
                if resp.status_code in (400, 404):
                    return None

                if resp.status_code in (429, 500, 502, 503, 504):
                    self.total_retries += 1
                    delay = self.config.backoff_base_sec * (2 ** (attempt - 1)) + random.uniform(0.05, 0.4)
                    logger.warning(
                        "HTTP %d on %s (attempt %d/%d); retrying in %.2fs...",
                        resp.status_code,
                        endpoint,
                        attempt,
                        self.config.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                last_err = exc
                self.total_retries += 1
                delay = self.config.backoff_base_sec * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                await asyncio.sleep(delay)

        logger.warning("Exhausted retries for CLOB endpoint %s: %s", endpoint, last_err)
        return None

    # --------------------------------------------------------------------------
    # Public Order Book & Historical Trajectory Endpoints
    # --------------------------------------------------------------------------
    async def fetch_order_book(self, token_id: str) -> Optional[RawClobOrderBook]:
        """
        Fetches the current Level-2 order book for a single ERC-1155 `token_id`
        via `GET /book?token_id=<token_id>`.
        """
        payload = await self._request("GET", "/book", params={"token_id": token_id})
        if not isinstance(payload, dict) or "bids" not in payload:
            return None
        payload.setdefault("asset_id", token_id)
        book = RawClobOrderBook.model_validate(payload)
        return book.sort_book_levels()

    async def fetch_order_books_batch(
        self, token_ids: Sequence[str]
    ) -> Dict[str, RawClobOrderBook]:
        """
        Fetches multiple L2 order books in a single round-trip via `POST /books`.
        """
        if not token_ids:
            return {}
        body = [{"token_id": str(tid)} for tid in token_ids]
        payload = await self._request("POST", "/books", json_body=body)
        results: Dict[str, RawClobOrderBook] = {}
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "asset_id" in item:
                    book = RawClobOrderBook.model_validate(item).sort_book_levels()
                    results[book.asset_token_id] = book
        return results

    async def fetch_midpoint_and_spread(self, token_id: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Queries `/midpoint` and `/spread` for quick lightweight polling during
        Stage 6 live execution.
        """
        mid_resp, spr_resp = await asyncio.gather(
            self._request("GET", "/midpoint", params={"token_id": token_id}),
            self._request("GET", "/spread", params={"token_id": token_id}),
        )
        mid_val = float(mid_resp["mid"]) if isinstance(mid_resp, dict) and "mid" in mid_resp else None
        spr_val = float(spr_resp["spread"]) if isinstance(spr_resp, dict) and "spread" in spr_resp else None
        return mid_val, spr_val

    async def fetch_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: str = "max",
        fidelity_minutes: int = 60,
    ) -> List[PriceHistoryPoint]:
        """
        Fetches historical implied probability ticks from `GET /prices-history`.
        Parameters:
          - `token_id`        : The CLOB ERC-1155 YES outcome token ID.
          - `start_ts`/`end_ts`: Optional UNIX epoch bounds in seconds.
          - `interval`        : `'1m'`, `'1h'`, `'6h'`, `'1d'`, `'1w'`, or `'max'`.
          - `fidelity_minutes`: Resolution in minutes (e.g. 1, 5, 15, 60).
        """
        params: Dict[str, Any] = {
            "market": str(token_id),
            "fidelity": int(fidelity_minutes),
        }
        if start_ts is not None and end_ts is not None:
            params["startTs"] = int(start_ts)
            params["endTs"] = int(end_ts)
        else:
            params["interval"] = interval

        payload = await self._request("GET", "/prices-history", params=params)
        if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
            return []

        points: List[PriceHistoryPoint] = []
        for raw_pt in payload["history"]:
            if isinstance(raw_pt, dict) and "t" in raw_pt and "p" in raw_pt:
                try:
                    points.append(PriceHistoryPoint.model_validate(raw_pt))
                except Exception:
                    continue

        points.sort(key=lambda pt: pt.timestamp_sec)
        return points


if __name__ == "__main__":
    # Offline + live self-test of RawClobOrderBook sorting and schema validation
    sample_book = RawClobOrderBook.model_validate(
        {
            "market": "0x" + "a1" * 32,
            "asset_id": "9918273645192837465",
            "timestamp": "1776000000000",
            "bids": [
                {"price": "0.61", "size": "250.0"},
                {"price": "0.63", "size": "180.5"},
                {"price": "0.60", "size": "900.0"},
            ],
            "asks": [
                {"price": "0.66", "size": "400.0"},
                {"price": "0.64", "size": "150.0"},
                {"price": "0.68", "size": "1200.0"},
            ],
        }
    ).sort_book_levels()
    print(
        json.dumps(
            {
                "asset_token_id": sample_book.asset_token_id,
                "best_bid": sample_book.bids[0].model_dump(),
                "best_ask": sample_book.asks[0].model_dump(),
                "bid_levels_count": len(sample_book.bids),
                "ask_levels_count": len(sample_book.asks),
            },
            indent=2,
        )
    )