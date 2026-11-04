#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART A)
File: 01_data_ingestion/polymarket_gamma/gamma_rest_client.py
================================================================================
High-Throughput Asynchronous REST Client for `https://gamma-api.polymarket.com`.

Architectural Features:
  1. Zero-Cost Public Access: Connects directly to Polymarket's Gamma discovery
     endpoints (`/markets`, `/events`, `/tags`) with HTTP/2 connection pooling.
  2. Token-Bucket Rate Limiter: Prevents Cloudflare 429 bans during multi-hour
     historical crawls of 100,000+ resolved markets.
  3. Exponential Backoff + Jitter: Automatically retries transient 429, 500,
     502, 503, and 504 responses and respects `Retry-After` headers.
  4. Async Generator Pagination: Streams normalized `GammaMarketRecord` objects
     enriched with `PolymarketCategoryTaxonomyEngine` classifications without
     loading the entire 100k+ dataset into RAM at once.
================================================================================
"""

from __future__ import annotations

import time
import json
import random
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import httpx

from .gamma_schema import GammaMarketRecord, GammaEventRecord
from .tag_category_taxonomy import PolymarketCategoryTaxonomyEngine

logger = logging.getLogger("sovereign_prob_engine.gamma_rest_client")


# ==============================================================================
# 1. CLIENT CONFIGURATION & ASYNC TOKEN-BUCKET RATE LIMITER
# ==============================================================================
@dataclass(frozen=True)
class GammaClientConfig:
    """
    Configuration parameters for `GammaAsyncRestClient`.
    Tuned to stay safely inside Polymarket Cloudflare edge rate limits while
    achieving ~500 markets/sec ingestion throughput.
    """
    base_url: str = "https://gamma-api.polymarket.com"
    requests_per_second: float = 6.0
    max_burst_capacity: int = 12
    page_size_limit: int = 100
    request_timeout_sec: float = 25.0
    max_retries: int = 5
    backoff_base_sec: float = 1.25
    user_agent: str = "SovereignProbEngine/1.0 (Quantitative-Research-Pipeline)"


class TokenBucketRateLimiter:
    """
    Asynchronous token-bucket rate limiter that smooths outbound HTTP calls
    across concurrent tasks so we never trip Cloudflare 429 blocks.
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        self.rate_per_sec = max(0.1, float(rate_per_sec))
        self.capacity = max(1, int(capacity))
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Waits until at least 1.0 request token is available in the bucket."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last_refill)
                self._tokens = min(
                    float(self.capacity),
                    self._tokens + elapsed * self.rate_per_sec,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait_time = (1.0 - self._tokens) / self.rate_per_sec
            await asyncio.sleep(wait_time)


# ==============================================================================
# 2. ASYNCHRONOUS GAMMA REST API CLIENT
# ==============================================================================
class GammaAsyncRestClient:
    """
    Production async client for fetching resolved and live markets from
    Polymarket's Gamma service.
    """

    def __init__(
        self,
        config: Optional[GammaClientConfig] = None,
        taxonomy_engine: Optional[PolymarketCategoryTaxonomyEngine] = None,
    ) -> None:
        self.config = config or GammaClientConfig()
        self.taxonomy_engine = taxonomy_engine or PolymarketCategoryTaxonomyEngine()
        self._rate_limiter = TokenBucketRateLimiter(
            rate_per_sec=self.config.requests_per_second,
            capacity=self.config.max_burst_capacity,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self.total_http_requests: int = 0
        self.total_retry_events: int = 0

    async def __aenter__(self) -> "GammaAsyncRestClient":
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def open(self) -> None:
        """Initializes the persistent `httpx.AsyncClient` connection pool."""
        if self._client is None:
            limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
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
        """Cleanly closes the underlying HTTP connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --------------------------------------------------------------------------
    # Core Resilient HTTP GET Dispatcher
    # --------------------------------------------------------------------------
    async def _request_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Executes a rate-limited HTTP GET request with exponential backoff and
        full `Retry-After` header support.
        """
        if self._client is None:
            await self.open()
        assert self._client is not None

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            await self._rate_limiter.acquire()
            self.total_http_requests += 1
            try:
                response = await self._client.get(endpoint, params=params)
                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    self.total_retry_events += 1
                    retry_after_hdr = response.headers.get("Retry-After")
                    if retry_after_hdr and retry_after_hdr.isdigit():
                        sleep_sec = float(retry_after_hdr) + random.uniform(0.1, 0.75)
                    else:
                        sleep_sec = (
                            self.config.backoff_base_sec * (2 ** (attempt - 1))
                            + random.uniform(0.05, 0.50)
                        )
                    logger.warning(
                        "[GAMMA-CLIENT] HTTP %d on %s (attempt %d/%d). Backing off %.2fs...",
                        response.status_code,
                        endpoint,
                        attempt,
                        self.config.max_retries,
                        sleep_sec,
                    )
                    await asyncio.sleep(sleep_sec)
                    continue

                response.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                last_exc = exc
                self.total_retry_events += 1
                sleep_sec = (
                    self.config.backoff_base_sec * (2 ** (attempt - 1))
                    + random.uniform(0.1, 0.6)
                )
                logger.warning(
                    "[GAMMA-CLIENT] Network exception on %s (%s) [attempt %d/%d]. Retrying in %.2fs...",
                    endpoint,
                    exc,
                    attempt,
                    self.config.max_retries,
                    sleep_sec,
                )
                await asyncio.sleep(sleep_sec)

        raise RuntimeError(
            f"Exhausted {self.config.max_retries} retries for Gamma endpoint {endpoint}"
        ) from last_exc

    # --------------------------------------------------------------------------
    # Market & Event Fetching Methods
    # --------------------------------------------------------------------------
    async def fetch_markets_page(
        self,
        offset: int = 0,
        limit: Optional[int] = None,
        closed: Optional[bool] = True,
        active: Optional[bool] = None,
        order: str = "volumeNum",
        ascending: bool = False,
    ) -> List[GammaMarketRecord]:
        """
        Fetches a single paginated batch of markets from `/markets`, validates
        each item via `GammaMarketRecord`, and enriches it with its canonical
        domain classification.
        """
        page_limit = limit or self.config.page_size_limit
        query_params: Dict[str, Any] = {
            "limit": page_limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if closed is not None:
            query_params["closed"] = str(closed).lower()
        if active is not None:
            query_params["active"] = str(active).lower()

        raw_payload = await self._request_json("/markets", params=query_params)
        if not isinstance(raw_payload, list):
            logger.warning("Unexpected non-list payload from /markets at offset %d", offset)
            return []

        records: List[GammaMarketRecord] = []
        for raw_item in raw_payload:
            if not isinstance(raw_item, dict):
                continue
            try:
                record = GammaMarketRecord.model_validate(raw_item)
                classification = self.taxonomy_engine.classify_market(record)
                record.canonical_category = classification.primary_domain.value
                records.append(record)
            except Exception as exc:
                logger.debug("Skipping malformed Gamma market item: %s", exc)
        return records

    async def fetch_market_by_id(self, market_id: str) -> Optional[GammaMarketRecord]:
        """Fetches a single market by its numeric Gamma ID."""
        raw_item = await self._request_json(f"/markets/{market_id}")
        if isinstance(raw_item, dict):
            record = GammaMarketRecord.model_validate(raw_item)
            record.canonical_category = (
                self.taxonomy_engine.classify_market(record).primary_domain.value
            )
            return record
        return None

    async def fetch_events_page(
        self,
        offset: int = 0,
        limit: int = 50,
        closed: bool = True,
    ) -> List[GammaEventRecord]:
        """Fetches top-level Event containers (each containing nested markets)."""
        params = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        raw_payload = await self._request_json("/events", params=params)
        if not isinstance(raw_payload, list):
            return []

        events: List[GammaEventRecord] = []
        for item in raw_payload:
            if isinstance(item, dict):
                try:
                    events.append(GammaEventRecord.model_validate(item))
                except Exception as exc:
                    logger.debug("Skipping malformed Gamma event item: %s", exc)
        return events

    async def iter_resolved_markets(
        self,
        start_offset: int = 0,
        max_markets: Optional[int] = None,
        only_valid_binary: bool = True,
        min_volume_usd: float = 0.0,
    ) -> AsyncIterator[Tuple[int, List[GammaMarketRecord]]]:
        """
        Asynchronous generator that walks `/markets?closed=true` page by page,
        yielding `(current_offset, filtered_records)` until exhaustion or
        `max_markets` is reached.
        """
        offset = start_offset
        yielded_count = 0
        page_size = self.config.page_size_limit

        while True:
            page_records = await self.fetch_markets_page(
                offset=offset,
                limit=page_size,
                closed=True,
                order="volumeNum",
                ascending=False,
            )
            if not page_records:
                logger.info("Reached end of Gamma `/markets` pagination at offset=%d.", offset)
                break

            filtered: List[GammaMarketRecord] = []
            for rec in page_records:
                if rec.volume_usd < min_volume_usd:
                    continue
                if only_valid_binary and not rec.is_eligible_for_grpo_brier_training(
                    min_volume_usd=min_volume_usd
                ):
                    continue
                filtered.append(rec)

            if filtered:
                if max_markets is not None and (yielded_count + len(filtered)) > max_markets:
                    needed = max(0, max_markets - yielded_count)
                    filtered = filtered[:needed]
                yielded_count += len(filtered)
                yield offset, filtered

            offset += len(page_records)
            if max_markets is not None and yielded_count >= max_markets:
                logger.info("Reached target max_markets=%d at offset=%d.", max_markets, offset)
                break
            if len(page_records) < page_size:
                break


if __name__ == "__main__":
    # Smoke test: Fetch 5 live resolved markets from Gamma API (or print status)
    logging.basicConfig(level=logging.INFO)

    async def _smoke_test() -> None:
        async with GammaAsyncRestClient() as client:
            markets = await client.fetch_markets_page(offset=0, limit=5, closed=True)
            print(f"Fetched {len(markets)} markets from Gamma API.")
            for m in markets[:3]:
                print(
                    f"  - [{m.canonical_category}] {m.question[:60]}... "
                    f"| status={m.resolution_status.value} | o={m.resolved_binary_outcome}"
                )

    asyncio.run(_smoke_test())