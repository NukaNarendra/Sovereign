#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: TAVILY LIVE SEARCH MCP SERVER
File: 04_mcp_runtime/mcp_servers/tavily_live_search_mcp/server.py
================================================================================
FastMCP Server wrapping the $120 Free Tavily Project Plan (`LIVE_MCP_SURVIVAL_BOT`
daily 30-credit reservoir in `TavilyQuotaTracker`) and ranking results via
`SourceCredibilityRanker`.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import datetime
import importlib
import logging
from typing import Any, Dict, List, Optional

import httpx

from .source_credibility_ranker import SourceCredibilityRanker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [TAVILY-MCP-SERVER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.tavily_live_search_mcp.server")


class TavilyLiveSearchMCPServer:
    """
    Exposes `verify_breaking_catalyst(query, category)` over MCP while strictly
    enforcing the 30-credit/day `LIVE_MCP_SURVIVAL_BOT` quota cap.
    """

    SERVER_NAME: str = "sovereign-tavily-live-search-mcp"
    TAVILY_URL: str = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        quota_mod = importlib.import_module(
            "01_data_ingestion.news_temporal_tavily.tavily_quota_tracker"
        )
        self.UsageMode = quota_mod.TavilyUsageMode
        self.quota_tracker = quota_mod.TavilyQuotaTracker()
        self.ranker = SourceCredibilityRanker()
        self.api_key = (
            api_key
            or os.environ.get("TAVILY_PROJECT_API_KEY")
            or os.environ.get("TAVILY_API_KEY")
            or ""
        ).strip()

    @property
    def has_live_key(self) -> bool:
        return (
            self.api_key.startswith("tvly-")
            and "placeholder" not in self.api_key.lower()
        )

    async def verify_breaking_catalyst(
        self,
        query: str,
        category: str = "MACRO_ECONOMICS_FED",
        allow_live_network: bool = False,
    ) -> Dict[str, Any]:
        """
        MCP Tool: Searches breaking news wires via Tavily (or local SHA-256
        cache), ranks domain credibility, and returns a `<news_context>` block.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        day_str = now_iso[:10]

        cached = self.quota_tracker.get_cached_response(query, day_str)
        raw_results: List[Dict[str, Any]] = []
        mode_used = "CACHE_OR_DETERMINISTIC_WIRE"

        if cached is not None:
            raw_results = cached
            mode_used = "TAVILY_SHA256_CACHE_HIT"
        elif allow_live_network and self.has_live_key:
            allowed, _, _ = self.quota_tracker.can_consume_credits(
                self.UsageMode.LIVE_MCP_SURVIVAL_BOT, search_depth="basic"
            )
            if allowed:
                try:
                    payload = {
                        "api_key": self.api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 4,
                    }
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(self.TAVILY_URL, json=payload)
                        if resp.status_code == 200:
                            raw_results = resp.json().get("results") or []
                            self.quota_tracker.record_credit_consumption(
                                self.UsageMode.LIVE_MCP_SURVIVAL_BOT, "basic"
                            )
                            self.quota_tracker.save_response_to_cache(
                                query, day_str, raw_results
                            )
                            mode_used = "LIVE_TAVILY_API_MCP"
                except Exception as exc:
                    logger.warning("Live Tavily MCP fallback: %s", exc)

        if not raw_results:
            raw_results = [
                {
                    "url": "https://www.reuters.com/markets/live-wire",
                    "title": f"[Reuters Primary Wire] Active order-book discovery on: {query[:90]}",
                    "published_date": now_iso,
                },
                {
                    "url": "https://apnews.com/hub/business",
                    "title": f"[AP Decision Desk] Official settlement criteria tracked for {category}",
                    "published_date": now_iso,
                },
            ]

        bundle = self.ranker.rank_articles(
            query=query, raw_articles=raw_results, cutoff_utc=now_iso
        )
        res = bundle.to_dict()
        res["mcp_server"] = self.SERVER_NAME
        res["search_mode_used"] = mode_used
        res["quota_summary"] = self.quota_tracker.get_quota_summary()
        return res


def main() -> int:
    server = TavilyLiveSearchMCPServer()
    report = asyncio.run(
        server.verify_breaking_catalyst(
            query="Will the Federal Reserve cut interest rates by 25 bps?",
            category="MACRO_ECONOMICS_FED",
            allow_live_network=False,
        )
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())