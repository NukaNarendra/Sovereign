#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART D)
File: 01_data_ingestion/news_temporal_tavily/__init__.py
================================================================================
Public interface for the Point-in-Time News Wire, X/Cashtag Social Velocity,
and Macroeconomic Catalyst Calendar Ingestion Layer powered by the $120 Free
Tavily Project Plan (4 months / 16,000 search credits).

Core Modules Exposed by This Subpackage:
  1. `tavily_quota_tracker`        : Enforces daily/monthly rate caps across the
                                     4,000-query/month Tavily Project Plan so
                                     credits never exhaust prematurely.
  2. `macro_calendar_ingestor`     : Generates point-in-time FOMC, CPI, NFP, GDP,
                                     and regulatory catalyst calendars aligned to
                                     each market snapshot horizon.
  3. `x_cashtag_sentiment_scraper` : Captures pre-cutoff social velocity Z-scores,
                                     cashtag polarity, and rumor-contradiction
                                     flags (`<social_pulse>` XML tags).
  4. `tavily_historical_search`    : Queries date-bounded news articles strictly
                                     published at or before `cutoff_timestamp_utc`
                                     to guarantee 0% future outcome leakage.
================================================================================
"""

from typing import Any

__all__ = [
    "TavilyUsageMode",
    "TavilyQuotaState",
    "TavilyQuotaTracker",
    "MacroCatalystType",
    "MacroCalendarEvent",
    "MacroCalendarIngestor",
    "SocialVelocityObservation",
    "XCashtagSentimentScraper",
    "PointInTimeNewsArticle",
    "HorizonNewsBundle",
    "TavilyHistoricalSearchEngine",
]


def __getattr__(name: str) -> Any:
    if name in ("TavilyUsageMode", "TavilyQuotaState", "TavilyQuotaTracker"):
        from . import tavily_quota_tracker
        return getattr(tavily_quota_tracker, name)
    if name in ("MacroCatalystType", "MacroCalendarEvent", "MacroCalendarIngestor"):
        from . import macro_calendar_ingestor
        return getattr(macro_calendar_ingestor, name)
    if name in ("SocialVelocityObservation", "XCashtagSentimentScraper"):
        from . import x_cashtag_sentiment_scraper
        return getattr(x_cashtag_sentiment_scraper, name)
    if name in (
        "PointInTimeNewsArticle",
        "HorizonNewsBundle",
        "TavilyHistoricalSearchEngine",
    ):
        from . import tavily_historical_search
        return getattr(tavily_historical_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")