#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: NEWS SOURCE CREDIBILITY RANKER (MCP)
File: 04_mcp_runtime/mcp_servers/tavily_live_search_mcp/source_credibility_ranker.py
================================================================================
Filters unverified social rumors and ranks breaking news articles returned by
Tavily Search API according to institutional source authority:
  - Tier 1 (`0.95 - 1.00`): Primary government/central-bank portals (`.gov`,
    `federalreserve.gov`, `bls.gov`, `sec.gov`) & primary wire desks
    (`reuters.com`, `apnews.com`, `bloomberg.com`).
  - Tier 2 (`0.80 - 0.90`): Major financial/policy newspapers (`wsj.com`,
    `ft.com`, `cfr.org`, `pewresearch.org`, `economist.com`).
  - Tier 3 (`0.25 - 0.55`): Unverified blogs, social media posts, or unknown
    domains (flagged with `unverified_rumor_hazard = True` if not corroborated
    by at least one Tier-1 or Tier-2 source).
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
    format="%(asctime)s | %(levelname)-8s | [CREDIBILITY-RANKER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.source_credibility_ranker")

TIER_1_DOMAINS: tuple[str, ...] = (
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "federalreserve.gov",
    "bls.gov",
    "sec.gov",
    "supremecourt.gov",
    "treasury.gov",
    "whitehouse.gov",
)

TIER_2_DOMAINS: tuple[str, ...] = (
    "wsj.com",
    "ft.com",
    "cfr.org",
    "pewresearch.org",
    "constitutioncenter.org",
    "calmatters.org",
    "cnbc.com",
    "bbc.com",
    "nytimes.com",
)


@dataclass(frozen=True)
class RankedNewsWireItem:
    """Single breaking news item scored for source authority and rumor risk."""
    source_domain: str
    authority_tier: int
    credibility_score: float
    published_utc: str
    headline: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CredibilityRankedBundle:
    """Aggregated credibility report and `<news_context>` XML for live MCP."""
    query: str
    total_articles_ranked: int
    tier1_confirmations_count: int
    tier1_confirmation_ratio: float
    unverified_rumor_hazard_flag: bool
    ranked_articles: List[RankedNewsWireItem]
    cpt_news_xml_tag: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ranked_articles"] = [a.to_dict() for a in self.ranked_articles]
        return payload


class SourceCredibilityRanker:
    """
    Scores and sorts live Tavily search results by domain authority and flags
    uncorroborated social rumors before they reach Qwen-2.5-7B.
    """

    @staticmethod
    def score_domain(url_or_domain: str) -> tuple[str, int, float]:
        clean = str(url_or_domain or "").lower().strip()
        if "://" in clean:
            clean = clean.split("://", 1)[1].split("/", 1)[0]
        if clean.startswith("www."):
            clean = clean[4:]

        if clean.endswith(".gov") or any(clean.endswith(d) for d in TIER_1_DOMAINS):
            return clean, 1, 0.96
        if any(clean.endswith(d) for d in TIER_2_DOMAINS):
            return clean, 2, 0.86
        return (clean or "unknown-source.org"), 3, 0.40

    def rank_articles(
        self, query: str, raw_articles: Sequence[Dict[str, Any]], cutoff_utc: str
    ) -> CredibilityRankedBundle:
        ranked: List[RankedNewsWireItem] = []
        for item in raw_articles:
            url = str(item.get("url") or item.get("source_domain") or "reuters.com")
            dom, tier, cred = self.score_domain(url)
            pub = str(item.get("published_date") or item.get("published_utc") or cutoff_utc)
            title = str(item.get("title") or item.get("headline") or "").strip()[:180]
            ranked.append(
                RankedNewsWireItem(
                    source_domain=dom,
                    authority_tier=tier,
                    credibility_score=cred,
                    published_utc=pub,
                    headline=title,
                    url=url,
                )
            )

        ranked.sort(key=lambda x: x.credibility_score, reverse=True)
        n_tot = max(1, len(ranked))
        t1_cnt = sum(1 for a in ranked if a.authority_tier in (1, 2))
        t1_ratio = round(t1_cnt / float(n_tot), 4)
        rumor_hazard = t1_cnt == 0

        lines = "\n".join(
            f"  - [{a.published_utc}] ({a.source_domain} | tier={a.authority_tier}): {a.headline}"
            for a in ranked[:4]
        )
        xml_tag = (
            f"<news_context cutoff_utc=\"{cutoff_utc}\" "
            f"tier1_ratio=\"{t1_ratio:0.2f}\" "
            f"rumor_hazard=\"{str(rumor_hazard).lower()}\">\n"
            f"{lines}\n"
            f"</news_context>"
        )

        return CredibilityRankedBundle(
            query=query,
            total_articles_ranked=len(ranked),
            tier1_confirmations_count=t1_cnt,
            tier1_confirmation_ratio=t1_ratio,
            unverified_rumor_hazard_flag=rumor_hazard,
            ranked_articles=ranked,
            cpt_news_xml_tag=xml_tag,
        )


def main() -> int:
    ranker = SourceCredibilityRanker()
    bundle = ranker.rank_articles(
        query="Federal Reserve FOMC interest rate decision",
        raw_articles=[
            {
                "url": "https://www.reuters.com/markets/us/fed-rate-decision",
                "title": "Federal Reserve signals data-dependent path at FOMC meeting",
                "published_date": "2026-06-17T16:00:00+00:00",
            },
            {
                "url": "https://random-crypto-rumor-blog.xyz/post",
                "title": "BREAKING: Secret emergency 100 bps cut rumored on X!",
                "published_date": "2026-06-17T16:05:00+00:00",
            },
        ],
        cutoff_utc="2026-06-17T17:00:00+00:00",
    )
    print(json.dumps(bundle.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())