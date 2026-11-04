#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART A)
File: 01_data_ingestion/polymarket_gamma/tag_category_taxonomy.py
================================================================================
Deterministic Multi-Signal Taxonomy Classifier & Correlation Cluster Mapper.

Why This Module Exists:
  1. Polymarket's raw `tags` array is noisy and inconsistent across years (e.g.,
     FOMC rate-cut markets may be tagged `'Business'`, `'Fed'`, `'Economy'`, or
     have no tags at all).
  2. In Stage 4 (SFT) and Stage 5 (GRPO), our Qwen-2.5-7B model uses domain-
     specific base rates (Politics vs. Macro/Fed vs. Crypto vs. Geopolitics).
  3. In Stage 6 (`05_execution_risk_engine/fractional_kelly_sizer/`), our $5.00
     survival bot enforces a strict **30% correlated cluster cap** so it never
     bets its entire bankroll on 4 overlapping crypto or election contracts.
  4. In Stage 7 (`07_institutional_scale/`), B2B hedge-fund and corporate
     treasury subscribers filter feeds specifically by Macro, Tariff, Regulatory,
     and Geopolitical Supply-Chain domains.
================================================================================
"""

from __future__ import annotations

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Pattern, Tuple

from .gamma_schema import GammaMarketRecord

logger = logging.getLogger("sovereign_prob_engine.tag_category_taxonomy")


# ==============================================================================
# 1. CANONICAL QUANTITATIVE PREDICTION MARKET DOMAINS
# ==============================================================================
class CanonicalDomain(str, Enum):
    """
    Eight canonical domains with distinct historical base rates, news catalyst
    velocities, and UMA resolution dispute profiles.
    """
    MACRO_ECONOMICS_FED = "MACRO_ECONOMICS_FED"
    US_GLOBAL_POLITICS = "US_GLOBAL_POLITICS"
    GEOPOLITICS_CONFLICT = "GEOPOLITICS_CONFLICT"
    CRYPTO_DIGITAL_ASSETS = "CRYPTO_DIGITAL_ASSETS"
    CORPORATE_TECH_AI = "CORPORATE_TECH_AI"
    LAW_REGULATION_SEC = "LAW_REGULATION_SEC"
    SPORTS_ATHLETICS = "SPORTS_ATHLETICS"
    CULTURE_WEATHER_SCIENCE = "CULTURE_WEATHER_SCIENCE"


@dataclass(frozen=True)
class DomainTaxonomyProfile:
    """
    Quantitative metadata attached to each CanonicalDomain for downstream
    training stratification and live Fractional Kelly risk caps.
    """
    domain: CanonicalDomain
    historical_yes_base_rate: float
    uma_dispute_risk_weight: float
    b2b_institutional_relevance: float
    survival_bot_priority_tier: int
    default_tavily_search_topic: str


# Grounded domain priors used for SFT base-rate anchoring and Kelly dampening
DOMAIN_PROFILES: Dict[CanonicalDomain, DomainTaxonomyProfile] = {
    CanonicalDomain.MACRO_ECONOMICS_FED: DomainTaxonomyProfile(
        domain=CanonicalDomain.MACRO_ECONOMICS_FED,
        historical_yes_base_rate=0.38,
        uma_dispute_risk_weight=0.05,  # Official BLS / Fed releases rarely dispute
        b2b_institutional_relevance=1.00,
        survival_bot_priority_tier=1,
        default_tavily_search_topic="finance",
    ),
    CanonicalDomain.US_GLOBAL_POLITICS: DomainTaxonomyProfile(
        domain=CanonicalDomain.US_GLOBAL_POLITICS,
        historical_yes_base_rate=0.34,
        uma_dispute_risk_weight=0.18,
        b2b_institutional_relevance=0.90,
        survival_bot_priority_tier=1,
        default_tavily_search_topic="news",
    ),
    CanonicalDomain.GEOPOLITICS_CONFLICT: DomainTaxonomyProfile(
        domain=CanonicalDomain.GEOPOLITICS_CONFLICT,
        historical_yes_base_rate=0.26,  # Sensational geopolitical events resolve NO ~74% of the time
        uma_dispute_risk_weight=0.35,  # High ambiguity around 'ceasefire' or 'strike' definitions
        b2b_institutional_relevance=0.95,
        survival_bot_priority_tier=1,
        default_tavily_search_topic="news",
    ),
    CanonicalDomain.CRYPTO_DIGITAL_ASSETS: DomainTaxonomyProfile(
        domain=CanonicalDomain.CRYPTO_DIGITAL_ASSETS,
        historical_yes_base_rate=0.41,
        uma_dispute_risk_weight=0.08,  # Binance/Coinbase candle highs are deterministic
        b2b_institutional_relevance=0.85,
        survival_bot_priority_tier=1,
        default_tavily_search_topic="finance",
    ),
    CanonicalDomain.CORPORATE_TECH_AI: DomainTaxonomyProfile(
        domain=CanonicalDomain.CORPORATE_TECH_AI,
        historical_yes_base_rate=0.35,
        uma_dispute_risk_weight=0.15,
        b2b_institutional_relevance=0.88,
        survival_bot_priority_tier=2,
        default_tavily_search_topic="news",
    ),
    CanonicalDomain.LAW_REGULATION_SEC: DomainTaxonomyProfile(
        domain=CanonicalDomain.LAW_REGULATION_SEC,
        historical_yes_base_rate=0.32,
        uma_dispute_risk_weight=0.12,
        b2b_institutional_relevance=0.92,
        survival_bot_priority_tier=1,
        default_tavily_search_topic="news",
    ),
    CanonicalDomain.SPORTS_ATHLETICS: DomainTaxonomyProfile(
        domain=CanonicalDomain.SPORTS_ATHLETICS,
        historical_yes_base_rate=0.48,
        uma_dispute_risk_weight=0.03,
        b2b_institutional_relevance=0.20,
        survival_bot_priority_tier=3,
        default_tavily_search_topic="general",
    ),
    CanonicalDomain.CULTURE_WEATHER_SCIENCE: DomainTaxonomyProfile(
        domain=CanonicalDomain.CULTURE_WEATHER_SCIENCE,
        historical_yes_base_rate=0.31,
        uma_dispute_risk_weight=0.14,
        b2b_institutional_relevance=0.45,
        survival_bot_priority_tier=3,
        default_tavily_search_topic="general",
    ),
}


@dataclass(frozen=True)
class DomainTaxonomyClassification:
    """
    Complete classification output for a single Polymarket contract, including
    its primary domain, correlation cluster ID (for Kelly risk capping), and
    quantitative priors.
    """
    primary_domain: CanonicalDomain
    secondary_domain: Optional[CanonicalDomain]
    correlation_cluster_key: str
    confidence_score: float
    historical_yes_base_rate: float
    uma_dispute_risk_weight: float
    b2b_institutional_relevance: float
    matched_keywords: List[str]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["primary_domain"] = self.primary_domain.value
        payload["secondary_domain"] = (
            self.secondary_domain.value if self.secondary_domain else None
        )
        return payload


# ==============================================================================
# 2. DETERMINISTIC MULTI-SIGNAL TAXONOMY ENGINE
# ==============================================================================
class PolymarketCategoryTaxonomyEngine:
    """
    Classifies any `GammaMarketRecord` into one of the 8 canonical domains by
    scoring explicit Gamma tags, question tokens, and UMA resolution sources.
    Also extracts a deterministic `correlation_cluster_key` so the execution
    risk engine never over-allocates the $5 survival bankroll to identical risks.
    """

    def __init__(self) -> None:
        self._compiled_rules: Dict[CanonicalDomain, List[Tuple[Pattern[str], float, str]]] = (
            self._build_weighted_regex_rules()
        )
        self._cluster_entity_patterns: List[Tuple[Pattern[str], str]] = (
            self._build_correlation_cluster_patterns()
        )

    @staticmethod
    def _build_weighted_regex_rules() -> Dict[CanonicalDomain, List[Tuple[Pattern[str], float, str]]]:
        """
        Compiles weighted keyword patterns per domain. Higher weights apply to
        unambiguous institutional terms (e.g. 'FOMC', 'CPI', 'Binance BTC/USDT').
        """
        raw_specs: Dict[CanonicalDomain, List[Tuple[str, float, str]]] = {
            CanonicalDomain.MACRO_ECONOMICS_FED: [
                (r"\b(fomc|federal reserve|fed funds|rate cut|rate hike|powell|bps)\b", 4.0, "fed_rates"),
                (r"\b(cpi|inflation|core pce|nonfarm payrolls|nfp|unemployment rate|gdp)\b", 3.8, "macro_print"),
                (r"\b(recession|treasury yield|10-year|tariff|trade deficit|ecb|boj)\b", 3.2, "macro_policy"),
            ],
            CanonicalDomain.US_GLOBAL_POLITICS: [
                (r"\b(presidential election|senate|house of representatives|electoral college|primary)\b", 4.0, "election"),
                (r"\b(trump|biden|harris|vance|newsom|desantis|starmer|macron|modi)\b", 3.2, "political_leader"),
                (r"\b(approval rating|impeach|cabinet|veto|congress|parliament|poll)\b", 2.8, "governance"),
            ],
            CanonicalDomain.GEOPOLITICS_CONFLICT: [
                (r"\b(ceasefire|invasion|military strike|missile|nato|nuclear|airstrike)\b", 4.0, "military_action"),
                (r"\b(ukraine|russia|israel|gaza|iran|taiwan|china|strait of hormuz|red sea)\b", 3.5, "geopolitical_zone"),
                (r"\b(sanctions|embargo|peace treaty|hostage|un security council)\b", 3.0, "diplomacy"),
            ],
            CanonicalDomain.CRYPTO_DIGITAL_ASSETS: [
                (r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|dogecoin|bnb)\b", 4.0, "crypto_asset"),
                (r"\b(binance|coinbase|hyperliquid|polymarket|airdrop|fdv|market cap|all-time high|ath)\b", 3.5, "crypto_venue"),
                (r"\b(usdc|tether|usdt|stablecoin|halving|spot etf|on-chain|tvl)\b", 3.2, "crypto_infra"),
            ],
            CanonicalDomain.CORPORATE_TECH_AI: [
                (r"\b(openai|gpt-5|claude|anthropic|gemini|deepseek|grok|xai|llama)\b", 4.0, "frontier_ai"),
                (r"\b(nvidia|nvda|apple|aapl|tesla|tsla|microsoft|amd|meta|earnings|ipo)\b", 3.5, "equities_tech"),
                (r"\b(agi|swe-bench|chatbot arena|lmsys|ceo|acquisition|merger)\b", 3.0, "corporate_event"),
            ],
            CanonicalDomain.LAW_REGULATION_SEC: [
                (r"\b(supreme court|scotus|sec|cftc|doj|ftc|indictment|antitrust)\b", 4.0, "regulatory_body"),
                (r"\b(convicted|guilty|sentenced|lawsuit|ruling|ban|executive order|fda approval)\b", 3.4, "legal_ruling"),
            ],
            CanonicalDomain.SPORTS_ATHLETICS: [
                (r"\b(nfl|nba|mlb|nhl|premier league|ucl|champions league|fifa|world cup|ipl)\b", 4.0, "sports_league"),
                (r"\b(super bowl|playoffs|mvp|grand slam|ufc|formula 1|f1|olympics)\b", 3.8, "sports_event"),
            ],
            CanonicalDomain.CULTURE_WEATHER_SCIENCE: [
                (r"\b(hurricane|noaa|temperature|celsius|earthquake|nasa|spacex|starship)\b", 3.6, "weather_space"),
                (r"\b(oscars|grammys|box office|billboard|spotify|gta|nobel prize)\b", 3.2, "culture_media"),
            ],
        }

        compiled: Dict[CanonicalDomain, List[Tuple[Pattern[str], float, str]]] = {}
        for domain, rules in raw_specs.items():
            compiled[domain] = [
                (re.compile(pattern, re.IGNORECASE), weight, label)
                for pattern, weight, label in rules
            ]
        return compiled

    @staticmethod
    def _build_correlation_cluster_patterns() -> List[Tuple[Pattern[str], str]]:
        """
        Maps specific recurring underlying assets/events to a shared cluster ID
        so the Fractional Kelly sizer caps cumulative exposure across sibling markets.
        """
        specs = [
            (r"\b(bitcoin|btc)\b", "CLUSTER_CRYPTO_BTC"),
            (r"\b(ethereum|eth)\b", "CLUSTER_CRYPTO_ETH"),
            (r"\b(solana|sol)\b", "CLUSTER_CRYPTO_SOL"),
            (r"\b(fomc|federal reserve|fed funds|rate cut|powell)\b", "CLUSTER_MACRO_FED_RATES"),
            (r"\b(cpi|inflation|pce)\b", "CLUSTER_MACRO_US_INFLATION"),
            (r"\b(tariff|trade war)\b", "CLUSTER_MACRO_TARIFFS"),
            (r"\b(ukraine|russia|zelensky|putin)\b", "CLUSTER_GEO_RU_UA"),
            (r"\b(israel|gaza|iran|lebanon|hormuz)\b", "CLUSTER_GEO_MENA"),
            (r"\b(taiwan|china|south china sea)\b", "CLUSTER_GEO_TW_CN"),
            (r"\b(openai|gpt|anthropic|claude|gemini|deepseek|grok)\b", "CLUSTER_TECH_LLM_RACE"),
        ]
        return [(re.compile(pat, re.IGNORECASE), cluster_id) for pat, cluster_id in specs]

    def classify_market(self, record: GammaMarketRecord) -> DomainTaxonomyClassification:
        """
        Scores the market's tags, question title, and resolution description to
        assign a deterministic `DomainTaxonomyClassification`.
        """
        tag_string = " ".join(f"{t.label} {t.slug}" for t in record.tags)
        combined_corpus = f"{tag_string} | {record.question} | {record.resolution_source} | {record.description[:600]}"

        domain_scores: Dict[CanonicalDomain, float] = {d: 0.0 for d in CanonicalDomain}
        matched_labels: List[str] = []

        for domain, rules in self._compiled_rules.items():
            for regex_pat, weight, label in rules:
                # Question & tag hits receive a 1.5x multiplier over body text hits
                header_text = f"{tag_string} {record.question}"
                if regex_pat.search(header_text):
                    domain_scores[domain] += weight * 1.5
                    matched_labels.append(f"{domain.value}:{label}")
                elif regex_pat.search(combined_corpus):
                    domain_scores[domain] += weight
                    matched_labels.append(f"{domain.value}:{label}")

        ranked = sorted(domain_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_domain, best_score = ranked[0]
        second_domain, second_score = ranked[1]

        if best_score <= 0.0:
            best_domain = CanonicalDomain.CULTURE_WEATHER_SCIENCE
            confidence = 0.35
        else:
            total_positive = sum(s for s in domain_scores.values() if s > 0.0)
            confidence = round(min(0.99, best_score / max(total_positive, 1e-6)), 4)

        secondary = second_domain if second_score >= (0.45 * best_score) and second_score > 0 else None
        profile = DOMAIN_PROFILES[best_domain]
        cluster_key = self._resolve_correlation_cluster(record, best_domain)

        return DomainTaxonomyClassification(
            primary_domain=best_domain,
            secondary_domain=secondary,
            correlation_cluster_key=cluster_key,
            confidence_score=confidence,
            historical_yes_base_rate=profile.historical_yes_base_rate,
            uma_dispute_risk_weight=profile.uma_dispute_risk_weight,
            b2b_institutional_relevance=profile.b2b_institutional_relevance,
            matched_keywords=matched_labels[:8],
        )

    def _resolve_correlation_cluster(
        self,
        record: GammaMarketRecord,
        primary_domain: CanonicalDomain,
    ) -> str:
        """
        Determines the correlation cluster ID for portfolio exposure capping.
        If the market belongs to a `neg_risk_market_id` group on Polymarket,
        that group ID takes precedence.
        """
        if record.neg_risk and record.neg_risk_market_id:
            return f"CLUSTER_NEGRISK_{record.neg_risk_market_id[:16]}"

        search_target = f"{record.question} {' '.join(t.label for t in record.tags)}"
        for pattern, cluster_id in self._cluster_entity_patterns:
            if pattern.search(search_target):
                return cluster_id

        # Default fallback cluster grouped by primary domain and expiry month
        expiry_month = (
            record.end_date_utc.strftime("%Y_%m")
            if record.end_date_utc is not None
            else "OPEN_HORIZON"
        )
        return f"CLUSTER_{primary_domain.value}_{expiry_month}"


if __name__ == "__main__":
    # Self-test of the taxonomy classifier across Macro and Crypto markets
    engine = PolymarketCategoryTaxonomyEngine()
    test_market = GammaMarketRecord(
        id="9001",
        conditionId="0x" + "11" * 32,
        question="Will the Federal Reserve cut interest rates by 25 bps at the June 2026 FOMC meeting?",
        description="Resolves to Yes if the FOMC statement lowers the target federal funds rate range by 25 bps.",
        closed=True,
        volumeNum=3200000.0,
        outcomes=["Yes", "No"],
        outcomePrices=[1.0, 0.0],
        clobTokenIds=["101", "102"],
    )
    result = engine.classify_market(test_market)
    print(json.dumps(result.to_dict(), indent=2))