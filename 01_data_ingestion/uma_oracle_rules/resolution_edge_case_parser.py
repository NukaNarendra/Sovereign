#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART C)
File: 01_data_ingestion/uma_oracle_rules/resolution_edge_case_parser.py
================================================================================
Extracts Deterministic Settlement Clauses from UMA Ancillary Rules:
  1. Exact Timezone & Deadline Cutoffs (`ET`, `UTC`, `PT`, `GMT`, `11:59 PM`)
  2. Designated Primary Source of Truth (`Binance BTC/USDT`, `BLS.gov`,
     `FederalReserve.gov`, `AP`, `Fox`, `NBC`, `Official Joint Statement`)
  3. 50-50 Tie-Breaker & Cancellation Clauses (`postponed`, `canceled`,
     `voided`, `resolves 50-50`)
  4. Quantitative Ambiguity Hazard Score (`[0.0, 1.0]`) used in Stage 6 by
     `uma_rule_checker_mcp` to automatically block our $5.00 survival bot from
     betting on subjective "trap" markets.

Outputs compressed Parquet shards (`data/raw_lake/uma_oracle_rules/uma_parsed_rules_shard_*.parquet`)
and formatted `<uma_rule>` blocks for Stage 4 Continual Pre-Training (CPT).
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .ancillary_data_decoder import UmaAncillaryDataDecoder
from .dispute_history_scraper import UmaDisputeHistoryScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [UMA-RULE-PARSER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.resolution_edge_case_parser")

TIMEZONE_REGEX = re.compile(
    r"\b(UTC|GMT|EST|EDT|ET|CST|CDT|CT|PST|PDT|PT|IST|CET)\b",
    re.IGNORECASE,
)
EXPLICIT_CLOCK_REGEX = re.compile(
    r"\b([0-1]?[0-9]:[0-5][0-9](?::[0-5][0-9])?\s*(?:AM|PM|am|pm)?)\b"
)
URL_DOMAIN_REGEX = re.compile(
    r"https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    re.IGNORECASE,
)

AUTHORITY_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(binance|coinbase|kraken|coingecko|coinmarketcap)\b", re.I), "CRYPTO_EXCHANGE_INDEX"),
    (re.compile(r"\b(federal reserve|fomc|bureau of labor statistics|bls|bea|treasury)\b", re.I), "US_FEDERAL_AGENCY"),
    (re.compile(r"\b(associated press|\bap\b|fox news|nbc|decision desk|reuters|bloomberg)\b", re.I), "TIER1_NEWS_DECISION_DESK"),
    (re.compile(r"\b(official .*? statement|press release|federal register|supreme court|sec\.gov)\b", re.I), "OFFICIAL_LEGAL_PUBLICATION"),
    (re.compile(r"\b(nba\.com|nfl\.com|fifa|espn|ufc)\b", re.I), "OFFICIAL_SPORTS_LEAGUE"),
    (re.compile(r"\b(consensus of credible|credible reporting|media consensus)\b", re.I), "UMA_CREDIBLE_MEDIA_CONSENSUS"),
]

TIE_BREAKER_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(50-50|50/50|0\.50|resolve to 50%)\b", re.I), "EXPLICIT_50_50_SPLIT"),
    (re.compile(r"\b(postponed|delayed|rescheduled|canceled|cancelled)\b", re.I), "POSTPONEMENT_OR_CANCELLATION_CLAUSE"),
    (re.compile(r"\b(otherwise,? this market will resolve to no|otherwise resolves to no)\b", re.I), "STRICT_DEFAULT_NO_FALLBACK"),
    (re.compile(r"\b(announcement alone is not sufficient|rumors .*? do not count)\b", re.I), "ANNOUNCEMENT_INSUFFICIENT_GUARD"),
]


# ==============================================================================
# 1. STRUCTURED RESOLUTION RULE SCHEMAS
# ==============================================================================
@dataclass(frozen=True)
class TimezoneCutoffSpec:
    """Captures the explicit timezone and clock cutoff inside a UMA rule."""
    detected_timezone: str
    explicit_clock_time: str
    is_strictly_utc_normalized: bool


@dataclass(frozen=True)
class ParsedUmaResolutionRule:
    """
    Complete machine-readable decomposition of a Polymarket contract's UMA
    Optimistic Oracle resolution rules.
    """
    market_id: str
    condition_id: str
    question_id: str
    question: str
    primary_authority_type: str
    referenced_domains: List[str]
    detected_timezone: str
    explicit_clock_cutoff: str
    tie_breaker_clauses: List[str]
    has_strict_default_no: bool
    has_announcement_exclusion_trap: bool
    dispute_trigger_category: str
    ambiguity_hazard_score: float
    safe_for_survival_5usd_bot: bool
    ancillary_sha256_digest: str
    cpt_formatted_rule_tag: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["referenced_domains_json"] = json.dumps(self.referenced_domains)
        payload["tie_breaker_clauses_json"] = json.dumps(self.tie_breaker_clauses)
        del payload["referenced_domains"]
        del payload["tie_breaker_clauses"]
        return payload


# ==============================================================================
# 2. CORE RESOLUTION EDGE-CASE PARSER
# ==============================================================================
class ResolutionEdgeCaseParser:
    """
    Parses raw Polymarket descriptions and UMA `ancillaryData` blobs into
    deterministic edge-case flags, authority types, and `<uma_rule>` XML tags.
    """

    def __init__(self, max_allowed_survival_hazard: float = 0.45) -> None:
        self.max_allowed_survival_hazard = max_allowed_survival_hazard
        self.decoder = UmaAncillaryDataDecoder()

    def extract_timezone_and_clock(self, text: str) -> TimezoneCutoffSpec:
        tz_matches = TIMEZONE_REGEX.findall(text)
        clock_matches = EXPLICIT_CLOCK_REGEX.findall(text)

        detected_tz = tz_matches[0].upper() if tz_matches else "IMPLICIT_ET_OR_UTC"
        clock_str = clock_matches[0].strip() if clock_matches else "23:59:59"
        is_utc = detected_tz in ("UTC", "GMT")

        return TimezoneCutoffSpec(
            detected_timezone=detected_tz,
            explicit_clock_time=clock_str,
            is_strictly_utc_normalized=is_utc,
        )

    def parse_market_resolution_rules(
        self, market_row: Dict[str, Any]
    ) -> ParsedUmaResolutionRule:
        """
        Extracts all settlement clauses, authority sources, and ambiguity hazard
        metrics from a Stage 2 Part A market record.
        """
        decoded = self.decoder.synthesize_from_gamma_market_row(market_row)
        q_text = decoded.extracted_title
        desc_text = decoded.extracted_description
        res_src = str(market_row.get("resolution_source") or "")
        full_corpus = f"{q_text}\n{desc_text}\n{res_src}"

        # 1. Identify primary resolution authority
        authorities: List[str] = []
        for pattern, auth_label in AUTHORITY_PATTERNS:
            if pattern.search(full_corpus):
                authorities.append(auth_label)
        primary_authority = authorities[0] if authorities else "UNSPECIFIED_GENERAL_ORACLE"

        # 2. Extract referenced web domains
        domains = sorted(set(m.lower() for m in URL_DOMAIN_REGEX.findall(full_corpus)))

        # 3. Extract timezone & clock cutoff
        tz_spec = self.extract_timezone_and_clock(full_corpus)

        # 4. Extract tie-breaker and trap clauses
        tie_clauses: List[str] = []
        for pattern, clause_label in TIE_BREAKER_PATTERNS:
            if pattern.search(full_corpus):
                tie_clauses.append(clause_label)

        has_default_no = "STRICT_DEFAULT_NO_FALLBACK" in tie_clauses
        has_announcement_trap = "ANNOUNCEMENT_INSUFFICIENT_GUARD" in tie_clauses

        # 5. Compute quantitative Ambiguity Hazard Score in [0.05, 0.95]
        trig_cat, elevated_dispute = UmaDisputeHistoryScraper.classify_dispute_trigger_from_text(
            q_text, desc_text
        )
        hazard = 0.10
        if elevated_dispute:
            hazard += 0.30
        if primary_authority in ("UMA_CREDIBLE_MEDIA_CONSENSUS", "UNSPECIFIED_GENERAL_ORACLE"):
            hazard += 0.15
        if tz_spec.detected_timezone == "IMPLICIT_ET_OR_UTC":
            hazard += 0.10
        if len(desc_text) < 60:
            hazard += 0.20
        if domains or primary_authority in ("US_FEDERAL_AGENCY", "CRYPTO_EXCHANGE_INDEX"):
            hazard -= 0.10

        hazard_clamped = round(max(0.05, min(0.95, hazard)), 4)
        is_safe_for_survival = hazard_clamped <= self.max_allowed_survival_hazard

        # 6. Build the compact `<uma_rule>` XML block for Stage 4 CPT & Stage 5 GRPO prompts
        cpt_tag = (
            f"<uma_rule authority=\"{primary_authority}\" "
            f"tz=\"{tz_spec.detected_timezone}\" "
            f"cutoff=\"{tz_spec.explicit_clock_time}\" "
            f"hazard=\"{hazard_clamped}\">\n"
            f"  {desc_text}\n"
            f"</uma_rule>"
        )

        return ParsedUmaResolutionRule(
            market_id=str(market_row.get("market_id") or ""),
            condition_id=str(market_row.get("condition_id") or ""),
            question_id=decoded.question_id,
            question=q_text,
            primary_authority_type=primary_authority,
            referenced_domains=domains,
            detected_timezone=tz_spec.detected_timezone,
            explicit_clock_cutoff=tz_spec.explicit_clock_time,
            tie_breaker_clauses=tie_clauses,
            has_strict_default_no=has_default_no,
            has_announcement_exclusion_trap=has_announcement_trap,
            dispute_trigger_category=trig_cat,
            ambiguity_hazard_score=hazard_clamped,
            safe_for_survival_5usd_bot=is_safe_for_survival,
            ancillary_sha256_digest=decoded.ancillary_sha256_digest,
            cpt_formatted_rule_tag=cpt_tag,
        )

    # --------------------------------------------------------------------------
    # Batch Processing over Stage 2 Part A Gamma Shards
    # --------------------------------------------------------------------------
    def process_gamma_lake_to_parquet(
        self,
        gamma_input_dir: pathlib.Path = pathlib.Path("data/raw_lake/polymarket_gamma"),
        uma_output_dir: pathlib.Path = pathlib.Path("data/raw_lake/uma_oracle_rules"),
    ) -> Dict[str, Any]:
        uma_output_dir.mkdir(parents=True, exist_ok=True)
        parquet_files = sorted(gamma_input_dir.glob("*.parquet"))

        market_rows: List[Dict[str, Any]] = []
        if parquet_files:
            import polars as pl  # type: ignore

            for pf in parquet_files:
                market_rows.extend(pl.read_parquet(pf).to_dicts())

        parsed_rows: List[Dict[str, Any]] = []
        safe_count = 0
        for mkt in market_rows:
            parsed = self.parse_market_resolution_rules(mkt)
            if parsed.safe_for_survival_5usd_bot:
                safe_count += 1
            parsed_rows.append(parsed.to_dict())

        out_path = uma_output_dir / "uma_parsed_rules_shard_00000.parquet"
        if parsed_rows:
            import polars as pl  # type: ignore

            pl.from_dicts(parsed_rows).write_parquet(out_path, compression="zstd")
            logger.info("Wrote Polars UMA parsed rules shard: %s (%d rows)", out_path, len(parsed_rows))

        return {
            "stage": "STAGE_2_PART_C_UMA_EDGE_CASE_PARSER",
            "markets_parsed": len(parsed_rows),
            "safe_for_5usd_survival_bot": safe_count,
            "blocked_high_ambiguity_markets": len(parsed_rows) - safe_count,
            "output_parquet_shard": str(out_path),
        }


if __name__ == "__main__":
    parser_engine = ResolutionEdgeCaseParser()
    summary = parser_engine.process_gamma_lake_to_parquet()
    print(json.dumps(summary, indent=2))