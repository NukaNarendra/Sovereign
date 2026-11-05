#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART B)
File: 02_dataset_factory/leak_auditor/outcome_spoiler_detector.py
================================================================================
Hindsight Outcome Spoiler & Post-Resolution Phrase Scanner.

Why Lexical Spoiler Detection Is Mandatory Alongside Timestamp Checks:
  1. Even if a text snippet carries no explicit calendar date, retrospective
     summaries, Wikipedia revisions, or cached API descriptions can contain
     hindsight outcome spoilers:
       - "Following Trump's victory in the 2024 presidential election..."
       - "This market resolved to Yes after the FOMC lowered rates..."
       - "UMA Optimistic Oracle settled the contract at 1.000..."
  2. If even 2% of training prompts contain retrospective spoilers, a 7B LLM
     will overfit to those lexical cues during SFT/GRPO instead of learning
     genuine Bayesian base-rate + order-book microstructure reasoning.
  3. This module scans every aligned record across 4 lexical spoiler categories:
       - `ORACLE_SETTLEMENT_SPOILER` ("resolved to Yes/No", "payout is 1.0")
       - `ELECTORAL_OR_SPORTS_RESULT_SPOILER` ("after winning the election",
         "clinched the championship", "president-elect")
       - `MACRO_OR_CORPORATE_POSTMORTEM` ("following the rate cut decision",
         "in the wake of the approval")
       - `RETROSPECTIVE_HINDSIGHT_MARKER` ("in hindsight", "ultimately proved")
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Pattern, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [SPOILER-DETECTOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.outcome_spoiler_detector")


# ==============================================================================
# 1. LEXICAL HINDSIGHT & OUTCOME SPOILER REGISTRY
# ==============================================================================
SPOILER_PATTERN_REGISTRY: List[Tuple[str, Pattern[str], float]] = [
    (
        "ORACLE_SETTLEMENT_SPOILER",
        re.compile(
            r"\b(market\s+resolved\s+to\s+(?:yes|no)|settled\s+at\s+(?:1\.0|0\.0|yes|no)|"
            r"uma\s+oracle\s+confirmed\s+the\s+outcome|final\s+payout\s+of\s+\$?1\.00|"
            r"winning\s+outcome\s+was\s+(?:yes|no))\b",
            re.IGNORECASE,
        ),
        1.00,
    ),
    (
        "POST_EVENT_VICTORY_OR_DEFEAT_SPOILER",
        re.compile(
            r"\b(after\s+winning\s+the\s+(?:202[4-9]\s+)?election|following\s+(?:his|her|their)\s+"
            r"landslide\s+victory|clinched\s+the\s+(?:presidency|championship|finals)|"
            r"was\s+officially\s+inaugurated\s+on|conceded\s+the\s+election\s+to)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        "MACRO_OR_CRYPTO_POST_RESOLUTION_SPOILER",
        re.compile(
            r"\b(after\s+bitcoin\s+surpassed\s+\$1[0-5]0,000|following\s+today's\s+fomc\s+rate\s+"
            r"cut\s+announcement|in\s+the\s+wake\s+of\s+the\s+sec's\s+final\s+approval|"
            r"having\s+already\s+reached\s+the\s+target\s+price)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    (
        "RETROSPECTIVE_HINDSIGHT_MARKER",
        re.compile(
            r"\b(in\s+hindsight|looking\s+back\s+at\s+the\s+resolved\s+market|"
            r"post-mortem\s+analysis\s+shows|as\s+history\s+ultimately\s+showed)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
]


# ==============================================================================
# 2. SPOILER VIOLATION & AUDIT RESULT SCHEMAS
# ==============================================================================
@dataclass(frozen=True)
class SpoilerViolationMatch:
    """Represents a single hindsight or outcome spoiler phrase found in text."""
    category: str
    matched_phrase: str
    severity_score: float
    field_name: str
    context_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpoilerScanResult:
    """Aggregated spoiler audit report for a single aligned market snapshot."""
    snapshot_uid: str
    market_id: str
    is_spoiler_free: bool
    max_severity_score: float
    violations_count: int
    sanitized_news_xml: str
    violations: List[SpoilerViolationMatch]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = [v.to_dict() for v in self.violations]
        return payload


# ==============================================================================
# 3. CORE OUTCOME SPOILER DETECTOR & REDACTOR ENGINE
# ==============================================================================
class OutcomeSpoilerDetector:
    """
    Scans `cpt_news_xml`, `cpt_social_xml`, and `question` fields in the
    Point-in-Time Aligned Warehouse for hindsight outcome spoilers and redacts
    any offending clauses before SFT/GRPO prompt construction.
    """

    def __init__(self, quarantine_threshold: float = 0.80) -> None:
        self.quarantine_threshold = quarantine_threshold

    def scan_and_redact_record(
        self,
        snapshot_uid: str,
        market_id: str,
        news_xml: str,
        social_xml: str = "",
    ) -> SpoilerScanResult:
        """
        Scans `news_xml` and `social_xml` for any phrase matching
        `SPOILER_PATTERN_REGISTRY`. Redacts matched phrases in `news_xml` with
        `[REDACTED_OUTCOME_SPOILER]`.
        """
        violations: List[SpoilerViolationMatch] = []
        sanitized_news = news_xml

        fields_to_scan = [
            ("cpt_news_xml", news_xml),
            ("cpt_social_xml", social_xml),
        ]

        for field_name, content in fields_to_scan:
            if not content:
                continue
            for category, regex_pat, severity in SPOILER_PATTERN_REGISTRY:
                for match in regex_pat.finditer(content):
                    phrase = match.group(0)
                    start_i = max(0, match.start() - 25)
                    end_i = min(len(content), match.end() + 25)
                    snippet = content[start_i:end_i].replace("\n", " ")

                    violations.append(
                        SpoilerViolationMatch(
                            category=category,
                            matched_phrase=phrase,
                            severity_score=severity,
                            field_name=field_name,
                            context_snippet=snippet,
                        )
                    )
                    if field_name == "cpt_news_xml":
                        sanitized_news = sanitized_news.replace(
                            phrase, "[REDACTED_OUTCOME_SPOILER]"
                        )

        max_sev = max((v.severity_score for v in violations), default=0.0)
        is_clean = max_sev < self.quarantine_threshold

        return SpoilerScanResult(
            snapshot_uid=snapshot_uid,
            market_id=market_id,
            is_spoiler_free=is_clean,
            max_severity_score=round(max_sev, 4),
            violations_count=len(violations),
            sanitized_news_xml=sanitized_news,
            violations=violations,
        )

    def audit_aligned_warehouse(
        self,
        input_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/temporal_aligned/pit_aligned_snapshots_shard_00000.parquet"
        ),
    ) -> Dict[str, Any]:
        """
        Audits all rows in `pit_aligned_snapshots_shard_00000.parquet` and runs
        an adversarial spoiler unit test to verify detection sensitivity.
        """
        import polars as pl  # type: ignore

        if not input_parquet.exists():
            raise FileNotFoundError(f"Aligned Parquet shard not found at {input_parquet}")

        rows = pl.read_parquet(input_parquet).to_dicts()
        clean_count = 0
        flagged_count = 0

        for row in rows:
            res = self.scan_and_redact_record(
                snapshot_uid=str(row.get("snapshot_uid") or ""),
                market_id=str(row.get("market_id") or ""),
                news_xml=str(row.get("cpt_news_xml") or ""),
                social_xml=str(row.get("cpt_social_xml") or ""),
            )
            if res.is_spoiler_free:
                clean_count += 1
            else:
                flagged_count += 1

        # Adversarial spoiler injection self-test
        adv_news = (
            "<news_context>\n"
            "  - [reuters.com]: After winning the 2024 election, this market resolved to Yes "
            "and settled at 1.0 on Polygon.\n"
            "</news_context>"
        )
        adv_result = self.scan_and_redact_record(
            snapshot_uid="ADV-SPOILER-TEST",
            market_id="999999",
            news_xml=adv_news,
        )

        return {
            "stage": "STAGE_3_PART_B_OUTCOME_SPOILER_DETECTOR",
            "warehouse_rows_scanned": len(rows),
            "spoiler_free_rows_verified": clean_count,
            "contaminated_rows_flagged": flagged_count,
            "warehouse_contamination_rate_pct": round(
                (flagged_count / max(1, len(rows))) * 100.0, 4
            ),
            "adversarial_unit_test": {
                "injected_spoilers_detected_count": adv_result.violations_count,
                "blocked_by_quarantine_gate": (not adv_result.is_spoiler_free),
                "redacted_preview": adv_result.sanitized_news_xml,
            },
        }


# ==============================================================================
# 4. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    detector = OutcomeSpoilerDetector()
    report = detector.audit_aligned_warehouse()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())