#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART B)
File: 02_dataset_factory/leak_auditor/future_date_regex_scrubber.py
================================================================================
Regex & AST Calendar Date Extractor and Post-Cutoff Temporal Leak Scrubber.

Why Timestamp Metadata Alone Is Not Enough to Prevent Data Leakage:
  1. Even when a news article's metadata `published_date` says `2024-10-25`, web
     scrapers frequently pull updated "Live Blog" pages or retrofitted summaries
     whose body text contains sentences added later:
       "Update (Nov 6, 2024): Following Tuesday's decisive victory..."
  2. If Qwen-2.5-7B sees a date inside `<news_context>` that is chronologically
     later than `cutoff_timestamp_utc` (e.g., `2024-11-06` inside a `2024-10-30`
     snapshot), it immediately detects the temporal anomaly and exploits it.
  3. Importantly, prediction market questions and UMA rules *legitimately*
     mention future deadline dates (e.g., "Will Bitcoin reach $100k before
     July 1, 2026?"). Therefore, this module:
       - Strictly enforces `mentioned_date_epoch <= cutoff_epoch_sec` inside
         historical news articles, headlines, and social snippets (`cpt_news_xml`).
       - Redacts any post-cutoff date clause in news text with
         `[REDACTED_POST_CUTOFF_DATE]` or quarantines the contaminated article.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
import argparse
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [DATE-SCRUBBER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.future_date_regex_scrubber")

# ==============================================================================
# 1. MULTI-FORMAT CALENDAR DATE REGEX PATTERNS
# ==============================================================================
# Pattern A: ISO-8601 dates & timestamps (e.g. 2026-03-18 or 2026-03-18T15:30:00Z)
ISO_DATE_PATTERN = re.compile(
    r"\b(?P<year>20[2-3][0-9])-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12][0-9]|3[01])"
    r"(?:[T ](?P<hour>[01][0-9]|2[0-3]):(?P<minute>[0-5][0-9])(?::(?P<second>[0-5][0-9]))?)?"
)

# Pattern B: Month-name Day, Year (e.g., "November 5, 2024" or "Mar 18, 2026")
MONTH_NAME_FIRST_PATTERN = re.compile(
    r"\b(?P<month_str>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?,\s*(?P<year>20[2-3][0-9])\b",
    re.IGNORECASE,
)

# Pattern C: Day Month-name Year (e.g., "18 March 2026" or "5 Nov 2024")
DAY_FIRST_MONTH_PATTERN = re.compile(
    r"\b(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<month_str>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?,?\s+(?P<year>20[2-3][0-9])\b",
    re.IGNORECASE,
)

MONTH_ABBR_TO_INT: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Context phrases that legitimately precede a future contract resolution deadline
LEGITIMATE_DEADLINE_CONTEXT_REGEX = re.compile(
    r"\b(before|by|prior to|on or before|scheduled for|expires on|close on|until|deadline)\s*$",
    re.IGNORECASE,
)


# ==============================================================================
# 2. DATA STRUCTURES FOR DATE MENTIONS & SCRUBBING AUDITS
# ==============================================================================
@dataclass(frozen=True)
class FutureDateMention:
    """
    Records a single calendar date detected inside a text block that falls
    chronologically after the snapshot's `cutoff_epoch_sec`.
    """
    matched_text: str
    parsed_iso_utc: str
    parsed_epoch_sec: int
    cutoff_epoch_sec: int
    hours_after_cutoff: float
    is_legitimate_deadline_reference: bool
    surrounding_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DateScrubAuditResult:
    """
    Complete audit result for a single `PointInTimeAlignedRecord` after scanning
    its news context for post-cutoff date leaks.
    """
    snapshot_uid: str
    cutoff_timestamp_utc: str
    cutoff_epoch_sec: int
    total_dates_extracted: int
    unlawful_future_dates_count: int
    legitimate_deadline_dates_count: int
    was_scrubbed: bool
    is_clean_after_scrub: bool
    sanitized_news_xml: str
    violations: List[FutureDateMention]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = [v.to_dict() for v in self.violations]
        return payload


# ==============================================================================
# 3. CORE POST-CUTOFF FUTURE DATE REGEX SCRUBBER ENGINE
# ==============================================================================
class FutureDateRegexScrubber:
    """
    Extracts all calendar dates from text fields, compares them against the
    snapshot's `cutoff_epoch_sec`, distinguishes legitimate contract expiry
    deadlines from unlawful post-cutoff news updates, and redacts any leak.
    """

    def __init__(self, allow_deadline_context_in_news: bool = True) -> None:
        self.allow_deadline_context_in_news = allow_deadline_context_in_news

    @staticmethod
    def _safe_build_utc_epoch(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
    ) -> Optional[Tuple[int, str]]:
        """Safely constructs a timezone-aware UTC datetime and epoch."""
        try:
            dt = datetime.datetime(
                year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc
            )
            return int(dt.timestamp()), dt.isoformat()
        except ValueError:
            return None

    def extract_all_date_mentions(
        self, text: str
    ) -> List[Tuple[str, int, str, int, int]]:
        """
        Scans `text` across ISO, Month-First, and Day-First date patterns.
        Returns a list of `(matched_str, epoch_sec, iso_utc, start_idx, end_idx)`.
        """
        results: List[Tuple[str, int, str, int, int]] = []

        # 1. ISO-8601 dates/timestamps
        for match in ISO_DATE_PATTERN.finditer(text):
            yr = int(match.group("year"))
            mo = int(match.group("month"))
            da = int(match.group("day"))
            hr = int(match.group("hour") or 0)
            mi = int(match.group("minute") or 0)
            se = int(match.group("second") or 0)
            built = self._safe_build_utc_epoch(yr, mo, da, hr, mi, se)
            if built is not None:
                results.append(
                    (match.group(0), built[0], built[1], match.start(), match.end())
                )

        # 2. Month-name first ("November 5, 2024")
        for match in MONTH_NAME_FIRST_PATTERN.finditer(text):
            mo_prefix = match.group("month_str")[:3].lower()
            mo = MONTH_ABBR_TO_INT.get(mo_prefix, 1)
            da = int(match.group("day"))
            yr = int(match.group("year"))
            built = self._safe_build_utc_epoch(yr, mo, da)
            if built is not None:
                results.append(
                    (match.group(0), built[0], built[1], match.start(), match.end())
                )

        # 3. Day first ("18 March 2026")
        for match in DAY_FIRST_MONTH_PATTERN.finditer(text):
            da = int(match.group("day"))
            mo_prefix = match.group("month_str")[:3].lower()
            mo = MONTH_ABBR_TO_INT.get(mo_prefix, 1)
            yr = int(match.group("year"))
            built = self._safe_build_utc_epoch(yr, mo, da)
            if built is not None:
                results.append(
                    (match.group(0), built[0], built[1], match.start(), match.end())
                )

        results.sort(key=lambda item: item[3])
        return results

    def audit_and_scrub_text(
        self,
        snapshot_uid: str,
        text: str,
        cutoff_epoch_sec: int,
        cutoff_timestamp_utc: str,
    ) -> DateScrubAuditResult:
        """
        Audits `text` (such as `cpt_news_xml`) against `cutoff_epoch_sec`.
        Any date strictly greater than `cutoff_epoch_sec` that is NOT preceded
        by a legitimate contract-deadline preposition ("before", "by", "until")
        is flagged as a temporal leak and replaced with `[REDACTED_POST_CUTOFF_DATE]`.
        """
        mentions = self.extract_all_date_mentions(text)
        violations: List[FutureDateMention] = []
        legit_deadlines = 0
        sanitized_text = text

        for matched_str, parsed_epoch, parsed_iso, start_pos, end_pos in mentions:
            if parsed_epoch <= cutoff_epoch_sec:
                continue

            # Inspect the 32 characters immediately preceding the date mention
            prefix_window = text[max(0, start_pos - 32) : start_pos]
            is_deadline_ref = bool(
                self.allow_deadline_context_in_news
                and LEGITIMATE_DEADLINE_CONTEXT_REGEX.search(prefix_window)
            )

            hrs_after = round((parsed_epoch - cutoff_epoch_sec) / 3600.0, 2)
            snippet = text[max(0, start_pos - 25) : min(len(text), end_pos + 35)].replace(
                "\n", " "
            )

            if is_deadline_ref:
                legit_deadlines += 1
            else:
                violations.append(
                    FutureDateMention(
                        matched_text=matched_str,
                        parsed_iso_utc=parsed_iso,
                        parsed_epoch_sec=parsed_epoch,
                        cutoff_epoch_sec=cutoff_epoch_sec,
                        hours_after_cutoff=hrs_after,
                        is_legitimate_deadline_reference=False,
                        surrounding_snippet=snippet,
                    )
                )
                sanitized_text = sanitized_text.replace(
                    matched_str, "[REDACTED_POST_CUTOFF_DATE]"
                )

        # Verify zero remaining unlawful future dates after scrubbing
        post_check_mentions = self.extract_all_date_mentions(sanitized_text)
        remaining_unlawful = 0
        for _, p_epoch, _, s_pos, _ in post_check_mentions:
            if p_epoch > cutoff_epoch_sec:
                pref = sanitized_text[max(0, s_pos - 32) : s_pos]
                if not (
                    self.allow_deadline_context_in_news
                    and LEGITIMATE_DEADLINE_CONTEXT_REGEX.search(pref)
                ):
                    remaining_unlawful += 1

        return DateScrubAuditResult(
            snapshot_uid=snapshot_uid,
            cutoff_timestamp_utc=cutoff_timestamp_utc,
            cutoff_epoch_sec=cutoff_epoch_sec,
            total_dates_extracted=len(mentions),
            unlawful_future_dates_count=len(violations),
            legitimate_deadline_dates_count=legit_deadlines,
            was_scrubbed=len(violations) > 0,
            is_clean_after_scrub=(remaining_unlawful == 0),
            sanitized_news_xml=sanitized_text,
            violations=violations,
        )

    def audit_aligned_parquet_shard(
        self,
        input_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/temporal_aligned/pit_aligned_snapshots_shard_00000.parquet"
        ),
    ) -> Dict[str, Any]:
        """
        Runs the post-cutoff future date scrubber across every row in the
        Stage 3 Part A aligned Parquet shard.
        """
        import polars as pl  # type: ignore

        if not input_parquet.exists():
            raise FileNotFoundError(f"Aligned Parquet shard not found at {input_parquet}")

        rows = pl.read_parquet(input_parquet).to_dicts()
        total_dates = 0
        total_violations = 0
        total_legit_deadlines = 0
        all_clean = True

        for row in rows:
            res = self.audit_and_scrub_text(
                snapshot_uid=str(row.get("snapshot_uid") or ""),
                text=str(row.get("cpt_news_xml") or ""),
                cutoff_epoch_sec=int(row.get("cutoff_epoch_sec") or 0),
                cutoff_timestamp_utc=str(row.get("cutoff_timestamp_utc") or ""),
            )
            total_dates += res.total_dates_extracted
            total_violations += res.unlawful_future_dates_count
            total_legit_deadlines += res.legitimate_deadline_dates_count
            if not res.is_clean_after_scrub:
                all_clean = False

        # Run a synthetic adversarial test to prove the scrubber catches injected future dates
        adv_sample = (
            "<news_context cutoff_utc=\"2026-03-10T12:00:00+00:00\">\n"
            "  - [2026-03-10T09:00:00+00:00] (reuters.com): Contract expires before July 1, 2026.\n"
            "  - Update added on March 25, 2026: Official statement released.\n"
            "</news_context>"
        )
        adv_res = self.audit_and_scrub_text(
            snapshot_uid="ADVERSARIAL-TEST-01",
            text=adv_sample,
            cutoff_epoch_sec=1773144000,  # 2026-03-10T12:00:00Z
            cutoff_timestamp_utc="2026-03-10T12:00:00+00:00",
        )

        return {
            "stage": "STAGE_3_PART_B_FUTURE_DATE_REGEX_SCRUBBER",
            "rows_audited": len(rows),
            "total_calendar_dates_inspected": total_dates,
            "unlawful_post_cutoff_dates_in_warehouse": total_violations,
            "legitimate_deadline_mentions_allowed": total_legit_deadlines,
            "warehouse_zero_future_date_leak_verified": (total_violations == 0 and all_clean),
            "adversarial_unit_test": {
                "injected_future_date_caught": adv_res.unlawful_future_dates_count == 1,
                "legitimate_july_deadline_preserved": adv_res.legitimate_deadline_dates_count == 1,
                "sanitized_output_clean": adv_res.is_clean_after_scrub,
                "sanitized_text_preview": adv_res.sanitized_news_xml,
            },
        }


# ==============================================================================
# 4. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    scrubber = FutureDateRegexScrubber()
    report = scrubber.audit_aligned_parquet_shard()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())