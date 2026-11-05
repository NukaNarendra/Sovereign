#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART A)
File: 02_dataset_factory/temporal_aligner/clock_skew_corrector.py
================================================================================
Multi-Source Timestamp Normalizer & Anti-Leakage Clock-Skew Guard.

Why Clock-Skew Correction Is Mandatory in Quantitative Prediction Markets:
  1. Our Stage 2 raw data lake (`data/raw_lake/`) ingests timestamps from four
     physically distinct clocks:
       - Polymarket Gamma & CLOB REST APIs (ISO-8601 UTC strings & ms epochs)
       - Polygon PoS Mainnet Chain ID 137 blocks (~2.15s block times with
         validator timestamp drift of +/- 15 seconds)
       - UMA Optimistic Oracle resolution rules (frequently written in US
         Eastern Time `ET` — `EST` UTC-5 in winter vs `EDT` UTC-4 in summer)
       - News Wire & RSS/Tavily crawlers (where `published_date` may reflect
         midnight `00:00:00Z` date-only truncation or up to 300 seconds of
         CDN syndication delay).
  2. If a news wire timestamp is truncated to `2026-03-18T00:00:00Z` when the
     article was actually published at `2026-03-18T19:30:00Z` (after the 18:00Z
     FOMC announcement), a naive join `pub_ts <= cutoff_ts` will leak the
     future FOMC outcome into an afternoon snapshot!
  3. This module enforces:
       - Exact US Eastern (`ET`) Daylight Saving Time (`EDT`/`EST`) to `UTC`
         conversion rules.
       - A configurable **Wire Propagation Safety Buffer** (`wire_safety_lag_sec`,
         default 300s / 5 minutes) and **Date-Only Midnight Penalty** (+86,399s
         pushed to `23:59:59Z`) so truncated date strings are treated as end-of-day
         publications and can never leak intra-day outcomes.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import math
import logging
import pathlib
import argparse
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [CLOCK-SKEW] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.clock_skew_corrector")

# Regex to detect date-only strings ("2025-11-05") or midnight-truncated ISO strings
DATE_ONLY_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MIDNIGHT_TRUNCATED_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]00:00:00(?:\.0+)?(?:Z|\+00:00)?$")


# ==============================================================================
# 1. CONFIGURATION & NORMALIZED TIMESTAMP SCHEMAS
# ==============================================================================
@dataclass(frozen=True)
class ClockSkewConfig:
    """
    Governs safety buffers applied when aligning asynchronous market, blockchain,
    and news wire timestamps into a single canonical UTC timeline.
    """
    # Minimum wire syndication lag buffer (5 minutes = 300 seconds)
    wire_propagation_safety_lag_sec: int = 300
    # Polygon PoS (Chain 137) block timestamp drift tolerance (15 seconds)
    polygon_block_drift_guard_sec: int = 15
    # Reference Polygon block and epoch for block-to-timestamp interpolation
    reference_polygon_block: int = 68_500_000
    reference_polygon_epoch_sec: int = 1_740_000_000
    polygon_avg_block_time_sec: float = 2.15
    # Push ambiguous 00:00:00 date-only news stamps to 23:59:59 UTC to prevent intra-day leaks
    penalize_midnight_truncated_news: bool = True


@dataclass(frozen=True)
class NormalizedTimestampResult:
    """
    Auditable record of a timestamp after timezone normalization and safety-lag
    adjustment.
    """
    raw_input: str
    source_type: str
    detected_timezone: str
    raw_epoch_sec: int
    safety_adjustment_sec: int
    effective_safe_epoch_sec: int
    effective_safe_iso_utc: str
    was_midnight_penalized: bool
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE CLOCK-SKEW CORRECTOR & TIMEZONE NORMALIZATION ENGINE
# ==============================================================================
class ClockSkewCorrector:
    """
    Normalizes timestamps across Gamma, CLOB v2, Polygon Chain 137 blocks,
    UMA Eastern Time cutoffs, and Tavily news wires into strict anti-leakage
    effective UTC epochs.
    """

    # Static timezone offsets (in seconds from UTC) for explicit abbreviations
    STATIC_TZ_OFFSETS_SEC: Dict[str, int] = {
        "UTC": 0,
        "GMT": 0,
        "EST": -5 * 3600,
        "EDT": -4 * 3600,
        "CST": -6 * 3600,
        "CDT": -5 * 3600,
        "PST": -8 * 3600,
        "PDT": -7 * 3600,
        "CET": +1 * 3600,
        "IST": +5 * 3600 + 1800,
    }

    def __init__(self, config: Optional[ClockSkewConfig] = None) -> None:
        self.config = config or ClockSkewConfig()

    # --------------------------------------------------------------------------
    # US Eastern Time (`ET`) Daylight Saving Time (DST) Resolver
    # --------------------------------------------------------------------------
    @staticmethod
    def is_us_eastern_dst(dt_utc_naive: datetime.datetime) -> bool:
        """
        Determines whether a given date falls inside US Daylight Saving Time
        (`EDT`, UTC-4: Second Sunday in March to First Sunday in November) or
        Standard Time (`EST`, UTC-5).
        """
        year = dt_utc_naive.year
        # Second Sunday in March
        march_1 = datetime.datetime(year, 3, 1)
        days_to_first_sun_mar = (6 - march_1.weekday()) % 7
        dst_start = datetime.datetime(year, 3, 1 + days_to_first_sun_mar + 7, 7, 0, 0)

        # First Sunday in November
        nov_1 = datetime.datetime(year, 11, 1)
        days_to_first_sun_nov = (6 - nov_1.weekday()) % 7
        dst_end = datetime.datetime(year, 11, 1 + days_to_first_sun_nov, 6, 0, 0)

        return dst_start <= dt_utc_naive < dst_end

    def resolve_tz_offset_seconds(
        self, tz_label: str, reference_dt: datetime.datetime
    ) -> int:
        """
        Returns the signed offset in seconds from UTC (`local_time - offset = UTC`).
        Handles dynamic `ET` / `IMPLICIT_ET_OR_UTC` based on US DST rules.
        """
        clean_tz = str(tz_label or "UTC").strip().upper()
        if clean_tz in self.STATIC_TZ_OFFSETS_SEC:
            return self.STATIC_TZ_OFFSETS_SEC[clean_tz]

        if clean_tz in ("ET", "US/EASTERN", "AMERICA/NEW_YORK"):
            return -4 * 3600 if self.is_us_eastern_dst(reference_dt) else -5 * 3600

        return 0

    # --------------------------------------------------------------------------
    # Core ISO / Epoch / Block Timestamp Parser with Anti-Leak Safety Buffers
    # --------------------------------------------------------------------------
    def normalize_timestamp(
        self,
        raw_value: Any,
        source_type: str = "CLOB_SNAPSHOT",
        explicit_tz: str = "UTC",
    ) -> NormalizedTimestampResult:
        """
        Parses an ISO-8601 string, UNIX epoch (seconds or milliseconds), or
        date-only string and applies source-specific anti-leakage safety buffers:
          - `CLOB_SNAPSHOT` / `GAMMA_CLOSE`: No forward penalty (`adjustment = 0`)
          - `NEWS_ARTICLE`                 : Adds `+wire_propagation_safety_lag_sec`
                                             (300s) so an article is only usable
                                             5 minutes after its stated timestamp.
                                             If midnight-truncated (`00:00:00`),
                                             pushes to `23:59:59Z` (`+86,399s`).
          - `POLYGON_BLOCK`                : Adds `+polygon_block_drift_guard_sec` (15s).
        """
        raw_str = str(raw_value or "").strip()
        if not raw_str:
            return NormalizedTimestampResult(
                raw_input="",
                source_type=source_type,
                detected_timezone=explicit_tz,
                raw_epoch_sec=0,
                safety_adjustment_sec=0,
                effective_safe_epoch_sec=0,
                effective_safe_iso_utc="1970-01-01T00:00:00+00:00",
                was_midnight_penalized=False,
                is_valid=False,
            )

        was_midnight_penalized = False
        raw_epoch: int = 0

        try:
            # Case 1: Pure numeric epoch (seconds or milliseconds)
            if raw_str.replace(".", "", 1).isdigit():
                num_val = float(raw_str)
                if num_val > 10_000_000_000:
                    num_val = num_val / 1000.0
                raw_epoch = int(num_val)
            else:
                # Case 2: ISO-8601 or YYYY-MM-DD string
                is_date_only = bool(DATE_ONLY_REGEX.match(raw_str))
                cleaned_iso = raw_str.replace("Z", "+00:00")
                dt = datetime.datetime.fromisoformat(cleaned_iso)

                if dt.tzinfo is None:
                    tz_offset = self.resolve_tz_offset_seconds(explicit_tz, dt)
                    tz_info = datetime.timezone(datetime.timedelta(seconds=tz_offset))
                    dt = dt.replace(tzinfo=tz_info)

                dt_utc = dt.astimezone(datetime.timezone.utc)
                raw_epoch = int(dt_utc.timestamp())

                if source_type == "NEWS_ARTICLE" and self.config.penalize_midnight_truncated_news:
                    if is_date_only or MIDNIGHT_TRUNCATED_REGEX.match(raw_str):
                        was_midnight_penalized = True
        except Exception as exc:
            logger.debug("Failed to parse timestamp %r: %s", raw_str, exc)
            return NormalizedTimestampResult(
                raw_input=raw_str,
                source_type=source_type,
                detected_timezone=explicit_tz,
                raw_epoch_sec=0,
                safety_adjustment_sec=0,
                effective_safe_epoch_sec=0,
                effective_safe_iso_utc="1970-01-01T00:00:00+00:00",
                was_midnight_penalized=False,
                is_valid=False,
            )

        # Compute source-specific anti-leakage safety adjustment
        safety_adj = 0
        if source_type == "NEWS_ARTICLE":
            if was_midnight_penalized:
                # Push ambiguous date-only news to the end of that UTC calendar day
                safety_adj += 86399
            else:
                safety_adj += self.config.wire_propagation_safety_lag_sec
        elif source_type == "POLYGON_BLOCK":
            safety_adj += self.config.polygon_block_drift_guard_sec

        effective_epoch = raw_epoch + safety_adj
        effective_iso = datetime.datetime.fromtimestamp(
            effective_epoch, tz=datetime.timezone.utc
        ).isoformat()

        return NormalizedTimestampResult(
            raw_input=raw_str,
            source_type=source_type,
            detected_timezone=explicit_tz,
            raw_epoch_sec=raw_epoch,
            safety_adjustment_sec=safety_adj,
            effective_safe_epoch_sec=effective_epoch,
            effective_safe_iso_utc=effective_iso,
            was_midnight_penalized=was_midnight_penalized,
            is_valid=True,
        )

    def estimate_epoch_from_polygon_block(self, block_number: int) -> int:
        """
        Converts a Polygon PoS Mainnet (Chain ID 137) block number into a
        safety-buffered UTC epoch timestamp.
        """
        delta_blocks = int(block_number) - self.config.reference_polygon_block
        est_epoch = int(
            self.config.reference_polygon_epoch_sec
            + (delta_blocks * self.config.polygon_avg_block_time_sec)
        )
        return est_epoch + self.config.polygon_block_drift_guard_sec

    def verify_strict_causality(
        self,
        feature_timestamp: Any,
        snapshot_cutoff_timestamp: Any,
        feature_source_type: str = "NEWS_ARTICLE",
    ) -> Tuple[bool, float, NormalizedTimestampResult, NormalizedTimestampResult]:
        """
        Verifies that `effective_feature_epoch <= snapshot_cutoff_epoch`.
        Returns `(is_causal_clean, lead_time_hours, norm_feature, norm_cutoff)`.
        """
        norm_feat = self.normalize_timestamp(
            feature_timestamp, source_type=feature_source_type
        )
        norm_cut = self.normalize_timestamp(
            snapshot_cutoff_timestamp, source_type="CLOB_SNAPSHOT"
        )
        if not norm_feat.is_valid or not norm_cut.is_valid:
            return False, -999.0, norm_feat, norm_cut

        delta_sec = norm_cut.effective_safe_epoch_sec - norm_feat.effective_safe_epoch_sec
        lead_hours = round(delta_sec / 3600.0, 4)
        is_causal = delta_sec >= 0
        return is_causal, lead_hours, norm_feat, norm_cut


# ==============================================================================
# 3. CLI & SELF-TEST ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Stage 3 Part A: Clock-Skew Corrector"
    )
    parser.add_argument(
        "--clob-shard",
        type=pathlib.Path,
        default=pathlib.Path(
            "data/raw_lake/polymarket_clob_v2/clob_horizon_snapshots_shard_00000.parquet"
        ),
    )
    args = parser.parse_args()

    corrector = ClockSkewCorrector()

    # 1. Test standard news wire vs midnight-truncated date-only wire
    cutoff = "2026-03-18T18:00:00Z"
    clean_wire = "2026-03-18T15:30:00Z"
    truncated_wire = "2026-03-18T00:00:00Z"

    ok_clean, lead_clean, _, _ = corrector.verify_strict_causality(clean_wire, cutoff)
    ok_trunc, lead_trunc, norm_trunc, _ = corrector.verify_strict_causality(
        truncated_wire, cutoff
    )

    # 2. Audit Stage 2 Part B CLOB Parquet shard if present
    rows_audited = 0
    valid_cutoffs = 0
    if args.clob_shard.exists():
        import polars as pl  # type: ignore

        df = pl.read_parquet(args.clob_shard)
        rows_audited = len(df)
        for row in df.to_dicts():
            res = corrector.normalize_timestamp(
                row.get("cutoff_timestamp_utc"), source_type="CLOB_SNAPSHOT"
            )
            if res.is_valid and res.effective_safe_epoch_sec > 1_600_000_000:
                valid_cutoffs += 1

    report = {
        "stage": "STAGE_3_PART_A_CLOCK_SKEW_CORRECTOR",
        "clob_horizon_rows_audited": rows_audited,
        "valid_utc_cutoffs_verified": valid_cutoffs,
        "unit_tests": {
            "clean_intraday_wire_allowed": ok_clean,
            "clean_intraday_lead_hours": lead_clean,
            "midnight_truncated_wire_quarantined": (not ok_trunc),
            "midnight_penalized_effective_utc": norm_trunc.effective_safe_iso_utc,
            "midnight_lead_hours_after_penalty": lead_trunc,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())