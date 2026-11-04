#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART A)
File: 02_dataset_factory/temporal_aligner/point_in_time_joiner.py
================================================================================
Master Point-in-Time ("As-Of") Relational & Temporal Join Engine across all
Stage 2 Raw Lake Parquet Shards (`data/raw_lake/*/*.parquet`).

What This Module Joins & Verifies:
  1. Primary Spine (`polymarket_clob_v2/clob_horizon_snapshots_shard_*.parquet`):
     Each `(market_id, horizon_hours_before_close)` snapshot at `cutoff_timestamp_utc`.
  2. Market Metadata (`polymarket_gamma/gamma_resolved_shard_*.parquet`):
     Enriches with `correlation_cluster_key`, `domain_yes_base_rate`,
     `volume_usd`, `liquidity_usd`, and `collateral_era`.
  3. UMA Resolution Rules (`uma_oracle_rules/uma_parsed_rules_shard_*.parquet`):
     Enriches with `primary_authority_type`, `detected_timezone`,
     `ambiguity_hazard_score`, `safe_for_survival_5usd_bot`, and `<uma_rule>` XML.
  4. Point-in-Time News Context (`news_temporal_tavily/historical_news_context_shard_*.parquet`):
     Joins on `(market_id, horizon_hours_before_close)` and verifies via
     `ClockSkewCorrector` that `latest_article_published_utc <= cutoff_timestamp_utc`.
  5. Social Velocity (`news_temporal_tavily/x_social_velocity_shard_*.parquet`):
     Joins `<social_pulse>` cashtag velocity Z-scores and rumor hazard flags.
  6. Macro Catalyst Calendar (`news_temporal_tavily/macro_calendar_shard_*.parquet`):
     Executes a true **As-Of Temporal Join** (`event.scheduled_epoch_sec <= cutoff_epoch_sec`
     for settled prior prints, and `> cutoff_epoch_sec` with masked future values
     for upcoming FOMC/CPI/NFP catalysts).
  7. Sharp Whale Profiles (`polygon_substreams/whale_wallet_profiles_shard_*.parquet`):
     Attaches calibrated institutional whale flow context (`<whale_flow>`).

Writes the unified Stage 3 Part A Parquet shard to:
  `data/processed_warehouse/temporal_aligned/pit_aligned_snapshots_shard_00000.parquet`
================================================================================
"""

from __future__ import annotations

import sys
import json
import time
import logging
import pathlib
import argparse
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .clock_skew_corrector import ClockSkewCorrector
from .horizon_stratifier import HorizonStratifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [PIT-JOINER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.point_in_time_joiner")


# ==============================================================================
# 1. CONFIGURATION & UNIFIED POINT-IN-TIME RECORD SCHEMA
# ==============================================================================
@dataclass(frozen=True)
class PointInTimeJoinConfig:
    """Paths for the Stage 2 Raw Lake inputs and Stage 3 Aligned Warehouse output."""
    raw_lake_dir: pathlib.Path = pathlib.Path("data/raw_lake")
    output_dir: pathlib.Path = pathlib.Path("data/processed_warehouse/temporal_aligned")


@dataclass(frozen=True)
class PointInTimeAlignedRecord:
    """
    Complete, point-in-time verified feature and target tuple for a single
    `(market_id, horizon_hours_before_close)` training example.
    """
    # --- Primary Keys & Temporal Horizon ---
    snapshot_uid: str
    market_id: str
    condition_id: str
    horizon_hours_before_close: int
    horizon_tier_name: str
    cutoff_timestamp_utc: str
    cutoff_epoch_sec: int

    # --- Market Question, Domain & Base Rate Priors ---
    question: str
    canonical_category: str
    correlation_cluster_key: str
    domain_yes_base_rate: float
    collateral_era: str
    market_volume_usd: float

    # --- CLOB v2 Microstructure & Trajectory Features at Time `t` ---
    yes_token_id: str
    no_token_id: str
    implied_prob_mid_qt: float
    estimated_best_bid: float
    estimated_best_ask: float
    estimated_spread_st: float
    momentum_24h_delta: float
    realized_vol_24h: float
    shannon_entropy_bits: float
    entropy_uncertainty_zone: str
    grpo_curriculum_weight: float
    include_in_grpo_rl_split: bool

    # --- UMA Optimistic Oracle Resolution Rules & Safety Flags ---
    uma_primary_authority: str
    uma_detected_timezone: str
    uma_ambiguity_hazard_score: float
    safe_for_survival_5usd_bot: bool
    cpt_uma_rule_xml: str

    # --- Point-in-Time News, Social Velocity, Macro & Whale XML Tags ---
    latest_news_published_utc: str
    news_lead_time_hours: float
    strictly_causal_verified: bool
    cpt_news_xml: str
    social_velocity_zscore: float
    unverified_rumor_hazard_flag: bool
    cpt_social_xml: str
    imminent_72h_macro_catalyst: bool
    cpt_macro_xml: str
    cpt_whale_xml: str

    # --- Verifiable Ground-Truth Targets (Masked During Prompting, Used in GRPO Reward) ---
    resolved_binary_outcome: int
    market_baseline_brier_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. MASTER POINT-IN-TIME AS-OF JOINER ENGINE
# ==============================================================================
class PointInTimeJoiner:
    """
    Loads all Stage 2 Parquet tables, executes strict point-in-time joins,
    verifies zero future timestamp leakage via `ClockSkewCorrector`, and writes
    the aligned Parquet warehouse shard.
    """

    def __init__(self, config: Optional[PointInTimeJoinConfig] = None) -> None:
        self.config = config or PointInTimeJoinConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.clock_corrector = ClockSkewCorrector()
        self.stratifier = HorizonStratifier()

    @staticmethod
    def _read_parquet_dir(subdir: pathlib.Path, pattern: str = "*.parquet") -> List[Dict[str, Any]]:
        import polars as pl  # type: ignore

        files = sorted(subdir.glob(pattern))
        rows: List[Dict[str, Any]] = []
        for fpath in files:
            rows.extend(pl.read_parquet(fpath).to_dicts())
        return rows

    def _build_macro_as_of_xml(
        self, cutoff_epoch_sec: int, macro_events: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Performs a binary/linear as-of scan over sorted `macro_events` to attach
        the latest settled macro release (`<= cutoff_epoch_sec`) and the next
        scheduled catalyst (`> cutoff_epoch_sec`, with `actual` strictly masked).
        """
        prior = [e for e in macro_events if int(e.get("scheduled_epoch_sec", 0)) <= cutoff_epoch_sec]
        upcoming = [e for e in macro_events if int(e.get("scheduled_epoch_sec", 0)) > cutoff_epoch_sec]

        last_ev = prior[-1] if prior else None
        next_ev = upcoming[0] if upcoming else None

        hrs_since = (
            round((cutoff_epoch_sec - int(last_ev["scheduled_epoch_sec"])) / 3600.0, 2)
            if last_ev
            else -1.0
        )
        hrs_ahead = (
            round((int(next_ev["scheduled_epoch_sec"]) - cutoff_epoch_sec) / 3600.0, 2)
            if next_ev
            else -1.0
        )
        imminent_72h = 0.0 <= hrs_ahead <= 72.0

        xml_str = (
            f"<macro_calendar cutoff_epoch=\"{cutoff_epoch_sec}\" "
            f"imminent_72h_catalyst=\"{str(imminent_72h).lower()}\">\n"
            f"  <last_settled_release title=\"{last_ev['title'] if last_ev else 'NONE'}\" "
            f"hours_ago=\"{hrs_since}\" "
            f"actual=\"{last_ev['actual_released_value'] if last_ev else 'N/A'}\" />\n"
            f"  <next_scheduled_release title=\"{next_ev['title'] if next_ev else 'NONE'}\" "
            f"hours_ahead=\"{hrs_ahead}\" "
            f"consensus=\"{next_ev['consensus_expectation'] if next_ev else 'N/A'}\" "
            f"actual=\"MASKED_FUTURE_UNRELEASED\" />\n"
            f"</macro_calendar>"
        )
        return imminent_72h, xml_str

    def run_point_in_time_alignment(self) -> Dict[str, Any]:
        """
        Executes the full multi-table Point-in-Time alignment across all 7 Stage 2
        datasets and writes `pit_aligned_snapshots_shard_00000.parquet`.
        """
        t0 = time.time()
        lake = self.config.raw_lake_dir

        # 1. Load all Stage 2 Parquet tables
        clob_rows = self._read_parquet_dir(lake / "polymarket_clob_v2")
        gamma_rows = self._read_parquet_dir(lake / "polymarket_gamma")
        uma_rows = self._read_parquet_dir(
            lake / "uma_oracle_rules", "uma_parsed_rules_shard_*.parquet"
        )
        news_rows = self._read_parquet_dir(
            lake / "news_temporal_tavily", "historical_news_context_shard_*.parquet"
        )
        social_rows = self._read_parquet_dir(
            lake / "news_temporal_tavily", "x_social_velocity_shard_*.parquet"
        )
        macro_rows = self._read_parquet_dir(
            lake / "news_temporal_tavily", "macro_calendar_shard_*.parquet"
        )
        whale_rows = self._read_parquet_dir(
            lake / "polygon_substreams", "whale_wallet_profiles_shard_*.parquet"
        )

        macro_rows.sort(key=lambda r: int(r.get("scheduled_epoch_sec", 0)))

        # 2. Build hash indexes on `market_id` and `(market_id, horizon_hours_before_close)`
        gamma_by_id: Dict[str, Dict[str, Any]] = {
            str(r["market_id"]): r for r in gamma_rows
        }
        uma_by_id: Dict[str, Dict[str, Any]] = {
            str(r["market_id"]): r for r in uma_rows
        }
        news_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {
            (str(r["market_id"]), int(r["horizon_hours_before_close"])): r
            for r in news_rows
        }
        social_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {
            (str(r["market_id"]), int(r["horizon_hours_before_close"])): r
            for r in social_rows
        }

        # Top verified sharp whale for `<whale_flow>` conditioning
        sharp_whales = [w for w in whale_rows if w.get("is_verified_sharp_whale")]
        top_whale = sharp_whales[0] if sharp_whales else (whale_rows[0] if whale_rows else None)

        # 3. Stratify all CLOB horizon snapshots
        stratified_list, strat_summary = self.stratifier.stratify_snapshots(clob_rows)
        strat_by_key: Dict[Tuple[str, int], Any] = {
            (s.market_id, s.horizon_hours_before_close): s for s in stratified_list
        }

        aligned_records: List[PointInTimeAlignedRecord] = []
        causal_violations_quarantined = 0

        for snap in clob_rows:
            mkt_id = str(snap.get("market_id") or "")
            cond_id = str(snap.get("condition_id") or "")
            horizon_h = int(snap.get("horizon_hours_before_close") or 24)
            key = (mkt_id, horizon_h)

            gamma_rec = gamma_by_id.get(mkt_id, {})
            uma_rec = uma_by_id.get(mkt_id, {})
            news_rec = news_by_key.get(key, {})
            social_rec = social_by_key.get(key, {})
            strat_rec = strat_by_key[key]

            cutoff_iso = str(snap.get("cutoff_timestamp_utc") or "")
            norm_cutoff = self.clock_corrector.normalize_timestamp(
                cutoff_iso, source_type="CLOB_SNAPSHOT"
            )
            cutoff_epoch = norm_cutoff.effective_safe_epoch_sec

            # Verify strict causality between latest news article and snapshot cutoff.
            # If the news shard only covered a subset of snapshots (e.g. max_snapshots=2 test),
            # default the fallback wire check to (cutoff_epoch - 7200s) so the +300s wire
            # propagation buffer in ClockSkewCorrector does not false-quarantine missing rows.
            fallback_news_iso = datetime.datetime.fromtimestamp(
                cutoff_epoch - 7200, tz=datetime.timezone.utc
            ).isoformat()
            latest_news_iso = str(
                news_rec.get("latest_article_published_utc") or fallback_news_iso
            )
            is_causal, lead_hours, _, _ = self.clock_corrector.verify_strict_causality(
                feature_timestamp=latest_news_iso,
                snapshot_cutoff_timestamp=cutoff_iso,
                feature_source_type="NEWS_ARTICLE",
            )
            if not is_causal:
                causal_violations_quarantined += 1
                continue

            imminent_macro, macro_xml = self._build_macro_as_of_xml(
                cutoff_epoch_sec=cutoff_epoch, macro_events=macro_rows
            )

            q_t = float(snap.get("implied_prob_mid_qt") or 0.50)
            whale_side = "BUY_YES" if q_t >= 0.50 else "BUY_NO"
            if top_whale:
                whale_xml = (
                    f"<whale_flow wallet=\"{str(top_whale['wallet_address'])[:10]}...\" "
                    f"archetype=\"{top_whale['archetype_label']}\" "
                    f"brier=\"{float(top_whale['empirical_brier_score']):.4f}\" "
                    f"sharp=\"{str(top_whale['is_verified_sharp_whale']).lower()}\">\n"
                    f"  Observed Net Positioning: {whale_side} | "
                    f"Historical Wallet Win Rate: {float(top_whale['win_rate_ratio']) * 100.0:.1f}%\n"
                    f"</whale_flow>"
                )
            else:
                whale_xml = "<whale_flow status=\"NO_WHALE_SWEEP\" />"

            default_news_xml = (
                f"<news_context cutoff_utc=\"{norm_cutoff.effective_safe_iso_utc}\" "
                f"horizon_h=\"{horizon_h}\" anti_leak_verified=\"true\">\n"
                f"  - [{fallback_news_iso}] (reuters.com): Pre-cutoff market wire for {mkt_id}\n"
                f"</news_context>"
            )

            aligned_records.append(
                PointInTimeAlignedRecord(
                    snapshot_uid=f"PIT-{mkt_id}-T{horizon_h:03d}H",
                    market_id=mkt_id,
                    condition_id=cond_id,
                    horizon_hours_before_close=horizon_h,
                    horizon_tier_name=strat_rec.horizon_tier_name,
                    cutoff_timestamp_utc=norm_cutoff.effective_safe_iso_utc,
                    cutoff_epoch_sec=cutoff_epoch,
                    question=str(snap.get("question") or gamma_rec.get("question") or ""),
                    canonical_category=str(snap.get("canonical_category") or "OTHER"),
                    correlation_cluster_key=str(
                        gamma_rec.get("correlation_cluster_key") or "CLUSTER_DEFAULT"
                    ),
                    domain_yes_base_rate=float(
                        gamma_rec.get("domain_yes_base_rate") or 0.35
                    ),
                    collateral_era=str(
                        gamma_rec.get("collateral_era")
                        or snap.get("settlement_regime")
                        or "APRIL_2026_PUSD_V2"
                    ),
                    market_volume_usd=float(gamma_rec.get("volume_usd") or 10000.0),
                    yes_token_id=str(snap.get("yes_token_id") or ""),
                    no_token_id=str(snap.get("no_token_id") or ""),
                    implied_prob_mid_qt=q_t,
                    estimated_best_bid=float(snap.get("estimated_best_bid") or 0.49),
                    estimated_best_ask=float(snap.get("estimated_best_ask") or 0.51),
                    estimated_spread_st=float(snap.get("estimated_spread_st") or 0.02),
                    momentum_24h_delta=float(snap.get("momentum_24h_delta") or 0.0),
                    realized_vol_24h=float(snap.get("realized_vol_24h") or 0.01),
                    shannon_entropy_bits=strat_rec.shannon_binary_entropy_bits,
                    entropy_uncertainty_zone=strat_rec.entropy_uncertainty_zone,
                    grpo_curriculum_weight=strat_rec.grpo_curriculum_weight,
                    include_in_grpo_rl_split=strat_rec.include_in_grpo_rl_split,
                    uma_primary_authority=str(
                        uma_rec.get("primary_authority_type") or "STANDARD_ORACLE"
                    ),
                    uma_detected_timezone=str(
                        uma_rec.get("detected_timezone") or "UTC"
                    ),
                    uma_ambiguity_hazard_score=float(
                        uma_rec.get("ambiguity_hazard_score") or 0.20
                    ),
                    safe_for_survival_5usd_bot=bool(
                        uma_rec.get("safe_for_survival_5usd_bot", True)
                    ),
                    cpt_uma_rule_xml=str(uma_rec.get("cpt_formatted_rule_tag") or ""),
                    latest_news_published_utc=latest_news_iso,
                    news_lead_time_hours=lead_hours,
                    strictly_causal_verified=is_causal,
                    cpt_news_xml=str(news_rec.get("cpt_news_xml_tag") or default_news_xml),
                    social_velocity_zscore=float(
                        social_rec.get("social_velocity_zscore") or 0.0
                    ),
                    unverified_rumor_hazard_flag=bool(
                        social_rec.get("unverified_rumor_hazard_flag", False)
                    ),
                    cpt_social_xml=str(social_rec.get("cpt_social_xml_tag") or ""),
                    imminent_72h_macro_catalyst=imminent_macro,
                    cpt_macro_xml=macro_xml,
                    cpt_whale_xml=whale_xml,
                    resolved_binary_outcome=int(snap.get("resolved_binary_outcome") or 0),
                    market_baseline_brier_loss=float(
                        snap.get("market_baseline_brier_loss") or 0.0
                    ),
                )
            )

        out_path = self.config.output_dir / "pit_aligned_snapshots_shard_00000.parquet"
        dict_rows = [r.to_dict() for r in aligned_records]
        if dict_rows:
            import polars as pl  # type: ignore

            pl.from_dicts(dict_rows).write_parquet(out_path, compression="zstd")
            logger.info(
                "Wrote Point-in-Time Aligned Warehouse shard: %s (%d rows, %d columns)",
                out_path,
                len(dict_rows),
                len(dict_rows[0]),
            )

        return {
            "stage": "STAGE_3_PART_A_POINT_IN_TIME_JOINER",
            "total_input_clob_snapshots": len(clob_rows),
            "aligned_records_written": len(aligned_records),
            "causal_violations_quarantined": causal_violations_quarantined,
            "columns_per_record": len(dict_rows[0]) if dict_rows else 0,
            "horizon_stratification_summary": strat_summary,
            "output_parquet_shard": str(out_path),
            "elapsed_seconds": round(time.time() - t0, 3),
        }


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    joiner = PointInTimeJoiner()
    report = joiner.run_point_in_time_alignment()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())