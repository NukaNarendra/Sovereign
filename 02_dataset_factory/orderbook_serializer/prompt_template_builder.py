#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART C)
File: 02_dataset_factory/orderbook_serializer/prompt_template_builder.py
================================================================================
Master Prompt Compiler for Stage 4 CPT, Stage 4 SFT, Stage 5 GRPO, and Stage 6
Live MCP Inference.

Why a Single Unified Prompt Contract Is Mandatory:
  1. A fatal mistake in quantitative LLM engineering is training the model on
     one prompt format during SFT/GRPO and feeding it a slightly different format
     in production via MCP servers.
  2. This module compiles every leak-certified market snapshot into a single
     canonical prompt architecture containing 8 strictly ordered XML blocks:
       - `<market_meta>`    : Question, category, base-rate prior, cutoff UTC
       - `<uma_rule>`       : Optimistic Oracle v3 resolution source, timezone,
                              and ambiguity hazard score
       - `<book>`           : 5-level L2 CLOB depth table, midpoint `q_t`,
                              spread `s_t`, and minimum edge hurdle
       - `<trajectory>`     : 24h pre-cutoff price path & regime tokens
       - `<whale_flow>`     : On-chain Polygon sharp whale positioning & Brier
       - `<macro_calendar>` : Prior settled macro print & next scheduled catalyst
       - `<social_pulse>`   : X cashtag velocity Z-score & rumor hazard flag
       - `<news_context>`   : Date-scrubbed, spoiler-free pre-cutoff news wires
  3. Writes the compiled Stage 3 Part C Parquet shard to:
     `data/processed_warehouse/serialized_prompts/serialized_prompts_shard_00000.parquet`
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .l2_book_to_markdown import L2BookToMarkdownSerializer
from .price_trajectory_tokenizer import PriceTrajectoryTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [PROMPT-BUILDER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.prompt_template_builder")

SOVEREIGN_SYSTEM_PROMPT: str = (
    "You are Sovereign-Prob-7B, an institutional quantitative prediction-market "
    "forecasting engine calibrated via Group Relative Policy Optimization (GRPO) "
    "on Polymarket CLOB v2 order books and UMA Optimistic Oracle resolution rules.\n"
    "Analyze the provided point-in-time market state strictly as of the cutoff "
    "timestamp. First, decompose the historical domain base rate (Outside View), "
    "the UMA Optimistic Oracle resolution clauses and timezone/source edge cases, "
    "the Level-2 CLOB bid-ask spread and whale order flow, and the pre-cutoff news "
    "catalysts (Inside View) inside <reasoning>...</reasoning>.\n"
    "Then output your final calibrated probability of the YES outcome in [0.0001, 0.9999] "
    "inside <prob>0.XXXX</prob> and the signed net edge (p - q_t) inside <edge>+0.XXXX</edge>."
)


# ==============================================================================
# 1. UNIFIED FORECASTING PROMPT RECORD SCHEMA
# ==============================================================================
@dataclass(frozen=True)
class UnifiedForecastingPrompt:
    """
    Complete serialized prompt record containing the CPT document, the SFT/GRPO
    chat messages, and the masked ground-truth verification targets (`o`, `q_t`,
    `s_t`, `min_edge_hurdle`) for Stage 5 reward calculation.
    """
    snapshot_uid: str
    market_id: str
    condition_id: str
    dataset_split: str
    horizon_hours_before_close: int
    cutoff_timestamp_utc: str
    canonical_category: str
    correlation_cluster_key: str
    question: str
    domain_yes_base_rate: float
    implied_prob_mid_qt: float
    estimated_spread_st: float
    min_required_edge_to_trade: float
    uma_ambiguity_hazard_score: float
    safe_for_survival_5usd_bot: bool
    grpo_curriculum_weight: float
    include_in_grpo_rl_split: bool
    cpt_book_xml: str
    cpt_trajectory_xml: str
    system_prompt: str
    user_prompt: str
    cpt_unsupervised_text: str
    estimated_prompt_tokens: int
    resolved_binary_outcome: int
    market_baseline_brier_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. MASTER PROMPT TEMPLATE COMPILER ENGINE
# ==============================================================================
class PromptTemplateBuilder:
    """
    Reads the leak-certified warehouse shard (`leak_certified_snapshots_shard_00000.parquet`),
    serializes the L2 order book and 24h price trajectory, and builds the
    unified prompt dataset for CPT, SFT, and GRPO.
    """

    def __init__(
        self,
        input_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/leak_audited/leak_certified_snapshots_shard_00000.parquet"
        ),
        output_dir: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/serialized_prompts"
        ),
    ) -> None:
        self.input_parquet = input_parquet
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.book_serializer = L2BookToMarkdownSerializer()
        self.traj_tokenizer = PriceTrajectoryTokenizer()

    def build_prompt_for_record(
        self, row: Dict[str, Any]
    ) -> UnifiedForecastingPrompt:
        """
        Compiles all 8 XML context blocks into `user_prompt` and `cpt_unsupervised_text`.
        Strictly masks `resolved_binary_outcome` from the prompt text so the
        model never sees the target label inside its input context.
        """
        book_block = self.book_serializer.serialize_snapshot_book(row)
        traj_block = self.traj_tokenizer.tokenize_snapshot_trajectory(row)

        snap_uid = str(row.get("snapshot_uid") or "")
        mkt_id = str(row.get("market_id") or "")
        cond_id = str(row.get("condition_id") or "")
        split = str(row.get("dataset_split") or "TRAIN")
        horizon_h = int(row.get("horizon_hours_before_close") or 24)
        cutoff_iso = str(row.get("cutoff_timestamp_utc") or "")
        category = str(row.get("canonical_category") or "OTHER")
        cluster_key = str(row.get("correlation_cluster_key") or "CLUSTER_DEFAULT")
        question = str(row.get("question") or "")
        base_rate = float(row.get("domain_yes_base_rate") or 0.35)

        uma_xml = str(row.get("cpt_uma_rule_xml") or "<uma_rule />")
        whale_xml = str(row.get("cpt_whale_xml") or "<whale_flow />")
        macro_xml = str(row.get("cpt_macro_xml") or "<macro_calendar />")
        social_xml = str(row.get("cpt_social_xml") or "<social_pulse />")
        news_xml = str(row.get("cpt_news_xml") or "<news_context />")

        market_meta_xml = (
            f"<market_meta snapshot_uid=\"{snap_uid}\" "
            f"market_id=\"{mkt_id}\" "
            f"cutoff_utc=\"{cutoff_iso}\" "
            f"horizon_h=\"{horizon_h}\">\n"
            f"  <question>{question}</question>\n"
            f"  <domain category=\"{category}\" "
            f"cluster=\"{cluster_key}\" "
            f"historical_yes_base_rate=\"{base_rate:0.2f}\" />\n"
            f"</market_meta>"
        )

        user_prompt = (
            f"{market_meta_xml}\n"
            f"{uma_xml}\n"
            f"{book_block.cpt_book_xml_markdown}\n"
            f"{traj_block.cpt_trajectory_xml}\n"
            f"{whale_xml}\n"
            f"{macro_xml}\n"
            f"{social_xml}\n"
            f"{news_xml}\n\n"
            f"Task: Estimate the true probability p of YES resolution as of {cutoff_iso} "
            f"(T-{horizon_h}h before close). Provide structured <reasoning>, <prob>, and <edge>."
        )

        cpt_doc = (
            f"<|im_start|>system\n{SOVEREIGN_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>"
        )

        # Approximate token count (~3.8 characters per token for XML/numbers)
        est_tokens = max(128, int(round(len(cpt_doc) / 3.8)))

        return UnifiedForecastingPrompt(
            snapshot_uid=snap_uid,
            market_id=mkt_id,
            condition_id=cond_id,
            dataset_split=split,
            horizon_hours_before_close=horizon_h,
            cutoff_timestamp_utc=cutoff_iso,
            canonical_category=category,
            correlation_cluster_key=cluster_key,
            question=question,
            domain_yes_base_rate=base_rate,
            implied_prob_mid_qt=book_block.midpoint_qt,
            estimated_spread_st=book_block.spread_abs_st,
            min_required_edge_to_trade=book_block.min_required_edge_to_trade,
            uma_ambiguity_hazard_score=float(
                row.get("uma_ambiguity_hazard_score") or 0.20
            ),
            safe_for_survival_5usd_bot=bool(
                row.get("safe_for_survival_5usd_bot", True)
            ),
            grpo_curriculum_weight=float(row.get("grpo_curriculum_weight") or 1.0),
            include_in_grpo_rl_split=bool(row.get("include_in_grpo_rl_split", True)),
            cpt_book_xml=book_block.cpt_book_xml_markdown,
            cpt_trajectory_xml=traj_block.cpt_trajectory_xml,
            system_prompt=SOVEREIGN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cpt_unsupervised_text=cpt_doc,
            estimated_prompt_tokens=est_tokens,
            resolved_binary_outcome=int(row.get("resolved_binary_outcome") or 0),
            market_baseline_brier_loss=float(
                row.get("market_baseline_brier_loss") or 0.0
            ),
        )

    def run_prompt_serialization_pipeline(self) -> Dict[str, Any]:
        """
        Compiles all leak-certified snapshots into `UnifiedForecastingPrompt`
        records and writes `serialized_prompts_shard_00000.parquet`.
        """
        import polars as pl  # type: ignore

        if not self.input_parquet.exists():
            raise FileNotFoundError(f"Missing leak-certified shard: {self.input_parquet}")

        rows = pl.read_parquet(self.input_parquet).to_dicts()
        compiled_prompts: List[UnifiedForecastingPrompt] = [
            self.build_prompt_for_record(r) for r in rows
        ]

        out_path = self.output_dir / "serialized_prompts_shard_00000.parquet"
        dict_rows = [p.to_dict() for p in compiled_prompts]
        if dict_rows:
            pl.from_dicts(dict_rows).write_parquet(out_path, compression="zstd")
            logger.info(
                "Wrote Serialized Prompts Parquet shard: %s (%d prompts)",
                out_path,
                len(dict_rows),
            )

        avg_tokens = (
            int(
                round(
                    sum(p.estimated_prompt_tokens for p in compiled_prompts)
                    / max(1, len(compiled_prompts))
                )
            )
            if compiled_prompts
            else 0
        )

        return {
            "stage": "STAGE_3_PART_C_PROMPT_TEMPLATE_BUILDER",
            "total_prompts_compiled": len(compiled_prompts),
            "average_estimated_prompt_tokens": avg_tokens,
            "all_8_xml_sections_present": True,
            "output_parquet_shard": str(out_path),
            "sample_prompt_preview": compiled_prompts[0].user_prompt[:850] + "\n..."
            if compiled_prompts
            else "",
        }


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    builder = PromptTemplateBuilder()
    report = builder.run_prompt_serialization_pipeline()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())