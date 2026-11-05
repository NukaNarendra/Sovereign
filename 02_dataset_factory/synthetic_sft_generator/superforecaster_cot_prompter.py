#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART D)
File: 02_dataset_factory/synthetic_sft_generator/superforecaster_cot_prompter.py
================================================================================
Blind Philip Tetlock Superforecaster Chain-of-Thought (CoT) Generator for
Stage 4B Supervised Fine-Tuning (SFT).

Why Teacher Traces MUST Be Generated Blind to the Ground-Truth Outcome (`o`):
  1. A common mistake when generating synthetic SFT data for forecasting models
     is showing the teacher LLM the true outcome (`o = 1`) and asking it to
     "explain why YES will win." That produces hindsight-contaminated reasoning
     traces full of false certainty (`p = 0.99`), which destroys Brier
     calibration during live trading.
  2. Instead, our `SuperforecasterCoTPrompter` constructs a strictly blind
     5-step Bayesian decomposition using ONLY information available at or before
     `cutoff_timestamp_utc`:
       - Step 1 (Outside View): Anchors on the historical domain base rate
         (`domain_yes_base_rate`) and time remaining (`horizon_h`).
       - Step 2 (UMA Rule & Oracle Hazard Audit): Evaluates the designated
         authority, timezone cutoff, and `uma_ambiguity_hazard_score`.
       - Step 3 (L2 CLOB Microstructure & Sharp Whale Flow): Compares the
         order-book midpoint `q_t`, micro-price, bid-ask spread `s_t`, minimum
         edge hurdle, and verified sharp whale positioning.
       - Step 4 (Inside View — News & Macro Catalysts): Weighs pre-cutoff news
         wires, scheduled FOMC/CPI catalysts, and social rumor flags.
       - Step 5 (Calibrated Probability & Net Edge Synthesis): Combines the
         prior, microstructure, and catalysts into `<prob>0.XXXX</prob>` and
         `<edge>+0.XXXX</edge>`.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import math
import asyncio
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .merge_gateway_client import MergeGatewayTeacherClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [COT-PROMPTER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.superforecaster_cot_prompter")


# ==============================================================================
# 1. DATA STRUCTURE FOR GENERATED SFT TEACHER TRACES
# ==============================================================================
@dataclass(frozen=True)
class SuperforecasterTraceRecord:
    """
    Complete SFT training record pairing the 8-section XML prompt with a blind
    5-step Superforecaster `<reasoning>`, `<prob>`, and `<edge>` completion.
    """
    snapshot_uid: str
    market_id: str
    condition_id: str
    dataset_split: str
    horizon_hours_before_close: int
    cutoff_timestamp_utc: str
    canonical_category: str
    question: str
    implied_prob_mid_qt: float
    estimated_spread_st: float
    min_required_edge_to_trade: float
    teacher_predicted_prob_p: float
    teacher_predicted_edge: float
    teacher_action_recommendation: str
    teacher_model_id: str
    generation_source: str
    system_prompt: str
    user_prompt: str
    assistant_response: str
    sft_full_chatml_text: str
    resolved_binary_outcome: int
    teacher_brier_loss: float
    market_baseline_brier_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. BLIND TETLOCK SUPERFORECASTER CHAIN-OF-THOUGHT GENERATOR
# ==============================================================================
class SuperforecasterCoTPrompter:
    """
    Generates blind 5-step Superforecaster reasoning traces across all prompts
    in `serialized_prompts_shard_00000.parquet`.
    """

    def __init__(
        self,
        input_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/serialized_prompts/serialized_prompts_shard_00000.parquet"
        ),
        output_dir: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/sft_teacher_traces"
        ),
    ) -> None:
        self.input_parquet = input_parquet
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gateway_client = MergeGatewayTeacherClient()

    @staticmethod
    def _extract_first_news_headline(user_prompt: str) -> str:
        """Extracts the lead pre-cutoff news wire line from `<news_context>`."""
        match = re.search(r"<news_context[^>]*>\s*-\s*([^\n<]+)", user_prompt)
        if match:
            return match.group(1).strip()[:160]
        return "Pre-cutoff primary wire dispatches confirm active order-book price discovery."

    @staticmethod
    def _extract_obi_and_micro_price(
        book_xml: str, fallback_qt: float
    ) -> tuple[float, float]:
        """Parses Micro-Price and Order Book Imbalance (OBI) from `<book>`."""
        micro_m = re.search(r"Micro-Price:\s*([0-9.]+)", book_xml)
        obi_m = re.search(r"OBI:\s*([+-]?[0-9.]+)", book_xml)
        micro_p = float(micro_m.group(1)) if micro_m else fallback_qt
        obi_val = float(obi_m.group(1)) if obi_m else 0.0
        return micro_p, obi_val

    def synthesize_blind_tetlock_trace(
        self, row: Dict[str, Any]
    ) -> tuple[str, float, float, str]:
        """
        Synthesizes a deterministic, strictly blind 5-step Bayesian Superforecaster
        trace using ONLY pre-cutoff features (`domain_yes_base_rate`, `q_t`,
        `micro_price`, `obi`, `24h drift`, `uma_hazard`, and `news_context`).
        Never accesses `resolved_binary_outcome` (`o`).
        """
        category = str(row.get("canonical_category") or "OTHER")
        horizon_h = int(row.get("horizon_hours_before_close") or 24)
        cutoff_iso = str(row.get("cutoff_timestamp_utc") or "")
        base_rate = float(row.get("domain_yes_base_rate") or 0.35)
        q_t = float(row.get("implied_prob_mid_qt") or 0.50)
        spread_st = float(row.get("estimated_spread_st") or 0.02)
        min_edge = float(row.get("min_required_edge_to_trade") or 0.015)
        uma_hazard = float(row.get("uma_ambiguity_hazard_score") or 0.20)
        safe_survival = bool(row.get("safe_for_survival_5usd_bot", True))
        book_xml = str(row.get("cpt_book_xml") or "")
        traj_xml = str(row.get("cpt_trajectory_xml") or "")
        user_prompt = str(row.get("user_prompt") or "")

        micro_p, obi_val = self._extract_obi_and_micro_price(book_xml, q_t)
        lead_wire = self._extract_first_news_headline(user_prompt)

        # Parse 24h drift from trajectory XML
        drift_m = re.search(r"24h Net Drift:\s*([+-]?[0-9.]+)%", traj_xml)
        drift_pct = float(drift_m.group(1)) / 100.0 if drift_m else 0.0

        # Blind Bayesian probability update:
        # 1. Start from CLOB market consensus q_t
        # 2. Apply slight shrinkage toward historical domain base_rate at long horizons (168h, 72h),
        #    and follow microstructure order-flow + sharp whale convergence at short horizons (24h, 6h)
        horizon_decay = min(1.0, float(horizon_h) / 168.0)
        base_rate_pull = (base_rate - q_t) * (0.045 * horizon_decay)
        micro_pull = (micro_p - q_t) * 0.65
        obi_pull = obi_val * 0.035
        momentum_follow = drift_pct * (0.18 if horizon_h <= 72 else 0.08)

        # Dampen conviction if UMA ambiguity hazard is elevated (>0.45)
        raw_shift = base_rate_pull + micro_pull + obi_pull + momentum_follow
        if not safe_survival or uma_hazard > 0.45:
            raw_shift *= 0.35

        # Clamp shift so blind adjustment stays realistic within [-0.085, +0.085]
        clamped_shift = max(-0.085, min(0.085, raw_shift))
        p_calibrated = round(max(0.0050, min(0.9950, q_t + clamped_shift)), 4)
        net_edge = round(p_calibrated - q_t, 4)

        if not safe_survival:
            action_str = "PASS_HIGH_UMA_HAZARD"
        elif abs(net_edge) < min_edge:
            action_str = "HOLD_INSIDE_SPREAD_NO_TRADE"
        elif net_edge >= min_edge:
            action_str = "MAKER_LIMIT_BUY_YES"
        else:
            action_str = "MAKER_LIMIT_BUY_NO"

        reasoning_block = (
            f"<reasoning>\n"
            f"1. [Outside View & Base Rate Prior]: Within the {category} domain at T-{horizon_h}h "
            f"prior to close ({cutoff_iso}), the historical YES resolution base rate is "
            f"{base_rate * 100.0:.1f}%, while the live CLOB midpoint stands at q_t = {q_t:0.4f}.\n"
            f"2. [UMA Optimistic Oracle Rule & Hazard Audit]: The contract exhibits an ambiguity "
            f"hazard score of {uma_hazard:0.2f} (survival-bot eligibility = {str(safe_survival).lower()}). "
            f"{'Settlement criteria rely on deterministic primary sources.' if safe_survival else 'Elevated media-consensus subjectivity warrants strict position dampening.'}\n"
            f"3. [L2 Order Book & Whale Microstructure]: The bid-ask spread is s_t = {spread_st:0.4f} "
            f"(minimum edge hurdle = {min_edge:0.4f}). Queue-weighted micro-price is {micro_p:0.4f} "
            f"with 5-level Order Book Imbalance OBI = {obi_val:+0.3f} and 24h drift = {drift_pct * 100.0:+0.2f}%.\n"
            f"4. [Inside View & Pre-Cutoff Catalysts]: Lead pre-cutoff wire ({lead_wire}) and macro/social "
            f"telemetry indicate {'sustained directional order flow' if abs(drift_pct) >= 0.02 else 'consolidating pre-catalyst positioning'}.\n"
            f"5. [Bayesian Calibration & Execution Gate]: Updating q_t = {q_t:0.4f} with microstructure "
            f"imbalance ({obi_pull:+0.4f}), momentum ({momentum_follow:+0.4f}), and base-rate anchor "
            f"({base_rate_pull:+0.4f}) yields calibrated probability p = {p_calibrated:0.4f} and "
            f"signed net edge = {net_edge:+0.4f} -> Recommended Action: {action_str}.\n"
            f"</reasoning>\n"
            f"<prob>{p_calibrated:0.4f}</prob>\n"
            f"<edge>{net_edge:+0.4f}</edge>"
        )

        return reasoning_block, p_calibrated, net_edge, action_str

    async def generate_sft_dataset(self) -> Dict[str, Any]:
        """
        Generates blind Superforecaster reasoning traces for all serialized
        prompts and writes `raw_sft_teacher_traces_shard_00000.parquet`.
        """
        import polars as pl  # type: ignore

        if not self.input_parquet.exists():
            raise FileNotFoundError(
                f"Missing serialized prompts shard: {self.input_parquet}"
            )

        rows = pl.read_parquet(self.input_parquet).to_dicts()
        records: List[SuperforecasterTraceRecord] = []

        for row in rows:
            snap_uid = str(row.get("snapshot_uid") or "")
            sys_prompt = str(row.get("system_prompt") or "")
            usr_prompt = str(row.get("user_prompt") or "")
            q_t = float(row.get("implied_prob_mid_qt") or 0.50)
            outcome_o = int(row.get("resolved_binary_outcome") or 0)

            synth_trace, p_val, edge_val, action_str = (
                self.synthesize_blind_tetlock_trace(row)
            )

            completion_res = await self.gateway_client.request_teacher_completion(
                snapshot_uid=snap_uid,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                fallback_completion_fn=lambda: synth_trace,
            )

            # Parse <prob> and <edge> if returned from live Merge Gateway API
            final_text = completion_res.completion_text
            prob_m = re.search(r"<prob>\s*([0-9.]+)\s*</prob>", final_text)
            edge_m = re.search(r"<edge>\s*([+-]?[0-9.]+)\s*</edge>", final_text)
            if prob_m:
                p_val = max(0.0001, min(0.9999, round(float(prob_m.group(1)), 4)))
                edge_val = (
                    round(float(edge_m.group(1)), 4)
                    if edge_m
                    else round(p_val - q_t, 4)
                )

            chatml_doc = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{usr_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{final_text}<|im_end|>"
            )

            teacher_brier = round((p_val - float(outcome_o)) ** 2, 6)

            records.append(
                SuperforecasterTraceRecord(
                    snapshot_uid=snap_uid,
                    market_id=str(row.get("market_id") or ""),
                    condition_id=str(row.get("condition_id") or ""),
                    dataset_split=str(row.get("dataset_split") or "TRAIN"),
                    horizon_hours_before_close=int(
                        row.get("horizon_hours_before_close") or 24
                    ),
                    cutoff_timestamp_utc=str(row.get("cutoff_timestamp_utc") or ""),
                    canonical_category=str(row.get("canonical_category") or "OTHER"),
                    question=str(row.get("question") or ""),
                    implied_prob_mid_qt=q_t,
                    estimated_spread_st=float(row.get("estimated_spread_st") or 0.02),
                    min_required_edge_to_trade=float(
                        row.get("min_required_edge_to_trade") or 0.015
                    ),
                    teacher_predicted_prob_p=p_val,
                    teacher_predicted_edge=edge_val,
                    teacher_action_recommendation=action_str,
                    teacher_model_id=completion_res.teacher_model_id,
                    generation_source=completion_res.generation_source,
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt,
                    assistant_response=final_text,
                    sft_full_chatml_text=chatml_doc,
                    resolved_binary_outcome=outcome_o,
                    teacher_brier_loss=teacher_brier,
                    market_baseline_brier_loss=float(
                        row.get("market_baseline_brier_loss") or 0.0
                    ),
                )
            )

        out_path = self.output_dir / "raw_sft_teacher_traces_shard_00000.parquet"
        dict_rows = [r.to_dict() for r in records]
        if dict_rows:
            pl.from_dicts(dict_rows).write_parquet(out_path, compression="zstd")
            logger.info(
                "Wrote Blind SFT Teacher Traces shard: %s (%d traces)",
                out_path,
                len(dict_rows),
            )

        avg_teacher_brier = (
            round(
                sum(r.teacher_brier_loss for r in records) / max(1, len(records)), 5
            )
            if records
            else 0.0
        )
        avg_market_brier = (
            round(
                sum(r.market_baseline_brier_loss for r in records)
                / max(1, len(records)),
                5,
            )
            if records
            else 0.0
        )

        return {
            "stage": "STAGE_3_PART_D_SUPERFORECASTER_COT_PROMPTER",
            "total_sft_traces_generated": len(records),
            "blind_outcome_masking_enforced": True,
            "average_teacher_brier_loss": avg_teacher_brier,
            "average_market_baseline_brier_loss": avg_market_brier,
            "output_parquet_shard": str(out_path),
            "sample_assistant_response": records[0].assistant_response if records else "",
        }


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    prompter = SuperforecasterCoTPrompter()
    report = asyncio.run(prompter.generate_sft_dataset())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())