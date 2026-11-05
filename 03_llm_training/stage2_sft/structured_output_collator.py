#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 4B: SUPERVISED FINE-TUNING (SFT)
File: 03_llm_training/stage2_sft/structured_output_collator.py
================================================================================
Completion-Only Loss Masking & Token-Level Gradient Weighting Collator for
Stage 4B Supervised Fine-Tuning (`sft_reasoning`).

Why Completion-Only Loss Masking + Token Weighting Is Critical for SFT:
  1. In every Stage 4B training document (`sft_full_chatml_text`), the `<|im_start|>system`
     and `<|im_start|>user` prompt accounts for ~1,314 tokens, whereas the
     `<|im_start|>assistant` Superforecaster completion accounts for ~320 tokens.
  2. Without completion-only loss masking, 80% of the SFT cross-entropy gradient
     is spent predicting the user's question and order book rather than learning
     how to write `<reasoning>...</reasoning><prob>0.XXXX</prob><edge>+0.XXXX</edge>`.
  3. This collator locates the exact token boundary of `<|im_start|>assistant\n`
     and sets `labels[i] = -100` (`loss_weight[i] = 0.0`) for all preceding
     prompt tokens.
  4. Furthermore, within the active assistant completion, it assigns a 3-tier
     token importance multiplier:
       - `w = 1.00` for standard 5-step Superforecaster reasoning prose.
       - `w = 1.50` for structural XML tags (`<reasoning>`, `</reasoning>`,
         `<prob>`, `</prob>`, `<edge>`, `</edge>`).
       - `w = 2.00` for the numeric forecast spans inside `<prob>0.XXXX</prob>`
         and `<edge>+0.XXXX</edge>` so Qwen-2.5-7B masters exact 4-decimal
         subtraction (`edge = p - q_t`) before entering Stage 5 GRPO.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import logging
import pathlib
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..stage1_cpt.domain_tokenizer_extender import (
    DomainTokenizerExtender,
    ALL_DOMAIN_SPECIAL_TOKENS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [SFT-COLLATOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.structured_output_collator")

COMPLETION_XML_EXTRACT_REGEX = re.compile(
    r"<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<prob>(?P<prob>[0-9.]+)</prob>\s*"
    r"<edge>(?P<edge>[+-]?[0-9.]+)</edge>",
    re.DOTALL,
)


# ==============================================================================
# 1. COLLATED SFT EXAMPLE & AUDIT REPORT DATA STRUCTURES
# ==============================================================================
@dataclass(frozen=True)
class CollatedSFTExample:
    """
    Represents a single tokenized, completion-masked, and token-weighted SFT
    training example ready for forward/backward passes.
    """
    snapshot_uid: str
    input_ids: List[int]
    labels: List[int]
    token_loss_weights: List[float]
    total_sequence_tokens: int
    masked_prompt_tokens_count: int
    active_completion_tokens_count: int
    upweighted_numeric_tokens_count: int
    prompt_mask_ratio_pct: float
    target_prob_p: float
    target_edge: float
    midpoint_qt: float
    xml_structure_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CollatorAuditReport:
    """
    Summary verification report confirming 100% prompt masking (`-100`) and
    proper 2.0x numeric token upweighting across the SFT Parquet warehouse.
    """
    stage: str
    parquet_shard_audited: str
    total_examples_collated: int
    all_prompts_masked_with_neg100: bool
    all_completions_xml_valid: bool
    average_total_tokens: float
    average_masked_prompt_tokens: float
    average_active_completion_tokens: float
    average_upweighted_numeric_tokens: float
    average_prompt_mask_ratio_pct: float
    sample_token_weights_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE COMPLETION-ONLY & STRUCTURED OUTPUT COLLATOR ENGINE
# ==============================================================================
class StructuredOutputCollator:
    """
    Tokenizes `sft_full_chatml_text` records, masks all prompt tokens before
    `<|im_start|>assistant\\n` with `-100`, and applies 1.5x / 2.0x loss weights
    to XML tags and `<prob>` / `<edge>` numeric spans.
    """

    IGNORE_INDEX: int = -100
    ASSISTANT_MARKER: str = "<|im_start|>assistant\n"

    WEIGHT_MASKED_PROMPT: float = 0.00
    WEIGHT_REASONING_PROSE: float = 1.00
    WEIGHT_XML_DELIMITER: float = 1.50
    WEIGHT_NUMERIC_PROB_EDGE: float = 2.00

    COMPLETION_XML_TAGS: Tuple[str, ...] = (
        "<reasoning>",
        "</reasoning>",
        "<prob>",
        "</prob>",
        "<edge>",
        "</edge>",
    )

    def __init__(
        self,
        max_seq_len: int = 2048,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            extender = DomainTokenizerExtender()
            tok, _, _ = extender.load_or_build_tokenizer(prefer_local_fast=True)
            tok.add_special_tokens(
                {"additional_special_tokens": list(ALL_DOMAIN_SPECIAL_TOKENS)}
            )
            self.tokenizer = tok

        # Resolve atomic integer token IDs for the 6 completion XML tags
        self.xml_tag_token_ids: Dict[str, int] = {}
        for tag in self.COMPLETION_XML_TAGS:
            enc = self.tokenizer.encode(tag)
            if enc:
                self.xml_tag_token_ids[tag] = enc[0]

    def collate_single_record(self, row: Dict[str, Any]) -> CollatedSFTExample:
        """
        Splits a single SFT record at `<|im_start|>assistant\\n`, masks the
        instruction prefix with `IGNORE_INDEX = -100`, and assigns token-level
        loss weights across `<reasoning>`, `<prob>`, and `<edge>`.
        """
        snap_uid = str(row.get("snapshot_uid") or "SFT-UNKNOWN")
        full_text = str(row.get("sft_full_chatml_text") or "")
        assistant_resp = str(row.get("assistant_response") or "")
        p_target = float(row.get("teacher_predicted_prob_p") or 0.50)
        edge_target = float(row.get("teacher_predicted_edge") or 0.0)
        q_t = round(p_target - edge_target, 4)

        marker_idx = full_text.find(self.ASSISTANT_MARKER)
        if marker_idx != -1:
            prompt_prefix = full_text[: marker_idx + len(self.ASSISTANT_MARKER)]
            completion_suffix = full_text[marker_idx + len(self.ASSISTANT_MARKER) :]
        else:
            sys_p = str(row.get("system_prompt") or "")
            usr_p = str(row.get("user_prompt") or "")
            prompt_prefix = (
                f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                f"<|im_start|>user\n{usr_p}<|im_end|>\n"
                f"{self.ASSISTANT_MARKER}"
            )
            completion_suffix = f"{assistant_resp}<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt_prefix)
        completion_ids = self.tokenizer.encode(completion_suffix)

        # Left-truncate prompt if combined sequence exceeds max_seq_len so the
        # assistant completion is never truncated!
        if len(prompt_ids) + len(completion_ids) > self.max_seq_len:
            keep_prompt_len = max(64, self.max_seq_len - len(completion_ids))
            prompt_ids = prompt_ids[-keep_prompt_len:]

        input_ids = prompt_ids + completion_ids
        n_prompt = len(prompt_ids)
        n_comp = len(completion_ids)

        labels = ([self.IGNORE_INDEX] * n_prompt) + list(completion_ids)
        loss_weights = [self.WEIGHT_MASKED_PROMPT] * n_prompt

        # Assign token-level weights inside the assistant completion
        prob_open_id = self.xml_tag_token_ids.get("<prob>", -1)
        prob_close_id = self.xml_tag_token_ids.get("</prob>", -2)
        edge_open_id = self.xml_tag_token_ids.get("<edge>", -3)
        edge_close_id = self.xml_tag_token_ids.get("</edge>", -4)
        all_xml_ids = set(self.xml_tag_token_ids.values())

        in_numeric_span = False
        upweighted_numeric_cnt = 0

        for tid in completion_ids:
            if tid in (prob_open_id, edge_open_id):
                loss_weights.append(self.WEIGHT_XML_DELIMITER)
                in_numeric_span = True
            elif tid in (prob_close_id, edge_close_id):
                loss_weights.append(self.WEIGHT_XML_DELIMITER)
                in_numeric_span = False
            elif tid in all_xml_ids:
                loss_weights.append(self.WEIGHT_XML_DELIMITER)
            elif in_numeric_span:
                loss_weights.append(self.WEIGHT_NUMERIC_PROB_EDGE)
                upweighted_numeric_cnt += 1
            else:
                loss_weights.append(self.WEIGHT_REASONING_PROSE)

        xml_match = COMPLETION_XML_EXTRACT_REGEX.search(assistant_resp)
        xml_ok = xml_match is not None

        mask_ratio = round((n_prompt / max(1, len(input_ids))) * 100.0, 2)

        return CollatedSFTExample(
            snapshot_uid=snap_uid,
            input_ids=input_ids,
            labels=labels,
            token_loss_weights=loss_weights,
            total_sequence_tokens=len(input_ids),
            masked_prompt_tokens_count=n_prompt,
            active_completion_tokens_count=n_comp,
            upweighted_numeric_tokens_count=upweighted_numeric_cnt,
            prompt_mask_ratio_pct=mask_ratio,
            target_prob_p=p_target,
            target_edge=edge_target,
            midpoint_qt=q_t,
            xml_structure_valid=xml_ok,
        )

    def collate_parquet_shard(
        self,
        parquet_path: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/train_sft_shard_00000.parquet"
        ),
    ) -> Tuple[List[CollatedSFTExample], CollatorAuditReport]:
        """
        Loads an SFT Parquet shard, collates every row, and produces a
        `CollatorAuditReport` verifying completion-only loss masking.
        """
        import polars as pl  # type: ignore

        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing SFT Parquet shard at {parquet_path}")

        rows = pl.read_parquet(parquet_path).to_dicts()
        examples = [self.collate_single_record(r) for r in rows]

        all_masked = all(
            ex.masked_prompt_tokens_count > 0
            and all(lbl == self.IGNORE_INDEX for lbl in ex.labels[: ex.masked_prompt_tokens_count])
            and all(lbl != self.IGNORE_INDEX for lbl in ex.labels[ex.masked_prompt_tokens_count :])
            for ex in examples
        )
        all_xml_ok = all(ex.xml_structure_valid for ex in examples)

        n_ex = max(1, len(examples))
        avg_total = round(sum(e.total_sequence_tokens for e in examples) / n_ex, 2)
        avg_masked = round(sum(e.masked_prompt_tokens_count for e in examples) / n_ex, 2)
        avg_active = round(sum(e.active_completion_tokens_count for e in examples) / n_ex, 2)
        avg_num = round(sum(e.upweighted_numeric_tokens_count for e in examples) / n_ex, 2)
        avg_ratio = round(sum(e.prompt_mask_ratio_pct for e in examples) / n_ex, 2)

        sample_ex = examples[0] if examples else None
        sample_summary = (
            {
                "snapshot_uid": sample_ex.snapshot_uid,
                "total_tokens": sample_ex.total_sequence_tokens,
                "masked_prompt_tokens_neg100": sample_ex.masked_prompt_tokens_count,
                "active_completion_tokens": sample_ex.active_completion_tokens_count,
                "upweighted_2x_numeric_tokens": sample_ex.upweighted_numeric_tokens_count,
                "target_prob_p": sample_ex.target_prob_p,
                "target_edge": sample_ex.target_edge,
            }
            if sample_ex
            else {}
        )

        report = CollatorAuditReport(
            stage="STAGE_4B_STRUCTURED_OUTPUT_COLLATOR",
            parquet_shard_audited=str(parquet_path),
            total_examples_collated=len(examples),
            all_prompts_masked_with_neg100=all_masked,
            all_completions_xml_valid=all_xml_ok,
            average_total_tokens=avg_total,
            average_masked_prompt_tokens=avg_masked,
            average_active_completion_tokens=avg_active,
            average_upweighted_numeric_tokens=avg_num,
            average_prompt_mask_ratio_pct=avg_ratio,
            sample_token_weights_summary=sample_summary,
        )
        return examples, report


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Stage 4B Structured Output Collator"
    )
    parser.add_argument(
        "--shard",
        type=pathlib.Path,
        default=pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/train_sft_shard_00000.parquet"
        ),
    )
    args = parser.parse_args()

    collator = StructuredOutputCollator()
    _, report = collator.collate_parquet_shard(parquet_path=args.shard)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())