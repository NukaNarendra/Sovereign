#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 4B: SUPERVISED FINE-TUNING (SFT)
File: 03_llm_training/stage2_sft/__init__.py
================================================================================
Public interface for Stage 4B Supervised Fine-Tuning (SFT) on Verified Blind
5-Step Philip Tetlock Superforecaster Reasoning Traces:
  1. `sft_lora_config.yaml`       : Declarative QLoRA/LoRA rank r=64, alpha=128
                                    configuration targeting all attention and
                                    SwiGLU MLP projections plus `embed_tokens`
                                    and `lm_head` on AMD Instinct MI300X.
  2. `structured_output_collator` : Completion-only loss masking collator
                                    (`label = -100` on `<|im_start|>system` and
                                    `<|im_start|>user`) with 2.0x token-level
                                    gradient upweighting on `<prob>` and `<edge>`.
  3. `run_sft_trainer`            : Stage 4B SFT trainer that initializes from
                                    the Stage 4A CPT checkpoint, trains over
                                    `sft_reasoning` Parquet shards, verifies
                                    XML schema and algebraic edge precision, and
                                    writes the Stage 4B Gate Certificate.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 03_llm_training.stage2_sft.<module>` never triggers a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "CollatedSFTExample",
    "CollatorAuditReport",
    "StructuredOutputCollator",
    "SFTStepTelemetry",
    "SFTTrainingMetrics",
    "SovereignSFTTrainer",
]


def __getattr__(name: str) -> Any:
    if name in (
        "CollatedSFTExample",
        "CollatorAuditReport",
        "StructuredOutputCollator",
    ):
        from . import structured_output_collator
        return getattr(structured_output_collator, name)
    if name in ("SFTStepTelemetry", "SFTTrainingMetrics", "SovereignSFTTrainer"):
        from . import run_sft_trainer
        return getattr(run_sft_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")