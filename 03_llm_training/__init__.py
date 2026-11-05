#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 4 & STAGE 5: LLM POST-TRAINING PIPELINE
File: 03_llm_training/__init__.py
================================================================================
Root package initializer for the 3-Stage Quantitative LLM Post-Training Suite
(`03_llm_training/`) targeting Qwen-2.5-7B on AMD Instinct MI300X (192GB HBM3):

  1. `stage1_cpt`            : Continual Pre-Training (CPT) on Polymarket UMA
                               Optimistic Oracle rules, Level-2 CLOB order-book
                               tables, 24h volatility trajectories, and point-
                               in-time news wires with custom domain vocabulary
                               extension (`domain_tokenizer_extender.py`).
  2. `stage2_sft`            : Supervised Fine-Tuning (SFT) with QLoRA/LoRA and
                               completion-only loss masking on verified blind
                               5-step Tetlock Superforecaster Chain-of-Thought
                               traces (`<reasoning>`, `<prob>`, `<edge>`).
  3. `stage3_grpo_brier_edge`: Group Relative Policy Optimization (GRPO) with
                               Verifiable Rewards combining Proper Brier Score
                               Calibration, CLOB Spread-Adjusted PnL Edge, and
                               XML Schema Compliance.
  4. `model_registry`        : Checkpoint manager, LoRA weight merger, and 4-bit
                               AWQ/GGUF quantizer for the $5.00 survival bot.
================================================================================
"""

from typing import Final

__version__: Final[str] = "1.0.0-sovereign-stage4-5"
__stage__: Final[str] = "STAGE_4_AND_5_LLM_TRAINING"
__base_model__: Final[str] = "Qwen/Qwen2.5-7B"

SUBMODULES: Final[tuple[str, ...]] = (
    "stage1_cpt",
    "stage2_sft",
    "stage3_grpo_brier_edge",
    "model_registry",
)

__all__ = [
    "__version__",
    "__stage__",
    "__base_model__",
    "SUBMODULES",
]