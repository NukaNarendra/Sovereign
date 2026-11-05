#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 4A: CONTINUAL PRE-TRAINING (CPT)
File: 03_llm_training/stage1_cpt/__init__.py
================================================================================
Public interface for Stage 4A Domain Vocabulary Extension and Continual
Pre-Training (CPT):
  1. `domain_tokenizer_extender` : Extends Qwen-2.5-7B's tokenizer with 43
                                   atomic prediction-market XML tags (`<book>`,
                                   `<uma_rule>`, `<prob>`, `<edge>`, etc.) and
                                   microstructure regime tokens, initializing
                                   new embedding rows via mean-subword semantic
                                   averaging and padding to multiples of 64 for
                                   AMD Instinct MI300X Matrix Cores.
  2. `cpt_hyperparams.yaml`      : Declarative hyperparameter configuration for
                                   both bare-metal MI300X (192GB HBM3) runs and
                                   local WSL2 micro-architecture smoke tests.
  3. `run_cpt_trainer`           : Causal Language Modeling (CLM) trainer with
                                   constant-length sequence packing (4,096
                                   tokens), system-preamble loss masking, domain
                                   curriculum weighting, and SIGUSR1 emergency
                                   checkpointing.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 03_llm_training.stage1_cpt.<module>` never triggers a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "TokenizerExtensionReport",
    "DomainTokenizerExtender",
    "CPTTrainingMetrics",
    "SovereignCPTTrainer",
]


def __getattr__(name: str) -> Any:
    if name in ("TokenizerExtensionReport", "DomainTokenizerExtender"):
        from . import domain_tokenizer_extender
        return getattr(domain_tokenizer_extender, name)
    if name in ("CPTTrainingMetrics", "SovereignCPTTrainer"):
        from . import run_cpt_trainer
        return getattr(run_cpt_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")