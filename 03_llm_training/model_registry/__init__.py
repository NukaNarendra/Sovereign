#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: MODEL REGISTRY & 4-BIT QUANTIZATION
File: 03_llm_training/model_registry/__init__.py
================================================================================
Public interface for merging Stage 4A CPT + Stage 4B SFT + Stage 5 GRPO LoRA
adapters into unified BF16 safetensors (`merge_lora_adapters.py`), exporting
4-bit AWQ & GGUF quantized models (`export_awq_gguf.sh`), and deploying to
Fireworks AI (`fireworks_upload_deployer.py`).
================================================================================
"""

from typing import Any

__all__ = [
    "MergedLayerAudit",
    "MergedModelManifest",
    "LoRAWeightMerger",
    "FireworksDeploymentCertificate",
    "FireworksLoRADeployer",
]


def __getattr__(name: str) -> Any:
    if name in ("MergedLayerAudit", "MergedModelManifest", "LoRAWeightMerger"):
        from . import merge_lora_adapters
        return getattr(merge_lora_adapters, name)
    if name in ("FireworksDeploymentCertificate", "FireworksLoRADeployer"):
        from . import fireworks_upload_deployer
        return getattr(fireworks_upload_deployer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")