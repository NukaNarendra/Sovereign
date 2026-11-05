#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART E)
File: 02_dataset_factory/parquet_warehouse/__init__.py
================================================================================
Public interface for the Stage 3 Apache Parquet Training Warehouse, Schema
Registry Validator, and Chronological Train/Val/Test Dataset Splitter:
  1. `schema_registry.json`         : Canonical column definitions, data types,
                                      numeric bounds, and target-masking rules
                                      for Stage 4A CPT, Stage 4B SFT, and
                                      Stage 5 GRPO datasets.
  2. `dataset_splitter.py`          : Validates compiled prompts and verified
                                      SFT teacher traces against the schema
                                      registry and writes partitioned `TRAIN`,
                                      `VAL`, and `TEST` Parquet shards with
                                      SHA-256 integrity hashes.
  3. `cpt_corpus_manifest.yaml`     : Stage 4A Continual Pre-Training manifest.
  4. `sft_reasoning_manifest.yaml`  : Stage 4B Supervised Fine-Tuning manifest.
  5. `grpo_verifiable_manifest.yaml`: Stage 5 Verifiable GRPO RL manifest.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 02_dataset_factory.parquet_warehouse.<module>` never triggers a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "SplitShardMetadata",
    "WarehouseGateCertificate",
    "ParquetWarehouseSplitter",
]


def __getattr__(name: str) -> Any:
    if name in (
        "SplitShardMetadata",
        "WarehouseGateCertificate",
        "ParquetWarehouseSplitter",
    ):
        from . import dataset_splitter
        return getattr(dataset_splitter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")