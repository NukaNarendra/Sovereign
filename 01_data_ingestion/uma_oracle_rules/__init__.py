#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART C)
File: 01_data_ingestion/uma_oracle_rules/__init__.py
================================================================================
Public interface for the UMA Optimistic Oracle v2/v3 & Polymarket CTF Adapter
Ancillary Rule Extraction and Dispute History Ingestion Layer.

Why This Subpackage Is the #1 Edge Over Retail LLM Bots:
  1. Retail LLM bots trade on breaking Twitter/X headlines ("Trump announces
     tariff truce!") without reading the actual UMA Optimistic Oracle contract
     fine print ("Market only resolves YES upon official publication in the US
     Federal Register by 11:59 PM ET").
  2. This subpackage decodes raw hexadecimal `ancillaryData` payloads from
     Polygon Chain ID 137 (`UMA_CTF_ADAPTER_V2` & `UMA_OPTIMISTIC_ORACLE_V3`),
     scrapes historical DVM/OO disputes to identify high-ambiguity phrasing,
     and parses every market description into structured `<uma_rule>` clauses
     for Stage 4 Continual Pre-Training (CPT) and Stage 5 GRPO RL.
================================================================================
"""

from .ancillary_data_decoder import (
    DecodedAncillaryPayload,
    UmaAncillaryDataDecoder,
)
from .dispute_history_scraper import (
    UmaDisputeStatus,
    UmaOracleDisputeRecord,
    DisputeScraperConfig,
    UmaDisputeHistoryScraper,
)
from .resolution_edge_case_parser import (
    TimezoneCutoffSpec,
    ParsedUmaResolutionRule,
    ResolutionEdgeCaseParser,
)

__all__ = [
    "DecodedAncillaryPayload",
    "UmaAncillaryDataDecoder",
    "UmaDisputeStatus",
    "UmaOracleDisputeRecord",
    "DisputeScraperConfig",
    "UmaDisputeHistoryScraper",
    "TimezoneCutoffSpec",
    "ParsedUmaResolutionRule",
    "ResolutionEdgeCaseParser",
]