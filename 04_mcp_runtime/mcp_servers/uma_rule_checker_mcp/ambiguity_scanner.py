#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: UMA AMBIGUITY TRAP SCANNER (MCP)
File: 04_mcp_runtime/mcp_servers/uma_rule_checker_mcp/ambiguity_scanner.py
================================================================================
Real-time UMA Optimistic Oracle v3 ancillary rule verifier that scans contract
descriptions for subjective dispute triggers and blocks execution on high-risk
markets (`ambiguity_hazard_score > 0.45`).
================================================================================
"""

from __future__ import annotations

import sys
import json
import importlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [UMA-AMBIGUITY-SCAN] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.ambiguity_scanner")


@dataclass(frozen=True)
class LiveUmaHazardVerdict:
    """
    Deterministic execution veto/approval verdict returned by `AmbiguityScanner`.
    """
    market_id: str
    question: str
    primary_authority_type: str
    detected_timezone: str
    dispute_trigger_category: str
    ambiguity_hazard_score: float
    safe_for_survival_5usd_bot: bool
    execution_veto_reason: str
    cpt_uma_rule_xml: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AmbiguityScanner:
    """
    Wraps `ResolutionEdgeCaseParser` from `01_data_ingestion.uma_oracle_rules`
    to audit live Polymarket contracts before any $5.00 survival bet is placed.
    """

    def __init__(self, max_allowed_hazard: float = 0.45) -> None:
        parser_mod = importlib.import_module(
            "01_data_ingestion.uma_oracle_rules.resolution_edge_case_parser"
        )
        self._parser = parser_mod.ResolutionEdgeCaseParser(
            max_allowed_survival_hazard=max_allowed_hazard
        )

    def scan_contract_rules(
        self, market_row: Dict[str, Any]
    ) -> LiveUmaHazardVerdict:
        parsed = self._parser.parse_market_resolution_rules(market_row)
        veto = (
            "APPROVED_DETERMINISTIC_ORACLE_RULE"
            if parsed.safe_for_survival_5usd_bot
            else f"BLOCKED_UMA_DISPUTE_HAZARD_{parsed.dispute_trigger_category}"
        )
        return LiveUmaHazardVerdict(
            market_id=parsed.market_id,
            question=parsed.question,
            primary_authority_type=parsed.primary_authority_type,
            detected_timezone=parsed.detected_timezone,
            dispute_trigger_category=parsed.dispute_trigger_category,
            ambiguity_hazard_score=parsed.ambiguity_hazard_score,
            safe_for_survival_5usd_bot=parsed.safe_for_survival_5usd_bot,
            execution_veto_reason=veto,
            cpt_uma_rule_xml=parsed.cpt_formatted_rule_tag,
        )


def main() -> int:
    scanner = AmbiguityScanner()
    safe_verdict = scanner.scan_contract_rules(
        {
            "market_id": "253591",
            "condition_id": "0x" + "11" * 32,
            "question": "Will the Federal Reserve cut interest rates by 25 bps at the June 2026 FOMC meeting?",
            "description": "Resolves to Yes if the official FOMC statement at federalreserve.gov lowers the target range by 25 bps by 2:00 PM ET. Otherwise resolves to No.",
            "resolution_source": "https://www.federalreserve.gov",
        }
    )
    trap_verdict = scanner.scan_contract_rules(
        {
            "market_id": "999999",
            "condition_id": "0x" + "22" * 32,
            "question": "Will a secret ceasefire rumour be reported?",
            "description": "Resolves based on a consensus of credible media reporting.",
            "resolution_source": "",
        }
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_6_UMA_AMBIGUITY_SCANNER",
                "safe_contract_approved": safe_verdict.safe_for_survival_5usd_bot,
                "trap_contract_blocked": not trap_verdict.safe_for_survival_5usd_bot,
                "safe_verdict": safe_verdict.to_dict(),
                "trap_verdict": trap_verdict.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())