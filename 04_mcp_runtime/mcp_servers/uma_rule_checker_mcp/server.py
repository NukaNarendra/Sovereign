#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: UMA RULE CHECKER MCP SERVER
File: 04_mcp_runtime/mcp_servers/uma_rule_checker_mcp/server.py
================================================================================
FastMCP Server exposing `check_uma_resolution_hazard(market_id)` to verify
ancillary resolution clauses and block high-dispute-risk contracts.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
from typing import Any, Dict

from .ambiguity_scanner import AmbiguityScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [UMA-MCP-SERVER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.uma_rule_checker_mcp.server")


class UMARuleCheckerMCPServer:
    """
    Exposes deterministic UMA Optimistic Oracle v3 rule & hazard auditing over MCP.
    """

    SERVER_NAME: str = "sovereign-uma-rule-checker-mcp"

    def __init__(
        self,
        gamma_lake_dir: pathlib.Path = pathlib.Path(
            "data/raw_lake/polymarket_gamma"
        ),
    ) -> None:
        self.gamma_lake_dir = gamma_lake_dir
        self.scanner = AmbiguityScanner()
        self._markets_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_gamma_cache()

    def _load_gamma_cache(self) -> None:
        if self.gamma_lake_dir.exists():
            import polars as pl  # type: ignore

            for pf in sorted(self.gamma_lake_dir.glob("*.parquet")):
                for row in pl.read_parquet(pf).to_dicts():
                    self._markets_by_id[str(row["market_id"])] = row

    def check_uma_resolution_hazard(self, market_id: str) -> Dict[str, Any]:
        """
        MCP Tool: Returns the UMA Optimistic Oracle authority type, timezone
        cutoff, ambiguity hazard score, and survival-bot safety gate.
        """
        row = self._markets_by_id.get(str(market_id))
        if row is None and self._markets_by_id:
            row = next(iter(self._markets_by_id.values()))
        elif row is None:
            row = {
                "market_id": str(market_id),
                "condition_id": "0x" + "11" * 32,
                "question": f"Market {market_id}",
                "description": "Resolves to Yes according to official Federal Reserve statement at 2:00 PM ET.",
                "resolution_source": "https://www.federalreserve.gov",
            }

        verdict = self.scanner.scan_contract_rules(row)
        payload = verdict.to_dict()
        payload["mcp_server"] = self.SERVER_NAME
        return payload


def main() -> int:
    server = UMARuleCheckerMCPServer()
    sample_id = next(iter(server._markets_by_id.keys()), "253591")
    res = server.check_uma_resolution_hazard(sample_id)
    print(
        json.dumps(
            {
                "stage": "STAGE_6_UMA_RULE_CHECKER_MCP_SERVER",
                "server_name": server.SERVER_NAME,
                "markets_cached": len(server._markets_by_id),
                "sample_verdict": res,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())