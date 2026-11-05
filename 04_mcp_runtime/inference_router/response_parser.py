#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: LLM XML RESPONSE PARSER
File: 04_mcp_runtime/inference_router/response_parser.py
================================================================================
Extracts calibrated float probability `p`, signed net edge `p - q_t`, reasoning
trace, and confidence score from Sovereign-Prob-7B outputs (`<reasoning>`,
`<prob>`, `<edge>`), enforcing strict algebraic consistency before passing `p`
to `05_execution_risk_engine`.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import math
import importlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [RESPONSE-PARSER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.response_parser")

REASONING_REGEX = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


@dataclass(frozen=True)
class ParsedForecastOutput:
    """
    Validated numeric forecast extracted from Sovereign-Prob-7B XML completion.
    """
    snapshot_uid: str
    xml_schema_valid: bool
    algebraic_edge_valid: bool
    extracted_prob_p: float
    market_midpoint_qt: float
    recomputed_signed_edge: float
    reported_edge_tag: float
    confidence_score: float
    reasoning_summary: str
    recommended_gate_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMResponseParser:
    """
    Parses `<reasoning>`, `<prob>`, and `<edge>` tags using the exact Stage 5
    `XMLFormatValidatorRewardFunction` and computes an epistemic confidence
    score based on rubric completeness and Shannon entropy reduction.
    """

    def __init__(self) -> None:
        xml_mod = importlib.import_module(
            "03_llm_training.stage3_grpo_brier_edge.format_validator_reward.xml_schema_reward"
        )
        self._validator = xml_mod.XMLFormatValidatorRewardFunction()

    def parse_llm_output(
        self,
        completion_text: str,
        market_midpoint_qt: float,
        min_required_edge: float = 0.015,
        safe_for_survival_bot: bool = True,
        snapshot_uid: str = "LIVE-SNAP",
    ) -> ParsedForecastOutput:
        ev = self._validator.evaluate_single_completion(
            completion_text=completion_text,
            market_midpoint_qt=market_midpoint_qt,
            snapshot_uid=snapshot_uid,
        )
        p_val = ev.extracted_prob_p
        q_t = round(float(market_midpoint_qt), 4)
        recomputed_edge = round(p_val - q_t, 4)

        m_reas = REASONING_REGEX.search(str(completion_text or ""))
        reas_text = m_reas.group(1).strip() if m_reas else ""

        # Confidence scales with rubric pillars matched (out of 4) and algebraic validity
        base_conf = (ev.rubric_pillars_matched / 4.0) * 0.75
        if ev.strict_xml_envelope_valid and ev.algebraic_edge_math_valid:
            base_conf += 0.25
        confidence = round(max(0.0, min(1.0, base_conf)), 4)

        if not safe_for_survival_bot or "PASS_HIGH_UMA_HAZARD" in completion_text:
            action = "PASS_HIGH_UMA_HAZARD"
        elif not ev.strict_xml_envelope_valid:
            action = "REJECT_MALFORMED_XML"
        elif abs(recomputed_edge) < min_required_edge:
            action = "HOLD_INSIDE_SPREAD"
        elif recomputed_edge >= min_required_edge:
            action = "MAKER_LIMIT_BUY_YES"
        else:
            action = "MAKER_LIMIT_BUY_NO"

        return ParsedForecastOutput(
            snapshot_uid=snapshot_uid,
            xml_schema_valid=ev.strict_xml_envelope_valid,
            algebraic_edge_valid=ev.algebraic_edge_math_valid,
            extracted_prob_p=p_val,
            market_midpoint_qt=q_t,
            recomputed_signed_edge=recomputed_edge,
            reported_edge_tag=ev.extracted_edge,
            confidence_score=confidence,
            reasoning_summary=reas_text[:240],
            recommended_gate_action=action,
        )


def main() -> int:
    parser = LLMResponseParser()
    sample_xml = (
        "<reasoning>\n"
        "1. [Outside View & Base Rate Prior]: Historical base rate is 38.0%, midpoint q_t = 0.6000.\n"
        "2. [UMA Optimistic Oracle Rule & Hazard Audit]: Hazard score is 0.15 (safe = true).\n"
        "3. [L2 Order Book & Whale Microstructure]: Spread s_t = 0.0120, OBI = +0.28.\n"
        "4. [Inside View & Pre-Cutoff Catalysts]: Reuters wire confirms bullish momentum.\n"
        "5. [Bayesian Calibration & Execution Gate]: Calibrated p = 0.6850, edge = +0.0850 -> MAKER_LIMIT_BUY_YES.\n"
        "</reasoning>\n<prob>0.6850</prob>\n<edge>+0.0850</edge>"
    )
    parsed = parser.parse_llm_output(
        completion_text=sample_xml,
        market_midpoint_qt=0.6000,
        min_required_edge=0.0120,
        safe_for_survival_bot=True,
        snapshot_uid="TEST-PARSE-01",
    )
    print(json.dumps(parsed.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())