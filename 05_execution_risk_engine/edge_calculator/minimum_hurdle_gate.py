#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: MINIMUM EDGE HURDLE GATE
File: 05_execution_risk_engine/edge_calculator/minimum_hurdle_gate.py
================================================================================
Deterministic Pre-Trade Safety Filter that rejects any trade where:
  1. Net per-share Expected Value (`net_ev_per_share_usd`) is below the minimum
     survival safety threshold (`min_net_edge_hurdle = 0.035` / `3.5%`, or the
     contract's dynamic spread hurdle `max(0.035, s_t + 0.005)`).
  2. The contract's bid-ask spread `s_t` exceeds `max_allowed_spread = 0.060`.
  3. The UMA Optimistic Oracle ambiguity hazard score exceeds `0.45` (`safe_for_survival_5usd_bot == False`).
  4. The LLM response parser flagged an unverified social rumor hazard or XML
     schema violation.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .net_ev_calculator import NetEVEvaluation, NetExpectedValueCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [HURDLE-GATE] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.minimum_hurdle_gate")


@dataclass(frozen=True)
class HurdleGateDecision:
    """
    Immutable pass/block decision emitted before Fractional Kelly position sizing.
    """
    snapshot_uid: str
    market_id: str
    approved_for_kelly_sizing: bool
    recommended_side: str
    net_ev_per_share_usd: float
    required_net_ev_hurdle_usd: float
    excess_edge_above_hurdle_usd: float
    uma_hazard_score: float
    clob_spread_st: float
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MinimumHurdleGate:
    """
    Enforces the 3.5% (`0.035`) minimum net EV safety hurdle and UMA/spread
    guardrails before allowing the $5.00 survival bankroll to risk capital.
    """

    def __init__(
        self,
        min_net_ev_hurdle: float = 0.035,
        max_allowed_spread_st: float = 0.060,
        max_allowed_uma_hazard: float = 0.45,
        min_confidence_score: float = 0.60,
    ) -> None:
        self.min_net_ev_hurdle = float(min_net_ev_hurdle)
        self.max_allowed_spread_st = float(max_allowed_spread_st)
        self.max_allowed_uma_hazard = float(max_allowed_uma_hazard)
        self.min_confidence_score = float(min_confidence_score)

    def evaluate_gate(
        self,
        ev_eval: NetEVEvaluation,
        uma_hazard_score: float = 0.15,
        safe_for_survival_bot: bool = True,
        xml_schema_valid: bool = True,
        confidence_score: float = 0.90,
        unverified_rumor_hazard: bool = False,
    ) -> HurdleGateDecision:
        reasons: List[str] = []

        if not xml_schema_valid:
            reasons.append("MALFORMED_LLM_XML_OUTPUT")
        if not safe_for_survival_bot or float(uma_hazard_score) > self.max_allowed_uma_hazard:
            reasons.append(f"HIGH_UMA_ORACLE_HAZARD:{uma_hazard_score:0.2f}")
        if unverified_rumor_hazard:
            reasons.append("UNCORROBORATED_SOCIAL_RUMOR_HAZARD")
        if float(confidence_score) < self.min_confidence_score:
            reasons.append(f"LOW_EPISTEMIC_CONFIDENCE:{confidence_score:0.2f}")
        if ev_eval.clob_spread_st > self.max_allowed_spread_st:
            reasons.append(f"SPREAD_TOO_WIDE:{ev_eval.clob_spread_st:0.4f}")

        dynamic_hurdle = round(
            max(self.min_net_ev_hurdle, ev_eval.clob_spread_st + 0.005), 4
        )
        excess_edge = round(ev_eval.net_ev_per_share_usd - dynamic_hurdle, 4)

        if ev_eval.net_ev_per_share_usd < dynamic_hurdle:
            reasons.append(
                f"NET_EV_BELOW_HURDLE:{ev_eval.net_ev_per_share_usd:+0.4f}<{dynamic_hurdle:0.4f}"
            )

        approved = len(reasons) == 0
        side = ev_eval.candidate_side if approved else "HOLD_NO_TRADE"

        return HurdleGateDecision(
            snapshot_uid=ev_eval.snapshot_uid,
            market_id=ev_eval.market_id,
            approved_for_kelly_sizing=approved,
            recommended_side=side,
            net_ev_per_share_usd=ev_eval.net_ev_per_share_usd,
            required_net_ev_hurdle_usd=dynamic_hurdle,
            excess_edge_above_hurdle_usd=excess_edge,
            uma_hazard_score=round(float(uma_hazard_score), 4),
            clob_spread_st=ev_eval.clob_spread_st,
            rejection_reasons=reasons,
        )


def main() -> int:
    calc = NetExpectedValueCalculator()
    gate = MinimumHurdleGate(min_net_ev_hurdle=0.035)

    strong_ev = calc.evaluate_net_ev(0.0050, 0.3147, 0.0162, "PIT-558938-T168H")
    pass_decision = gate.evaluate_gate(strong_ev, uma_hazard_score=0.15)

    weak_ev = calc.evaluate_net_ev(0.5200, 0.5000, 0.0200, "WEAK-EDGE-01")
    block_decision = gate.evaluate_gate(weak_ev, uma_hazard_score=0.15)

    print(
        json.dumps(
            {
                "stage": "STAGE_7_MINIMUM_HURDLE_GATE",
                "strong_edge_approved": pass_decision.approved_for_kelly_sizing,
                "weak_edge_blocked": not block_decision.approved_for_kelly_sizing,
                "approved_sample": pass_decision.to_dict(),
                "blocked_sample": block_decision.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())