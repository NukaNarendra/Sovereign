#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: INFERENCE COST METER
File: 04_mcp_runtime/inference_router/inference_cost_meter.py
================================================================================
Logs exact token cost (`$0.0002/call` virtual compute burn and `$0.20/1M tokens`
Fireworks AI credit draw) to the survival ledger (`state/survival_nav_state.json`)
so the Reaper Watchdog enforces honest "Earn-or-Die" economics.
================================================================================
"""

from __future__ import annotations

import sys
import json
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [INFERENCE-METER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.inference_cost_meter")


@dataclass(frozen=True)
class InferenceCallReceipt:
    """Auditable receipt generated for a single Sovereign-Prob-7B forecast."""
    receipt_id: str
    snapshot_uid: str
    provider_tier_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    fireworks_credit_deducted_usd: float
    virtual_survival_burn_deducted_usd: float
    remaining_fireworks_credit_usd: float
    remaining_net_survival_nav_usd: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VirtualCostLedgerState:
    """Persistent state tracking Fireworks $50 credit and $5 survival burn."""
    fireworks_initial_credit_usd: float = 50.00
    fireworks_safety_buffer_usd: float = 1.50
    fireworks_consumed_credit_usd: float = 0.00
    price_per_million_tokens_usd: float = 0.20
    virtual_survival_charge_per_call_usd: float = 0.00020
    local_gguf_electricity_charge_per_call_usd: float = 0.00005
    total_serverless_calls: int = 0
    total_local_gguf_fallback_calls: int = 0
    total_prompt_tokens_processed: int = 0
    total_completion_tokens_generated: int = 0
    accrued_virtual_survival_burn_usd: float = 0.00
    gross_survival_bankroll_usd: float = 5.00
    last_updated_utc: str = ""

    @property
    def remaining_fireworks_credit_usd(self) -> float:
        return max(
            0.0,
            round(
                self.fireworks_initial_credit_usd - self.fireworks_consumed_credit_usd,
                6,
            ),
        )

    @property
    def can_use_fireworks_serverless(self) -> bool:
        return self.remaining_fireworks_credit_usd > self.fireworks_safety_buffer_usd

    @property
    def net_liquid_survival_nav_usd(self) -> float:
        return round(
            self.gross_survival_bankroll_usd - self.accrued_virtual_survival_burn_usd,
            6,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["remaining_fireworks_credit_usd"] = self.remaining_fireworks_credit_usd
        payload["can_use_fireworks_serverless"] = self.can_use_fireworks_serverless
        payload["net_liquid_survival_nav_usd"] = self.net_liquid_survival_nav_usd
        return payload


class InferenceCostMeter:
    """
    Meters every LLM inference call, updates `state/virtual_cost_ledger.json`,
    and syncs `accrued_compute_burn_usd` with `state/survival_nav_state.json`.
    """

    def __init__(
        self,
        ledger_path: pathlib.Path = pathlib.Path("state/virtual_cost_ledger.json"),
        survival_nav_path: pathlib.Path = pathlib.Path("state/survival_nav_state.json"),
    ) -> None:
        self.ledger_path = ledger_path
        self.survival_nav_path = survival_nav_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.survival_nav_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_init_ledger()

    def _load_or_init_ledger(self) -> VirtualCostLedgerState:
        if self.ledger_path.exists():
            try:
                raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                filtered = {
                    k: v
                    for k, v in raw.items()
                    if k in VirtualCostLedgerState.__dataclass_fields__
                }
                return VirtualCostLedgerState(**filtered)
            except Exception:
                pass
        st = VirtualCostLedgerState(
            last_updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        self._save_ledgers(st)
        return st

    def _save_ledgers(self, state: Optional[VirtualCostLedgerState] = None) -> None:
        target = state or self.state
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        target.last_updated_utc = now_iso

        tmp_cost = self.ledger_path.with_suffix(".json.tmp")
        tmp_cost.write_text(json.dumps(target.to_dict(), indent=2), encoding="utf-8")
        tmp_cost.replace(self.ledger_path)

        nav_payload = {
            "agent_id": "sovereign_survival_agent",
            "initial_seed_usd": 5.00,
            "liquid_pusd_balance": round(target.gross_survival_bankroll_usd, 6),
            "unrealized_ctf_value_usd": 0.00,
            "accrued_compute_burn_usd": round(
                target.accrued_virtual_survival_burn_usd, 6
            ),
            "net_liquid_nav_usd": target.net_liquid_survival_nav_usd,
            "status": (
                "ALIVE_EARNING"
                if target.net_liquid_survival_nav_usd > 0.0
                else "REAPER_DEATH_FLOOR_BREACHED"
            ),
            "last_audit_timestamp_utc": now_iso,
        }
        tmp_nav = self.survival_nav_path.with_suffix(".json.tmp")
        tmp_nav.write_text(json.dumps(nav_payload, indent=2), encoding="utf-8")
        tmp_nav.replace(self.survival_nav_path)

    def meter_inference_call(
        self,
        snapshot_uid: str,
        provider_tier: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> InferenceCallReceipt:
        p_tok = max(1, int(prompt_tokens))
        c_tok = max(1, int(completion_tokens))
        tot_tok = p_tok + c_tok

        self.state.total_prompt_tokens_processed += p_tok
        self.state.total_completion_tokens_generated += c_tok

        if "FIREWORKS" in provider_tier.upper():
            self.state.total_serverless_calls += 1
            fw_cost = round(
                (tot_tok / 1_000_000.0) * self.state.price_per_million_tokens_usd, 6
            )
            virt_burn = self.state.virtual_survival_charge_per_call_usd
            self.state.fireworks_consumed_credit_usd = round(
                self.state.fireworks_consumed_credit_usd + fw_cost, 6
            )
        else:
            self.state.total_local_gguf_fallback_calls += 1
            fw_cost = 0.0
            virt_burn = self.state.local_gguf_electricity_charge_per_call_usd

        self.state.accrued_virtual_survival_burn_usd = round(
            self.state.accrued_virtual_survival_burn_usd + virt_burn, 6
        )
        self._save_ledgers()

        call_num = (
            self.state.total_serverless_calls
            + self.state.total_local_gguf_fallback_calls
        )
        return InferenceCallReceipt(
            receipt_id=f"RCPT-{call_num:06d}",
            snapshot_uid=snapshot_uid,
            provider_tier_used=provider_tier,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=tot_tok,
            fireworks_credit_deducted_usd=fw_cost,
            virtual_survival_burn_deducted_usd=virt_burn,
            remaining_fireworks_credit_usd=self.state.remaining_fireworks_credit_usd,
            remaining_net_survival_nav_usd=self.state.net_liquid_survival_nav_usd,
            timestamp_utc=self.state.last_updated_utc,
        )


def main() -> int:
    meter = InferenceCostMeter()
    rcpt = meter.meter_inference_call(
        snapshot_uid="PIT-253591-T168H",
        provider_tier="FIREWORKS_SERVERLESS_LORA",
        prompt_tokens=1286,
        completion_tokens=214,
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_6_INFERENCE_COST_METER",
                "ledger_state": meter.state.to_dict(),
                "sample_receipt": rcpt.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())