#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 8: MITOSIS $50.00 CLONE SPAWNER
File: 06_survival_watchdog/reaper_controller/mitosis_clone_spawner.py
================================================================================
When the parent survival agent compounds its initial `$5.00` seed by `10x` to
`Net Liquid NAV >= $50.00`, this controller:
  1. Carves out `$5.00` of realized pUSD profit as seed capital for a new
     isolated child container (`sovereign_survival_agent_clone_XX`).
  2. Retains `$45.00` in the parent wallet (resetting the parent's next mitosis
     watermark to `$95.00`) so parent and child compound in parallel across
     uncorrelated Polymarket domains!
================================================================================
"""

from __future__ import annotations

import sys
import json
import hashlib
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [MITOSIS-SPAWNER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.mitosis_clone_spawner")


@dataclass(frozen=True)
class MitosisCloneReceipt:
    """Manifest emitted when a parent agent hits 10x ($50) and spawns a clone."""
    parent_agent_id: str
    child_clone_container_name: str
    assigned_domain_specialty: str
    parent_nav_before_mitosis_usd: float
    child_seed_capital_allocated_usd: float
    parent_nav_after_mitosis_usd: float
    next_parent_mitosis_watermark_usd: float
    docker_spawn_command: str
    spawn_status: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MitosisCloneSpawner:
    """
    Spawns specialized child agent containers seeded with $5.00 whenever a
    parent agent reaches its $50.00 (10x) Net Asset Value watermark.
    """

    SPECIALTIES = (
        "MACRO_ECONOMICS_FED",
        "CRYPTO_DIGITAL_ASSETS",
        "GEOPOLITICS_CONFLICT",
        "US_GLOBAL_POLITICS",
    )

    def __init__(
        self,
        mitosis_trigger_usd: float = 50.00,
        child_seed_usd: float = 5.00,
        registry_path: pathlib.Path = pathlib.Path(
            "state/mitosis_clones_registry.json"
        ),
    ) -> None:
        self.mitosis_trigger_usd = float(mitosis_trigger_usd)
        self.child_seed_usd = float(child_seed_usd)
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_and_spawn_if_eligible(
        self,
        parent_agent_id: str,
        current_net_nav_usd: float,
        clone_generation_index: int = 1,
    ) -> MitosisCloneReceipt:
        nav = float(current_net_nav_usd)
        if nav < self.mitosis_trigger_usd:
            raise ValueError(
                f"NAV ${nav:.2f} has not reached mitosis threshold ${self.mitosis_trigger_usd:.2f}"
            )

        specialty = self.SPECIALTIES[
            (clone_generation_index - 1) % len(self.SPECIALTIES)
        ]
        child_name = f"sovereign_survival_clone_{clone_generation_index:02d}_{specialty[:5].lower()}"
        parent_after = round(nav - self.child_seed_usd, 4)
        next_watermark = round(parent_after + 50.00, 2)

        cmd = (
            f"docker run -d --name {child_name} "
            f"-e SURVIVAL_INITIAL_BANKROLL_USD={self.child_seed_usd:.2f} "
            f"-e DOMAIN_SPECIALTY={specialty} "
            f"sovereign_survival_agent:latest"
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        receipt = MitosisCloneReceipt(
            parent_agent_id=parent_agent_id,
            child_clone_container_name=child_name,
            assigned_domain_specialty=specialty,
            parent_nav_before_mitosis_usd=round(nav, 4),
            child_seed_capital_allocated_usd=self.child_seed_usd,
            parent_nav_after_mitosis_usd=parent_after,
            next_parent_mitosis_watermark_usd=next_watermark,
            docker_spawn_command=cmd,
            spawn_status="MITOSIS_CHILD_SPEC_REGISTERED",
            timestamp_utc=now_iso,
        )
        self.registry_path.write_text(
            json.dumps(receipt.to_dict(), indent=2), encoding="utf-8"
        )
        return receipt


def main() -> int:
    spawner = MitosisCloneSpawner()
    rcpt = spawner.evaluate_and_spawn_if_eligible(
        parent_agent_id="sovereign_survival_agent",
        current_net_nav_usd=52.40,
        clone_generation_index=1,
    )
    print(json.dumps(rcpt.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())