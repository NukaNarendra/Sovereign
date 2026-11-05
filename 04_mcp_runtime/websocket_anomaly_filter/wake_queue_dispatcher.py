#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: PRIORITY WAKE QUEUE DISPATCHER
File: 04_mcp_runtime/websocket_anomaly_filter/wake_queue_dispatcher.py
================================================================================
Priority-Ranked Asynchronous & Persistent JSONL Wake Queue Dispatcher Connecting
the Zero-Cost WebSocket Anomaly Filter to the 7B LLM Inference Router.

How the Wake Queue Protects Latency & Survival Bankroll Capital:
  1. During a major macro event (e.g., an FOMC rate announcement or US CPI print),
     dozens of Polymarket contracts spike simultaneously within the same 500ms
     window.
  2. Rather than firing 40 concurrent LLM requests or processing markets in
     arbitrary arrival order, `WakeQueueDispatcher` ranks every triggered
     `AnomalyTriggerEvent` by its composite `priority_urgency_score`:
         $$S_{\text{urgency}} = Z_{\text{vol}} + 25.0 \cdot |\Delta q_t| + |\text{OBI}_t|$$
  3. Deduplicates by `market_id` (keeping only the freshest snapshot for any
     given contract) and persists the prioritized queue to:
     `state/wake_queue/active_wake_queue.jsonl`
================================================================================
"""

from __future__ import annotations

import sys
import json
import heapq
import asyncio
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clob_ws_listener import CLOBWSListener, CombinedAnomalyWakeSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [WAKE-DISPATCHER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.wake_queue_dispatcher")


# ==============================================================================
# 1. WAKE QUEUE ENVELOPE & DISPATCH TELEMETRY DATA STRUCTURES
# ==============================================================================
@dataclass(frozen=True)
class WakeQueueEnvelope:
    """
    Prioritized task envelope popped by `HybridModelRouter` to run MCP tool
    enrichment and 7B LLM probability forecasting.
    """
    queue_rank: int
    trigger_id: str
    snapshot_uid: str
    market_id: str
    condition_id: str
    canonical_category: str
    question: str
    priority_urgency_score: float
    trigger_reasons: List[str]
    implied_prob_mid_qt: float
    spread_st: float
    enqueued_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WakeQueueDispatchReport:
    """
    Summary audit report for the Stage 6 Priority Wake Queue Dispatcher.
    """
    stage: str
    timestamp_utc: str
    raw_anomalies_received: int
    unique_markets_enqueued: int
    duplicate_market_bursts_coalesced: int
    queue_state_file_path: str
    highest_urgency_score: float
    top_dispatched_envelopes: List[WakeQueueEnvelope]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_dispatched_envelopes"] = [
            e.to_dict() for e in self.top_dispatched_envelopes
        ]
        return payload


# ==============================================================================
# 2. CORE PRIORITY WAKE QUEUE DISPATCHER ENGINE
# ==============================================================================
class WakeQueueDispatcher:
    """
    Coalesces burst anomalies per `market_id`, orders them in a max-urgency
    heap, and writes the atomic `active_wake_queue.jsonl` ledger for the
    Stage 6 MCP + LLM inference loop.
    """

    def __init__(
        self,
        queue_dir: pathlib.Path = pathlib.Path("state/wake_queue"),
        max_queue_depth: int = 128,
    ) -> None:
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.queue_dir / "active_wake_queue.jsonl"
        self.max_queue_depth = max(8, int(max_queue_depth))
        self._by_market_id: Dict[str, CombinedAnomalyWakeSignal] = {}
        self._coalesced_count: int = 0

    def ingest_anomaly_events(
        self, events: Sequence[CombinedAnomalyWakeSignal]
    ) -> int:
        """
        Ingests a batch of `CombinedAnomalyWakeSignal` items, keeping the highest-urgency
        (or freshest) trigger per unique `market_id`.
        """
        for ev in events:
            existing = self._by_market_id.get(ev.market_id)
            if existing is not None:
                self._coalesced_count += 1
                if ev.priority_urgency_score >= existing.priority_urgency_score:
                    self._by_market_id[ev.market_id] = ev
            else:
                self._by_market_id[ev.market_id] = ev
        return len(self._by_market_id)

    def flush_prioritized_queue(self) -> List[WakeQueueEnvelope]:
        """
        Sorts all active market triggers by descending `priority_urgency_score`,
        caps at `max_queue_depth`, and atomically writes `active_wake_queue.jsonl`.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sorted_events = sorted(
            self._by_market_id.values(),
            key=lambda x: x.priority_urgency_score,
            reverse=True,
        )[: self.max_queue_depth]

        envelopes: List[WakeQueueEnvelope] = []
        lines: List[str] = []

        for rank_idx, ev in enumerate(sorted_events, start=1):
            env = WakeQueueEnvelope(
                queue_rank=rank_idx,
                trigger_id=ev.trigger_id,
                snapshot_uid=ev.snapshot_uid,
                market_id=ev.market_id,
                condition_id=ev.condition_id,
                canonical_category=ev.canonical_category,
                question=ev.question,
                priority_urgency_score=ev.priority_urgency_score,
                trigger_reasons=ev.trigger_reasons,
                implied_prob_mid_qt=ev.implied_prob_mid_qt,
                spread_st=ev.spread_st,
                enqueued_at_utc=now_iso,
            )
            envelopes.append(env)
            lines.append(json.dumps(env.to_dict()))

        tmp_path = self.queue_file.with_suffix(".jsonl.tmp")
        tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp_path.replace(self.queue_file)
        return envelopes

    async def run_dispatch_pipeline(self) -> WakeQueueDispatchReport:
        """
        Streams the out-of-time test shard through `CLOBWSListener`,
        coalesces burst triggers by `market_id`, and writes the priority wake queue.
        """
        listener = CLOBWSListener()
        raw_events = await listener.stream_replay_from_parquet(max_rows=600)
        self.ingest_anomaly_events(raw_events)
        envelopes = self.flush_prioritized_queue()

        top_score = envelopes[0].priority_urgency_score if envelopes else 0.0
        report = WakeQueueDispatchReport(
            stage="STAGE_6_WAKE_QUEUE_DISPATCHER",
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            raw_anomalies_received=len(raw_events),
            unique_markets_enqueued=len(envelopes),
            duplicate_market_bursts_coalesced=self._coalesced_count,
            queue_state_file_path=str(self.queue_file),
            highest_urgency_score=top_score,
            top_dispatched_envelopes=envelopes[:4],
        )
        logger.info(
            "Wake Queue Dispatched | Raw Spikes: %d -> Unique Markets: %d (Coalesced: %d) | Max Urgency: %.2f",
            report.raw_anomalies_received,
            report.unique_markets_enqueued,
            report.duplicate_market_bursts_coalesced,
            report.highest_urgency_score,
        )
        return report


def main() -> int:
    dispatcher = WakeQueueDispatcher()
    report = asyncio.run(dispatcher.run_dispatch_pipeline())
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())