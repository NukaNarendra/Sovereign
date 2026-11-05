#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 6: 3-SIGMA VOLUME ANOMALY DETECTOR
File: 04_mcp_runtime/websocket_anomaly_filter/zscore_volume_detector.py
================================================================================
Triggers a wake signal when 5-minute rolling Polymarket CLOB v2 notional USD
volume exceeds $3.0\sigma$ above the contract's rolling baseline mean:
    $$Z_{\text{vol}}(t) = \frac{V_{5\text{m}}(t) - \mu_V}{\max(\sigma_V, \sigma_{\min})} \ge 3.0$$
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [ZSCORE-VOL-DETECTOR] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.zscore_volume_detector")


@dataclass(frozen=True)
class VolumeObservation:
    """Represents a 5-minute rolling notional USD volume observation on CLOB v2."""
    snapshot_uid: str
    market_id: str
    condition_id: str
    timestamp_epoch_sec: float
    five_min_volume_usd: float
    spread_st: float
    uma_hazard_score: float
    safe_for_survival_5usd_bot: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VolumeSpikeTrigger:
    """Wake signal emitted when 5-min volume exceeds 3-sigma above mean."""
    snapshot_uid: str
    market_id: str
    condition_id: str
    timestamp_epoch_sec: float
    five_min_volume_usd: float
    rolling_mean_volume_usd: float
    rolling_std_volume_usd: float
    volume_zscore: float
    exceeded_3sigma_gate: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ZScoreVolumeDetector:
    """
    Maintains bounded rolling 5-minute volume histories per `market_id` and
    triggers a wake signal whenever `Z_vol >= 3.0`.
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        min_std_floor_usd: float = 25.0,
        window_size: int = 24,
        max_allowed_spread_st: float = 0.060,
    ) -> None:
        self.zscore_threshold = float(zscore_threshold)
        self.min_std_floor_usd = float(min_std_floor_usd)
        self.window_size = max(4, int(window_size))
        self.max_allowed_spread_st = float(max_allowed_spread_st)
        self._windows: Dict[str, Deque[float]] = {}
        self.total_evaluated: int = 0
        self.total_triggered: int = 0

    def seed_market_volumes(self, market_id: str, volumes_usd: Sequence[float]) -> None:
        win = self._windows.setdefault(market_id, deque(maxlen=self.window_size))
        for v in volumes_usd:
            win.append(float(v))

    def evaluate_volume_tick(
        self, obs: VolumeObservation
    ) -> Optional[VolumeSpikeTrigger]:
        """
        Computes the rolling volume Z-score prior to appending the new tick.
        Returns a `VolumeSpikeTrigger` if `Z_vol >= zscore_threshold` on a
        survival-safe market, otherwise `None`.
        """
        self.total_evaluated += 1
        if (
            not obs.safe_for_survival_5usd_bot
            or obs.uma_hazard_score > 0.45
            or obs.spread_st > self.max_allowed_spread_st
        ):
            return None

        win = self._windows.setdefault(obs.market_id, deque(maxlen=self.window_size))
        if len(win) >= 3:
            mean_v = sum(win) / len(win)
            var_v = sum((x - mean_v) ** 2 for x in win) / len(win)
            std_v = max(self.min_std_floor_usd, math.sqrt(var_v))
            z_val = (obs.five_min_volume_usd - mean_v) / std_v
        else:
            mean_v = obs.five_min_volume_usd
            std_v = self.min_std_floor_usd
            z_val = 0.0

        win.append(float(obs.five_min_volume_usd))

        if z_val >= self.zscore_threshold:
            self.total_triggered += 1
            return VolumeSpikeTrigger(
                snapshot_uid=obs.snapshot_uid,
                market_id=obs.market_id,
                condition_id=obs.condition_id,
                timestamp_epoch_sec=obs.timestamp_epoch_sec,
                five_min_volume_usd=round(obs.five_min_volume_usd, 2),
                rolling_mean_volume_usd=round(mean_v, 2),
                rolling_std_volume_usd=round(std_v, 2),
                volume_zscore=round(z_val, 4),
                exceeded_3sigma_gate=True,
            )
        return None


def main() -> int:
    detector = ZScoreVolumeDetector()
    detector.seed_market_volumes("253591", [180.0, 195.0, 210.0, 185.0, 200.0])

    quiet = detector.evaluate_volume_tick(
        VolumeObservation(
            snapshot_uid="QUIET-VOL-1",
            market_id="253591",
            condition_id="0x" + "11" * 32,
            timestamp_epoch_sec=1776000100.0,
            five_min_volume_usd=215.0,
            spread_st=0.012,
            uma_hazard_score=0.15,
            safe_for_survival_5usd_bot=True,
        )
    )
    burst = detector.evaluate_volume_tick(
        VolumeObservation(
            snapshot_uid="BURST-VOL-2",
            market_id="253591",
            condition_id="0x" + "11" * 32,
            timestamp_epoch_sec=1776000200.0,
            five_min_volume_usd=2450.0,
            spread_st=0.012,
            uma_hazard_score=0.15,
            safe_for_survival_5usd_bot=True,
        )
    )
    print(
        json.dumps(
            {
                "stage": "STAGE_6_ZSCORE_VOLUME_DETECTOR",
                "quiet_tick_suppressed": quiet is None,
                "three_sigma_burst_triggered": burst is not None,
                "sample_trigger": burst.to_dict() if burst else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())