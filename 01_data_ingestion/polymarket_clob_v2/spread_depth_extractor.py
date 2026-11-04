#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART B)
File: 01_data_ingestion/polymarket_clob_v2/spread_depth_extractor.py
================================================================================
Quantitative Level-2 Order Book Microstructure & Slippage Analyzer.

Why This Module Is Critical:
  1. General LLMs only look at the last traded price (`0.55`). They ignore that
     the best bid is `0.48` and best ask is `0.62` (a 14-cent spread that makes
     taker trading negative EV).
  2. This module computes institutional microstructure features from any
     `RawClobOrderBook` or historical price trajectory:
       - `best_bid`, `best_ask`, `midpoint`, and `spread_abs`
       - `micro_price` (volume-weighted mid-price biased toward queue pressure)
       - `order_book_imbalance_top1` and `order_book_imbalance_5pct`
       - `bid_depth_1pct_usd`, `ask_depth_1pct_usd`, `bid_depth_5pct_usd`,
         `ask_depth_5pct_usd`
       - `walk_the_book_vwap`: Simulates the exact Volume-Weighted Average Price
         and slippage if our $5.00 survival bot (or a $5,000 institutional
         vault) sweeps the ask or bid ladder.
================================================================================
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from .clob_v2_rest_client import OrderBookLevel, RawClobOrderBook, PriceHistoryPoint

logger = logging.getLogger("sovereign_prob_engine.spread_depth_extractor")


# ==============================================================================
# 1. DATA STRUCTURES FOR SLIPPAGE & MICROSTRUCTURE METRICS
# ==============================================================================
@dataclass(frozen=True)
class WalkTheBookResult:
    """
    Captures the exact execution impact of sweeping `target_notional_usd` across
    the resting CLOB limit order ladder (`BUY_YES` sweeps asks; `SELL_YES` sweeps bids).
    """
    side: str
    target_notional_usd: float
    filled_notional_usd: float
    filled_shares: float
    vwap_execution_price: float
    reference_best_price: float
    slippage_abs: float
    slippage_bps: float
    is_fully_fillable: bool
    levels_consumed: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrderBookMicrostructureMetrics:
    """
    Complete microstructure feature vector extracted from a Polymarket CLOB L2
    order book snapshot. Used in Stage 3 (`orderbook_serializer`), Stage 5
    (`clob_spread_edge_reward`), and Stage 6 (`fractional_kelly_sizer`).
    """
    asset_token_id: str
    timestamp_ms: int
    best_bid: float
    best_ask: float
    midpoint: float
    micro_price: float
    spread_abs: float
    spread_bps: float
    top_bid_size_shares: float
    top_ask_size_shares: float
    obi_top_level: float
    obi_5pct_band: float
    bid_depth_1pct_usd: float
    ask_depth_1pct_usd: float
    bid_depth_5pct_usd: float
    ask_depth_5pct_usd: float
    total_book_liquidity_usd: float
    survival_5usd_buy_vwap: float
    survival_5usd_slippage_bps: float
    inst_1000usd_buy_vwap: float
    inst_1000usd_slippage_bps: float
    is_tradable_under_survival_guardrails: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ==============================================================================
# 2. CORE MICROSTRUCTURE & SPREAD-DEPTH EXTRACTOR ENGINE
# ==============================================================================
class SpreadDepthExtractor:
    """
    Extracts Level-2 microstructure features, simulates VWAP order-book sweeps,
    and reconstructs synthetic historical spreads when only `/prices-history`
    tick series are available for older resolved markets.
    """

    def __init__(
        self,
        max_allowed_survival_spread: float = 0.060,
        min_required_top_depth_usd: float = 15.0,
    ) -> None:
        self.max_allowed_survival_spread = max_allowed_survival_spread
        self.min_required_top_depth_usd = min_required_top_depth_usd

    # --------------------------------------------------------------------------
    # Walk-the-Book VWAP Slippage Simulator
    # --------------------------------------------------------------------------
    @staticmethod
    def simulate_walk_the_book(
        levels: Sequence[OrderBookLevel],
        target_notional_usd: float,
        side: str = "BUY_ASK",
    ) -> WalkTheBookResult:
        """
        Walks sorted L2 order book levels (`asks` for BUY, `bids` for SELL) to
        compute the exact VWAP execution price and price impact for a given USD
        order notional.
        """
        if not levels or target_notional_usd <= 0.0:
            return WalkTheBookResult(
                side=side,
                target_notional_usd=target_notional_usd,
                filled_notional_usd=0.0,
                filled_shares=0.0,
                vwap_execution_price=0.0,
                reference_best_price=0.0,
                slippage_abs=0.0,
                slippage_bps=0.0,
                is_fully_fillable=False,
                levels_consumed=0,
            )

        ref_best_price = levels[0].price
        remaining_usd = float(target_notional_usd)
        accumulated_usd = 0.0
        accumulated_shares = 0.0
        levels_used = 0

        for lvl in levels:
            if remaining_usd <= 1e-9:
                break
            if lvl.price <= 0.0 or lvl.size <= 0.0:
                continue

            level_max_usd = lvl.price * lvl.size
            take_usd = min(remaining_usd, level_max_usd)
            take_shares = take_usd / lvl.price

            accumulated_usd += take_usd
            accumulated_shares += take_shares
            remaining_usd -= take_usd
            levels_used += 1

        if accumulated_shares <= 1e-9:
            return WalkTheBookResult(
                side=side,
                target_notional_usd=target_notional_usd,
                filled_notional_usd=0.0,
                filled_shares=0.0,
                vwap_execution_price=ref_best_price,
                reference_best_price=ref_best_price,
                slippage_abs=0.0,
                slippage_bps=0.0,
                is_fully_fillable=False,
                levels_consumed=0,
            )

        vwap_price = accumulated_usd / accumulated_shares
        slippage_abs = abs(vwap_price - ref_best_price)
        slippage_bps = round(slippage_abs * 10_000.0, 2)
        fully_filled = remaining_usd <= 0.01

        return WalkTheBookResult(
            side=side,
            target_notional_usd=round(target_notional_usd, 4),
            filled_notional_usd=round(accumulated_usd, 4),
            filled_shares=round(accumulated_shares, 4),
            vwap_execution_price=round(vwap_price, 6),
            reference_best_price=round(ref_best_price, 6),
            slippage_abs=round(slippage_abs, 6),
            slippage_bps=slippage_bps,
            is_fully_fillable=fully_filled,
            levels_consumed=levels_used,
        )

    # --------------------------------------------------------------------------
    # Full L2 Order Book Feature Extraction
    # --------------------------------------------------------------------------
    def extract_from_order_book(
        self, book: RawClobOrderBook
    ) -> OrderBookMicrostructureMetrics:
        """
        Computes the complete `OrderBookMicrostructureMetrics` feature vector
        from a `RawClobOrderBook`.
        """
        book.sort_book_levels()

        best_bid = book.bids[0].price if book.bids else 0.0
        best_ask = book.asks[0].price if book.asks else 1.0
        top_bid_sz = book.bids[0].size if book.bids else 0.0
        top_ask_sz = book.asks[0].size if book.asks else 0.0

        # Ensure spread is strictly non-negative even on crossed snapshot ticks
        spread_abs = max(0.0, round(best_ask - best_bid, 6))
        midpoint = round((best_bid + best_ask) / 2.0, 6) if (book.bids and book.asks) else (best_bid or best_ask)

        # Micro-price weights the best bid and best ask by the opposite queue's size
        top_total_sz = top_bid_sz + top_ask_sz
        if top_total_sz > 0.0 and book.bids and book.asks:
            micro_price = round(
                (best_bid * top_ask_sz + best_ask * top_bid_sz) / top_total_sz,
                6,
            )
            obi_top = round((top_bid_sz - top_ask_sz) / top_total_sz, 4)
        else:
            micro_price = midpoint
            obi_top = 0.0

        # Compute cumulative USD depth within 1% (0.01) and 5% (0.05) probability bands
        bid_1pct_usd = sum(
            b.price * b.size for b in book.bids if (best_bid - b.price) <= 0.01001
        )
        ask_1pct_usd = sum(
            a.price * a.size for a in book.asks if (a.price - best_ask) <= 0.01001
        )
        bid_5pct_usd = sum(
            b.price * b.size for b in book.bids if (best_bid - b.price) <= 0.05001
        )
        ask_5pct_usd = sum(
            a.price * a.size for a in book.asks if (a.price - best_ask) <= 0.05001
        )

        band_5pct_total = bid_5pct_usd + ask_5pct_usd
        obi_5pct = (
            round((bid_5pct_usd - ask_5pct_usd) / band_5pct_total, 4)
            if band_5pct_total > 0.0
            else 0.0
        )

        total_book_usd = sum(b.price * b.size for b in book.bids) + sum(
            a.price * a.size for a in book.asks
        )

        # Simulate $5.00 survival order (max 15% Kelly = $0.75 to $5.00 sweep) and $1,000 institutional sweep
        sweep_5usd = self.simulate_walk_the_book(book.asks, target_notional_usd=5.0, side="BUY_ASK")
        sweep_1000usd = self.simulate_walk_the_book(book.asks, target_notional_usd=1000.0, side="BUY_ASK")

        is_tradable = (
            bool(book.bids)
            and bool(book.asks)
            and spread_abs <= self.max_allowed_survival_spread
            and ask_1pct_usd >= self.min_required_top_depth_usd
        )

        return OrderBookMicrostructureMetrics(
            asset_token_id=book.asset_token_id,
            timestamp_ms=book.timestamp_ms,
            best_bid=round(best_bid, 4),
            best_ask=round(best_ask, 4),
            midpoint=round(midpoint, 4),
            micro_price=round(micro_price, 4),
            spread_abs=round(spread_abs, 4),
            spread_bps=round(spread_abs * 10_000.0, 1),
            top_bid_size_shares=round(top_bid_sz, 2),
            top_ask_size_shares=round(top_ask_sz, 2),
            obi_top_level=obi_top,
            obi_5pct_band=obi_5pct,
            bid_depth_1pct_usd=round(bid_1pct_usd, 2),
            ask_depth_1pct_usd=round(ask_1pct_usd, 2),
            bid_depth_5pct_usd=round(bid_5pct_usd, 2),
            ask_depth_5pct_usd=round(ask_5pct_usd, 2),
            total_book_liquidity_usd=round(total_book_usd, 2),
            survival_5usd_buy_vwap=round(sweep_5usd.vwap_execution_price, 4),
            survival_5usd_slippage_bps=sweep_5usd.slippage_bps,
            inst_1000usd_buy_vwap=round(sweep_1000usd.vwap_execution_price, 4),
            inst_1000usd_slippage_bps=sweep_1000usd.slippage_bps,
            is_tradable_under_survival_guardrails=is_tradable,
        )

    # --------------------------------------------------------------------------
    # Historical Trajectory Volatility & Synthetic Spread Estimator
    # --------------------------------------------------------------------------
    def estimate_historical_snapshot_from_ticks(
        self,
        token_id: str,
        history: Sequence[PriceHistoryPoint],
        target_ts_sec: int,
        market_volume_usd: float,
        tick_size: float = 0.01,
    ) -> Dict[str, float]:
        """
        Given a `/prices-history` time series for a resolved market and a target
        historical cutoff `target_ts_sec` (e.g. T-24h before resolution), finds
        the strictly prior price `q_t` (`t <= target_ts_sec` to guarantee 0%
        future data leakage) and estimates the effective spread `s_t` and 24h
        realized volatility.
        """
        prior_points = [pt for pt in history if pt.timestamp_sec <= target_ts_sec]
        if not prior_points:
            return {
                "snapshot_ts_sec": float(target_ts_sec),
                "implied_prob_mid_qt": 0.50,
                "estimated_best_bid": 0.49,
                "estimated_best_ask": 0.51,
                "estimated_spread_st": 0.02,
                "momentum_24h_delta": 0.0,
                "realized_vol_24h": 0.0,
                "prior_ticks_count": 0.0,
            }

        latest_pt = prior_points[-1]
        q_t = max(0.005, min(0.995, latest_pt.price))

        # Look back 24 hours (86,400 seconds) strictly prior to latest_pt
        window_start = latest_pt.timestamp_sec - 86400
        window_pts = [pt.price for pt in prior_points if pt.timestamp_sec >= window_start]

        if len(window_pts) >= 2:
            momentum_24h = q_t - window_pts[0]
            diffs = [abs(window_pts[i] - window_pts[i - 1]) for i in range(1, len(window_pts))]
            realized_vol = math.sqrt(sum(d * d for d in diffs))
        else:
            momentum_24h = 0.0
            realized_vol = 0.01

        # Empirical spread estimator: scales inversely with log10(volume_usd) and directly with local volatility
        vol_factor = max(1.0, math.log10(max(market_volume_usd, 1000.0)) - 2.5)
        raw_spread = max(tick_size, min(0.06, (0.04 / vol_factor) + 0.15 * realized_vol))
        half_spread = raw_spread / 2.0

        est_bid = max(0.001, round(q_t - half_spread, 4))
        est_ask = min(0.999, round(q_t + half_spread, 4))
        eff_spread = round(est_ask - est_bid, 4)

        return {
            "snapshot_ts_sec": float(latest_pt.timestamp_sec),
            "implied_prob_mid_qt": round(q_t, 4),
            "estimated_best_bid": est_bid,
            "estimated_best_ask": est_ask,
            "estimated_spread_st": eff_spread,
            "momentum_24h_delta": round(momentum_24h, 4),
            "realized_vol_24h": round(realized_vol, 4),
            "prior_ticks_count": float(len(prior_points)),
        }


if __name__ == "__main__":
    # Self-test of Level-2 microstructure extraction and VWAP walk-the-book math
    extractor = SpreadDepthExtractor()
    test_book = RawClobOrderBook(
        market="0x" + "12" * 32,
        asset_id="9918273645192837465",
        timestamp=1776000000000,
        bids=[
            OrderBookLevel(price=0.58, size=400.0),
            OrderBookLevel(price=0.57, size=850.0),
            OrderBookLevel(price=0.55, size=2500.0),
        ],
        asks=[
            OrderBookLevel(price=0.60, size=300.0),
            OrderBookLevel(price=0.61, size=900.0),
            OrderBookLevel(price=0.63, size=3000.0),
        ],
    )
    metrics = extractor.extract_from_order_book(test_book)
    print(json.dumps(metrics.to_dict(), indent=2))