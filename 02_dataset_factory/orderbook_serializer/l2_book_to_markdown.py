#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART C)
File: 02_dataset_factory/orderbook_serializer/l2_book_to_markdown.py
================================================================================
Level-2 Central Limit Order Book (CLOB) to Structured Markdown/XML Serializer.

Why Numeric Order Books Must Be Serialized into Structured Tokens for LLMs:
  1. Off-the-shelf language models only see a single scalar probability (e.g.
     "Market odds: 62%"). They have zero visibility into whether that 62%
     midpoint is backed by $85,000 of tight 1-cent institutional depth or a
     hollow 9-cent wide bid-ask spread that will destroy a taker order.
  2. During Stage 5 GRPO training (`clob_spread_edge_reward`), our reward
     function penalizes the model if it triggers a trade when its edge
     `|p - q_t|` is smaller than the effective half-spread plus slippage `s_t`.
     Therefore, the prompt itself must expose the exact L2 bid/ask ladder,
     Order Book Imbalance (OBI), and VWAP sweep cost inside a deterministic
     `<book>` block.
  3. This module reconstructs a canonical 5-level Bid/Ask ladder from each
     leak-certified snapshot's microstructure parameters (`implied_prob_mid_qt`,
     `estimated_best_bid`, `estimated_best_ask`, `estimated_spread_st`,
     `market_volume_usd`, `momentum_24h_delta`) and formats it into compact
     Markdown/XML tables with zero future information leakage.
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import hashlib
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [L2-SERIALIZER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.l2_book_to_markdown")


# ==============================================================================
# 1. DATA STRUCTURES FOR SERIALIZED LEVEL-2 ORDER BOOKS
# ==============================================================================
@dataclass(frozen=True)
class L2LadderLevel:
    """
    Represents a single resting price level (`ASK` or `BID`) in the 5-level
    serialized CLOB depth table.
    """
    side: str
    level_index: int
    price_usd: float
    size_shares: float
    notional_usd: float
    cumulative_notional_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SerializedOrderBookBlock:
    """
    Complete serialized `<book>` artifact for a single `(market_id, horizon_h)`
    snapshot, ready for injection into Stage 4 CPT/SFT and Stage 5 GRPO prompts.
    """
    snapshot_uid: str
    market_id: str
    horizon_hours_before_close: int
    midpoint_qt: float
    best_bid_price: float
    best_ask_price: float
    spread_abs_st: float
    spread_bps: float
    micro_price: float
    order_book_imbalance: float
    liquidity_regime_token: str
    bid_depth_5lvl_usd: float
    ask_depth_5lvl_usd: float
    survival_5usd_vwap_ask: float
    inst_1000usd_vwap_ask: float
    min_required_edge_to_trade: float
    ask_levels: List[L2LadderLevel]
    bid_levels: List[L2LadderLevel]
    cpt_book_xml_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ask_levels"] = [a.to_dict() for a in self.ask_levels]
        payload["bid_levels"] = [b.to_dict() for b in self.bid_levels]
        return payload


# ==============================================================================
# 2. CORE LEVEL-2 ORDER BOOK TO MARKDOWN/XML SERIALIZER ENGINE
# ==============================================================================
class L2BookToMarkdownSerializer:
    """
    Constructs a 5-level Level-2 CLOB depth table from aligned snapshot
    microstructure features and serializes it into a compact, token-efficient
    `<book>` block for Qwen-2.5-7B.
    """

    def __init__(
        self,
        num_levels: int = 5,
        tick_size: float = 0.01,
        fee_buffer_abs: float = 0.005,
    ) -> None:
        self.num_levels = max(3, int(num_levels))
        self.tick_size = float(tick_size)
        self.fee_buffer_abs = float(fee_buffer_abs)

    @staticmethod
    def _simulate_vwap_on_ladder(
        levels: Sequence[L2LadderLevel], target_usd: float
    ) -> float:
        """
        Walks a 5-level ladder to compute the exact Volume-Weighted Average
        Price (VWAP) for a `target_usd` sweep order.
        """
        if not levels:
            return 0.50
        rem_usd = float(target_usd)
        spent_usd = 0.0
        bought_shares = 0.0

        for lvl in levels:
            if rem_usd <= 1e-6:
                break
            take_usd = min(rem_usd, lvl.notional_usd)
            take_shares = take_usd / max(0.001, lvl.price_usd)
            spent_usd += take_usd
            bought_shares += take_shares
            rem_usd -= take_usd

        if bought_shares <= 1e-6:
            return levels[0].price_usd
        return round(spent_usd / bought_shares, 4)

    def _build_5level_ladder(
        self,
        snapshot_uid: str,
        best_bid: float,
        best_ask: float,
        market_volume_usd: float,
        momentum_24h: float,
    ) -> Tuple[List[L2LadderLevel], List[L2LadderLevel], float]:
        """
        Synthesizes a realistic, deterministic 5-level Bid/Ask depth profile
        anchored to the market's cumulative USD volume, best bid/ask prices,
        and 24h order-flow momentum skew.
        """
        digest = hashlib.sha256(snapshot_uid.encode("utf-8")).digest()

        # Base top-of-book depth scales sub-linearly with market volume
        base_top_usd = max(
            45.0,
            min(25000.0, math.sqrt(max(1000.0, market_volume_usd)) * 0.85),
        )

        # Positive 24h momentum skews resting liquidity toward thicker bid support
        imbalance_skew = max(-0.45, min(0.45, momentum_24h * 2.2))
        bid_mult = 1.0 + imbalance_skew
        ask_mult = 1.0 - imbalance_skew

        ask_levels: List[L2LadderLevel] = []
        bid_levels: List[L2LadderLevel] = []

        cum_ask_usd = 0.0
        cum_bid_usd = 0.0

        for idx in range(self.num_levels):
            # Depth grows geometrically away from the midpoint as limit makers stack ticks
            level_growth = 1.0 + (idx * 0.65) + ((digest[idx] / 255.0) * 0.25)

            ask_p = min(0.999, round(best_ask + (idx * self.tick_size), 4))
            ask_usd = round(base_top_usd * ask_mult * level_growth, 2)
            ask_sz = round(ask_usd / max(0.001, ask_p), 1)
            cum_ask_usd = round(cum_ask_usd + ask_usd, 2)

            ask_levels.append(
                L2LadderLevel(
                    side="ASK",
                    level_index=idx + 1,
                    price_usd=ask_p,
                    size_shares=ask_sz,
                    notional_usd=ask_usd,
                    cumulative_notional_usd=cum_ask_usd,
                )
            )

            bid_p = max(0.001, round(best_bid - (idx * self.tick_size), 4))
            bid_usd = round(base_top_usd * bid_mult * level_growth, 2)
            bid_sz = round(bid_usd / max(0.001, bid_p), 1)
            cum_bid_usd = round(cum_bid_usd + bid_usd, 2)

            bid_levels.append(
                L2LadderLevel(
                    side="BID",
                    level_index=idx + 1,
                    price_usd=bid_p,
                    size_shares=bid_sz,
                    notional_usd=bid_usd,
                    cumulative_notional_usd=cum_bid_usd,
                )
            )

        total_5lvl = cum_bid_usd + cum_ask_usd
        obi = (
            round((cum_bid_usd - cum_ask_usd) / total_5lvl, 4)
            if total_5lvl > 0.0
            else 0.0
        )
        return ask_levels, bid_levels, obi

    def serialize_snapshot_book(
        self, row: Dict[str, Any]
    ) -> SerializedOrderBookBlock:
        """
        Converts a single leak-certified snapshot dictionary into a complete
        `SerializedOrderBookBlock` containing the `<book>` Markdown/XML table.
        """
        snap_uid = str(row.get("snapshot_uid") or "PIT-UNKNOWN")
        mkt_id = str(row.get("market_id") or "0")
        horizon_h = int(row.get("horizon_hours_before_close") or 24)
        q_t = float(row.get("implied_prob_mid_qt") or 0.50)
        best_bid = float(row.get("estimated_best_bid") or max(0.01, q_t - 0.01))
        best_ask = float(row.get("estimated_best_ask") or min(0.99, q_t + 0.01))
        spread_st = max(0.001, round(best_ask - best_bid, 4))
        spread_bps = round(spread_st * 10000.0, 1)
        vol_usd = float(row.get("market_volume_usd") or 25000.0)
        mom_24h = float(row.get("momentum_24h_delta") or 0.0)
        era = str(row.get("collateral_era") or "APRIL_2026_PUSD_V2")

        ask_lvls, bid_lvls, obi = self._build_5level_ladder(
            snapshot_uid=snap_uid,
            best_bid=best_bid,
            best_ask=best_ask,
            market_volume_usd=vol_usd,
            momentum_24h=mom_24h,
        )

        # Micro-price weighted by top-of-book opposing queue sizes
        top_bid_sz = bid_lvls[0].size_shares
        top_ask_sz = ask_lvls[0].size_shares
        top_sum = top_bid_sz + top_ask_sz
        micro_p = (
            round((best_bid * top_ask_sz + best_ask * top_bid_sz) / top_sum, 4)
            if top_sum > 0
            else q_t
        )

        bid_5lvl_usd = bid_lvls[-1].cumulative_notional_usd
        ask_5lvl_usd = ask_lvls[-1].cumulative_notional_usd
        total_depth = bid_5lvl_usd + ask_5lvl_usd

        if total_depth >= 50000.0 and spread_st <= 0.015:
            liq_token = "LIQ_DEEP_INSTITUTIONAL"
        elif total_depth >= 5000.0 and spread_st <= 0.030:
            liq_token = "LIQ_STANDARD_TRADABLE"
        else:
            liq_token = "LIQ_THIN_WIDE_SPREAD"

        vwap_5usd = self._simulate_vwap_on_ladder(ask_lvls, target_usd=5.0)
        vwap_1000usd = self._simulate_vwap_on_ladder(ask_lvls, target_usd=1000.0)

        # Minimum absolute probability edge |p - q_t| required to overcome half-spread + fee buffer
        min_edge = round((spread_st / 2.0) + self.fee_buffer_abs, 4)

        # Format the 5-level Markdown/XML table (Asks descending from Ask-3 to Ask-1, then Bids)
        ask_rows_md: List[str] = []
        for lvl in reversed(ask_lvls[:3]):
            ask_rows_md.append(
                f"  | ASK-{lvl.level_index} | ${lvl.price_usd:0.4f} | "
                f"{lvl.size_shares:9.1f} | ${lvl.notional_usd:9.2f} | ${lvl.cumulative_notional_usd:10.2f} |"
            )

        bid_rows_md: List[str] = []
        for lvl in bid_lvls[:3]:
            bid_rows_md.append(
                f"  | BID-{lvl.level_index} | ${lvl.price_usd:0.4f} | "
                f"{lvl.size_shares:9.1f} | ${lvl.notional_usd:9.2f} | ${lvl.cumulative_notional_usd:10.2f} |"
            )

        book_xml = (
            f"<book snapshot_uid=\"{snap_uid}\" era=\"{era}\" regime=\"{liq_token}\">\n"
            f"  Midpoint q_t: {q_t:0.4f} ({q_t * 100.0:0.2f}%) | "
            f"Micro-Price: {micro_p:0.4f} | "
            f"Spread s_t: {spread_st:0.4f} ({spread_bps:0.0f} bps) | "
            f"OBI: {obi:+0.3f}\n"
            f"  Min Edge Hurdle |p - q_t| >= {min_edge:0.4f} | "
            f"$5 Survival Buy VWAP: ${vwap_5usd:0.4f} | "
            f"$1k Inst Buy VWAP: ${vwap_1000usd:0.4f}\n"
            f"  | Level |  Price  |   Shares  |  Level USD | Cumul. USD  |\n"
            f"  |-------|---------|-----------|------------|-------------|\n"
            + "\n".join(ask_rows_md)
            + f"\n  |--- MIDPOINT q_t = ${q_t:0.4f} (Spread = ${spread_st:0.4f}) ---|\n"
            + "\n".join(bid_rows_md)
            + "\n</book>"
        )

        return SerializedOrderBookBlock(
            snapshot_uid=snap_uid,
            market_id=mkt_id,
            horizon_hours_before_close=horizon_h,
            midpoint_qt=q_t,
            best_bid_price=best_bid,
            best_ask_price=best_ask,
            spread_abs_st=spread_st,
            spread_bps=spread_bps,
            micro_price=micro_p,
            order_book_imbalance=obi,
            liquidity_regime_token=liq_token,
            bid_depth_5lvl_usd=bid_5lvl_usd,
            ask_depth_5lvl_usd=ask_5lvl_usd,
            survival_5usd_vwap_ask=vwap_5usd,
            inst_1000usd_vwap_ask=vwap_1000usd,
            min_required_edge_to_trade=min_edge,
            ask_levels=ask_lvls,
            bid_levels=bid_lvls,
            cpt_book_xml_markdown=book_xml,
        )


# ==============================================================================
# 3. CLI & SELF-TEST ENTRYPOINT
# ==============================================================================
def main() -> int:
    import polars as pl  # type: ignore

    input_path = pathlib.Path(
        "data/processed_warehouse/leak_audited/leak_certified_snapshots_shard_00000.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing leak-certified shard at {input_path}")

    rows = pl.read_parquet(input_path).to_dicts()
    serializer = L2BookToMarkdownSerializer()
    serialized_books = [serializer.serialize_snapshot_book(r) for r in rows]

    sample_book = serialized_books[0] if serialized_books else None
    report = {
        "stage": "STAGE_3_PART_C_L2_BOOK_SERIALIZER",
        "total_books_serialized": len(serialized_books),
        "sample_snapshot_uid": sample_book.snapshot_uid if sample_book else None,
        "sample_liquidity_regime": sample_book.liquidity_regime_token if sample_book else None,
        "sample_min_edge_hurdle": sample_book.min_required_edge_to_trade if sample_book else None,
        "sample_book_xml_preview": sample_book.cpt_book_xml_markdown if sample_book else "",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())