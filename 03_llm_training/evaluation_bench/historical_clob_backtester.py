#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: HISTORICAL CLOB OUT-OF-SAMPLE BACKTESTER
File: 03_llm_training/evaluation_bench/historical_clob_backtester.py
================================================================================
Self-Contained Out-of-Sample Walk-Forward Backtester simulating the $5.00
"Earn-or-Die" Survival Trading Bot with 0.25x Fractional Kelly Position Sizing
on the Held-Out Out-of-Time `TEST` Split (`test_grpo_shard_00000.parquet`).

How This Backtester Enforces Realistic Polymarket CLOB v2 Execution Physics:
  1. **Zero Hindsight Leakage:** Evaluates strictly on `test_grpo_shard_00000.parquet`
     (the most recent 15% of markets chronologically disjoint from `TRAIN` & `VAL`).
  2. **UMA Ambiguity Trap Guardrail:** Automatically passes (`PASS_UMA_TRAP`) on
     any contract where `safe_for_survival_5usd_bot == False` or
     `uma_ambiguity_hazard_score > 0.45`.
  3. **Spread + Fee Hurdle Gate:** Only triggers an order when the model's net
     probability edge exceeds the contract's half-spread plus fee buffer.
  4. **Dampened Fractional Kelly Sizing ($c = 0.25$, max $15\%$ per trade):**
     - For **BUY YES** at effective entry cost $c_{\text{YES}}$:
         $$f^*_{\text{YES}} = 0.25 \cdot \left(\frac{p - c_{\text{YES}}}{1.0 - c_{\text{YES}}}\right)$$
     - For **BUY NO** at effective entry cost $c_{\text{NO}}$:
         $$f^*_{\text{NO}} = 0.25 \cdot \left(\frac{(1 - p) - c_{\text{NO}}}{1.0 - c_{\text{NO}}}\right)$$
     - Clamped to `max_single_market_fraction = 0.15` ($15\%$ of current liquid
       bankroll) so a single loss can never trigger the Reaper Watchdog!
================================================================================
"""

from __future__ import annotations

import sys
import json
import math
import logging
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from ..stage3_grpo_brier_edge.run_grpo_pipeline import SovereignGRPOPipelineTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [HIST-CLOB-BACKTEST] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.historical_clob_backtester")


# ==============================================================================
# 1. BACKTEST TRADE RECORD & OUT-OF-TIME PERFORMANCE REPORT SCHEMAS
# ==============================================================================
@dataclass(frozen=True)
class BacktestTradeExecution:
    """Detailed execution telemetry for a single simulated Out-of-Time CLOB trade."""
    snapshot_uid: str
    market_id: str
    canonical_category: str
    side: str
    predicted_prob_p: float
    market_midpoint_qt: float
    effective_entry_price: float
    kelly_fraction_allocated: float
    stake_usd: float
    resolved_outcome_o: int
    won_trade: bool
    realized_pnl_usd: float
    bankroll_after_trade_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutOfTimeBacktestReport:
    """
    Comprehensive Out-of-Time Walk-Forward Backtest Certificate verifying Stage 5
    exit gate criteria (`Sharpe >= 1.50`, `Max Drawdown < 25%`, `Net NAV > $5.00`).
    """
    stage: str
    timestamp_utc: str
    test_parquet_shard: str
    total_test_snapshots_scanned: int
    uma_traps_filtered_count: int
    inside_spread_holds_count: int
    executed_trades_count: int
    winning_trades_count: int
    losing_trades_count: int
    win_rate_pct: float
    initial_survival_bankroll_usd: float
    final_ending_bankroll_usd: float
    total_net_profit_usd: float
    cumulative_return_pct: float
    annualized_sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    reaper_death_floor_breached: bool
    mitosis_50usd_clone_threshold_reached: bool
    sharpe_exit_gate_passed: bool
    sample_trades: List[BacktestTradeExecution]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sample_trades"] = [t.to_dict() for t in self.sample_trades]
        return payload


# ==============================================================================
# 2. SELF-CONTAINED HISTORICAL CLOB WALK-FORWARD BACKTESTER ENGINE
# ==============================================================================
class HistoricalCLOBBacktester:
    """
    Runs a chronological walk-forward simulation of the $5.00 survival bankroll
    across `test_grpo_shard_00000.parquet` using 0.25x Fractional Kelly sizing.
    """

    def __init__(
        self,
        test_grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet"
        ),
        grpo_checkpoint_dir: pathlib.Path = pathlib.Path(
            "03_llm_training/model_registry/checkpoints/stage3_grpo"
        ),
        initial_bankroll_usd: float = 5.00,
        kelly_dampener_c: float = 0.25,
        max_single_trade_fraction: float = 0.15,
        maker_spread_capture_factor: float = 0.35,
    ) -> None:
        self.test_grpo_parquet = test_grpo_parquet
        self.grpo_checkpoint_dir = grpo_checkpoint_dir
        self.grpo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initial_bankroll_usd = float(initial_bankroll_usd)
        self.kelly_dampener_c = float(kelly_dampener_c)
        self.max_single_trade_fraction = float(max_single_trade_fraction)
        self.maker_spread_capture_factor = float(maker_spread_capture_factor)

    def _load_learned_grpo_weights(self) -> Dict[str, float]:
        ckpt_file = self.grpo_checkpoint_dir / "grpo_policy_adapter_checkpoint.json"
        if not ckpt_file.exists():
            SovereignGRPOPipelineTrainer().run_grpo_training_pipeline(num_steps=20)
        raw = json.loads(ckpt_file.read_text(encoding="utf-8"))
        return dict(raw["learned_bayesian_policy_weights"])

    def run_historical_replay(self) -> OutOfTimeBacktestReport:
        """
        Executes the chronological walk-forward backtest on the `TEST` split.
        """
        import polars as pl  # type: ignore

        weights = self._load_learned_grpo_weights()
        test_rows = pl.read_parquet(self.test_grpo_parquet).to_dicts()
        trainer = SovereignGRPOPipelineTrainer()

        bankroll = self.initial_bankroll_usd
        peak_bankroll = bankroll
        max_drawdown_pct = 0.0
        reaper_breached = False

        uma_traps_skipped = 0
        inside_spread_holds = 0
        trades: List[BacktestTradeExecution] = []
        trade_returns_pct: List[float] = []
        traded_market_ids: set[str] = set()

        for row in test_rows:
            mkt_id = str(row["market_id"])
            feats = trainer.extract_prompt_features(row)
            if feats["safe_flag"] < 0.5 or feats["uma_hazard"] > 0.45:
                uma_traps_skipped += 1
                continue

            # Institutional risk rule: at most 1 position per unique market_id
            # (prevents quadruple-compounding across T-168h, T-72h, T-24h, T-6h)
            if mkt_id in traded_market_ids:
                inside_spread_holds += 1
                continue

            p_pred = trainer.predict_calibrated_probability(feats, weights)
            q_t = feats["q_t"]
            s_t = feats["spread_st"]
            min_edge = feats["min_edge"]
            o_val = int(row["resolved_binary_outcome"])

            signed_edge = p_pred - q_t
            if abs(signed_edge) < min_edge:
                inside_spread_holds += 1
                continue

            # Resting POST_ONLY limit maker execution captures part of the half-spread
            effective_slip = (s_t / 2.0) * self.maker_spread_capture_factor

            if signed_edge >= min_edge:
                side = "MAKER_LIMIT_BUY_YES"
                entry_price = min(0.990, round(q_t + effective_slip, 4))
                net_prob_edge = p_pred - entry_price
                if net_prob_edge <= 0.005:
                    inside_spread_holds += 1
                    continue
                raw_kelly = net_prob_edge / max(0.01, 1.0 - entry_price)
                won = o_val == 1
            else:
                side = "MAKER_LIMIT_BUY_NO"
                p_no = 1.0 - p_pred
                q_no = 1.0 - q_t
                entry_price = min(0.990, round(q_no + effective_slip, 4))
                net_prob_edge = p_no - entry_price
                if net_prob_edge <= 0.005:
                    inside_spread_holds += 1
                    continue
                raw_kelly = net_prob_edge / max(0.01, 1.0 - entry_price)
                won = o_val == 0

            traded_market_ids.add(mkt_id)
            kelly_frac = min(
                self.max_single_trade_fraction,
                max(0.01, self.kelly_dampener_c * raw_kelly),
            )
            stake_usd = round(bankroll * kelly_frac, 4)
            shares = stake_usd / entry_price

            payout_usd = shares * 1.0 if won else 0.0
            pnl_usd = round(payout_usd - stake_usd, 4)
            ret_pct = pnl_usd / max(1e-6, bankroll)
            trade_returns_pct.append(ret_pct)

            bankroll = round(max(0.0, bankroll + pnl_usd), 4)
            if bankroll <= 0.0:
                reaper_breached = True
                break

            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            dd = ((peak_bankroll - bankroll) / max(1e-6, peak_bankroll)) * 100.0
            if dd > max_drawdown_pct:
                max_drawdown_pct = round(dd, 2)

            trades.append(
                BacktestTradeExecution(
                    snapshot_uid=str(row["snapshot_uid"]),
                    market_id=str(row["market_id"]),
                    canonical_category=str(row["canonical_category"]),
                    side=side,
                    predicted_prob_p=p_pred,
                    market_midpoint_qt=q_t,
                    effective_entry_price=entry_price,
                    kelly_fraction_allocated=round(kelly_frac, 4),
                    stake_usd=stake_usd,
                    resolved_outcome_o=o_val,
                    won_trade=won,
                    realized_pnl_usd=pnl_usd,
                    bankroll_after_trade_usd=bankroll,
                )
            )

        n_trades = len(trades)
        wins = sum(1 for t in trades if t.won_trade)
        losses = n_trades - wins
        win_rate = round((wins / max(1, n_trades)) * 100.0, 2)
        net_profit = round(bankroll - self.initial_bankroll_usd, 4)
        cum_ret_pct = round((net_profit / self.initial_bankroll_usd) * 100.0, 2)

        if len(trade_returns_pct) >= 2:
            mean_r = sum(trade_returns_pct) / len(trade_returns_pct)
            var_r = sum((r - mean_r) ** 2 for r in trade_returns_pct) / len(
                trade_returns_pct
            )
            std_r = max(1e-6, math.sqrt(var_r))
            downside_sq = [min(0.0, r) ** 2 for r in trade_returns_pct]
            down_std = max(1e-6, math.sqrt(sum(downside_sq) / len(downside_sq)))

            ann_factor = math.sqrt(min(252, max(36, len(trade_returns_pct))))
            sharpe = round((mean_r / std_r) * ann_factor, 3)
            sortino = round((mean_r / down_std) * ann_factor, 3)
        else:
            sharpe = 0.0
            sortino = 0.0

        sharpe_gate_ok = (sharpe >= 1.50) and (not reaper_breached) and (net_profit > 0.0)

        report = OutOfTimeBacktestReport(
            stage="STAGE_5_HISTORICAL_CLOB_BACKTESTER",
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            test_parquet_shard=str(self.test_grpo_parquet),
            total_test_snapshots_scanned=len(test_rows),
            uma_traps_filtered_count=uma_traps_skipped,
            inside_spread_holds_count=inside_spread_holds,
            executed_trades_count=n_trades,
            winning_trades_count=wins,
            losing_trades_count=losses,
            win_rate_pct=win_rate,
            initial_survival_bankroll_usd=self.initial_bankroll_usd,
            final_ending_bankroll_usd=bankroll,
            total_net_profit_usd=net_profit,
            cumulative_return_pct=cum_ret_pct,
            annualized_sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown_pct,
            reaper_death_floor_breached=reaper_breached,
            mitosis_50usd_clone_threshold_reached=(bankroll >= 50.0),
            sharpe_exit_gate_passed=sharpe_gate_ok,
            sample_trades=trades[:5],
        )

        out_json = self.grpo_checkpoint_dir / "out_of_time_backtest_report.json"
        out_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Out-of-Time CLOB Replay | Trades: %d | WinRate: %.1f%% | NAV: $%.2f -> $%.2f | Sharpe: %.2f",
            report.executed_trades_count,
            report.win_rate_pct,
            report.initial_survival_bankroll_usd,
            report.final_ending_bankroll_usd,
            report.annualized_sharpe_ratio,
        )
        return report


def main() -> int:
    backtester = HistoricalCLOBBacktester()
    report = backtester.run_historical_replay()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())