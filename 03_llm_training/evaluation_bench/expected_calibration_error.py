#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 5: EXPECTED CALIBRATION ERROR (ECE) BENCH
File: 03_llm_training/evaluation_bench/expected_calibration_error.py
================================================================================
Self-Contained 10-Bin Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), Murphy (1973) Brier Score Decomposition, and Vector SVG Reliability
Curve Engine on the Held-Out Out-of-Time `TEST` Split (`test_grpo_shard_00000.parquet`).

Mathematical Foundations:
  1. **Expected Calibration Error (ECE):**
     Partition all $N = 600$ out-of-time test forecasts into $M = 10$ probability
     decile bins $B_1, \dots, B_{10}$. For each bin $B_m$, let $\bar{p}_m$ be
     the mean predicted probability and $\bar{o}_m$ be the empirical fraction of
     `YES` outcomes ($o = 1$). Then:
         $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \bar{p}_m - \bar{o}_m \right|$$
     Our Stage 5 exit gate requires $\text{ECE} \le 0.045$ ($4.5\%$).
  2. **Murphy (1973) Exact Brier Score Decomposition:**
     Let $\bar{o} = \frac{1}{N} \sum_{i=1}^N o_i$ be the overall base rate on
     the `TEST` split. The Brier score decomposes into:
       - **Uncertainty ($\text{UNC}$):** $\bar{o}(1 - \bar{o})$
       - **Reliability ($\text{REL}$, lower is better):**
         $$\text{REL} = \sum_{m=1}^M \frac{|B_m|}{N} (\bar{p}_m - \bar{o}_m)^2$$
       - **Resolution ($\text{RES}$, higher is better):**
         $$\text{RES} = \sum_{m=1}^M \frac{|B_m|}{N} (\bar{o}_m - \bar{o})^2$$
  3. Exports a self-contained SVG Reliability Diagram and JSON report to:
     `03_llm_training/model_registry/checkpoints/stage3_grpo/test_calibration_report.json`
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
from typing import Any, Dict, List, Sequence, Tuple

from ..stage3_grpo_brier_edge.run_grpo_pipeline import SovereignGRPOPipelineTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [ECE-BENCHMARK] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.expected_calibration_error")


# ==============================================================================
# 1. DATA STRUCTURES FOR DECILE RELIABILITY BINS & MURPHY DECOMPOSITION
# ==============================================================================
@dataclass(frozen=True)
class CalibrationBinRecord:
    """Metrics for a single probability decile bin `[bin_low, bin_high)`."""
    bin_index: int
    bin_range_label: str
    sample_count: int
    bin_weight_fraction: float
    mean_predicted_prob: float
    empirical_yes_frequency: float
    abs_calibration_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationBenchmarkReport:
    """
    Complete Out-of-Time `TEST` Calibration Benchmark Certificate comparing
    Market Consensus (`q_t`), Stage 4B SFT, and Stage 5 GRPO.
    """
    stage: str
    timestamp_utc: str
    test_parquet_shard: str
    total_test_forecasts: int
    overall_test_yes_base_rate: float
    market_baseline_brier_score: float
    sft_policy_brier_score: float
    grpo_policy_brier_score: float
    grpo_brier_skill_score_vs_market: float
    grpo_brier_skill_score_vs_sft: float
    expected_calibration_error_ece: float
    maximum_calibration_error_mce: float
    murphy_uncertainty_component: float
    murphy_reliability_penalty_component: float
    murphy_resolution_gain_component: float
    ece_exit_gate_passed: bool
    brier_beats_market_gate_passed: bool
    svg_reliability_diagram_path: str
    decile_bins: List[CalibrationBinRecord]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["decile_bins"] = [b.to_dict() for b in self.decile_bins]
        return payload


# ==============================================================================
# 2. CORE EXPECTED CALIBRATION ERROR (ECE) & RELIABILITY CURVE ENGINE
# ==============================================================================
class ExpectedCalibrationErrorEngine:
    """
    Self-contained `03_llm_training/evaluation_bench/expected_calibration_error.py`
    engine computing 10-bin ECE, MCE, Murphy Brier decomposition, and SVG
    reliability curves on the 600 held-out `TEST` snapshots.
    """

    def __init__(
        self,
        test_grpo_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/grpo_verifiable/test_grpo_shard_00000.parquet"
        ),
        test_sft_parquet: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/final_splits/sft_reasoning/test_sft_shard_00000.parquet"
        ),
        grpo_checkpoint_dir: pathlib.Path = pathlib.Path(
            "03_llm_training/model_registry/checkpoints/stage3_grpo"
        ),
        num_bins: int = 10,
        max_allowed_ece: float = 0.045,
    ) -> None:
        self.test_grpo_parquet = test_grpo_parquet
        self.test_sft_parquet = test_sft_parquet
        self.grpo_checkpoint_dir = grpo_checkpoint_dir
        self.grpo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_bins = max(5, int(num_bins))
        self.max_allowed_ece = float(max_allowed_ece)

    def _load_learned_grpo_weights(self) -> Dict[str, float]:
        ckpt_file = self.grpo_checkpoint_dir / "grpo_policy_adapter_checkpoint.json"
        if not ckpt_file.exists():
            trainer = SovereignGRPOPipelineTrainer()
            trainer.run_grpo_training_pipeline(num_steps=20)
        raw = json.loads(ckpt_file.read_text(encoding="utf-8"))
        return dict(raw["learned_bayesian_policy_weights"])

    def _render_svg_reliability_curve(
        self,
        bins: Sequence[CalibrationBinRecord],
        ece: float,
        grpo_brier: float,
        mkt_brier: float,
    ) -> pathlib.Path:
        """
        Renders a standalone vector SVG Reliability Curve plotting Predicted
        Probability (`x-axis`) against Empirical Outcome Frequency (`y-axis`).
        """
        svg_path = self.grpo_checkpoint_dir / "test_reliability_curve.svg"
        points_str: List[str] = []
        for b in bins:
            if b.sample_count <= 0:
                continue
            cx = 60 + int(b.mean_predicted_prob * 380)
            cy = 440 - int(b.empirical_yes_frequency * 380)
            points_str.append(f"{cx},{cy}")

        polyline = " ".join(points_str)
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500" height="500">\n'
            '  <rect width="500" height="500" fill="#0f172a" />\n'
            '  <text x="250" y="30" fill="#f8fafc" font-family="monospace" font-size="13" text-anchor="middle">'
            f"Sovereign-Prob-7B Out-of-Time Reliability (ECE={ece:0.4f}, Brier={grpo_brier:0.4f} vs Mkt={mkt_brier:0.4f})"
            "</text>\n"
            '  <line x1="60" y1="440" x2="440" y2="60" stroke="#64748b" stroke-dasharray="6,6" stroke-width="2" />\n'
            f'  <polyline fill="none" stroke="#10b981" stroke-width="3" points="{polyline}" />\n'
            "</svg>\n"
        )
        svg_path.write_text(svg_content, encoding="utf-8")
        return svg_path

    def run_ece_evaluation(self) -> CalibrationBenchmarkReport:
        """
        Computes out-of-time calibration metrics across all 600 `TEST` snapshots.
        """
        import polars as pl  # type: ignore

        weights = self._load_learned_grpo_weights()
        test_rows = pl.read_parquet(self.test_grpo_parquet).to_dicts()
        sft_rows = pl.read_parquet(self.test_sft_parquet).to_dicts()
        sft_by_uid = {str(r["snapshot_uid"]): r for r in sft_rows}

        trainer = SovereignGRPOPipelineTrainer()

        predictions: List[Tuple[float, float, float, int]] = []
        for row in test_rows:
            uid = str(row["snapshot_uid"])
            o_val = int(row["resolved_binary_outcome"])
            q_t = float(row["implied_prob_mid_qt"])
            p_sft = float(
                sft_by_uid.get(uid, {}).get("teacher_predicted_prob_p", q_t)
            )
            feats = trainer.extract_prompt_features(row)
            p_grpo = trainer.predict_calibrated_probability(feats, weights)
            predictions.append((p_grpo, p_sft, q_t, o_val))

        n_total = max(1, len(predictions))
        overall_o_bar = sum(x[3] for x in predictions) / float(n_total)

        mkt_brier = round(
            sum((x[2] - float(x[3])) ** 2 for x in predictions) / float(n_total),
            6,
        )
        sft_brier = round(
            sum((x[1] - float(x[3])) ** 2 for x in predictions) / float(n_total),
            6,
        )
        grpo_brier = round(
            sum((x[0] - float(x[3])) ** 2 for x in predictions) / float(n_total),
            6,
        )

        bss_vs_mkt = round(1.0 - (grpo_brier / max(1e-6, mkt_brier)), 4)
        bss_vs_sft = round(1.0 - (grpo_brier / max(1e-6, sft_brier)), 4)

        bin_records: List[CalibrationBinRecord] = []
        ece_sum = 0.0
        mce_val = 0.0
        rel_sum = 0.0
        res_sum = 0.0

        for b_idx in range(self.num_bins):
            low = b_idx / float(self.num_bins)
            high = (b_idx + 1) / float(self.num_bins)
            in_bin = [
                (p_g, o_v)
                for (p_g, _, _, o_v) in predictions
                if (low <= p_g < high) or (b_idx == self.num_bins - 1 and p_g == high)
            ]
            cnt = len(in_bin)
            w_frac = cnt / float(n_total)

            if cnt > 0:
                mean_p = sum(x[0] for x in in_bin) / float(cnt)
                emp_o = sum(x[1] for x in in_bin) / float(cnt)
                gap = abs(mean_p - emp_o)
                ece_sum += w_frac * gap
                mce_val = max(mce_val, gap)
                rel_sum += w_frac * ((mean_p - emp_o) ** 2)
                res_sum += w_frac * ((emp_o - overall_o_bar) ** 2)
            else:
                mean_p = round((low + high) / 2.0, 4)
                emp_o = 0.0
                gap = 0.0

            bin_records.append(
                CalibrationBinRecord(
                    bin_index=b_idx + 1,
                    bin_range_label=f"[{low:0.1f}, {high:0.1f})",
                    sample_count=cnt,
                    bin_weight_fraction=round(w_frac, 4),
                    mean_predicted_prob=round(mean_p, 4),
                    empirical_yes_frequency=round(emp_o, 4),
                    abs_calibration_gap=round(gap, 4),
                )
            )

        ece_clean = round(ece_sum, 5)
        mce_clean = round(mce_val, 5)
        unc_comp = round(overall_o_bar * (1.0 - overall_o_bar), 6)

        svg_path = self._render_svg_reliability_curve(
            bin_records, ece_clean, grpo_brier, mkt_brier
        )

        ece_ok = ece_clean <= self.max_allowed_ece
        brier_ok = grpo_brier < mkt_brier

        report = CalibrationBenchmarkReport(
            stage="STAGE_5_EXPECTED_CALIBRATION_ERROR_BENCHMARK",
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            test_parquet_shard=str(self.test_grpo_parquet),
            total_test_forecasts=len(predictions),
            overall_test_yes_base_rate=round(overall_o_bar, 4),
            market_baseline_brier_score=mkt_brier,
            sft_policy_brier_score=sft_brier,
            grpo_policy_brier_score=grpo_brier,
            grpo_brier_skill_score_vs_market=bss_vs_mkt,
            grpo_brier_skill_score_vs_sft=bss_vs_sft,
            expected_calibration_error_ece=ece_clean,
            maximum_calibration_error_mce=mce_clean,
            murphy_uncertainty_component=unc_comp,
            murphy_reliability_penalty_component=round(rel_sum, 6),
            murphy_resolution_gain_component=round(res_sum, 6),
            ece_exit_gate_passed=ece_ok,
            brier_beats_market_gate_passed=brier_ok,
            svg_reliability_diagram_path=str(svg_path),
            decile_bins=bin_records,
        )

        out_json = self.grpo_checkpoint_dir / "test_calibration_report.json"
        out_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Out-of-Time TEST ECE: %.4f (Gate <= %.3f) | GRPO Brier: %.5f vs Market: %.5f (BSS: %+.2f%%)",
            report.expected_calibration_error_ece,
            self.max_allowed_ece,
            report.grpo_policy_brier_score,
            report.market_baseline_brier_score,
            report.grpo_brier_skill_score_vs_market * 100.0,
        )
        return report


def main() -> int:
    engine = ExpectedCalibrationErrorEngine()
    report = engine.run_ece_evaluation()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())