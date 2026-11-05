#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — REPOSITORY-WIDE AUTOMATED REGRESSION TEST SUITE
File: tests/test_repository_audit_suite.py
================================================================================
Validates syntax, JSON/YAML schemas, anti-leakage timestamp guards, tokenizer
vocabulary alignment, completion-only loss masking, strictly proper Brier &
CLOB Spread-Edge GRPO rewards, AWQ salient-channel math, and Stage 6 MCP tools.
================================================================================
"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import importlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestSovereignRepositoryAudit(unittest.TestCase):
    """End-to-end regression suite covering Stages 1 through 6."""

    def test_01_all_python_files_compile_ast_cleanly(self) -> None:
        """Verifies zero SyntaxErrors across every .py file in the repository."""
        py_files = [
            p
            for p in REPO_ROOT.rglob("*.py")
            if ".venv" not in p.parts and "sovereign_env" not in p.parts
        ]
        self.assertGreaterEqual(len(py_files), 35)
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                self.fail(f"SyntaxError in {py_file}: {exc}")

    def test_02_all_json_configs_parse_strictly(self) -> None:
        """Verifies zero invalid escape sequences in JSON schemas & manifests."""
        json_targets = [
            REPO_ROOT / "00_infra_bootstrap/amd_rocm_cluster/deepspeed_zero3_rocm.json",
            REPO_ROOT / "00_infra_bootstrap/amd_rocm_cluster/credit_ledger_state.json",
            REPO_ROOT / "02_dataset_factory/parquet_warehouse/schema_registry.json",
            REPO_ROOT / "03_llm_training/model_registry/registry_manifest.json",
        ]
        for jpath in json_targets:
            if jpath.exists():
                parsed = json.loads(jpath.read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)

    def test_03_clock_skew_and_anti_leakage_guards(self) -> None:
        """Verifies midnight-truncated news penalties and post-cutoff date redaction."""
        cs_mod = importlib.import_module(
            "02_dataset_factory.temporal_aligner.clock_skew_corrector"
        )
        corrector = cs_mod.ClockSkewCorrector()
        ok_clean, _, _, _ = corrector.verify_strict_causality(
            "2026-03-18T15:00:00Z", "2026-03-18T18:00:00Z"
        )
        ok_trunc, _, _, _ = corrector.verify_strict_causality(
            "2026-03-18T00:00:00Z", "2026-03-18T18:00:00Z"
        )
        self.assertTrue(ok_clean)
        self.assertFalse(ok_trunc)

    def test_04_tokenizer_extension_and_sft_loss_masking(self) -> None:
        """Verifies 151,744 hardware-aligned vocabulary and -100 prompt masking."""
        tok_mod = importlib.import_module(
            "03_llm_training.stage1_cpt.domain_tokenizer_extender"
        )
        extender = tok_mod.DomainTokenizerExtender()
        report = extender.run_extension_pipeline(prefer_local_fast=True)
        self.assertEqual(report.added_domain_tokens_count, 43)
        self.assertEqual(report.hardware_aligned_vocab_size % 64, 0)
        self.assertTrue(report.atomic_single_token_split_verified)

        col_mod = importlib.import_module(
            "03_llm_training.stage2_sft.structured_output_collator"
        )
        collator = col_mod.StructuredOutputCollator()
        ex = collator.collate_single_record(
            {
                "snapshot_uid": "UNIT-TEST-SFT",
                "system_prompt": "System",
                "user_prompt": "<market_meta><question>Q?</question></market_meta>",
                "assistant_response": "<reasoning>Step 1</reasoning><prob>0.6200</prob><edge>+0.0200</edge>",
                "teacher_predicted_prob_p": 0.6200,
                "teacher_predicted_edge": 0.0200,
            }
        )
        self.assertTrue(ex.xml_structure_valid)
        self.assertTrue(
            all(lbl == -100 for lbl in ex.labels[: ex.masked_prompt_tokens_count])
        )
        self.assertGreater(ex.upweighted_numeric_tokens_count, 0)

    def test_05_stage5_grpo_verifiable_rewards_and_awq_math(self) -> None:
        """Verifies Brier propriety, CLOB spread gate, XML validator, and AWQ MSE reduction."""
        brier_mod = importlib.import_module(
            "03_llm_training.stage3_grpo_brier_edge.brier_calibration_reward.brier_score_fn"
        )
        clob_mod = importlib.import_module(
            "03_llm_training.stage3_grpo_brier_edge.clob_spread_edge_reward.spread_adjusted_pnl_fn"
        )
        quant_mod = importlib.import_module(
            "03_llm_training.model_registry.quantize_awq_4bit"
        )

        brier_fn = brier_mod.BrierCalibrationRewardFunction()
        lazy = brier_fn.evaluate_single_forecast(
            "<prob>0.5000</prob>", resolved_outcome_o=1, market_midpoint_qt=0.5000
        )
        self.assertAlmostEqual(lazy.total_brier_reward, 0.0, places=5)

        clob_fn = clob_mod.CLOBSpreadAdjustedPnLRewardFunction()
        trap = clob_fn.evaluate_single_rollout(
            completion_text="<reasoning>PASS_HIGH_UMA_HAZARD</reasoning><prob>0.8000</prob><edge>+0.2000</edge>",
            resolved_outcome_o=1,
            market_midpoint_qt=0.6000,
            clob_spread_st=0.0200,
            min_required_edge_to_trade=0.0150,
            safe_for_survival_5usd_bot=False,
            uma_ambiguity_hazard_score=0.65,
        )
        self.assertEqual(trap.simulated_trade_action, "PASS_AVOIDED_UMA_TRAP")
        self.assertGreater(trap.total_clob_edge_reward, 0.0)

        quant = quant_mod.AWQAndGGUFQuantizer()
        mse_red = quant._verify_awq_salient_channel_scaling()
        self.assertGreater(mse_red, 25.0)

    def test_06_bash_orchestrators_syntax_valid(self) -> None:
        """Verifies zero Bash syntax or line-wrapping errors across all .sh scripts."""
        import subprocess

        sh_targets = [
            REPO_ROOT / "03_llm_training/model_registry/export_awq_gguf.sh",
            REPO_ROOT / "00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh",
            REPO_ROOT / "00_infra_bootstrap/gcp_gpu_cluster/run_dual_l40s_training.sh",
        ]
        for sh_path in sh_targets:
            self.assertTrue(sh_path.exists(), f"Missing script: {sh_path}")
            proc = subprocess.run(
                ["bash", "-n", str(sh_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Bash syntax error in {sh_path}: {proc.stderr}",
            )

    def test_07_cpt_and_sft_trainer_instantiation_and_gcp_runner_integrity(self) -> None:
        """Verifies CPT/SFT trainers instantiate cleanly and GCP script has all 6 stages."""
        cpt_mod = importlib.import_module("03_llm_training.stage1_cpt.run_cpt_trainer")
        sft_mod = importlib.import_module("03_llm_training.stage2_sft.run_sft_trainer")

        cpt_trainer = cpt_mod.SovereignCPTTrainer()
        self.assertTrue(hasattr(cpt_trainer, "tokenize_and_mask_cpt_document"))

        sft_trainer = sft_mod.SovereignSFTTrainer()
        self.assertTrue(hasattr(sft_trainer, "_verify_upstream_cpt_checkpoint"))

        gcp_sh = (
            REPO_ROOT / "00_infra_bootstrap/gcp_gpu_cluster/run_gcp_cuda_training.sh"
        ).read_text(encoding="utf-8")
        for required_step in (
            "stage4a_tokenizer",
            "stage4a_cpt",
            "stage4b_sft",
            "stage5_grpo",
            "stage5_evaluation",
            "stage5_quantization",
        ):
            self.assertIn(
                required_step,
                gcp_sh,
                f"Missing {required_step} in run_gcp_cuda_training.sh (truncated file)!",
            )


if __name__ == "__main__":
    unittest.main()