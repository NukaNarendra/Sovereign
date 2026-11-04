#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART D)
File: 02_dataset_factory/synthetic_sft_generator/merge_gateway_client.py
================================================================================
Multi-Model Teacher LLM Router & Quota Governor for the $10.00/Month Free Merge
Gateway Student Perk (`MERGE_GATEWAY_API_KEY`).

Why Stage 4B SFT Needs a Blind Teacher Before Stage 5 GRPO:
  1. If you throw a raw base model (Qwen-2.5-7B) directly into Stage 5 GRPO
     reinforcement learning, it wastes thousands of GPU rollouts discovering
     how to format `<reasoning>...</reasoning><prob>0.XXXX</prob><edge>+0.XXXX</edge>`
     and how to read our `<book>` and `<uma_rule>` XML tables.
  2. By first running Supervised Fine-Tuning (Stage 4B SFT) on structured
     Superforecaster Chain-of-Thought traces, the model enters Stage 5 GRPO
     already knowing how to parse L2 order books, cite UMA resolution clauses,
     and output exact 4-decimal probabilities.
  3. This client manages the $10.00/month Merge Gateway credit pool with:
       - A hard $9.50 monthly spend ceiling ($0.50 safety buffer).
       - Content-addressable SHA-256 disk caching so identical prompts never
         consume duplicate teacher tokens.
       - Seamless, zero-cost fallback to our deterministic Bayesian
         Superforecaster synthesizer when running offline or conserving credits.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import hashlib
import logging
import pathlib
import argparse
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [MERGE-GATEWAY] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.merge_gateway_client")


# ==============================================================================
# 1. QUOTA LEDGER & TEACHER COMPLETION DATA STRUCTURES
# ==============================================================================
@dataclass
class MergeGatewayQuotaState:
    """
    Tracks monthly token consumption and USD spend against the $10.00/month
    free Merge Gateway student developer credit pool.
    """
    provider_name: str = "Merge Gateway Multi-LLM Teacher Router ($10/mo Student Perk)"
    monthly_credit_limit_usd: float = 10.00
    safety_buffer_usd: float = 0.50
    blended_cost_per_1k_tokens_usd: float = 0.0015
    current_month_key: str = ""
    monthly_requests_count: int = 0
    monthly_prompt_tokens: int = 0
    monthly_completion_tokens: int = 0
    monthly_usd_consumed: float = 0.0
    total_cache_hits_saved: int = 0
    last_updated_utc: str = ""

    @property
    def remaining_credit_usd(self) -> float:
        return max(0.0, round(self.monthly_credit_limit_usd - self.monthly_usd_consumed, 4))

    @property
    def usable_credit_before_buffer_usd(self) -> float:
        max_spend = self.monthly_credit_limit_usd - self.safety_buffer_usd
        return max(0.0, round(max_spend - self.monthly_usd_consumed, 4))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["remaining_credit_usd"] = self.remaining_credit_usd
        payload["usable_credit_before_buffer_usd"] = self.usable_credit_before_buffer_usd
        return payload


@dataclass(frozen=True)
class TeacherCompletionResult:
    """
    Standardized response returned by `MergeGatewayTeacherClient` for a single
    market snapshot prompt.
    """
    snapshot_uid: str
    teacher_model_id: str
    generation_source: str
    completion_text: str
    prompt_tokens: int
    completion_tokens: int
    incurred_cost_usd: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CORE MERGE GATEWAY TEACHER CLIENT & QUOTA GOVERNOR
# ==============================================================================
class MergeGatewayTeacherClient:
    """
    Routes teacher SFT generation requests through Merge Gateway (`MERGE_GATEWAY_API_KEY`)
    while enforcing the $10/month student credit cap and caching responses on disk.
    """

    DEFAULT_GATEWAY_URL: str = "https://api.merge.dev/llm-gateway/v1/chat/completions"
    DEFAULT_TEACHER_MODEL: str = "deepseek-r1-distill-qwen-32b"

    def __init__(
        self,
        ledger_path: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/sft_teacher_traces/merge_gateway_quota_ledger.json"
        ),
        cache_dir: pathlib.Path = pathlib.Path(
            "data/processed_warehouse/sft_teacher_traces/teacher_cache"
        ),
        api_key: Optional[str] = None,
        gateway_url: Optional[str] = None,
        teacher_model: Optional[str] = None,
    ) -> None:
        self.ledger_path = ledger_path
        self.cache_dir = cache_dir
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = (
            api_key
            or os.environ.get("MERGE_GATEWAY_API_KEY")
            or ""
        ).strip()
        self.gateway_url = gateway_url or os.environ.get(
            "MERGE_GATEWAY_URL", self.DEFAULT_GATEWAY_URL
        )
        self.teacher_model = teacher_model or self.DEFAULT_TEACHER_MODEL
        self.state = self._load_or_init_ledger()
        self._roll_month_if_needed()

    @property
    def has_live_merge_key(self) -> bool:
        """Returns True if a real non-placeholder Merge Gateway key is configured."""
        return (
            len(self.api_key) >= 12
            and "placeholder" not in self.api_key.lower()
            and "verify-only" not in self.api_key.lower()
            and not self.api_key.startswith("ENC[")
        )

    def _load_or_init_ledger(self) -> MergeGatewayQuotaState:
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        month_key = now_dt.strftime("%Y-%m")
        if self.ledger_path.exists():
            try:
                raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                filtered = {
                    k: v
                    for k, v in raw.items()
                    if k in MergeGatewayQuotaState.__dataclass_fields__
                }
                return MergeGatewayQuotaState(**filtered)
            except Exception as exc:
                logger.warning("Corrupt Merge Gateway ledger (%s); re-initializing.", exc)

        state = MergeGatewayQuotaState(
            current_month_key=month_key,
            last_updated_utc=now_dt.isoformat(),
        )
        self._save_ledger(state)
        return state

    def _save_ledger(self, state: Optional[MergeGatewayQuotaState] = None) -> None:
        target = state or self.state
        target.last_updated_utc = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        tmp_path = self.ledger_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(target.to_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(self.ledger_path)

    def _roll_month_if_needed(self) -> None:
        now_month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
        if self.state.current_month_key != now_month:
            logger.info(
                "New UTC billing cycle (%s -> %s): resetting Merge Gateway $10/mo counter.",
                self.state.current_month_key,
                now_month,
            )
            self.state.current_month_key = now_month
            self.state.monthly_requests_count = 0
            self.state.monthly_prompt_tokens = 0
            self.state.monthly_completion_tokens = 0
            self.state.monthly_usd_consumed = 0.0
            self._save_ledger()

    # --------------------------------------------------------------------------
    # Content-Addressable SHA-256 Teacher Cache
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_cache_key(system_prompt: str, user_prompt: str, model_id: str) -> str:
        payload = f"{model_id.strip()}|{system_prompt.strip()}|{user_prompt.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached_completion(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        key = self.compute_cache_key(system_prompt, user_prompt, self.teacher_model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                self.state.total_cache_hits_saved += 1
                self._save_ledger()
                return data
            except Exception:
                return None
        return None

    def save_completion_to_cache(
        self,
        system_prompt: str,
        user_prompt: str,
        result: TeacherCompletionResult,
    ) -> None:
        key = self.compute_cache_key(system_prompt, user_prompt, self.teacher_model)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # --------------------------------------------------------------------------
    # Quota Gate & Live Merge Gateway API Call
    # --------------------------------------------------------------------------
    def can_spend_tokens(self, estimated_total_tokens: int = 1800) -> Tuple[bool, float, str]:
        self._roll_month_if_needed()
        est_cost = round(
            (estimated_total_tokens / 1000.0) * self.state.blended_cost_per_1k_tokens_usd,
            6,
        )
        max_allowed_spend = self.state.monthly_credit_limit_usd - self.state.safety_buffer_usd
        if (self.state.monthly_usd_consumed + est_cost) > max_allowed_spend:
            return False, est_cost, "MONTHLY_10USD_SAFETY_BUFFER_REACHED"
        return True, est_cost, "QUOTA_AVAILABLE"

    def record_token_usage(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        total_tok = max(0, int(prompt_tokens) + int(completion_tokens))
        cost_usd = round(
            (total_tok / 1000.0) * self.state.blended_cost_per_1k_tokens_usd, 6
        )
        self.state.monthly_requests_count += 1
        self.state.monthly_prompt_tokens += int(prompt_tokens)
        self.state.monthly_completion_tokens += int(completion_tokens)
        self.state.monthly_usd_consumed = round(
            self.state.monthly_usd_consumed + cost_usd, 6
        )
        self._save_ledger()
        return cost_usd

    async def request_teacher_completion(
        self,
        snapshot_uid: str,
        system_prompt: str,
        user_prompt: str,
        fallback_completion_fn: Any,
    ) -> TeacherCompletionResult:
        """
        Checks the local SHA-256 cache first, then calls Merge Gateway if a live
        key and quota are available, or invokes `fallback_completion_fn` for
        deterministic zero-cost Superforecaster trace synthesis.
        """
        t0 = time.perf_counter()
        cached = self.get_cached_completion(system_prompt, user_prompt)
        if cached is not None:
            return TeacherCompletionResult(
                snapshot_uid=snapshot_uid,
                teacher_model_id=str(cached.get("teacher_model_id", self.teacher_model)),
                generation_source="MERGE_GATEWAY_SHA256_CACHE",
                completion_text=str(cached.get("completion_text", "")),
                prompt_tokens=int(cached.get("prompt_tokens", 0)),
                completion_tokens=int(cached.get("completion_tokens", 0)),
                incurred_cost_usd=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        if self.has_live_merge_key:
            allowed, _, _ = self.can_spend_tokens(estimated_total_tokens=1800)
            if allowed:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    body = {
                        "model": self.teacher_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 700,
                    }
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(
                            self.gateway_url, headers=headers, json=body
                        )
                        if resp.status_code == 200:
                            payload = resp.json()
                            choices = payload.get("choices") or []
                            if choices:
                                content = str(
                                    choices[0].get("message", {}).get("content") or ""
                                )
                                usage = payload.get("usage") or {}
                                p_tok = int(usage.get("prompt_tokens") or 1300)
                                c_tok = int(usage.get("completion_tokens") or 380)
                                cost = self.record_token_usage(p_tok, c_tok)
                                res = TeacherCompletionResult(
                                    snapshot_uid=snapshot_uid,
                                    teacher_model_id=self.teacher_model,
                                    generation_source="LIVE_MERGE_GATEWAY_API",
                                    completion_text=content,
                                    prompt_tokens=p_tok,
                                    completion_tokens=c_tok,
                                    incurred_cost_usd=cost,
                                    latency_ms=round(
                                        (time.perf_counter() - t0) * 1000.0, 2
                                    ),
                                )
                                self.save_completion_to_cache(
                                    system_prompt, user_prompt, res
                                )
                                return res
                except Exception as exc:
                    logger.warning(
                        "Merge Gateway live request failed (%s); using deterministic Tetlock synthesizer.",
                        exc,
                    )

        # Deterministic zero-cost Tetlock Superforecaster synthesis
        synth_text = fallback_completion_fn()
        est_p_tok = max(128, int(round(len(user_prompt) / 3.8)))
        est_c_tok = max(64, int(round(len(synth_text) / 3.8)))

        return TeacherCompletionResult(
            snapshot_uid=snapshot_uid,
            teacher_model_id="tetlock-bayesian-synthesizer-v1",
            generation_source="DETERMINISTIC_SUPERFORECASTER_SYNTHESIS",
            completion_text=synth_text,
            prompt_tokens=est_p_tok,
            completion_tokens=est_c_tok,
            incurred_cost_usd=0.0,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sovereign Probability Engine — Merge Gateway $10/mo Teacher Client"
    )
    parser.add_argument(
        "--simulate-calls",
        type=int,
        default=0,
        help="Simulate N teacher calls to verify quota accounting.",
    )
    args = parser.parse_args()

    client = MergeGatewayTeacherClient()
    for _ in range(args.simulate_calls):
        client.record_token_usage(prompt_tokens=1314, completion_tokens=360)

    report = {
        "stage": "STAGE_3_PART_D_MERGE_GATEWAY_CLIENT",
        "live_merge_key_detected": client.has_live_merge_key,
        "quota_state": client.state.to_dict(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())