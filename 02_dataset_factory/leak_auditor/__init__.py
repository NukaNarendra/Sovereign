#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 3: DATASET FACTORY (PART B)
File: 02_dataset_factory/leak_auditor/__init__.py
================================================================================
Public interface for the 3-Stage Anti-Leakage & Contamination Audit Suite:
  1. `future_date_regex_scrubber` : Scans all prompt text fields (`cpt_news_xml`,
                                    `cpt_uma_rule_xml`, `cpt_macro_xml`) for
                                    explicit calendar dates (`YYYY-MM-DD`,
                                    `Month DD, YYYY`, `DD Month YYYY`) that fall
                                    after `cutoff_timestamp_utc` outside of
                                    legitimate contract deadline definitions.
  2. `outcome_spoiler_detector`   : Detects and scrubs hindsight spoiler phrases
                                    ("resolved to Yes", "after winning the
                                    election", "final tally confirmed", "settled
                                    at 1.0") that would give Qwen-2.5-7B an
                                    unfair shortcut during training.
  3. `entity_contamination_check` : Audits chronological train/val/test splits
                                    and `correlation_cluster_key` groups to
                                    guarantee 0.0% sibling-market outcome leakage
                                    across split boundaries.

Uses PEP 562 lazy attribute loading (`__getattr__`) so running modules via
`python3 -m 02_dataset_factory.leak_auditor.<module>` never emits a
`<frozen runpy>:128: RuntimeWarning`.
================================================================================
"""

from typing import Any

__all__ = [
    "FutureDateMention",
    "DateScrubAuditResult",
    "FutureDateRegexScrubber",
    "SpoilerViolationMatch",
    "SpoilerScanResult",
    "OutcomeSpoilerDetector",
    "SplitContaminationReport",
    "EntityContaminationChecker",
]


def __getattr__(name: str) -> Any:
    if name in ("FutureDateMention", "DateScrubAuditResult", "FutureDateRegexScrubber"):
        from . import future_date_regex_scrubber
        return getattr(future_date_regex_scrubber, name)
    if name in ("SpoilerViolationMatch", "SpoilerScanResult", "OutcomeSpoilerDetector"):
        from . import outcome_spoiler_detector
        return getattr(outcome_spoiler_detector, name)
    if name in ("SplitContaminationReport", "EntityContaminationChecker"):
        from . import entity_contamination_check
        return getattr(entity_contamination_check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")