#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `auto_settlement_Sweep`."""

from typing import Any

__all__ = [
    "RedeemableWinningPosition",
    "ResolvedPositionScanner",
    "CtfRedemptionTxReceipt",
    "CtfRedeemTxSender",
]


def __getattr__(name: str) -> Any:
    if name in ("RedeemableWinningPosition", "ResolvedPositionScanner"):
        from . import resolved_position_scanner
        return getattr(resolved_position_scanner, name)
    if name in ("CtfRedemptionTxReceipt", "CtfRedeemTxSender"):
        from . import ctf_redeem_tx_sender
        return getattr(ctf_redeem_tx_sender, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")