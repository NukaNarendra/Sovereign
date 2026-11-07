#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy-loaded package initializer for `polymarket_clob_executor`."""

from typing import Any

__all__ = [
    "SignedEIP712OrderPayload",
    "CLOBv2EIP712OrderSigner",
    "ActiveOrderState",
    "OrderLifecycleManager",
    "RelayerSubmissionReceipt",
    "PolymarketGaslessRelayerBridge",
]


def __getattr__(name: str) -> Any:
    if name in ("SignedEIP712OrderPayload", "CLOBv2EIP712OrderSigner"):
        from . import clob_v2_order_signer
        return getattr(clob_v2_order_signer, name)
    if name in ("ActiveOrderState", "OrderLifecycleManager"):
        from . import order_lifecycle_manager
        return getattr(order_lifecycle_manager, name)
    if name in ("RelayerSubmissionReceipt", "PolymarketGaslessRelayerBridge"):
        from . import relayer_gasless_bridge
        return getattr(relayer_gasless_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")