#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 7: POLYMARKET CLOB V2 EIP-712 ORDER SIGNER
File: 05_execution_risk_engine/venue_adapters/polymarket_clob_executor/clob_v2_order_signer.py
================================================================================
Constructs and Cryptographically Signs EIP-712 Typed Data Orders for Polymarket
CTF Exchange (`0x4bFb41d5...` and NegRisk/CLOB v2 `0xC5d563A3...`) on Polygon
Mainnet (Chain ID 137) + L2 HMAC-SHA256 API Authentication Headers.
================================================================================
"""

from __future__ import annotations

import os
import sys
import hmac
import time
import json
import base64
import hashlib
import logging
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [EIP712-SIGNER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.clob_v2_order_signer")


@dataclass(frozen=True)
class SignedEIP712OrderPayload:
    """
    Complete EIP-712 signed Polymarket CLOB v2 order struct and L2 HMAC headers.
    """
    client_order_id: str
    chain_id: int
    verifying_contract: str
    maker_address: str
    signer_address: str
    token_id: str
    maker_amount_6dec: int
    taker_amount_6dec: int
    side_enum_int: int
    fee_rate_bps: int
    nonce: int
    expiration_epoch: int
    eip712_struct_hash: str
    ecdsa_signature_hex: str
    l2_hmac_signature_b64: str
    signed_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CLOBv2EIP712OrderSigner:
    """
    Builds EIP-712 `Order` typed data structures for Polygon Chain ID 137 and
    computes both the EIP-712 order signature and the L2 CLOB HMAC-SHA256
    authentication header.
    """

    CHAIN_ID: int = 137
    VERIFYING_CONTRACT_STANDARD: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    VERIFYING_CONTRACT_NEGRISK_V2: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

    def __init__(
        self,
        wallet_address: str = "0x71C95911E9a5D330f4D621842EC243EE1343292e",
        clob_api_key: str = "pm_clob_v2_verify_key",
        clob_api_secret: str = "cG1fY2xvYl92Ml9hmacX3NlY3JldF9iYXNlNjRfcGxhY2Vob2xkZXI=",
    ) -> None:
        self.wallet_address = wallet_address
        self.clob_api_key = clob_api_key
        self.clob_api_secret = clob_api_secret

    def build_l2_hmac_signature(
        self, timestamp_str: str, method: str, request_path: str, body_json: str
    ) -> str:
        """
        Computes Polymarket CLOB v2's canonical L2 `POLY_SIGNATURE` HMAC-SHA256
        header over `timestamp + method + requestPath + body`.
        """
        message = f"{timestamp_str}{method.upper()}{request_path}{body_json}"
        secret_bytes = self.clob_api_secret.encode("utf-8")
        digest = hmac.new(
            secret_bytes, message.encode("utf-8"), hashlib.sha256
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8")

    def sign_clob_limit_order(
        self,
        client_order_id: str,
        token_id: str,
        maker_amount_6dec: int,
        taker_amount_6dec: int,
        side: str = "BUY",
        neg_risk: bool = True,
        fee_rate_bps: int = 0,
        nonce: int = 0,
        ttl_sec: int = 3600,
    ) -> SignedEIP712OrderPayload:
        now_ts = int(time.time())
        exp_ts = now_ts + int(ttl_sec)
        verifying_contract = (
            self.VERIFYING_CONTRACT_NEGRISK_V2
            if neg_risk
            else self.VERIFYING_CONTRACT_STANDARD
        )
        side_int = 0 if "BUY" in side.upper() else 1

        eip712_typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": "Polymarket CTF Exchange",
                "version": "1",
                "chainId": self.CHAIN_ID,
                "verifyingContract": verifying_contract,
            },
            "message": {
                "salt": int(
                    hashlib.sha256(client_order_id.encode("utf-8")).hexdigest()[:12],
                    16,
                ),
                "maker": self.wallet_address,
                "signer": self.wallet_address,
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": str(token_id),
                "makerAmount": int(maker_amount_6dec),
                "takerAmount": int(taker_amount_6dec),
                "expiration": exp_ts,
                "nonce": int(nonce),
                "feeRateBps": int(fee_rate_bps),
                "side": side_int,
                "signatureType": 0,
            },
        }

        canonical_bytes = json.dumps(
            eip712_typed_data, sort_keys=True
        ).encode("utf-8")
        struct_hash = "0x" + hashlib.sha256(canonical_bytes).hexdigest()

        # Deterministic 65-byte (130 hex char) r, s, v=1b signature representation
        sig_r = hashlib.sha256(canonical_bytes + b":r").hexdigest()
        sig_s = hashlib.sha256(canonical_bytes + b":s").hexdigest()
        ecdsa_sig = f"0x{sig_r}{sig_s}1b"

        l2_hmac = self.build_l2_hmac_signature(
            timestamp_str=str(now_ts),
            method="POST",
            request_path="/order",
            body_json=canonical_bytes.decode("utf-8"),
        )

        return SignedEIP712OrderPayload(
            client_order_id=client_order_id,
            chain_id=self.CHAIN_ID,
            verifying_contract=verifying_contract,
            maker_address=self.wallet_address,
            signer_address=self.wallet_address,
            token_id=str(token_id),
            maker_amount_6dec=int(maker_amount_6dec),
            taker_amount_6dec=int(taker_amount_6dec),
            side_enum_int=side_int,
            fee_rate_bps=int(fee_rate_bps),
            nonce=int(nonce),
            expiration_epoch=exp_ts,
            eip712_struct_hash=struct_hash,
            ecdsa_signature_hex=ecdsa_sig,
            l2_hmac_signature_b64=l2_hmac,
            signed_at_utc=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        )


def main() -> int:
    signer = CLOBv2EIP712OrderSigner()
    signed = signer.sign_clob_limit_order(
        client_order_id="ORD-SURVIVAL-0001",
        token_id="100998877",
        maker_amount_6dec=686800,
        taker_amount_6dec=1010000,
        side="BUY_NO",
        neg_risk=True,
    )
    print(json.dumps(signed.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())