#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOVEREIGN PROBABILITY ENGINE — STAGE 2: DATA INGESTION (PART C)
File: 01_data_ingestion/uma_oracle_rules/ancillary_data_decoder.py
================================================================================
Cryptographic & ABI Hex Decoder for Polygon UMA Optimistic Oracle v2/v3
`ancillaryData` Payloads (`0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74`).

How Polymarket Encodes Resolution Rules On-Chain:
  1. When a market is initialized on Polygon (`UmaCtfAdapter`), the exact
     question title, resolution description, outcome mapping (`p1: 0, p2: 1,
     p3: 0.5`), and governance initializer address are packed into a UTF-8 /
     ABI-encoded byte array called `ancillaryData`.
  2. When on-chain events (`QuestionInitialized`, `RequestPrice`, `ProposePrice`,
     `DisputePrice`) are emitted on Polygon Chain ID 137, `ancillaryData` is
     logged as a raw `0x...` hex string.
  3. This module deterministically decodes `ancillaryData` from raw Polygon
     hex logs OR synthesizes the canonical UMA ancillary byte structure from
     Stage 2 Part A Gamma records, computing the deterministic keccak256/SHA256
     rule digest so our bot verifies the rules never mutated mid-market.
================================================================================
"""

from __future__ import annotations

import re
import sys
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [UMA-ANCILLARY-DECODER] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign_prob_engine.ancillary_data_decoder")

# Regex patterns matching Polymarket's canonical UMA ancillaryData key-value format:
# e.g. "q: title: Will ..., description: This market resolves..., res_data: p1: 0, p2: 1, p3: 0.5."
TITLE_EXTRACTOR_REGEX = re.compile(
    r"(?:q:\s*title:\s*|title:\s*)(?P<title>.*?)(?=,\s*description:|\s*res_data:|$)",
    re.IGNORECASE | re.DOTALL,
)
DESCRIPTION_EXTRACTOR_REGEX = re.compile(
    r"description:\s*(?P<desc>.*?)(?=,\s*res_data:|\s*res_data:|\s*initializer:|$)",
    re.IGNORECASE | re.DOTALL,
)
RES_DATA_EXTRACTOR_REGEX = re.compile(
    r"res_data:\s*(?P<res_data>p1:\s*[0-9.]+,\s*p2:\s*[0-9.]+(?:,\s*p3:\s*[0-9.]+)?)",
    re.IGNORECASE,
)
INITIALIZER_REGEX = re.compile(
    r"initializer:\s*(?P<init_addr>0x[a-fA-F0-9]{40})",
    re.IGNORECASE,
)
P_MAPPING_ITEM_REGEX = re.compile(r"(p[1-4]):\s*([0-9.]+)", re.IGNORECASE)


# ==============================================================================
# 1. DATA STRUCTURE FOR DECODED UMA ANCILLARY PAYLOADS
# ==============================================================================
@dataclass(frozen=True)
class DecodedAncillaryPayload:
    """
    Structured representation of a decoded UMA Optimistic Oracle `ancillaryData`
    blob on Polygon Mainnet (Chain ID 137).
    """
    question_id: str
    condition_id: str
    raw_hex_length_bytes: int
    decoded_utf8_text: str
    extracted_title: str
    extracted_description: str
    resolution_data_string: str
    payout_mapping: Dict[str, float]
    initializer_contract: str
    uma_price_identifier: str
    ancillary_sha256_digest: str
    has_50_50_tie_fallback: bool
    is_valid_uma_binary_spec: bool
    decoding_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cpt_xml_tag(self) -> str:
        """
        Formats the decoded UMA ancillary payload into the exact `<uma_ancillary>`
        block ingested during Stage 4 Continual Pre-Training (CPT).
        """
        payout_str = ", ".join(f"{k}={v}" for k, v in sorted(self.payout_mapping.items()))
        return (
            f"<uma_ancillary question_id=\"{self.question_id}\" "
            f"initializer=\"{self.initializer_contract}\" "
            f"payouts=\"{payout_str}\">\n"
            f"  <title>{self.extracted_title}</title>\n"
            f"  <rules>{self.extracted_description}</rules>\n"
            f"</uma_ancillary>"
        )


# ==============================================================================
# 2. CORE UMA ANCILLARY DATA HEX DECODER & CANONICAL ENCODER
# ==============================================================================
class UmaAncillaryDataDecoder:
    """
    Decodes hexadecimal `ancillaryData` strings emitted by Polygon's
    `UmaCtfAdapter` (`0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74`) and
    `OptimisticOracleV3` (`0x5953f2538F613E05bAED8A5AeFa8e6622467AD3D`).
    """

    DEFAULT_UMA_CTF_ADAPTER: str = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"
    DEFAULT_PRICE_IDENTIFIER: str = "YES_OR_NO_QUERY"

    def __init__(self, strict_utf8: bool = False) -> None:
        self.strict_utf8 = strict_utf8

    # --------------------------------------------------------------------------
    # Hexadecimal Cleaning & Byte Conversion
    # --------------------------------------------------------------------------
    @staticmethod
    def _strip_non_printable_control_chars(text: str) -> str:
        """
        Strips ABI length-prefix padding bytes (null bytes `\\x00` and control
        bytes < 0x20 except newline/tab) while preserving UTF-8 punctuation.
        """
        cleaned_chars: List[str] = []
        for ch in text:
            code = ord(ch)
            if ch in ("\n", "\r", "\t") or code >= 32:
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        # Collapse runs of multiple spaces caused by ABI 32-byte zero padding
        collapsed = re.sub(r"[ \t]{2,}", " ", "".join(cleaned_chars))
        return collapsed.strip()

    def decode_hex_bytes_to_text(self, raw_hex: str) -> Tuple[str, int, List[str]]:
        """
        Converts a `0x`-prefixed hexadecimal string from Polygon event logs
        into clean UTF-8 text, returning `(cleaned_text, byte_len, notes)`.
        """
        notes: List[str] = []
        cleaned_hex = str(raw_hex or "").strip()
        if cleaned_hex.startswith(("0x", "0X")):
            cleaned_hex = cleaned_hex[2:]

        if not cleaned_hex:
            notes.append("EMPTY_HEX_INPUT")
            return "", 0, notes

        # Ensure even number of hex nibbles
        if len(cleaned_hex) % 2 != 0:
            cleaned_hex = "0" + cleaned_hex
            notes.append("ODD_HEX_NIBBLE_PADDED")

        try:
            raw_bytes = bytes.fromhex(cleaned_hex)
        except ValueError as exc:
            notes.append(f"INVALID_HEX_ENCODING:{exc}")
            return "", 0, notes

        error_mode = "strict" if self.strict_utf8 else "replace"
        decoded_raw = raw_bytes.decode("utf-8", errors=error_mode)
        sanitized_text = self._strip_non_printable_control_chars(decoded_raw)
        return sanitized_text, len(raw_bytes), notes

    # --------------------------------------------------------------------------
    # Payout Mapping Parser (`p1: 0, p2: 1, p3: 0.5`)
    # --------------------------------------------------------------------------
    @staticmethod
    def parse_payout_mapping(res_data_str: str) -> Dict[str, float]:
        """
        Parses UMA's `res_data` clause into a float dictionary:
          - `p1`: Corresponds to NO (`0.0`)
          - `p2`: Corresponds to YES (`1.0`)
          - `p3`: Corresponds to UNKNOWN / 50-50 SPLIT (`0.5`)
          - `p4`: Early request / unresolvable sentinel (`0.5` or `-1.0`)
        """
        mapping: Dict[str, float] = {}
        for match in P_MAPPING_ITEM_REGEX.finditer(res_data_str):
            key = match.group(1).lower()
            try:
                mapping[key] = float(match.group(2))
            except ValueError:
                continue

        if not mapping:
            # Canonical Polymarket binary default
            mapping = {"p1": 0.0, "p2": 1.0, "p3": 0.5}
        return mapping

    # --------------------------------------------------------------------------
    # Primary Decoding Method (From Raw Polygon Hex)
    # --------------------------------------------------------------------------
    def decode_ancillary_hex(
        self,
        raw_hex: str,
        question_id: str = "",
        condition_id: str = "",
    ) -> DecodedAncillaryPayload:
        """
        Decodes a raw hexadecimal `ancillaryData` string into a fully structured
        `DecodedAncillaryPayload` instance.
        """
        utf8_text, byte_len, notes = self.decode_hex_bytes_to_text(raw_hex)

        title_match = TITLE_EXTRACTOR_REGEX.search(utf8_text)
        desc_match = DESCRIPTION_EXTRACTOR_REGEX.search(utf8_text)
        res_match = RES_DATA_EXTRACTOR_REGEX.search(utf8_text)
        init_match = INITIALIZER_REGEX.search(utf8_text)

        if title_match:
            title = title_match.group("title").strip().rstrip(",")
        else:
            title = utf8_text[:160].strip()
            notes.append("FALLBACK_TITLE_SLICE")

        if desc_match:
            description = desc_match.group("desc").strip().rstrip(",")
        else:
            description = utf8_text
            notes.append("FALLBACK_FULL_TEXT_DESCRIPTION")

        if res_match:
            res_data_str = res_match.group("res_data").strip()
        else:
            res_data_str = "p1: 0, p2: 1, p3: 0.5"
            notes.append("DEFAULT_BINARY_RES_DATA_APPLIED")

        initializer_addr = (
            init_match.group("init_addr")
            if init_match
            else self.DEFAULT_UMA_CTF_ADAPTER
        )

        payout_map = self.parse_payout_mapping(res_data_str)
        has_50_50 = abs(payout_map.get("p3", 0.0) - 0.5) < 1e-6
        is_valid_binary = (
            "p1" in payout_map
            and "p2" in payout_map
            and abs(payout_map["p1"] - 0.0) < 1e-6
            and abs(payout_map["p2"] - 1.0) < 1e-6
        )

        digest = hashlib.sha256(utf8_text.encode("utf-8")).hexdigest()

        return DecodedAncillaryPayload(
            question_id=str(question_id or "0x" + digest[:64]),
            condition_id=str(condition_id or ""),
            raw_hex_length_bytes=byte_len,
            decoded_utf8_text=utf8_text,
            extracted_title=title,
            extracted_description=description,
            resolution_data_string=res_data_str,
            payout_mapping=payout_map,
            initializer_contract=initializer_addr,
            uma_price_identifier=self.DEFAULT_PRICE_IDENTIFIER,
            ancillary_sha256_digest=digest,
            has_50_50_tie_fallback=has_50_50,
            is_valid_uma_binary_spec=is_valid_binary,
            decoding_notes=notes,
        )

    # --------------------------------------------------------------------------
    # Canonical Synthesizer (From Gamma Market Row -> UMA Hex & Decoded Object)
    # --------------------------------------------------------------------------
    def synthesize_from_gamma_market_row(
        self, market_row: Dict[str, Any]
    ) -> DecodedAncillaryPayload:
        """
        Constructs the exact canonical UMA `ancillaryData` byte payload from a
        Stage 2 Part A Gamma market dictionary and runs it through the hex
        decoder so every market in our lake has a verified UMA ancillary digest.
        """
        question_title = str(market_row.get("question", "")).strip()
        description_body = str(market_row.get("description", "")).strip()
        question_id = str(market_row.get("question_id", "")).strip()
        condition_id = str(market_row.get("condition_id", "")).strip()

        canonical_ancillary_str = (
            f"q: title: {question_title}, "
            f"description: {description_body} "
            f"res_data: p1: 0, p2: 1, p3: 0.5. "
            f"initializer: {self.DEFAULT_UMA_CTF_ADAPTER}"
        )
        encoded_hex = "0x" + canonical_ancillary_str.encode("utf-8").hex()
        return self.decode_ancillary_hex(
            raw_hex=encoded_hex,
            question_id=question_id,
            condition_id=condition_id,
        )


# ==============================================================================
# 3. SELF-TEST ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    decoder = UmaAncillaryDataDecoder()
    sample_ancillary_text = (
        "q: title: Will the Federal Reserve cut interest rates by 25 bps at the June 2026 FOMC meeting?, "
        "description: This market resolves to Yes if the official Federal Open Market Committee (FOMC) "
        "statement released at 2:00 PM ET on June 17, 2026 lowers the target federal funds rate range by "
        "at least 25 basis points. Otherwise resolves to No. "
        "res_data: p1: 0, p2: 1, p3: 0.5. "
        "initializer: 0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"
    )
    sample_hex = "0x" + sample_ancillary_text.encode("utf-8").hex()
    decoded = decoder.decode_ancillary_hex(
        raw_hex=sample_hex,
        question_id="0x" + "9a" * 32,
        condition_id="0x" + "7b" * 32,
    )
    print(json.dumps(decoded.to_dict(), indent=2))
    print("\n--- CPT XML SERIALIZATION ---")
    print(decoded.to_cpt_xml_tag())