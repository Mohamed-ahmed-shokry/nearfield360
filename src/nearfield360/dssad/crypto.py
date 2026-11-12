"""Cryptographic SHA-256 and HMAC hash chaining engine for tamper-evident DSSAD/EDR audit trails."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from pydantic import BaseModel

from nearfield360.dssad.models import ChainVerificationResult, TamperProofBlock

GENESIS_PREV_HASH = "0" * 64


class HashChainer:
    """Cryptographic tamper-evident hash chaining engine for UN R157 data integrity."""

    def __init__(self, signing_key_salt: str = "nearfield360-dssad-secret-salt-2026") -> None:
        if len(signing_key_salt) < 8:
            raise ValueError("Signing key salt must have at least 8 characters.")
        self._salt = signing_key_salt.encode()
        self._blocks: list[TamperProofBlock] = []

    @property
    def total_blocks(self) -> int:
        """Total number of blocks chained in the audit trail."""
        return len(self._blocks)

    def latest_digest(self) -> str:
        """Return the block hash of the most recent block, or GENESIS_PREV_HASH if empty."""
        return self._blocks[-1].block_hash if self._blocks else GENESIS_PREV_HASH

    def all_blocks(self) -> list[TamperProofBlock]:
        """Return a copy of all blocks currently chained."""
        return list(self._blocks)

    @staticmethod
    def compute_payload_digest(payload: str | bytes | BaseModel | dict[str, Any]) -> str:
        """Compute the canonical SHA-256 digest of arbitrary payload objects."""
        if isinstance(payload, BaseModel):
            data_bytes = payload.model_dump_json().encode()
        elif isinstance(payload, dict):
            data_bytes = json.dumps(payload, sort_keys=True).encode()
        elif isinstance(payload, str):
            data_bytes = payload.encode()
        else:
            data_bytes = payload

        return hashlib.sha256(data_bytes).hexdigest()

    def create_block(
        self, payload: str | bytes | BaseModel | dict[str, Any], timestamp_ns: int
    ) -> TamperProofBlock:
        """Construct and append a new tamper-evident block into the cryptographic chain."""
        seq_id = len(self._blocks)
        prev_hash = self.latest_digest()
        payload_digest = self.compute_payload_digest(payload)

        # Composite block preimage
        preimage = f"{seq_id}:{prev_hash}:{timestamp_ns}:{payload_digest}".encode()
        block_hash = hashlib.sha256(preimage).hexdigest()

        # HMAC signature
        signature = hmac.new(self._salt, block_hash.encode(), hashlib.sha256).hexdigest()

        block = TamperProofBlock(
            sequence_id=seq_id,
            prev_hash=prev_hash,
            timestamp_ns=timestamp_ns,
            payload_digest=payload_digest,
            block_hash=block_hash,
            hmac_signature=signature,
        )
        self._blocks.append(block)
        return block

    def verify_chain(self, blocks: list[TamperProofBlock] | None = None) -> ChainVerificationResult:
        """Verify cryptographic sequence, parent hashes, block digests, and HMAC signatures."""
        chain = self._blocks if blocks is None else blocks

        if not chain:
            return ChainVerificationResult(valid=True, total_blocks=0)

        for idx, block in enumerate(chain):
            if block.sequence_id != idx:
                return ChainVerificationResult(
                    valid=False,
                    total_blocks=len(chain),
                    corrupted_index=idx,
                    error_message=(
                        f"Sequence gap or reordering at index {idx}: "
                        f"expected {idx}, got {block.sequence_id}"
                    ),
                )

            expected_prev = GENESIS_PREV_HASH if idx == 0 else chain[idx - 1].block_hash
            if block.prev_hash != expected_prev:
                return ChainVerificationResult(
                    valid=False,
                    total_blocks=len(chain),
                    corrupted_index=idx,
                    error_message=(
                        f"Broken parent hash link at index {idx}: "
                        f"expected {expected_prev}, got {block.prev_hash}"
                    ),
                )

            preimage = (
                f"{block.sequence_id}:{block.prev_hash}:{block.timestamp_ns}:{block.payload_digest}"
            ).encode()
            recomputed_hash = hashlib.sha256(preimage).hexdigest()
            if block.block_hash != recomputed_hash:
                return ChainVerificationResult(
                    valid=False,
                    total_blocks=len(chain),
                    corrupted_index=idx,
                    error_message=(
                        f"Block hash integrity mismatch at index {idx}: "
                        f"stored {block.block_hash}, computed {recomputed_hash}"
                    ),
                )

            recomputed_sig = hmac.new(
                self._salt, block.block_hash.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(block.hmac_signature, recomputed_sig):
                return ChainVerificationResult(
                    valid=False,
                    total_blocks=len(chain),
                    corrupted_index=idx,
                    error_message=f"Invalid HMAC signature authentication at index {idx}",
                )

        return ChainVerificationResult(valid=True, total_blocks=len(chain))
