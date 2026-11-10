"""Unit tests for cryptographic SHA-256 hash chaining and tamper-evident audit trail engine."""

from __future__ import annotations

import pytest

from nearfield360.dssad.crypto import GENESIS_PREV_HASH, HashChainer
from nearfield360.dssad.models import TamperProofBlock


def test_hash_chainer_genesis_block() -> None:
    chainer = HashChainer(signing_key_salt="secret-test-salt-12345")
    assert chainer.total_blocks == 0
    assert chainer.latest_digest() == GENESIS_PREV_HASH

    block0 = chainer.create_block(payload="Genesis event record", timestamp_ns=1000)
    assert block0.sequence_id == 0
    assert block0.prev_hash == GENESIS_PREV_HASH
    assert len(block0.block_hash) == 64
    assert len(block0.hmac_signature) == 64
    assert chainer.total_blocks == 1
    assert chainer.latest_digest() == block0.block_hash

    res = chainer.verify_chain()
    assert res.valid is True
    assert res.total_blocks == 1
    assert res.corrupted_index is None


def test_hash_chainer_multi_block_sequence() -> None:
    chainer = HashChainer()
    blocks = [
        chainer.create_block(payload={"event": f"event_{i}", "val": i * 10}, timestamp_ns=1000 + i)
        for i in range(5)
    ]

    assert len(blocks) == 5
    for i in range(1, 5):
        assert blocks[i].sequence_id == i
        assert blocks[i].prev_hash == blocks[i - 1].block_hash

    res = chainer.verify_chain()
    assert res.valid is True
    assert res.total_blocks == 5


def test_tamper_detection_modified_payload() -> None:
    chainer = HashChainer()
    for i in range(4):
        chainer.create_block(payload=f"Payload {i}", timestamp_ns=1000 + i)

    blocks = chainer.all_blocks()

    # Tamper with block 2 payload digest (simulating silent disk bit flip)
    tampered_block = TamperProofBlock(
        sequence_id=blocks[2].sequence_id,
        prev_hash=blocks[2].prev_hash,
        timestamp_ns=blocks[2].timestamp_ns,
        payload_digest="f" * 64,  # altered
        block_hash=blocks[2].block_hash,
        hmac_signature=blocks[2].hmac_signature,
    )
    tampered_chain = list(blocks)
    tampered_chain[2] = tampered_block

    res = chainer.verify_chain(tampered_chain)
    assert res.valid is False
    assert res.corrupted_index == 2
    assert "Block hash integrity mismatch" in (res.error_message or "")


def test_tamper_detection_broken_parent_link() -> None:
    chainer = HashChainer()
    for i in range(4):
        chainer.create_block(payload=f"Payload {i}", timestamp_ns=1000 + i)

    blocks = chainer.all_blocks()

    # Tamper with block 1 prev_hash
    tampered_block = TamperProofBlock(
        sequence_id=blocks[1].sequence_id,
        prev_hash="a" * 64,  # altered
        timestamp_ns=blocks[1].timestamp_ns,
        payload_digest=blocks[1].payload_digest,
        block_hash=blocks[1].block_hash,
        hmac_signature=blocks[1].hmac_signature,
    )
    tampered_chain = list(blocks)
    tampered_chain[1] = tampered_block

    res = chainer.verify_chain(tampered_chain)
    assert res.valid is False
    assert res.corrupted_index == 1
    assert "Broken parent hash link" in (res.error_message or "")


def test_tamper_detection_tampered_hmac() -> None:
    chainer = HashChainer()
    chainer.create_block(payload="Valid payload", timestamp_ns=1000)

    blocks = chainer.all_blocks()
    tampered_block = TamperProofBlock(
        sequence_id=blocks[0].sequence_id,
        prev_hash=blocks[0].prev_hash,
        timestamp_ns=blocks[0].timestamp_ns,
        payload_digest=blocks[0].payload_digest,
        block_hash=blocks[0].block_hash,
        hmac_signature="bad_sig" + "0" * 57,
    )

    res = chainer.verify_chain([tampered_block])
    assert res.valid is False
    assert res.corrupted_index == 0
    assert "Invalid HMAC signature" in (res.error_message or "")


def test_tamper_detection_sequence_gap() -> None:
    chainer = HashChainer()
    for i in range(4):
        chainer.create_block(payload=f"Payload {i}", timestamp_ns=1000 + i)

    blocks = chainer.all_blocks()
    # Remove block 1 to simulate unauthorized log record deletion
    gapped_chain = [blocks[0], blocks[2], blocks[3]]

    res = chainer.verify_chain(gapped_chain)
    assert res.valid is False
    assert res.corrupted_index == 1
    assert "Sequence gap or reordering" in (res.error_message or "")


def test_hash_chainer_salt_validation() -> None:
    with pytest.raises(ValueError, match="at least 8 characters"):
        HashChainer(signing_key_salt="short")
