"""Unit tests for cryptographic firmware verifier and anti-rollback engine."""

from __future__ import annotations

from nearfield360.config import OTAConfig
from nearfield360.ota.models import FirmwareManifest
from nearfield360.ota.verifier import FirmwareVerifier


def test_payload_hash_and_signature() -> None:
    verifier = FirmwareVerifier()
    payload = b"NEO-FIRMWARE-IMAGE-DATA-BYTES-TEST"
    sha = verifier.compute_payload_sha256(payload)
    assert len(sha) == 64

    sig = verifier.sign_payload(payload)
    assert len(sig) == 64
    assert verifier.verify_signature(payload, sig) is True

    # Tampered payload fails verification
    assert verifier.verify_signature(payload + b"TAMPERED", sig) is False

    # Wrong secret key fails verification
    dummy_wrong_key = "invalid-test-signing-key-value"
    assert verifier.verify_signature(payload, sig, secret_key=dummy_wrong_key) is False


def test_verify_manifest_happy_path() -> None:
    verifier = FirmwareVerifier()
    payload = b"VALID-CAN-GATEWAY-FIRMWARE-IMAGE-V2"
    sha = verifier.compute_payload_sha256(payload)
    sig = verifier.sign_payload(payload)

    manifest = FirmwareManifest(
        component_id="can_gateway",
        version="v2.0.0",
        monotonic_revision=2,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="can_gateway",
        payload_size_bytes=len(payload),
        sha256_hash=sha,
        signature_hex=sig,
    )

    ok, msg = verifier.verify_manifest(manifest, payload)
    assert ok is True
    assert "verified successfully" in msg


def test_verify_manifest_rejections() -> None:
    verifier = FirmwareVerifier(config=OTAConfig(max_package_size_mb=2.0))
    payload = b"TEST-REJECTIONS-PAYLOAD-BINARY"
    sha = verifier.compute_payload_sha256(payload)
    sig = verifier.sign_payload(payload)

    # 1. Size mismatch in manifest
    manifest_bad_size = FirmwareManifest(
        component_id="can_gateway",
        version="v2.0.0",
        monotonic_revision=2,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="can_gateway",
        payload_size_bytes=len(payload) + 5,
        sha256_hash=sha,
        signature_hex=sig,
    )
    ok1, msg1 = verifier.verify_manifest(manifest_bad_size, payload)
    assert ok1 is False
    assert "size mismatch" in msg1

    # 2. Package size exceeds configured limit (2.5 MB > 2.0 MB)
    large_payload = b"X" * int(2.5 * 1024 * 1024)
    large_sha = verifier.compute_payload_sha256(large_payload)
    large_sig = verifier.sign_payload(large_payload)
    manifest_large = FirmwareManifest(
        component_id="can_gateway",
        version="v2.0.0",
        monotonic_revision=2,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="can_gateway",
        payload_size_bytes=len(large_payload),
        sha256_hash=large_sha,
        signature_hex=large_sig,
    )
    ok2, msg2 = verifier.verify_manifest(manifest_large, large_payload)
    assert ok2 is False
    assert "exceeds configured limit" in msg2

    # 3. Hash mismatch
    manifest_bad_hash = FirmwareManifest(
        component_id="can_gateway",
        version="v2.0.0",
        monotonic_revision=2,
        rxswin="RXSWIN-E4-156R00-NF360-0042",
        target_ecu="can_gateway",
        payload_size_bytes=len(payload),
        sha256_hash="0" * 64,
        signature_hex=sig,
    )
    ok3, msg3 = verifier.verify_manifest(manifest_bad_hash, payload)
    assert ok3 is False
    assert "SHA-256 digest mismatch" in msg3


def test_anti_rollback_monotonic_check() -> None:
    verifier = FirmwareVerifier(config=OTAConfig(enforce_monotonic_version=True))

    # Higher revision succeeds
    ok_up, _ = verifier.verify_anti_rollback(candidate_revision=5, current_installed_revision=4)
    assert ok_up is True

    # Same revision fails
    ok_same, msg_same = verifier.verify_anti_rollback(
        candidate_revision=4, current_installed_revision=4
    )
    assert ok_same is False
    assert "Anti-rollback violation" in msg_same

    # Lower revision fails (downgrade attack)
    ok_down, msg_down = verifier.verify_anti_rollback(
        candidate_revision=3, current_installed_revision=4
    )
    assert ok_down is False
    assert "Anti-rollback violation" in msg_down

    # Disabled enforcement allows downgrade
    verifier_loose = FirmwareVerifier(config=OTAConfig(enforce_monotonic_version=False))
    ok_loose, _ = verifier_loose.verify_anti_rollback(
        candidate_revision=2, current_installed_revision=5
    )
    assert ok_loose is True
