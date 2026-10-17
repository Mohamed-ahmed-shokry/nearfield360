"""Unit tests for AccessCredentialValidator and ALPR matching."""

from __future__ import annotations

from nearfield360.config import GateConfig
from nearfield360.gate.models import (
    AccessDecisionStatus,
    CredentialType,
    GateAccessRequest,
    VehicleCredential,
    VehicleDimensionEnvelope,
)
from nearfield360.gate.validator import (
    AccessCredentialValidator,
    compute_plate_similarity,
    normalize_plate_string,
)


def _sample_envelope() -> VehicleDimensionEnvelope:
    return VehicleDimensionEnvelope(length_m=4.6, width_m=1.85, height_m=1.5)


def test_plate_normalization_and_similarity() -> None:
    assert normalize_plate_string("7-xyz 890") == "7XYZ890"
    assert normalize_plate_string("  california  ") == "CALIFORNIA"
    assert normalize_plate_string("") == ""

    assert compute_plate_similarity("7XYZ890", "7XYZ890") == 1.0
    assert compute_plate_similarity("7XYZ890", "7XYZ891") > 0.8
    assert compute_plate_similarity("", "") == 1.0
    assert compute_plate_similarity("ABC", "") == 0.0


def test_validator_credential_registration() -> None:
    config = GateConfig()
    cred = VehicleCredential(
        credential_id="cred-1",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="5ABC123",
        tenant_id="VIP",
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])
    assert len(validator.list_credentials()) == 1
    assert validator.get_credential("cred-1") == cred
    assert validator.find_by_plate("5-abc-123") == cred

    removed = validator.unregister_credential("cred-1")
    assert removed is True
    assert validator.get_credential("cred-1") is None
    assert validator.find_by_plate("5ABC123") is None


def test_validator_grants_valid_alpr_plate() -> None:
    config = GateConfig(alpr_confidence_threshold=0.85)
    cred = VehicleCredential(
        credential_id="cred-plate-42",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="9XYZ999",
        valid_from_s=0.0,
        valid_until_s=1000.0,
        tenant_id="AIRPORT_VALET",
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])

    req = GateAccessRequest(
        request_id="req-101",
        gate_id="GATE_ENTRY_1",
        timestamp_s=250.0,
        vehicle_id="veh-101",
        detected_plate="9-XYZ-999",
        plate_confidence=0.95,
        envelope=_sample_envelope(),
    )
    decision = validator.validate_request(req)

    assert decision.granted is True
    assert decision.status == AccessDecisionStatus.GRANTED
    assert "9XYZ999" in decision.message
    assert decision.issued_pass_id == "PASS-GATE_ENTRY_1-veh-101"  # noqa: S105


def test_validator_rejects_low_confidence_alpr() -> None:
    config = GateConfig(alpr_confidence_threshold=0.85)
    cred = VehicleCredential(
        credential_id="cred-1",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="9XYZ999",
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])

    req = GateAccessRequest(
        request_id="req-102",
        gate_id="GATE_ENTRY_1",
        timestamp_s=10.0,
        vehicle_id="veh-102",
        detected_plate="9XYZ999",
        plate_confidence=0.72,  # below threshold
        envelope=_sample_envelope(),
    )
    decision = validator.validate_request(req)

    assert decision.granted is False
    assert decision.status == AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL
    assert "below threshold" in decision.message


def test_validator_rejects_unknown_and_missing_plate() -> None:
    config = GateConfig()
    validator = AccessCredentialValidator(config)

    # Empty plate
    req_empty = GateAccessRequest(
        request_id="req-103",
        gate_id="GATE_ENTRY_1",
        timestamp_s=10.0,
        vehicle_id="veh-103",
        detected_plate="",
        plate_confidence=0.0,
        envelope=_sample_envelope(),
    )
    dec_empty = validator.validate_request(req_empty)
    assert dec_empty.granted is False
    assert dec_empty.status == AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL

    # Unknown plate
    req_unknown = GateAccessRequest(
        request_id="req-104",
        gate_id="GATE_ENTRY_1",
        timestamp_s=10.0,
        vehicle_id="veh-104",
        detected_plate="UNKNOWN99",
        plate_confidence=0.98,
        envelope=_sample_envelope(),
    )
    dec_unknown = validator.validate_request(req_unknown)
    assert dec_unknown.granted is False
    assert dec_unknown.status == AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL


def test_validator_rejects_expired_plate_reservation() -> None:
    config = GateConfig()
    cred = VehicleCredential(
        credential_id="cred-expired",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="EXPIRED1",
        valid_from_s=10.0,
        valid_until_s=100.0,
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])

    req = GateAccessRequest(
        request_id="req-105",
        gate_id="GATE_ENTRY_1",
        timestamp_s=150.0,  # Expired
        vehicle_id="veh-105",
        detected_plate="EXPIRED1",
        plate_confidence=0.95,
        envelope=_sample_envelope(),
    )
    decision = validator.validate_request(req)

    assert decision.granted is False
    assert decision.status == AccessDecisionStatus.REJECTED_EXPIRED_RESERVATION


def test_validator_direct_token_credential() -> None:
    config = GateConfig()
    cred = VehicleCredential(
        credential_id="cred-uwb-88",
        credential_type=CredentialType.UWB_DIGITAL_KEY,
        token_hash="hash-abc-123",  # noqa: S106
        valid_from_s=50.0,
        valid_until_s=500.0,
        vip_status=True,
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])

    req_valid = GateAccessRequest(
        request_id="req-106",
        gate_id="GATE_ENTRY_1",
        timestamp_s=120.0,
        vehicle_id="veh-106",
        credential=cred,
        envelope=_sample_envelope(),
    )
    dec_valid = validator.validate_request(req_valid)
    assert dec_valid.granted is True
    assert dec_valid.status == AccessDecisionStatus.GRANTED

    req_expired = GateAccessRequest(
        request_id="req-107",
        gate_id="GATE_ENTRY_1",
        timestamp_s=600.0,
        vehicle_id="veh-107",
        credential=cred,
        envelope=_sample_envelope(),
    )
    dec_expired = validator.validate_request(req_expired)
    assert dec_expired.granted is False
    assert dec_expired.status == AccessDecisionStatus.REJECTED_EXPIRED_RESERVATION
