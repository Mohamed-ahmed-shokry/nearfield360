"""Unit tests for DSSAD and EDR domain models, serialization, and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.dssad.models import (
    ChainVerificationResult,
    DSSADEventRecord,
    DSSADEventType,
    DSSADReport,
    EDRIncidentRecord,
    EventSeverity,
    TamperProofBlock,
    TelemetrySnapshot,
    event_severity_rank,
)


def test_dssad_event_types_and_severity_ranking() -> None:
    assert DSSADEventType.COLLISION_IMPACT_DETECTED == "COLLISION_IMPACT_DETECTED"
    assert DSSADEventType.MINIMUM_RISK_MANEUVER_INITIATED == "MINIMUM_RISK_MANEUVER_INITIATED"
    assert DSSADEventType.ODD_BOUNDARY_BREACH == "ODD_BOUNDARY_BREACH"
    assert DSSADEventType.SECURITY_INTRUSION_ALERT == "SECURITY_INTRUSION_ALERT"

    assert event_severity_rank(EventSeverity.INFO) < event_severity_rank(EventSeverity.WARNING)
    assert event_severity_rank(EventSeverity.WARNING) < event_severity_rank(EventSeverity.CRITICAL)
    assert event_severity_rank(EventSeverity.CRITICAL) < event_severity_rank(EventSeverity.FATAL)


def test_telemetry_snapshot_validation_and_bounds() -> None:
    snapshot = TelemetrySnapshot(
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        speed_mps=5.2,
        longitudinal_accel_mps2=-1.8,
        lateral_accel_mps2=0.1,
        yaw_rate_rads=0.02,
        steering_angle_rad=0.05,
        brake_pressure_bar=12.5,
        throttle_pct=0.0,
        gear="D",
        x_m=15.2,
        y_m=-3.4,
        yaw_rad=1.57,
        odd_state="NOMINAL",
        aeb_active=False,
        active_dtcs=["U0100"],
    )

    assert snapshot.speed_mps == 5.2
    assert snapshot.gear == "D"
    assert "U0100" in snapshot.active_dtcs

    # Serialization roundtrip
    data = snapshot.model_dump_json()
    reconstructed = TelemetrySnapshot.model_validate_json(data)
    assert reconstructed == snapshot

    # Bounds validation: negative speed is forbidden
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        TelemetrySnapshot(
            timestamp_ns=1000,
            timestamp_iso="2026-11-10T12:00:00Z",
            speed_mps=-1.0,
            longitudinal_accel_mps2=0.0,
            lateral_accel_mps2=0.0,
            yaw_rate_rads=0.0,
            steering_angle_rad=0.0,
            brake_pressure_bar=0.0,
            throttle_pct=0.0,
        )


def test_tamper_proof_block_validation() -> None:
    block = TamperProofBlock(
        sequence_id=0,
        prev_hash="0" * 64,
        timestamp_ns=1700000000000000000,
        payload_digest="a" * 64,
        block_hash="b" * 64,
        hmac_signature="c" * 64,
    )
    assert block.sequence_id == 0
    assert block.prev_hash == "0" * 64

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        TamperProofBlock(
            sequence_id=0,
            prev_hash="0" * 64,
            timestamp_ns=1000,
            payload_digest="a" * 64,
            block_hash="b" * 64,
            hmac_signature="c" * 64,
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_dssad_event_record_serialization() -> None:
    snapshot = TelemetrySnapshot(
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        speed_mps=0.0,
        longitudinal_accel_mps2=-4.5,
        lateral_accel_mps2=0.0,
        yaw_rate_rads=0.0,
        steering_angle_rad=0.0,
        brake_pressure_bar=45.0,
        throttle_pct=0.0,
        odd_state="ODD_BREACH_MRM",
        aeb_active=True,
    )
    record = DSSADEventRecord(
        record_id="EVT-20261110-0001",
        sequence_id=1,
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        event_type=DSSADEventType.EMERGENCY_BRAKING_TRIGGERED,
        severity=EventSeverity.CRITICAL,
        trigger_subsystem="odd_supervisor",
        description="ASIL-D MRM emergency braking triggered due to critical surface friction loss.",
        snapshot=snapshot,
        active_dtcs=["C1201"],
        block_hash="feedface" * 8,
    )

    assert record.record_id == "EVT-20261110-0001"
    assert record.event_type == DSSADEventType.EMERGENCY_BRAKING_TRIGGERED
    assert record.severity == EventSeverity.CRITICAL

    raw_json = record.model_dump_json()
    loaded = DSSADEventRecord.model_validate_json(raw_json)
    assert loaded == record


def test_edr_incident_record_and_report_aggregation() -> None:
    snapshot = TelemetrySnapshot(
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        speed_mps=2.0,
        longitudinal_accel_mps2=-3.0,
        lateral_accel_mps2=0.0,
        yaw_rate_rads=0.0,
        steering_angle_rad=0.0,
        brake_pressure_bar=20.0,
        throttle_pct=0.0,
    )
    event = DSSADEventRecord(
        record_id="EVT-001",
        sequence_id=1,
        timestamp_ns=1700000000000000000,
        timestamp_iso="2026-11-10T12:00:00Z",
        event_type=DSSADEventType.COLLISION_IMPACT_DETECTED,
        severity=EventSeverity.FATAL,
        trigger_subsystem="chassis_imu",
        description="Crash sensor impact exceeding 2.5g threshold.",
        snapshot=snapshot,
    )
    incident = EDRIncidentRecord(
        incident_id="INC-20261110-001",
        freeze_timestamp_iso="2026-11-10T12:00:00.100Z",
        trigger_event=event,
        pre_trigger_telemetry=[snapshot],
        post_trigger_telemetry=[snapshot],
        delta_v_mps=3.2,
        peak_accel_g=2.85,
        tamper_chain_digest="abc123" * 10,
        is_locked=True,
    )

    report = DSSADReport(
        report_id="REP-20261110-001",
        vehicle_id="AVP-TEST-01",
        generated_at="2026-11-10T12:05:00Z",
        total_events_recorded=1,
        total_incidents_frozen=1,
        chain_verified=True,
        events=[event],
        incidents=[incident],
    )

    assert report.total_events_recorded == 1
    assert report.incidents[0].peak_accel_g == 2.85

    verification = ChainVerificationResult(
        valid=True,
        total_blocks=10,
    )
    assert verification.valid is True
    assert verification.corrupted_index is None
