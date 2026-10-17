"""Unit tests for facility access control and gateway domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    BarrierKinematicsState,
    CredentialType,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    GateDirection,
    GateTransitEvent,
    InductiveLoopId,
    InductiveLoopState,
    VehicleCredential,
    VehicleDimensionEnvelope,
)


def test_enums_values() -> None:
    assert GateDirection.ENTRY.value == "entry"
    assert GateDirection.EXIT.value == "exit"
    assert GateDirection.BIDIRECTIONAL.value == "bidirectional"

    assert BarrierArmState.LOWERED_CLOSED.value == "lowered_closed"
    assert BarrierArmState.OPENING.value == "opening"
    assert BarrierArmState.RAISED_OPEN.value == "raised_open"
    assert BarrierArmState.CLOSING.value == "closing"
    assert BarrierArmState.SAFETY_REVERSING.value == "safety_reversing"
    assert BarrierArmState.FAULT_LOCKED.value == "fault_locked"

    assert CredentialType.ALPR_PLATE.value == "alpr_plate"
    assert CredentialType.UWB_DIGITAL_KEY.value == "uwb_digital_key"
    assert CredentialType.RFID_BADGE.value == "rfid_badge"
    assert CredentialType.QR_BARCODE.value == "qr_barcode"
    assert CredentialType.V2X_DIRECT_AUTH.value == "v2x_direct_auth"

    assert AccessDecisionStatus.GRANTED.value == "granted"
    assert AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL.value == "rejected_unknown_credential"
    assert AccessDecisionStatus.REJECTED_OVER_HEIGHT.value == "rejected_over_height"
    assert AccessDecisionStatus.REJECTED_OVER_WIDTH.value == "rejected_over_width"
    assert (
        AccessDecisionStatus.REJECTED_LOW_GROUND_CLEARANCE.value == "rejected_low_ground_clearance"
    )
    assert AccessDecisionStatus.REJECTED_EXPIRED_RESERVATION.value == "rejected_expired_reservation"
    assert AccessDecisionStatus.TAILGATING_LOCKOUT.value == "tailgating_lockout"

    assert InductiveLoopId.LOOP_A_APPROACH.value == "loop_a_approach"
    assert InductiveLoopId.LOOP_B_PRESENCE.value == "loop_b_presence"
    assert InductiveLoopId.LOOP_C_SAFETY.value == "loop_c_safety"
    assert InductiveLoopId.LOOP_D_DEPARTURE.value == "loop_d_departure"


def test_vehicle_dimension_envelope() -> None:
    env = VehicleDimensionEnvelope(
        length_m=4.8,
        width_m=1.9,
        height_m=1.6,
        ground_clearance_m=0.14,
        front_overhang_m=0.8,
        rear_overhang_m=0.9,
    )
    assert env.length_m == 4.8
    assert env.width_m == 1.9
    assert env.height_m == 1.6
    assert env.ground_clearance_m == 0.14

    with pytest.raises(ValidationError):
        VehicleDimensionEnvelope(length_m=-1.0, width_m=1.9, height_m=1.6)

    with pytest.raises(ValidationError):
        VehicleDimensionEnvelope(length_m=4.8, width_m=5.5, height_m=1.6)


def test_vehicle_credential_serialization() -> None:
    cred = VehicleCredential(
        credential_id="cred-001",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="7XYZ890",
        token_hash="a1b2c3d4",  # noqa: S106
        valid_from_s=100.0,
        valid_until_s=5000.0,
        tenant_id="CORP_A",
        vip_status=True,
    )
    dumped = cred.model_dump()
    assert dumped["plate_number"] == "7XYZ890"
    assert dumped["vip_status"] is True

    json_str = cred.model_dump_json()
    loaded = VehicleCredential.model_validate_json(json_str)
    assert loaded == cred


def test_inductive_loop_state() -> None:
    loop = InductiveLoopState(
        loop_id=InductiveLoopId.LOOP_A_APPROACH,
        occupied=True,
        activated_at_s=12.5,
        dwell_duration_s=2.0,
    )
    assert loop.loop_id == InductiveLoopId.LOOP_A_APPROACH
    assert loop.occupied is True
    assert loop.dwell_duration_s == 2.0


def test_gate_access_request_and_decision() -> None:
    env = VehicleDimensionEnvelope(length_m=4.5, width_m=1.85, height_m=1.55)
    cred = VehicleCredential(
        credential_id="cred-100",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="5ABC123",
    )
    req = GateAccessRequest(
        request_id="req-1",
        gate_id="GATE_NORTH_ENTRY",
        timestamp_s=10.0,
        vehicle_id="veh-42",
        detected_plate="5ABC123",
        plate_confidence=0.96,
        credential=cred,
        envelope=env,
        approach_speed_mps=1.8,
    )
    assert req.request_id == "req-1"
    assert req.detected_plate == "5ABC123"

    decision = GateAccessDecision(
        decision_id="dec-1",
        request_id="req-1",
        status=AccessDecisionStatus.GRANTED,
        granted=True,
        message="Access authorized",
        authorized_at_s=10.2,
        headroom_margin_m=0.55,
        width_margin_m=0.75,
        issued_pass_id="PASS-9988",  # noqa: S106
    )
    assert decision.granted is True
    assert decision.headroom_margin_m == 0.55

    event = GateTransitEvent(
        event_id="evt-1",
        gate_id="GATE_NORTH_ENTRY",
        vehicle_id="veh-42",
        decision=decision,
        duration_s=4.5,
        tailgating_detected=False,
        timestamp_s=15.0,
    )
    assert event.event_id == "evt-1"
    assert event.duration_s == 4.5


def test_barrier_kinematics_and_audit_report() -> None:
    kin = BarrierKinematicsState(
        barrier_angle_deg=45.0,
        angular_velocity_deg_s=25.0,
        state=BarrierArmState.OPENING,
        safety_interlock_active=False,
        obstruction_detected=False,
    )
    assert kin.barrier_angle_deg == 45.0
    assert kin.state == BarrierArmState.OPENING

    report = GateAuditReport(
        gate_id="GATE_MAIN",
        total_transits=50,
        granted_count=48,
        rejected_count=2,
        tailgating_incidents=1,
        clearance_violations=1,
        average_transit_duration_s=4.2,
    )
    assert report.total_transits == 50
    assert report.rejected_count == 2
