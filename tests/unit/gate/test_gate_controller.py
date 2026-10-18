"""Unit tests for BarrierInterlockController, loop sequencing, and anti-tailgating."""

from __future__ import annotations

from nearfield360.config import GateConfig
from nearfield360.gate.controller import BarrierInterlockController
from nearfield360.gate.envelope import VehicleEnvelopeProfiler
from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    CredentialType,
    GateAccessRequest,
    InductiveLoopId,
    VehicleCredential,
    VehicleDimensionEnvelope,
)
from nearfield360.gate.validator import AccessCredentialValidator


def _build_controller() -> BarrierInterlockController:
    config = GateConfig(
        portal_clearance_height_m=2.10,
        portal_clearance_width_m=2.60,
        ground_clearance_min_m=0.12,
        barrier_opening_time_s=1.0,
        barrier_closing_time_s=1.0,
        alpr_confidence_threshold=0.85,
    )
    cred = VehicleCredential(
        credential_id="cred-valid-1",
        credential_type=CredentialType.ALPR_PLATE,
        plate_number="CALIF77",
        valid_from_s=0.0,
        valid_until_s=1000.0,
        tenant_id="AIRPORT_T1",
    )
    validator = AccessCredentialValidator(config, initial_credentials=[cred])
    profiler = VehicleEnvelopeProfiler(config)
    return BarrierInterlockController(config, validator, profiler, gate_id="GATE_T1_ENTRY")


def test_controller_initial_state() -> None:
    controller = _build_controller()
    assert controller.gate_id == "GATE_T1_ENTRY"
    assert controller.barrier.state.value == BarrierArmState.LOWERED_CLOSED.value
    assert controller.barrier.barrier_angle_deg == 0.0
    assert controller.tailgating_alarm_active is False
    assert controller.clearance_violations_count == 0
    assert controller.tailgating_incidents_count == 0

    for loop_id in InductiveLoopId:
        assert controller.get_loop(loop_id).occupied is False


def test_controller_loop_state_and_dwell() -> None:
    controller = _build_controller()
    controller.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, True, timestamp_s=10.0)
    loop = controller.get_loop(InductiveLoopId.LOOP_A_APPROACH)
    assert loop.occupied is True
    assert loop.activated_at_s == 10.0

    controller.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, True, timestamp_s=12.5)
    assert loop.dwell_duration_s == 2.5

    controller.set_loop_state(InductiveLoopId.LOOP_A_APPROACH, False, timestamp_s=14.0)
    assert loop.occupied is False


def test_controller_safety_loop_reversal_interlock() -> None:
    controller = _build_controller()
    controller.command_open()
    # Step until fully open
    for _ in range(20):
        controller.step_kinematics(0.1)
    assert str(controller.barrier.state) == str(BarrierArmState.RAISED_OPEN)
    assert controller.barrier.barrier_angle_deg == 90.0

    # Start closing
    closed_initiated = controller.command_close()
    assert closed_initiated is True
    assert str(controller.barrier.state) == str(BarrierArmState.CLOSING)

    # Step partially downward
    controller.step_kinematics(0.4)
    assert controller.barrier.barrier_angle_deg < 90.0

    # Obstacle triggers Loop C under closing arm!
    controller.set_loop_state(InductiveLoopId.LOOP_C_SAFETY, True, timestamp_s=5.0)
    assert str(controller.barrier.state) == str(BarrierArmState.SAFETY_REVERSING)
    assert controller.barrier.safety_interlock_active is True
    assert controller.barrier.obstruction_detected is True
    assert controller.barrier.angular_velocity_deg_s > 0.0

    # Step until arm safely returns to 90 degrees
    for _ in range(20):
        controller.step_kinematics(0.1)
    assert str(controller.barrier.state) == str(BarrierArmState.RAISED_OPEN)
    assert controller.barrier.barrier_angle_deg == 90.0


def test_controller_anti_tailgating_detection() -> None:
    controller = _build_controller()
    controller.command_open()
    for _ in range(20):
        controller.step_kinematics(0.1)

    # Barrier is raised open, but NO vehicle is authorized!
    # A tailgater triggers Loop B
    controller.set_loop_state(InductiveLoopId.LOOP_B_PRESENCE, True, timestamp_s=1.0)
    assert controller.tailgating_alarm_active is True
    assert controller.tailgating_incidents_count == 1

    controller.reset_tailgating_alarm()
    assert controller.tailgating_alarm_active is False


def test_controller_process_access_request_rejection_paths() -> None:
    controller = _build_controller()

    # Rejection 1: Over-height vehicle
    bad_envelope = VehicleDimensionEnvelope(length_m=5.0, width_m=2.0, height_m=2.4)
    req_bad_height = GateAccessRequest(
        request_id="req-over-h",
        gate_id="GATE_T1_ENTRY",
        timestamp_s=1.0,
        vehicle_id="veh-tall",
        detected_plate="CALIF77",
        plate_confidence=0.95,
        envelope=bad_envelope,
    )
    dec_height = controller.process_access_request(req_bad_height)
    assert dec_height.granted is False
    assert dec_height.status == AccessDecisionStatus.REJECTED_OVER_HEIGHT
    assert controller.clearance_violations_count == 1
    assert controller.barrier.state.value == BarrierArmState.LOWERED_CLOSED.value

    # Rejection 2: Unregistered plate
    good_envelope = VehicleDimensionEnvelope(length_m=4.5, width_m=1.8, height_m=1.5)
    req_bad_plate = GateAccessRequest(
        request_id="req-unknown-p",
        gate_id="GATE_T1_ENTRY",
        timestamp_s=2.0,
        vehicle_id="veh-stranger",
        detected_plate="STRANGER1",
        plate_confidence=0.95,
        envelope=good_envelope,
    )
    dec_plate = controller.process_access_request(req_bad_plate)
    assert dec_plate.granted is False
    assert dec_plate.status == AccessDecisionStatus.REJECTED_UNKNOWN_CREDENTIAL
    assert controller.barrier.state.value == BarrierArmState.LOWERED_CLOSED.value


def test_controller_simulate_transit_happy_path_and_audit() -> None:
    controller = _build_controller()
    good_envelope = VehicleDimensionEnvelope(length_m=4.6, width_m=1.85, height_m=1.5)
    valid_req = GateAccessRequest(
        request_id="req-pass-1",
        gate_id="GATE_T1_ENTRY",
        timestamp_s=10.0,
        vehicle_id="veh-vip-1",
        detected_plate="CALIF77",
        plate_confidence=0.95,
        envelope=good_envelope,
    )

    event = controller.simulate_transit(valid_req)
    assert event.decision.granted is True
    assert event.duration_s > 0.0
    assert event.tailgating_detected is False
    assert controller.barrier.state.value == BarrierArmState.LOWERED_CLOSED.value

    report = controller.generate_audit_report()
    assert report.total_transits == 1
    assert report.granted_count == 1
    assert report.rejected_count == 0
    assert report.average_transit_duration_s > 0.0
