"""Unit tests for VehicleEnvelopeProfiler and physical portal clearance verification."""

from __future__ import annotations

from nearfield360.config import GateConfig
from nearfield360.gate.envelope import VehicleEnvelopeProfiler
from nearfield360.gate.models import AccessDecisionStatus, VehicleDimensionEnvelope


def test_envelope_profiler_grants_nominal_vehicle() -> None:
    config = GateConfig(
        portal_clearance_height_m=2.10,
        portal_clearance_width_m=2.60,
        ground_clearance_min_m=0.12,
    )
    profiler = VehicleEnvelopeProfiler(config)

    # Standard passenger sedan
    envelope = VehicleDimensionEnvelope(
        length_m=4.75,
        width_m=1.85,
        height_m=1.50,
        ground_clearance_m=0.15,
    )
    decision = profiler.evaluate_clearance(envelope, gate_id="GATE_NORTH", request_id="req-1")

    assert decision.granted is True
    assert decision.status == AccessDecisionStatus.GRANTED
    assert decision.headroom_margin_m == round(2.10 - 1.50, 3)
    assert decision.width_margin_m == round(2.60 - 1.85, 3)
    assert "Clearance verified" in decision.message
    assert decision.issued_pass_id.startswith("PASS-CLEARANCE-GATE_NORTH")


def test_envelope_profiler_rejects_over_height_vehicle() -> None:
    config = GateConfig(portal_clearance_height_m=2.10)
    profiler = VehicleEnvelopeProfiler(config)

    # High roof van or SUV with luggage pod
    envelope = VehicleDimensionEnvelope(
        length_m=5.20,
        width_m=2.00,
        height_m=2.35,  # 25cm over height limit
        ground_clearance_m=0.20,
    )
    decision = profiler.evaluate_clearance(envelope, request_id="req-over-height")

    assert decision.granted is False
    assert decision.status == AccessDecisionStatus.REJECTED_OVER_HEIGHT
    assert decision.headroom_margin_m < 0.0
    assert "exceeds portal limit" in decision.message
    assert "0.25m" in decision.message


def test_envelope_profiler_rejects_over_width_vehicle() -> None:
    config = GateConfig(portal_clearance_width_m=2.60)
    profiler = VehicleEnvelopeProfiler(config)

    # Wide commercial pickup or dually
    envelope = VehicleDimensionEnvelope(
        length_m=5.80,
        width_m=2.75,  # 15cm over width limit
        height_m=1.90,
        ground_clearance_m=0.22,
    )
    decision = profiler.evaluate_clearance(envelope, request_id="req-over-width")

    assert decision.granted is False
    assert decision.status == AccessDecisionStatus.REJECTED_OVER_WIDTH
    assert decision.width_margin_m < 0.0
    assert "exceeds portal limit" in decision.message
    assert "0.15m" in decision.message


def test_envelope_profiler_rejects_low_ground_clearance() -> None:
    config = GateConfig(ground_clearance_min_m=0.12)
    profiler = VehicleEnvelopeProfiler(config)

    # Low-slung track sports car
    envelope = VehicleDimensionEnvelope(
        length_m=4.50,
        width_m=1.95,
        height_m=1.20,
        ground_clearance_m=0.07,  # 5cm below required minimum
    )
    decision = profiler.evaluate_clearance(envelope, request_id="req-low-clearance")

    assert decision.granted is False
    assert decision.status == AccessDecisionStatus.REJECTED_LOW_GROUND_CLEARANCE
    assert "below required" in decision.message
    assert "0.05m" in decision.message


def test_envelope_profiler_exact_boundary_clearance() -> None:
    config = GateConfig(
        portal_clearance_height_m=2.10,
        portal_clearance_width_m=2.60,
        ground_clearance_min_m=0.12,
    )
    profiler = VehicleEnvelopeProfiler(config)

    # Vehicle matching boundaries exactly
    envelope = VehicleDimensionEnvelope(
        length_m=5.00,
        width_m=2.60,
        height_m=2.10,
        ground_clearance_m=0.12,
    )
    decision = profiler.evaluate_clearance(envelope, request_id="req-boundary")

    assert decision.granted is True
    assert decision.status == AccessDecisionStatus.GRANTED
    assert decision.headroom_margin_m == 0.0
    assert decision.width_margin_m == 0.0
