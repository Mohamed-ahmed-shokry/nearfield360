"""Unit tests for tactical gateway BEV dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.gate.models import (
    AccessDecisionStatus,
    BarrierArmState,
    BarrierKinematicsState,
    GateAccessDecision,
    GateAccessRequest,
    GateAuditReport,
    InductiveLoopId,
    InductiveLoopState,
    VehicleDimensionEnvelope,
)
from nearfield360.gate.viz import render_gate_dashboard


def test_render_gate_dashboard_basic_array(tmp_path: Path) -> None:
    barrier = BarrierKinematicsState(
        barrier_angle_deg=90.0,
        angular_velocity_deg_s=0.0,
        state=BarrierArmState.RAISED_OPEN,
    )
    loops = {
        lid: InductiveLoopState(loop_id=lid, occupied=(lid == InductiveLoopId.LOOP_D_DEPARTURE))
        for lid in InductiveLoopId
    }
    decision = GateAccessDecision(
        decision_id="dec-1",
        request_id="req-1",
        status=AccessDecisionStatus.GRANTED,
        granted=True,
        message="Access authorized for VIP",
        headroom_margin_m=0.55,
        width_margin_m=0.75,
        issued_pass_id="PASS-9988",  # noqa: S106
    )
    request = GateAccessRequest(
        request_id="req-1",
        gate_id="GATE_NORTH",
        vehicle_id="veh-vip",
        detected_plate="VIP999",
        plate_confidence=0.98,
        envelope=VehicleDimensionEnvelope(length_m=4.8, width_m=1.85, height_m=1.55),
    )
    audit = GateAuditReport(
        gate_id="GATE_NORTH",
        total_transits=25,
        granted_count=24,
        rejected_count=1,
        tailgating_incidents=0,
        clearance_violations=0,
        average_transit_duration_s=4.2,
    )

    out_file = tmp_path / "gate_dashboard.png"
    img = render_gate_dashboard(
        "GATE_NORTH",
        barrier,
        loops,
        decision=decision,
        request=request,
        audit=audit,
        tailgating_active=False,
        output_path=out_file,
    )

    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1280, 3)
    assert img.dtype == np.uint8

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000
    # Verify PNG magic header
    with out_file.open("rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n"


def test_render_gate_dashboard_tailgating_and_reversal(tmp_path: Path) -> None:
    barrier = BarrierKinematicsState(
        barrier_angle_deg=45.0,
        angular_velocity_deg_s=40.0,
        state=BarrierArmState.SAFETY_REVERSING,
        safety_interlock_active=True,
        obstruction_detected=True,
    )
    loops = {
        lid: InductiveLoopState(loop_id=lid, occupied=(lid == InductiveLoopId.LOOP_C_SAFETY))
        for lid in InductiveLoopId
    }
    decision = GateAccessDecision(
        decision_id="dec-2",
        request_id="req-2",
        status=AccessDecisionStatus.REJECTED_OVER_HEIGHT,
        granted=False,
        message="Vehicle height exceeds portal limit",
        headroom_margin_m=-0.25,
        width_margin_m=0.30,
    )

    out_file = tmp_path / "gate_tailgate_alert.png"
    img = render_gate_dashboard(
        "GATE_SOUTH",
        barrier,
        loops,
        decision=decision,
        tailgating_active=True,
        output_path=out_file,
    )

    assert img.shape == (720, 1280, 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 5000


def test_render_gate_dashboard_standby_idle() -> None:
    barrier = BarrierKinematicsState(
        barrier_angle_deg=0.0,
        angular_velocity_deg_s=0.0,
        state=BarrierArmState.LOWERED_CLOSED,
    )
    loops = {lid: InductiveLoopState(loop_id=lid, occupied=False) for lid in InductiveLoopId}

    img = render_gate_dashboard("GATE_STANDBY", barrier, loops)
    assert img.shape == (720, 1280, 3)
