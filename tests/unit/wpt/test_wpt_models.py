"""Unit tests for SAE J2954 WPT domain models, enums, and schemas."""

import json

import pytest
from pydantic import ValidationError

from nearfield360.wpt.models import (
    CoilGeometry,
    CoilOffset3D,
    FODStatus,
    WPTAlignmentStatus,
    WPTDockingStep,
    WPTEfficiencyReport,
    WPTInterlockState,
    WPTPowerClass,
    WPTReport,
    WPTZClass,
)


def test_wpt_power_and_z_class_enums() -> None:
    assert WPTPowerClass.WPT1 == "WPT1"
    assert WPTPowerClass.WPT2 == "WPT2"
    assert WPTPowerClass.WPT3 == "WPT3"
    assert WPTPowerClass.WPT4 == "WPT4"

    assert WPTZClass.Z1 == "Z1"
    assert WPTZClass.Z2 == "Z2"
    assert WPTZClass.Z3 == "Z3"

    assert WPTAlignmentStatus.ALIGNED == "ALIGNED"
    assert WPTAlignmentStatus.DOCKED == "DOCKED"
    assert WPTAlignmentStatus.MISALIGNED == "MISALIGNED"

    assert WPTInterlockState.NORMAL == "NORMAL"
    assert WPTInterlockState.FOD_FAULT == "FOD_FAULT"
    assert WPTInterlockState.LOD_FAULT == "LOD_FAULT"


def test_coil_geometry_validation() -> None:
    geom = CoilGeometry(length_mm=700.0, width_mm=550.0, num_turns=14, self_inductance_uh=82.0)
    assert geom.length_mm == 700.0
    assert geom.num_turns == 14

    with pytest.raises(ValidationError):
        CoilGeometry(length_mm=50.0)  # Too small


def test_coil_offset_3d() -> None:
    offset = CoilOffset3D(
        delta_x_mm=45.0,
        delta_y_mm=-25.0,
        delta_z_mm=165.0,
        delta_theta_deg=1.2,
        radial_offset_mm=51.48,
        alignment_status=WPTAlignmentStatus.ALIGNED,
        is_within_tolerance=True,
        confidence=0.95,
        timestamp_ns=1700000000000000,
    )
    assert offset.is_within_tolerance is True
    assert offset.alignment_status == WPTAlignmentStatus.ALIGNED
    assert offset.delta_z_mm == 165.0

    dump = offset.model_dump(mode="json")
    assert dump["alignment_status"] == "ALIGNED"
    assert dump["delta_x_mm"] == 45.0


def test_fod_status_serialization() -> None:
    fod = FODStatus(
        metal_object_detected=True,
        frequency_shift_khz=2.4,
        q_drop_percent=22.5,
        living_object_detected=False,
        interlock_state=WPTInterlockState.FOD_FAULT,
        power_inhibited=True,
        reason="Metallic debris detected on primary pad",
    )
    assert fod.metal_object_detected is True
    assert fod.power_inhibited is True
    assert fod.interlock_state == WPTInterlockState.FOD_FAULT


def test_efficiency_report() -> None:
    eff = WPTEfficiencyReport(
        coupling_factor_k=0.21,
        transfer_efficiency=0.895,
        commanded_power_kw=11.0,
        delivered_power_kw=9.845,
        thermal_loss_kw=1.155,
        power_class=WPTPowerClass.WPT3,
        z_class=WPTZClass.Z2,
    )
    assert eff.transfer_efficiency == 0.895
    assert eff.delivered_power_kw == 9.845


def test_docking_step_and_report_round_trip() -> None:
    offset = CoilOffset3D(
        delta_x_mm=10.0,
        delta_y_mm=5.0,
        delta_z_mm=160.0,
        delta_theta_deg=0.5,
        radial_offset_mm=11.18,
        alignment_status=WPTAlignmentStatus.DOCKED,
        is_within_tolerance=True,
    )
    step = WPTDockingStep(
        step_index=0,
        vehicle_x_m=0.01,
        vehicle_y_m=0.005,
        heading_deg=0.5,
        velocity_mps=0.04,
        steering_angle_deg=0.2,
        coil_offset=offset,
        docked=True,
    )
    fod = FODStatus(
        metal_object_detected=False,
        frequency_shift_khz=0.1,
        q_drop_percent=1.2,
        living_object_detected=False,
        interlock_state=WPTInterlockState.NORMAL,
        power_inhibited=False,
        reason="Clean primary pad and clear chassis perimeter",
    )
    eff = WPTEfficiencyReport(
        coupling_factor_k=0.24,
        transfer_efficiency=0.91,
        commanded_power_kw=11.0,
        delivered_power_kw=10.01,
        thermal_loss_kw=0.99,
    )
    report = WPTReport(
        vehicle_id="veh-porsche-taycan-avp",
        pad_id="ga-pad-bay-07",
        power_class=WPTPowerClass.WPT3,
        z_class=WPTZClass.Z2,
        offset=offset,
        fod=fod,
        efficiency=eff,
        docking_history=[step],
    )

    data = report.to_dict()
    assert data["pad_id"] == "ga-pad-bay-07"
    assert data["offset"]["alignment_status"] == "DOCKED"
    assert len(data["docking_history"]) == 1

    json_str = json.dumps(data)
    assert "ga-pad-bay-07" in json_str
