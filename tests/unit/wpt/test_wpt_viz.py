"""Tests for SAE J2954 WPT tactical 1280x720 HUD dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nearfield360.wpt.models import (
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
from nearfield360.wpt.viz import render_wpt_dashboard


def _create_sample_report(
    status: WPTAlignmentStatus = WPTAlignmentStatus.DOCKED,
    interlock: WPTInterlockState = WPTInterlockState.NORMAL,
) -> WPTReport:
    """Helper creating representative WPTReport fixture."""
    offset = CoilOffset3D(
        delta_x_mm=12.0,
        delta_y_mm=-8.0,
        delta_z_mm=168.0,
        radial_offset_mm=14.42,
        delta_theta_deg=0.4,
        is_within_tolerance=True,
        alignment_status=status,
    )
    fod = FODStatus(
        metal_object_detected=(interlock == WPTInterlockState.FOD_FAULT),
        frequency_shift_khz=0.15 if interlock == WPTInterlockState.NORMAL else 2.8,
        q_drop_percent=4.2 if interlock == WPTInterlockState.NORMAL else 26.0,
        living_object_detected=False,
        interlock_state=interlock,
        power_inhibited=(interlock != WPTInterlockState.NORMAL),
        reason="Coil magnetic flux nominal"
        if interlock == WPTInterlockState.NORMAL
        else "FOD metal trip",
    )
    eff = WPTEfficiencyReport(
        coupling_factor_k=0.255,
        transfer_efficiency=0.912,
        commanded_power_kw=11.0 if interlock == WPTInterlockState.NORMAL else 0.0,
        delivered_power_kw=10.03 if interlock == WPTInterlockState.NORMAL else 0.0,
        thermal_loss_kw=0.97 if interlock == WPTInterlockState.NORMAL else 0.0,
        power_class=WPTPowerClass.WPT3,
        z_class=WPTZClass.Z2,
    )
    step1 = WPTDockingStep(
        step_index=0,
        vehicle_x_m=0.20,
        vehicle_y_m=0.04,
        heading_deg=1.5,
        velocity_mps=0.08,
        steering_angle_deg=-1.5,
        coil_offset=offset,
        docked=False,
    )
    step2 = WPTDockingStep(
        step_index=1,
        vehicle_x_m=0.012,
        vehicle_y_m=-0.008,
        heading_deg=0.4,
        velocity_mps=0.0,
        steering_angle_deg=0.0,
        coil_offset=offset,
        docked=True,
    )

    return WPTReport(
        vehicle_id="EV-TEST-904",
        pad_id="PAD-BAY-03",
        power_class=WPTPowerClass.WPT3,
        z_class=WPTZClass.Z2,
        offset=offset,
        fod=fod,
        efficiency=eff,
        docking_history=[step1, step2],
    )


def test_render_wpt_dashboard_dimensions_and_dtype() -> None:
    """Verify render_wpt_dashboard returns exactly 1280x720 uint8 BGR canvas."""
    rep = _create_sample_report()
    canvas = render_wpt_dashboard(rep)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8
    # Canvas should not be empty/all-zero
    assert np.any(canvas > 0)


def test_render_wpt_dashboard_file_saving(tmp_path: Path) -> None:
    """Verify visualizer writes valid PNG image to disk and can be reloaded."""
    rep = _create_sample_report()
    out_file = tmp_path / "wpt_dashboard.png"

    canvas = render_wpt_dashboard(rep, output_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000

    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == (720, 1280, 3)
    assert np.array_equal(canvas, loaded)


def test_render_wpt_dashboard_fault_state() -> None:
    """Verify visualizer handles faulted interlock and empty trajectory history."""
    rep = _create_sample_report(
        status=WPTAlignmentStatus.MISALIGNED,
        interlock=WPTInterlockState.FOD_FAULT,
    )
    rep = WPTReport(
        vehicle_id=rep.vehicle_id,
        pad_id=rep.pad_id,
        power_class=rep.power_class,
        z_class=rep.z_class,
        offset=rep.offset,
        fod=rep.fod,
        efficiency=rep.efficiency,
        docking_history=[],
    )

    canvas = render_wpt_dashboard(rep)
    assert canvas.shape == (720, 1280, 3)
    assert np.any(canvas > 0)
