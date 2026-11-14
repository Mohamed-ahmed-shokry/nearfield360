"""Tests for SAE J2954 WPT coupling factor and electrical transfer efficiency model."""

from __future__ import annotations

import pytest

from nearfield360.config import WPTConfig
from nearfield360.wpt.efficiency import WPTEfficiencyModel
from nearfield360.wpt.models import (
    CoilOffset3D,
    FODStatus,
    WPTAlignmentStatus,
    WPTInterlockState,
    WPTPowerClass,
    WPTZClass,
)


def test_coupling_factor_nominal_and_distance_decay() -> None:
    """Verify coupling factor k matches nominal at origin and decays with radial/z offset."""
    model = WPTEfficiencyModel()
    k_nominal = model.compute_coupling_factor(delta_r_mm=0.0, air_gap_mm=170.0)
    assert 0.20 <= k_nominal <= 0.35

    # Radial misalignment reduces k
    k_offset = model.compute_coupling_factor(delta_r_mm=100.0, air_gap_mm=170.0)
    assert k_offset < k_nominal

    # Larger air gap reduces k
    k_high_z = model.compute_coupling_factor(delta_r_mm=0.0, air_gap_mm=230.0)
    assert k_high_z < k_nominal


def test_transfer_efficiency_curve() -> None:
    """Verify transfer efficiency matches resonant inductive formulas and exceeds 85% at nominal."""
    model = WPTEfficiencyModel()
    k_nominal = model.compute_coupling_factor(delta_r_mm=15.0, air_gap_mm=165.0)
    eff = model.compute_transfer_efficiency(k_nominal)

    # Efficiency should be >= 85% for nominal coupling
    assert eff >= 0.85
    assert eff <= 0.98

    # Low coupling drops efficiency
    eff_low = model.compute_transfer_efficiency(coupling_k=0.005)
    assert eff_low < 0.60


def test_power_governor_wpt3_and_docked_full_power() -> None:
    """Verify power governor delivers full 11 kW when docked and throttles on misalignment."""
    config = WPTConfig(max_power_kw=11.0)
    model = WPTEfficiencyModel(config=config)

    # Docked offset (within 30 mm threshold)
    offset_docked = CoilOffset3D(
        delta_x_mm=10.0,
        delta_y_mm=10.0,
        delta_z_mm=170.0,
        radial_offset_mm=14.14,
        delta_theta_deg=0.5,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.DOCKED,
    )
    rep = model.evaluate(offset=offset_docked, power_class=WPTPowerClass.WPT3)
    assert rep.commanded_power_kw == pytest.approx(11.0, abs=0.1)
    assert rep.delivered_power_kw > 9.0
    assert rep.z_class == WPTZClass.Z2

    # Offset within tolerance but not docked
    offset_tolerance = CoilOffset3D(
        delta_x_mm=80.0,
        delta_y_mm=50.0,
        delta_z_mm=170.0,
        radial_offset_mm=94.34,
        delta_theta_deg=2.0,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.ALIGNED,
    )
    rep_tol = model.evaluate(offset=offset_tolerance, power_class=WPTPowerClass.WPT3)
    assert rep_tol.commanded_power_kw < rep.commanded_power_kw
    assert rep_tol.commanded_power_kw >= 5.5

    # Out of tolerance -> beacon probing power (0.5 kW)
    offset_out = CoilOffset3D(
        delta_x_mm=180.0,
        delta_y_mm=120.0,
        delta_z_mm=170.0,
        radial_offset_mm=216.33,
        delta_theta_deg=5.0,
        is_within_tolerance=False,
        alignment_status=WPTAlignmentStatus.MISALIGNED,
    )
    rep_out = model.evaluate(offset=offset_out, power_class=WPTPowerClass.WPT3)
    assert rep_out.commanded_power_kw == 0.5


def test_power_governor_trips_to_zero_on_fod_fault() -> None:
    """Verify power output is strictly 0.0 kW whenever FOD trips interlock."""
    model = WPTEfficiencyModel()
    offset_docked = CoilOffset3D(
        delta_x_mm=0.0,
        delta_y_mm=0.0,
        delta_z_mm=170.0,
        radial_offset_mm=0.0,
        delta_theta_deg=0.0,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.DOCKED,
    )
    fod = FODStatus(
        metal_object_detected=True,
        living_object_detected=False,
        q_drop_percent=28.5,
        frequency_shift_khz=2.4,
        interlock_state=WPTInterlockState.FOD_FAULT,
        power_inhibited=True,
        reason="Resonant detuning exceeded safety limit",
    )
    rep = model.evaluate(offset=offset_docked, fod_status=fod, power_class=WPTPowerClass.WPT3)
    assert rep.commanded_power_kw == 0.0
    assert rep.delivered_power_kw == 0.0
    assert rep.thermal_loss_kw == 0.0


def test_z_class_classification() -> None:
    """Verify Z1, Z2, Z3 classification based on air gap."""
    model = WPTEfficiencyModel()
    z1_offset = CoilOffset3D(
        delta_x_mm=0.0,
        delta_y_mm=0.0,
        delta_z_mm=120.0,
        radial_offset_mm=0.0,
        delta_theta_deg=0.0,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.DOCKED,
    )
    z3_offset = CoilOffset3D(
        delta_x_mm=0.0,
        delta_y_mm=0.0,
        delta_z_mm=240.0,
        radial_offset_mm=0.0,
        delta_theta_deg=0.0,
        is_within_tolerance=True,
        alignment_status=WPTAlignmentStatus.DOCKED,
    )

    assert model.evaluate(z1_offset).z_class == WPTZClass.Z1
    assert model.evaluate(z3_offset).z_class == WPTZClass.Z3
