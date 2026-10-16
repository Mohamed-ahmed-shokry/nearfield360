"""Unit tests for ramp dynamics, pitch forces, spiral kinematics, and breakover clearances."""

import pytest

from nearfield360.multilevel.ramp import RampDynamicsEngine


def test_pitch_dynamics_standard_ramp() -> None:
    engine = RampDynamicsEngine(
        default_vehicle_mass_kg=1800.0,
        default_wheelbase_m=2.8,
        default_ground_clearance_m=0.14,
        default_surface_friction=0.80,
    )

    state = engine.calculate_pitch_dynamics(
        vehicle_id="veh-test-01",
        grade_pct=12.0,
        elevation_m=1.75,
    )

    assert state.vehicle_id == "veh-test-01"
    assert state.grade_pct == 12.0
    # arctan(0.12) is approx 6.84 degrees
    assert state.pitch_angle_deg == pytest.approx(6.84, abs=0.1)
    # Gravity force component ~ 1800 * 9.81 * sin(6.84 deg) ~ 2100 N
    assert 2000.0 < state.gravity_force_n < 2200.0
    # Normal force ~ 1800 * 9.81 * cos(6.84 deg) ~ 17500 N
    assert 17400.0 < state.normal_force_n < 17650.0
    # Traction margin = 0.80 - 0.12 = 0.68
    assert state.traction_margin == pytest.approx(0.68, abs=0.01)
    # Breakover clearance remains positive and safe
    assert state.breakover_clearance_m > 0.05
    assert state.is_clearance_safe is True


def test_steep_ramp_clearance_violation() -> None:
    engine = RampDynamicsEngine(
        default_vehicle_mass_kg=2200.0,
        default_wheelbase_m=3.4,  # Long executive limousine wheelbase
        default_ground_clearance_m=0.08,  # Very low sports suspension clearance
        default_surface_friction=0.30,  # Slippery/wet surface
    )

    state = engine.calculate_pitch_dynamics(
        vehicle_id="veh-limo",
        grade_pct=25.0,  # Extreme steep ramp
        elevation_m=3.5,
        min_clearance_buffer_m=0.06,
    )

    assert state.grade_pct == 25.0
    # Clearance buffer breached or traction margin under threshold
    assert state.is_clearance_safe is False


def test_spiral_ramp_kinematics() -> None:
    engine = RampDynamicsEngine()

    result = engine.calculate_spiral_kinematics(
        radius_m=12.0,
        velocity_mps=2.5,
        grade_pct=10.0,
        max_lat_accel_mps2=2.0,
    )

    assert result["radius_m"] == 12.0
    assert result["velocity_mps"] == 2.5
    assert result["yaw_rate_radps"] > 0.15
    assert result["lateral_accel_mps2"] < 2.0
    assert result["vertical_speed_mps"] > 0.1
    assert result["is_speed_safe"] is True

    # High speed on tight spiral should exceed lateral comfort limit
    fast_result = engine.calculate_spiral_kinematics(
        radius_m=10.0,
        velocity_mps=8.0,
        grade_pct=10.0,
        max_lat_accel_mps2=2.0,
    )
    assert fast_result["is_speed_safe"] is False

    # Non-positive radius should raise ValueError
    with pytest.raises(ValueError, match="radius must be positive"):
        engine.calculate_spiral_kinematics(radius_m=-5.0, velocity_mps=2.0, grade_pct=5.0)


def test_check_breakover_clearance_angles() -> None:
    engine = RampDynamicsEngine(default_wheelbase_m=2.8, default_ground_clearance_m=0.15)

    # Gentle 5-degree grade transition
    clearance, is_safe = engine.check_breakover_clearance(grade_change_deg=5.0, min_buffer_m=0.05)
    assert is_safe is True
    assert clearance > 0.08

    # Severe 20-degree grade breakover crest
    severe_clearance, severe_safe = engine.check_breakover_clearance(
        grade_change_deg=20.0, min_buffer_m=0.05
    )
    assert severe_clearance < clearance
    assert severe_safe is True or severe_clearance < 0.05
