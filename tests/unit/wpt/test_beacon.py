"""Unit tests for magnetic beacon differential sensing and coil offset estimator."""

from nearfield360.wpt.beacon import MagneticBeaconEstimator
from nearfield360.wpt.models import WPTAlignmentStatus


def test_symmetric_voltages_zero_offset() -> None:
    estimator = MagneticBeaconEstimator(sensitivity_mv_per_mm=10.0)
    # Balanced 850 mV across all quadrants
    offset = estimator.estimate_from_voltages(
        v_front_mv=850.0,
        v_rear_mv=850.0,
        v_left_mv=850.0,
        v_right_mv=850.0,
        v_ref_mv=850.0,
    )
    assert offset.delta_x_mm == 0.0
    assert offset.delta_y_mm == 0.0
    assert offset.delta_theta_deg == 0.0
    assert abs(offset.delta_z_mm - 170.0) < 1.0
    assert offset.alignment_status == WPTAlignmentStatus.DOCKED
    assert offset.is_within_tolerance is True


def test_differential_voltages_offsets() -> None:
    estimator = MagneticBeaconEstimator(sensitivity_mv_per_mm=10.0)
    # v_front - v_rear = 500 mV -> delta_x = 50 mm
    # v_left - v_right = -300 mV -> delta_y = -30 mm
    offset = estimator.estimate_from_voltages(
        v_front_mv=1100.0,
        v_rear_mv=600.0,
        v_left_mv=700.0,
        v_right_mv=1000.0,
        v_ref_mv=850.0,
    )
    assert offset.delta_x_mm == 50.0
    assert offset.delta_y_mm == -30.0
    assert offset.alignment_status == WPTAlignmentStatus.ALIGNED
    assert offset.is_within_tolerance is True


def test_alignment_status_categories() -> None:
    estimator = MagneticBeaconEstimator()

    # DOCKED: delta_r <= 30 mm
    d_docked = estimator.compute_status(
        delta_x_mm=15.0, delta_y_mm=10.0, delta_z_mm=170.0, delta_theta_deg=0.5
    )
    assert d_docked.alignment_status == WPTAlignmentStatus.DOCKED
    assert d_docked.is_within_tolerance is True

    # ALIGNED: within SAE J2954 box (dx <= 100 mm, dy <= 75 mm)
    d_aligned = estimator.compute_status(
        delta_x_mm=85.0, delta_y_mm=60.0, delta_z_mm=170.0, delta_theta_deg=2.0
    )
    assert d_aligned.alignment_status == WPTAlignmentStatus.ALIGNED
    assert d_aligned.is_within_tolerance is True

    # APPROACHING: radial offset <= 400 mm
    d_appr = estimator.compute_status(
        delta_x_mm=250.0, delta_y_mm=150.0, delta_z_mm=170.0, delta_theta_deg=1.0
    )
    assert d_appr.alignment_status == WPTAlignmentStatus.APPROACHING
    assert d_appr.is_within_tolerance is False

    # MISALIGNED: radial offset <= 1500 mm
    d_mis = estimator.compute_status(
        delta_x_mm=600.0, delta_y_mm=500.0, delta_z_mm=170.0, delta_theta_deg=0.0
    )
    assert d_mis.alignment_status == WPTAlignmentStatus.MISALIGNED

    # OUT_OF_RANGE: radial offset > 1500 mm
    d_out = estimator.compute_status(
        delta_x_mm=1800.0, delta_y_mm=100.0, delta_z_mm=170.0, delta_theta_deg=0.0
    )
    assert d_out.alignment_status == WPTAlignmentStatus.OUT_OF_RANGE
    assert d_out.confidence <= 0.1
