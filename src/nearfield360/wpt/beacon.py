"""Magnetic beacon differential sensing and coil offset estimator for SAE J2954 WPT."""

import math

from nearfield360.config import WPTConfig
from nearfield360.wpt.models import CoilOffset3D, WPTAlignmentStatus


class MagneticBeaconEstimator:
    """Estimates 3D coil offset and alignment status from magnetic beacon sensor voltages."""

    def __init__(
        self,
        config: WPTConfig | None = None,
        sensitivity_mv_per_mm: float = 12.5,
        angular_sensitivity_mv_per_deg: float = 45.0,
    ) -> None:
        self.config = config or WPTConfig()
        self.sensitivity_mv_per_mm = sensitivity_mv_per_mm
        self.angular_sensitivity = angular_sensitivity_mv_per_deg

    def estimate_from_voltages(
        self,
        v_front_mv: float,
        v_rear_mv: float,
        v_left_mv: float,
        v_right_mv: float,
        v_heading_mv: float = 0.0,
        v_ref_mv: float = 850.0,
        timestamp_ns: int = 0,
    ) -> CoilOffset3D:
        """Estimate 3D offset and orientation from 4-quadrant coil differential voltages."""
        # Longitudinal and lateral displacement
        diff_x = v_front_mv - v_rear_mv
        diff_y = v_left_mv - v_right_mv

        delta_x = round(diff_x / self.sensitivity_mv_per_mm, 2)
        delta_y = round(diff_y / self.sensitivity_mv_per_mm, 2)

        # Estimate vertical air gap from total magnetic coupling voltage
        v_sum = max(10.0, v_front_mv + v_rear_mv + v_left_mv + v_right_mv)
        # Empirical inverse cubic magnetic dipole relation normalized to nominal air gap
        ratio = (v_ref_mv * 4.0) / v_sum
        air_gap_mm = round(self.config.target_air_gap_mm * (ratio ** (1.0 / 3.0)), 1)
        clamped_gap = max(80.0, min(300.0, air_gap_mm))

        delta_theta = round(v_heading_mv / self.angular_sensitivity, 2)

        return self.compute_status(
            delta_x_mm=delta_x,
            delta_y_mm=delta_y,
            delta_z_mm=clamped_gap,
            delta_theta_deg=delta_theta,
            timestamp_ns=timestamp_ns,
        )

    def compute_status(
        self,
        delta_x_mm: float,
        delta_y_mm: float,
        delta_z_mm: float,
        delta_theta_deg: float,
        timestamp_ns: int = 0,
    ) -> CoilOffset3D:
        """Evaluate alignment tolerance and categorize alignment status."""
        radial_offset = round(math.hypot(delta_x_mm, delta_y_mm), 2)

        within_tolerance = (
            abs(delta_x_mm) <= self.config.max_longitudinal_offset_mm
            and abs(delta_y_mm) <= self.config.max_lateral_offset_mm
            and abs(delta_theta_deg) <= self.config.max_heading_error_deg
        )

        if radial_offset <= self.config.docked_radius_threshold_mm and abs(delta_theta_deg) <= 1.5:
            status = WPTAlignmentStatus.DOCKED
        elif within_tolerance:
            status = WPTAlignmentStatus.ALIGNED
        elif radial_offset <= 400.0:
            status = WPTAlignmentStatus.APPROACHING
        elif radial_offset <= 1500.0:
            status = WPTAlignmentStatus.MISALIGNED
        else:
            status = WPTAlignmentStatus.OUT_OF_RANGE

        confidence = max(0.1, min(1.0, 1.0 - (radial_offset / 1500.0)))

        return CoilOffset3D(
            delta_x_mm=delta_x_mm,
            delta_y_mm=delta_y_mm,
            delta_z_mm=delta_z_mm,
            delta_theta_deg=delta_theta_deg,
            radial_offset_mm=radial_offset,
            alignment_status=status,
            is_within_tolerance=within_tolerance,
            confidence=round(confidence, 3),
            timestamp_ns=timestamp_ns,
        )
