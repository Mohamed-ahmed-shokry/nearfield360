"""Controlled robustness evaluation, synthetic corruptions, and diagnostic plots."""

from __future__ import annotations

from nearfield360.robustness.benchmark import (
    OccupancyComparison,
    RobustnessReport,
    ZoneComparison,
    compare_occupancy_grids,
    compare_zone_risks,
    run_calibration_sweep,
    run_corruption_sweep,
)
from nearfield360.robustness.calibration import (
    euler_to_rotation_matrix,
    perturb_calibrated_camera,
    perturb_camera_calibration,
    perturb_rigid_transform,
    rotation_matrix_to_quaternion,
)
from nearfield360.robustness.corruptions import (
    CorruptionType,
    apply_fog,
    apply_lens_soiling,
    apply_low_light_noise,
    apply_rain,
    apply_sensor_corruption,
)

__all__ = [
    "CorruptionType",
    "OccupancyComparison",
    "RobustnessReport",
    "ZoneComparison",
    "apply_fog",
    "apply_lens_soiling",
    "apply_low_light_noise",
    "apply_rain",
    "apply_sensor_corruption",
    "compare_occupancy_grids",
    "compare_zone_risks",
    "euler_to_rotation_matrix",
    "perturb_calibrated_camera",
    "perturb_camera_calibration",
    "perturb_rigid_transform",
    "rotation_matrix_to_quaternion",
    "run_calibration_sweep",
    "run_corruption_sweep",
]
