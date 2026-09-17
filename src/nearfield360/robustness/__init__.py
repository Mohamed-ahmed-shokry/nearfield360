"""Controlled robustness evaluation, synthetic corruptions, and diagnostic plots."""

from __future__ import annotations

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
    "apply_fog",
    "apply_lens_soiling",
    "apply_low_light_noise",
    "apply_rain",
    "apply_sensor_corruption",
    "euler_to_rotation_matrix",
    "perturb_calibrated_camera",
    "perturb_camera_calibration",
    "perturb_rigid_transform",
    "rotation_matrix_to_quaternion",
]
