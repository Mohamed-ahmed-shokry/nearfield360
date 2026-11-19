"""Map-anchored online extrinsic self-calibration and camera drift monitoring."""

from __future__ import annotations

from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.models import (
    CalibrationEstimate,
    DriftReport,
    DriftStatus,
    GroundObservation,
    RotationDelta,
    rotation_matrix,
)
from nearfield360.selfcal.monitor import TRUSTED_STATUSES, DriftMonitor
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations
from nearfield360.selfcal.viz import render_selfcal_dashboard

__all__ = [
    "DEFAULT_CAMERA_ORIGIN",
    "TRUSTED_STATUSES",
    "CalibrationEstimate",
    "DriftMonitor",
    "DriftReport",
    "DriftStatus",
    "ExtrinsicSelfCalibrator",
    "GroundObservation",
    "RotationDelta",
    "render_selfcal_dashboard",
    "rotation_matrix",
    "synthesize_observations",
]
