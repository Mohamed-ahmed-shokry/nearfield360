"""Map-anchored online extrinsic self-calibration and camera drift monitoring."""

from __future__ import annotations

from nearfield360.selfcal.association import (
    NearestResult,
    PolylineMap,
    associate_pixel_detections,
)
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.icp import (
    AssociationRound,
    ImageCalibrationResult,
    ImageLandmarkCalibrator,
)
from nearfield360.selfcal.models import (
    CalibrationEstimate,
    DriftReport,
    DriftStatus,
    GroundObservation,
    RotationDelta,
    rotation_matrix,
)
from nearfield360.selfcal.monitor import TRUSTED_STATUSES, DriftMonitor
from nearfield360.selfcal.scene import (
    demo_fisheye_camera,
    demo_parking_aisle_map,
    demo_parking_aisle_polylines,
    synthesize_pixel_detections,
)
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations
from nearfield360.selfcal.viz import render_selfcal_dashboard

__all__ = [
    "DEFAULT_CAMERA_ORIGIN",
    "TRUSTED_STATUSES",
    "AssociationRound",
    "CalibrationEstimate",
    "DriftMonitor",
    "DriftReport",
    "DriftStatus",
    "ExtrinsicSelfCalibrator",
    "GroundObservation",
    "ImageCalibrationResult",
    "ImageLandmarkCalibrator",
    "NearestResult",
    "PolylineMap",
    "RotationDelta",
    "associate_pixel_detections",
    "demo_fisheye_camera",
    "demo_parking_aisle_map",
    "demo_parking_aisle_polylines",
    "render_selfcal_dashboard",
    "rotation_matrix",
    "synthesize_observations",
    "synthesize_pixel_detections",
]
