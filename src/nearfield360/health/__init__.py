"""Camera health monitoring, lens soiling detection, and fusion discounting."""

from __future__ import annotations

from nearfield360.health.detector import (
    assess_camera_health,
    calculate_blur_score,
    calculate_photometric_properties,
    detect_blockage,
    detect_lens_soiling,
)
from nearfield360.health.models import (
    CameraHealthMetrics,
    CameraHealthReport,
    CameraHealthStatus,
    HealthAnomaly,
)

__all__ = [
    "CameraHealthMetrics",
    "CameraHealthReport",
    "CameraHealthStatus",
    "HealthAnomaly",
    "assess_camera_health",
    "calculate_blur_score",
    "calculate_photometric_properties",
    "detect_blockage",
    "detect_lens_soiling",
]
