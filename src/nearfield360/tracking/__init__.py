"""Multi-camera 2D metric obstacle tracking, Kalman filtering, and dynamic risk."""

from __future__ import annotations

from nearfield360.tracking.models import (
    GroundFootprint,
    TrackedObstacle,
    TrackState,
    TrajectoryForecast,
)
from nearfield360.tracking.projection import (
    DEFAULT_CLASS_DIMENSIONS,
    project_detection_to_ground,
    project_detections,
)

__all__ = [
    "DEFAULT_CLASS_DIMENSIONS",
    "GroundFootprint",
    "TrackState",
    "TrackedObstacle",
    "TrajectoryForecast",
    "project_detection_to_ground",
    "project_detections",
]
