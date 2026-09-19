"""Multi-camera 2D metric obstacle tracking, Kalman filtering, and dynamic risk."""

from __future__ import annotations

from nearfield360.tracking.kalman import KalmanFilter2D
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
from nearfield360.tracking.risk import (
    forecast_all_trajectories,
    forecast_obstacle_trajectory,
)
from nearfield360.tracking.tracker import MultiObjectTracker

__all__ = [
    "DEFAULT_CLASS_DIMENSIONS",
    "GroundFootprint",
    "KalmanFilter2D",
    "MultiObjectTracker",
    "TrackState",
    "TrackedObstacle",
    "TrajectoryForecast",
    "forecast_all_trajectories",
    "forecast_obstacle_trajectory",
    "project_detection_to_ground",
    "project_detections",
]
