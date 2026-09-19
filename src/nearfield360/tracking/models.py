"""Data models for multi-camera 2D metric object tracking and dynamic risk."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TrackState(StrEnum):
    """Lifecycle state of an active obstacle track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class GroundFootprint(BaseModel):
    """Ground-projected metric obstacle footprint in vehicle coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    center_x: float = Field(description="Longitudinal X coordinate in metres (forward positive).")
    center_y: float = Field(description="Lateral Y coordinate in metres (left positive).")
    width: float = Field(gt=0.0, description="Obstacle lateral width in metres.")
    length: float = Field(gt=0.0, description="Obstacle longitudinal length in metres.")
    class_id: int = Field(ge=0, description="WoodScape detection class id.")
    class_name: str = Field(description="WoodScape class name (e.g. vehicles, person).")
    camera: str = Field(description="Originating camera identifier (e.g. FV).")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score.")


class TrackedObstacle(BaseModel):
    """State of an obstacle tracked across temporal frames in vehicle coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(ge=0, description="Unique persistent integer track identifier.")
    class_id: int = Field(ge=0, description="WoodScape detection class identifier.")
    class_name: str = Field(description="Class name of the tracked obstacle.")
    state: TrackState = Field(description="Track lifecycle status.")
    position: tuple[float, float] = Field(
        description="Current estimated position (x, y) in vehicle coordinates (metres)."
    )
    velocity: tuple[float, float] = Field(
        description="Current estimated velocity (vx, vy) in metres/second."
    )
    speed: float = Field(ge=0.0, description="Scalar obstacle speed in metres/second.")
    hits: int = Field(ge=1, description="Cumulative detection updates associated with track.")
    age: int = Field(ge=1, description="Total frames since track inception.")
    time_since_update: int = Field(ge=0, description="Frames elapsed since last detection update.")
    history: tuple[tuple[float, float], ...] = Field(
        default=(), description="History of past (x, y) positions in vehicle coordinates."
    )


class TrajectoryForecast(BaseModel):
    """Future projected motion and zone collision risk for a tracked obstacle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(description="Track ID of the evaluated obstacle.")
    class_name: str = Field(description="Class name of the evaluated obstacle.")
    predicted_positions: tuple[tuple[float, float, float], ...] = Field(
        default=(),
        description="Predicted (t_offset, x, y) states over forecast horizon.",
    )
    zone_intrusions: tuple[tuple[str, float], ...] = Field(
        default=(),
        description="Tuples of (zone_name, time_to_arrival_s) for predicted zone crossings.",
    )
    min_ttc_s: float | None = Field(
        default=None,
        description="Minimum Time-to-Collision in seconds to any safety zone (None if clear).",
    )


__all__ = [
    "GroundFootprint",
    "TrackState",
    "TrackedObstacle",
    "TrajectoryForecast",
]
