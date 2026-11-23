"""Domain models for extrinsic self-calibration and drift monitoring."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nearfield360.data.woodscape import CameraId


class RotationDelta(BaseModel):
    """Small residual rotation applied to vehicle-frame camera rays.

    The three components form a rotation vector (axis times angle, radians)
    about the vehicle X (roll), Y (pitch), and Z (yaw) axes. For the sub-degree
    to few-degree drifts of interest this is indistinguishable from Euler
    roll/pitch/yaw, and unlike Euler angles it composes without ambiguity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    roll_rad: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    pitch_rad: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    yaw_rad: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)

    def as_vector(self) -> npt.NDArray[np.float64]:
        """Return the rotation vector ``[roll, pitch, yaw]`` in radians."""
        return np.array([self.roll_rad, self.pitch_rad, self.yaw_rad], dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: npt.ArrayLike) -> Self:
        """Build a delta from a length-3 rotation vector in radians."""
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f"rotation vector must have shape (3,), got {values.shape}")
        return cls(roll_rad=float(values[0]), pitch_rad=float(values[1]), yaw_rad=float(values[2]))

    @classmethod
    def from_degrees(cls, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> Self:
        """Build a delta from roll/pitch/yaw expressed in degrees."""
        return cls(
            roll_rad=math.radians(roll),
            pitch_rad=math.radians(pitch),
            yaw_rad=math.radians(yaw),
        )

    @property
    def magnitude_rad(self) -> float:
        """Total rotation angle in radians."""
        return float(np.linalg.norm(self.as_vector()))

    @property
    def magnitude_deg(self) -> float:
        """Total rotation angle in degrees."""
        return math.degrees(self.magnitude_rad)

    def matrix(self) -> npt.NDArray[np.float64]:
        """Return the 3x3 rotation matrix via the Rodrigues formula."""
        return rotation_matrix(self.as_vector())


def rotation_matrix(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Rodrigues rotation matrix for a rotation vector (radians)."""
    rotvec = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    skew: npt.NDArray[np.float64] = np.array(
        [
            [0.0, -rotvec[2], rotvec[1]],
            [rotvec[2], 0.0, -rotvec[0]],
            [-rotvec[1], rotvec[0], 0.0],
        ],
        dtype=np.float64,
    )
    identity: npt.NDArray[np.float64] = np.eye(3, dtype=np.float64)
    if angle < 1e-12:
        return identity + skew
    sin_term = math.sin(angle) / angle
    cos_term = (1.0 - math.cos(angle)) / (angle * angle)
    rotation: npt.NDArray[np.float64] = identity + sin_term * skew + cos_term * (skew @ skew)
    return rotation


class GroundObservation(BaseModel):
    """One camera ray toward a landmark whose ground position is known.

    ``direction`` is the nominal vehicle-frame ray (from the camera origin) as
    produced by the factory calibration. ``reference_x_m``/``reference_y_m`` is
    the landmark position in the vehicle frame obtained from the HD map match
    or a calibration target. When ``reference_normal`` is given the landmark lies
    on a line with that vehicle-frame normal, and only the cross-line component
    of the ground residual is meaningful (point-to-line correspondence).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: CameraId
    direction: tuple[float, float, float]
    reference_x_m: float = Field(..., allow_inf_nan=False)
    reference_y_m: float = Field(..., allow_inf_nan=False)
    reference_normal: tuple[float, float] | None = None

    @field_validator("reference_normal")
    @classmethod
    def _normal_finite_nonzero(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if not all(math.isfinite(component) for component in value):
            raise ValueError("reference_normal components must be finite")
        if math.hypot(*value) == 0.0:
            raise ValueError("reference_normal must be non-zero")
        return value

    @field_validator("direction")
    @classmethod
    def _direction_finite_nonzero(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(component) for component in value):
            raise ValueError("direction components must be finite")
        if math.sqrt(sum(component * component for component in value)) == 0.0:
            raise ValueError("direction must be non-zero")
        return value


class DriftStatus(StrEnum):
    """Severity of a camera's estimated extrinsic misalignment."""

    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    RECALIBRATE = "RECALIBRATE"
    FAULT = "FAULT"
    UNOBSERVABLE = "UNOBSERVABLE"


class CalibrationEstimate(BaseModel):
    """Result of one robust extrinsic estimation for a single camera."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: CameraId
    delta: RotationDelta
    observable: bool
    converged: bool
    num_observations: int = Field(..., ge=0)
    inlier_ratio: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    rms_before_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    rms_after_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    iterations: int = Field(..., ge=0)
    sigma_deg: tuple[float, float, float] | None = None
    condition_number: float | None = None
    reason: str = ""


class DriftReport(BaseModel):
    """Smoothed drift state of one camera after a monitor update."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: CameraId
    status: DriftStatus
    smoothed_delta: RotationDelta
    magnitude_deg: float = Field(..., ge=0.0, allow_inf_nan=False)
    accepted_updates: int = Field(..., ge=0)
    rejected_updates: int = Field(..., ge=0)
