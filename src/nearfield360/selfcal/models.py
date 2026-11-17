"""Domain models for extrinsic self-calibration and drift monitoring."""

from __future__ import annotations

import math
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


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
