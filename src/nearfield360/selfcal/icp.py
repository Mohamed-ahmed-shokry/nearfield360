"""Coarse-to-fine point-to-line ICP driver for pixel-space extrinsic self-calibration."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from nearfield360.config import SelfCalConfig
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.selfcal.association import PolylineMap, associate_pixel_detections
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.models import CalibrationEstimate, RotationDelta

_ICP_CONVERGENCE_DEG = 0.01


class AssociationRound(BaseModel):
    """Telemetry recorded for one coarse-to-fine association and solve round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(..., ge=1)
    gate_m: float = Field(..., gt=0.0, allow_inf_nan=False)
    num_associated: int = Field(..., ge=0)
    inlier_ratio: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    rms_after_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    step_deg: float = Field(..., ge=0.0, allow_inf_nan=False)
    delta: RotationDelta
    observable: bool


class ImageCalibrationResult(BaseModel):
    """Outcome of a multi-round pixel-to-map ICP calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimate: CalibrationEstimate
    rounds: list[AssociationRound]
    converged: bool


class ImageLandmarkCalibrator:
    """Alternates fisheye pixel-to-polyline association with robust rotation estimation.

    Starts with a wide gate (``association_gate_start_m``) so drifted detections
    still match the map, then decays the gate geometrically down to
    ``association_gate_end_m`` to strip clutter and tighten the final estimate.
    """

    def __init__(
        self,
        config: SelfCalConfig | None = None,
        *,
        ground_z: float = 0.0,
        max_distance: float | None = 25.0,
    ) -> None:
        self.config = config or SelfCalConfig()
        self.ground_z = ground_z
        self.max_distance = max_distance
        self._estimator = ExtrinsicSelfCalibrator(self.config, ground_z=ground_z)

    def _gate_schedule(self) -> list[float]:
        rounds = self.config.max_association_rounds
        start = self.config.association_gate_start_m
        end = self.config.association_gate_end_m
        if rounds == 1:
            return [end]
        if rounds == 2:
            return [start, end]
        # Decay geometrically to `end` over rounds-1 steps, then stabilize at `end`.
        decay_steps = rounds - 1
        schedule = [start * ((end / start) ** (i / (decay_steps - 1))) for i in range(decay_steps)]
        schedule.append(end)
        return schedule

    def calibrate(
        self,
        camera: CalibratedCamera,
        pixels: npt.ArrayLike,
        polyline_map: PolylineMap,
        *,
        initial_delta: RotationDelta | None = None,
    ) -> ImageCalibrationResult:
        """Run coarse-to-fine point-to-line ICP from pixel detections to map polylines."""
        origin_vec = camera.camera_to_vehicle.translation
        origin = (float(origin_vec[0]), float(origin_vec[1]), float(origin_vec[2]))
        current_delta = initial_delta or RotationDelta()
        rounds: list[AssociationRound] = []
        last_estimate: CalibrationEstimate | None = None
        last_step_deg = math.inf

        for round_idx, gate_m in enumerate(self._gate_schedule(), start=1):
            observations = associate_pixel_detections(
                camera,
                pixels,
                polyline_map,
                current_delta=current_delta,
                gate_m=gate_m,
                ground_z=self.ground_z,
                max_distance=self.max_distance,
            )
            if len(observations) < self.config.min_correspondences:
                unobservable = CalibrationEstimate(
                    camera_id=camera.name,
                    delta=RotationDelta(),
                    observable=False,
                    converged=False,
                    num_observations=len(observations),
                    inlier_ratio=0.0,
                    rms_before_m=0.0,
                    rms_after_m=0.0,
                    iterations=0,
                    sigma_deg=None,
                    reason=(
                        f"round {round_idx}: only {len(observations)} detections associated "
                        f"within {gate_m:.2f} m gate "
                        f"(need >= {self.config.min_correspondences})"
                    ),
                )
                rounds.append(
                    AssociationRound(
                        round_index=round_idx,
                        gate_m=gate_m,
                        num_associated=len(observations),
                        inlier_ratio=0.0,
                        rms_after_m=0.0,
                        step_deg=0.0,
                        delta=current_delta,
                        observable=False,
                    )
                )
                return ImageCalibrationResult(estimate=unobservable, rounds=rounds, converged=False)

            estimate = self._estimator.estimate(origin, observations)
            step_rad = float(np.linalg.norm(estimate.delta.as_vector() - current_delta.as_vector()))
            last_step_deg = math.degrees(step_rad)
            rounds.append(
                AssociationRound(
                    round_index=round_idx,
                    gate_m=gate_m,
                    num_associated=len(observations),
                    inlier_ratio=estimate.inlier_ratio,
                    rms_after_m=estimate.rms_after_m,
                    step_deg=last_step_deg,
                    delta=estimate.delta,
                    observable=estimate.observable,
                )
            )
            last_estimate = estimate
            if not estimate.observable:
                return ImageCalibrationResult(estimate=estimate, rounds=rounds, converged=False)
            current_delta = estimate.delta

        if last_estimate is None:  # pragma: no cover - max_association_rounds >= 1
            raise RuntimeError("association schedule produced no estimate")
        converged = last_estimate.observable and last_step_deg <= _ICP_CONVERGENCE_DEG
        return ImageCalibrationResult(estimate=last_estimate, rounds=rounds, converged=converged)
