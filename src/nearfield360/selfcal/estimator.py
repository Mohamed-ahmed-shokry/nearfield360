"""Robust extrinsic rotation estimation from HD-map-anchored ground landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from nearfield360.config import SelfCalConfig
from nearfield360.selfcal.models import (
    CalibrationEstimate,
    GroundObservation,
    RotationDelta,
    rotation_matrix,
)

_MIN_DIRECTION_Z = 1e-9
_JACOBIAN_STEP_RAD = 1e-6
_INLIER_SCALES = 3.0
_SIGMA_FLOOR_M = 1e-4


@dataclass(frozen=True)
class _Problem:
    """Fixed inputs of one estimation: camera origin, nominal rays, and references."""

    origin: npt.NDArray[np.float64]
    directions: npt.NDArray[np.float64]
    references: npt.NDArray[np.float64]


class ExtrinsicSelfCalibrator:
    """Estimate a camera's residual rotation from known ground landmark positions.

    Each observation is a nominal vehicle-frame ray plus the landmark's known
    vehicle-frame ground position. The estimator finds the small rotation that,
    applied to every ray, makes the ground-plane intersections land on their
    reference positions. Levenberg-Marquardt solves the 3-parameter problem and
    Huber iteratively-reweighted least squares downweights wrong map matches.
    """

    def __init__(self, config: SelfCalConfig | None = None, *, ground_z: float = 0.0) -> None:
        self.config = config or SelfCalConfig()
        self.ground_z = ground_z

    def estimate(
        self,
        origin: tuple[float, float, float],
        observations: list[GroundObservation],
    ) -> CalibrationEstimate:
        """Estimate the residual rotation for the single camera in ``observations``."""
        if not observations:
            raise ValueError("observations must not be empty")
        camera_ids = {observation.camera_id for observation in observations}
        if len(camera_ids) != 1:
            raise ValueError("observations must all belong to one camera")
        camera_id = observations[0].camera_id
        total = len(observations)

        origin_array = np.asarray(origin, dtype=np.float64)
        directions = np.asarray([o.direction for o in observations], dtype=np.float64)
        references = np.asarray(
            [[o.reference_x_m, o.reference_y_m] for o in observations], dtype=np.float64
        )
        usable = (self.ground_z - origin_array[2]) / np.where(
            np.abs(directions[:, 2]) < _MIN_DIRECTION_Z, np.nan, directions[:, 2]
        ) > 0.0
        directions = directions[usable]
        references = references[usable]
        problem = _Problem(origin_array, directions, references)

        def unobservable(reason: str, num: int = total) -> CalibrationEstimate:
            return CalibrationEstimate(
                camera_id=camera_id,
                delta=RotationDelta(),
                observable=False,
                converged=False,
                num_observations=num,
                inlier_ratio=0.0,
                rms_before_m=0.0,
                rms_after_m=0.0,
                iterations=0,
                sigma_deg=None,
                reason=reason,
            )

        if len(directions) < self.config.min_correspondences:
            return unobservable(
                f"need >= {self.config.min_correspondences} usable correspondences, "
                f"got {len(directions)}"
            )

        solution = self._solve(problem)
        if solution is None:
            return unobservable("estimator diverged outside the supported rotation range")
        vector, iterations, converged, weights, jacobian = solution

        residual_norm = np.linalg.norm(self._residuals(vector, problem).reshape(-1, 2), axis=1)
        inliers = residual_norm <= _INLIER_SCALES * self.config.huber_delta_m
        inlier_ratio = float(np.count_nonzero(inliers)) / total
        zero_norm = np.linalg.norm(
            self._residuals(np.zeros(3), problem).reshape(-1, 2),
            axis=1,
        )
        rms_after = _rms(residual_norm[inliers])
        rms_before = _rms(zero_norm[inliers])

        weighted = jacobian * np.repeat(weights, 2)[:, None] ** 0.5
        singular = np.linalg.svd(weighted, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
        sigma_deg = self._sigma_deg(weighted, residual_norm, weights)

        reason = ""
        if not converged:
            reason = "estimator did not converge within the iteration budget"
        elif inlier_ratio < self.config.min_inlier_ratio:
            reason = f"inlier ratio {inlier_ratio:.2f} below {self.config.min_inlier_ratio:.2f}"
        elif condition > self.config.max_condition_number:
            reason = f"geometry is ill-conditioned (condition number {condition:.3g})"
        try:
            delta = RotationDelta.from_vector(vector)
        except ValidationError:
            return unobservable("estimated rotation outside the supported range")
        return CalibrationEstimate(
            camera_id=camera_id,
            delta=delta,
            observable=reason == "",
            converged=converged,
            num_observations=total,
            inlier_ratio=inlier_ratio,
            rms_before_m=rms_before,
            rms_after_m=rms_after,
            iterations=iterations,
            sigma_deg=sigma_deg,
            condition_number=condition if math.isfinite(condition) else None,
            reason=reason,
        )

    def _residuals(
        self,
        vector: npt.NDArray[np.float64],
        problem: _Problem,
    ) -> npt.NDArray[np.float64]:
        origin = problem.origin
        rotated = problem.directions @ rotation_matrix(vector).T
        with np.errstate(all="ignore"):
            scale = (self.ground_z - origin[2]) / rotated[:, 2]
        points = origin[:2] + scale[:, None] * rotated[:, :2]
        residual: npt.NDArray[np.float64] = (points - problem.references).reshape(-1)
        return residual

    def _jacobian(
        self,
        vector: npt.NDArray[np.float64],
        problem: _Problem,
    ) -> npt.NDArray[np.float64]:
        columns = []
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = _JACOBIAN_STEP_RAD
            plus = self._residuals(vector + step, problem)
            minus = self._residuals(vector - step, problem)
            columns.append((plus - minus) / (2.0 * _JACOBIAN_STEP_RAD))
        jacobian: npt.NDArray[np.float64] = np.stack(columns, axis=1)
        return jacobian

    def _huber_weights(self, residual: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        norms = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        delta = self.config.huber_delta_m
        weights: npt.NDArray[np.float64] = np.where(
            norms <= delta, 1.0, delta / np.maximum(norms, 1e-12)
        )
        return weights

    def _solve(
        self,
        problem: _Problem,
    ) -> (
        tuple[
            npt.NDArray[np.float64],
            int,
            bool,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
        | None
    ):
        vector = np.zeros(3)
        damping = 1e-3
        converged = False
        iterations = 0
        residual = self._residuals(vector, problem)
        weights = self._huber_weights(residual)
        cost = float(np.sum(np.repeat(weights, 2) * residual**2))
        jacobian = self._jacobian(vector, problem)
        for _ in range(self.config.max_iterations):
            iterations += 1
            row_weights = np.repeat(weights, 2)
            normal = jacobian.T @ (row_weights[:, None] * jacobian)
            gradient = jacobian.T @ (row_weights * residual)
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            if float(np.linalg.norm(step)) < self.config.convergence_tolerance_rad:
                converged = True
                break
            candidate = vector + step
            candidate_residual = self._residuals(candidate, problem)
            candidate_cost = float(
                np.sum(
                    np.repeat(self._huber_weights(candidate_residual), 2) * candidate_residual**2
                )
            )
            if np.isfinite(candidate_cost) and candidate_cost <= cost:
                vector, residual = candidate, candidate_residual
                weights = self._huber_weights(residual)
                cost = float(np.sum(np.repeat(weights, 2) * residual**2))
                jacobian = self._jacobian(vector, problem)
                damping = max(damping * 0.3, 1e-12)
            else:
                damping *= 10.0
                if damping > 1e12:
                    break
            if float(np.max(np.abs(vector))) > 1.0:
                return None
        return vector, iterations, converged, weights, jacobian

    def _sigma_deg(
        self,
        weighted_jacobian: npt.NDArray[np.float64],
        residual_norm: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        dof = max(2 * len(residual_norm) - 3, 1)
        variance = max(float(np.sum(weights * residual_norm**2)) / dof, _SIGMA_FLOOR_M**2)
        covariance = variance * np.linalg.pinv(weighted_jacobian.T @ weighted_jacobian)
        sigma = np.degrees(np.sqrt(np.maximum(np.diag(covariance), 0.0)))
        return (float(sigma[0]), float(sigma[1]), float(sigma[2]))


def _rms(values: npt.NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))
