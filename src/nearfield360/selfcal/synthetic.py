"""Seeded synthetic ground-landmark observations for self-calibration tests and demos."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.models import GroundObservation, RotationDelta, rotation_matrix

DEFAULT_CAMERA_ORIGIN: tuple[float, float, float] = (3.6, 0.0, 0.8)


def synthesize_observations(
    camera_id: CameraId,
    true_delta: RotationDelta,
    *,
    origin: tuple[float, float, float] = DEFAULT_CAMERA_ORIGIN,
    ground_z: float = 0.0,
    num_points: int = 40,
    x_range: tuple[float, float] = (5.0, 10.0),
    y_range: tuple[float, float] = (-4.0, 4.0),
    noise_deg: float = 0.02,
    outlier_fraction: float = 0.0,
    seed: int = 0,
) -> list[GroundObservation]:
    """Generate observations of a camera whose true extrinsics are off by ``true_delta``.

    Ground landmarks are drawn uniformly in the given vehicle-frame rectangle.
    The nominal rays handed to the estimator are the true rays rotated by the
    inverse of ``true_delta`` (so applying ``true_delta`` recovers the truth),
    perturbed by isotropic angular noise. A fraction of the landmark references
    is displaced by 0.5-2.0 m to model wrong map matches.
    """
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if not 0.0 <= outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    origin_array = np.asarray(origin, dtype=np.float64)
    points = np.column_stack(
        [
            rng.uniform(x_range[0], x_range[1], num_points),
            rng.uniform(y_range[0], y_range[1], num_points),
            np.full(num_points, ground_z),
        ]
    )
    true_rays = points - origin_array
    inverse = rotation_matrix(true_delta.as_vector()).T
    nominal = true_rays @ inverse.T
    noise_rad = np.radians(noise_deg)
    num_outliers = round(outlier_fraction * num_points)
    outlier_index = set(rng.choice(num_points, size=num_outliers, replace=False).tolist())
    observations: list[GroundObservation] = []
    for index in range(num_points):
        ray = _perturb(nominal[index], noise_rad, rng)
        reference = points[index, :2].copy()
        if index in outlier_index:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            reference += rng.uniform(0.5, 2.0) * np.array([np.cos(angle), np.sin(angle)])
        observations.append(
            GroundObservation(
                camera_id=camera_id,
                direction=(float(ray[0]), float(ray[1]), float(ray[2])),
                reference_x_m=float(reference[0]),
                reference_y_m=float(reference[1]),
            )
        )
    return observations


def _perturb(
    ray: npt.NDArray[np.float64], sigma_rad: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    if sigma_rad <= 0.0:
        return ray
    perturbation: npt.NDArray[np.float64] = rotation_matrix(rng.normal(0.0, sigma_rad, 3)) @ ray
    return perturbation
