"""Vectorized WoodScape radial-polynomial projection in camera coordinates.

The optical axis is positive Z; X points right and Y points down. Unlike the
OpenCV fisheye model, WoodScape uses consecutive powers of the incidence angle:
``rho(theta) = k1*theta + k2*theta**2 + k3*theta**3 + k4*theta**4``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.data.calibration import IntrinsicParameters

_FLOAT_EPSILON = np.finfo(np.float64).eps
_FLOAT_TINY = np.finfo(np.float64).tiny
_MAX_INVERSE_ITERATIONS = 80


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Pixel coordinates ``(..., 2)`` and a validity mask ``(...)``.

    Every invalid coordinate is NaN. A valid result is geometric validity, not
    an assertion about image content, occlusion, or calibration accuracy.
    """

    pixels: NDArray[np.float64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class UnprojectionResult:
    """Unit camera rays ``(..., 3)`` and a validity mask ``(...)``.

    Every component of an invalid ray is NaN. Rays encode direction, not depth.
    """

    rays: NDArray[np.float64]
    valid: NDArray[np.bool_]


def _coordinates(values: ArrayLike, dimension: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real coordinates")
    result = np.asarray(array, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != dimension:
        raise ValueError(f"{name} must have shape (..., {dimension}), got {result.shape}")
    return result


def _stationary_angles(coefficients: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Solve rho'' = 0 using a cancellation-resistant quadratic formula."""
    _, k2, k3, k4 = coefficients
    quadratic, linear, constant = 12.0 * k4, 6.0 * k3, 2.0 * k2
    if quadratic == 0.0:
        return () if linear == 0.0 else (-constant / linear,)
    discriminant = math.fsum((linear * linear, -4.0 * quadratic * constant))
    if discriminant < 0.0:
        return ()
    root_discriminant = math.sqrt(discriminant)
    numerator = -0.5 * (linear + math.copysign(root_discriminant, linear))
    if numerator == 0.0:
        return (0.0,)
    return (numerator / quadratic, constant / numerator)


@dataclass(frozen=True, slots=True)
class RadialPolynomialFisheye:
    """A calibrated fisheye model with an explicit usable angular domain.

    ``theta_max`` is REQUIRED, in radians, and lies strictly between zero and pi.
    It is a chosen/calibrated limit, not a field of view inferred from the image
    corners. The radial derivative must be strictly positive throughout
    ``[0, theta_max]``; flat or folding domains are rejected conservatively.

    Bounds filtering, enabled by default on both operations, uses the half-open
    image rectangle ``[0, width) x [0, height)``. Disable it to operate on the
    full angular model. The angular/radial limit is inclusive, with one floating
    point ULP of tolerance for round-off at the boundary.
    """

    intrinsics: IntrinsicParameters
    theta_max: float
    _coefficients: tuple[float, float, float, float] = field(init=False, repr=False)
    _coefficient_scale: float = field(init=False, repr=False)
    _radius_max: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.theta_max, bool)
            or not math.isfinite(self.theta_max)
            or not 0.0 < self.theta_max < math.pi
        ):
            raise ValueError("theta_max must be finite and strictly between zero and pi radians")

        scale = max(abs(value) for value in self.intrinsics.coefficients)
        coefficients = tuple(value / scale for value in self.intrinsics.coefficients)
        # The tuple is written explicitly to preserve its fixed length for typing.
        normalized = (coefficients[0], coefficients[1], coefficients[2], coefficients[3])
        object.__setattr__(self, "_coefficient_scale", scale)
        object.__setattr__(self, "_coefficients", normalized)

        candidates = (0.0, self.theta_max, *_stationary_angles(normalized))
        k1, k2, k3, k4 = normalized
        for theta in candidates:
            if 0.0 <= theta <= self.theta_max:
                derivative = math.fsum(
                    (k1, 2.0 * k2 * theta, 3.0 * k3 * theta**2, 4.0 * k4 * theta**3)
                )
                if derivative <= 0.0:
                    raise ValueError(
                        "radial polynomial must have a strictly positive derivative "
                        "throughout [0, theta_max]"
                    )

        radial_limit = (
            (((k4 * self.theta_max + k3) * self.theta_max + k2) * self.theta_max + k1)
            * self.theta_max
            * scale
        )
        if not math.isfinite(radial_limit) or radial_limit <= 0.0:
            raise ValueError("radial polynomial must have a finite, positive radius at theta_max")
        object.__setattr__(self, "_radius_max", radial_limit)

    @property
    def radius_max(self) -> float:
        """Maximum polynomial radius in pixels, before aspect scaling."""
        return self._radius_max

    def _radial(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        k1, k2, k3, k4 = self._coefficients
        return (((k4 * theta + k3) * theta + k2) * theta + k1) * theta

    def _derivative(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        k1, k2, k3, k4 = self._coefficients
        return ((4.0 * k4 * theta + 3.0 * k3) * theta + 2.0 * k2) * theta + k1

    def _image_bounds(self, pixels: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < self.intrinsics.width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < self.intrinsics.height)
        )

    def project(self, points: ArrayLike, *, check_image_bounds: bool = True) -> ProjectionResult:
        """Project camera-frame points/directions of shape ``(..., 3)``.

        Distance has no effect. Zero/non-finite vectors and the exact negative
        optical axis are invalid. Non-axis rays behind the camera are valid if
        their incidence angle is within the configured domain.
        """
        coordinates = _coordinates(points, 3, "points")
        flat = coordinates.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=1)
        safe = np.where(finite[:, None], flat, 0.0)
        # Scale before computing a norm: neither huge nor subnormal finite
        # direction vectors should overflow/underflow into an invalid norm.
        with np.errstate(all="ignore"):
            scale = np.max(np.abs(safe), axis=1)
            normalized = np.divide(
                safe, scale[:, None], out=np.zeros_like(safe), where=scale[:, None] > 0.0
            )
            transverse = np.hypot(normalized[:, 0], normalized[:, 1])
            theta = np.arctan2(transverse, normalized[:, 2])
            valid = (
                finite
                & (scale > 0.0)
                & ((transverse > 0.0) | (normalized[:, 2] > 0.0))
                & (theta <= np.nextafter(self.theta_max, np.inf))
            )
            directions = np.divide(
                normalized[:, :2],
                transverse[:, None],
                out=np.zeros((len(flat), 2), dtype=np.float64),
                where=transverse[:, None] > 0.0,
            )
            radius = self._radial(np.minimum(theta, self.theta_max)) * self._coefficient_scale
            pixels = directions * radius[:, None]
            pixels[:, 1] *= self.intrinsics.aspect_ratio
            pixels += self.intrinsics.principal_point
            valid &= np.isfinite(pixels).all(axis=1)
        if check_image_bounds:
            valid &= self._image_bounds(pixels)
        pixels[~valid] = np.nan
        return ProjectionResult(
            pixels=pixels.reshape((*coordinates.shape[:-1], 2)),
            valid=valid.reshape(coordinates.shape[:-1]),
        )

    def _invert_radius(
        self, radius: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Invert in a bracket, with at most 80 safeguarded Newton iterations."""
        angles = np.zeros_like(radius)
        converged = np.ones(radius.shape, dtype=np.bool_)
        boundary = radius == self.radius_max
        angles[boundary] = self.theta_max
        interior = (radius > 0.0) & ~boundary
        if not np.any(interior):
            return angles, converged

        target = radius[interior] / self._coefficient_scale
        lower = np.zeros_like(target)
        upper = np.full_like(target, self.theta_max)
        # The linear approximation is especially important near the optical
        # axis, where a fixed absolute-angle bisection tolerance loses precision.
        theta = np.minimum(target / self._coefficients[0], self.theta_max)
        done = np.zeros(target.shape, dtype=np.bool_)
        for _ in range(_MAX_INVERSE_ITERATIONS):
            residual = self._radial(theta) - target
            lower = np.where(residual <= 0.0, theta, lower)
            upper = np.where(residual >= 0.0, theta, upper)
            width = upper - lower
            done = width <= 4.0 * _FLOAT_EPSILON * np.maximum(np.abs(theta), _FLOAT_TINY)
            if np.all(done):
                break
            newton = theta - residual / self._derivative(theta)
            midpoint = lower + 0.5 * width
            safe_newton = np.isfinite(newton) & (newton > lower) & (newton < upper)
            theta = np.where(done, theta, np.where(safe_newton, newton, midpoint))

        angles[interior] = lower + 0.5 * (upper - lower)
        converged[interior] = done
        return angles, converged

    def unproject(
        self, pixels: ArrayLike, *, check_image_bounds: bool = True
    ) -> UnprojectionResult:
        """Unproject pixel coordinates ``(..., 2)`` to unit camera rays.

        This only supplies a ray; metric position requires a depth/ground-plane
        assumption elsewhere. Non-finite, out-of-domain, and non-convergent
        inputs are invalid rather than extrapolated through the polynomial.
        """
        coordinates = _coordinates(pixels, 2, "pixels")
        flat = coordinates.reshape(-1, 2)
        finite = np.isfinite(flat).all(axis=1)
        safe = np.where(finite[:, None], flat, self.intrinsics.principal_point)
        with np.errstate(all="ignore"):
            offsets = safe - self.intrinsics.principal_point
            offsets[:, 1] /= self.intrinsics.aspect_ratio
            radius = np.hypot(offsets[:, 0], offsets[:, 1])
            valid = finite & np.isfinite(radius) & (radius <= np.nextafter(self.radius_max, np.inf))
            if check_image_bounds:
                valid &= self._image_bounds(flat)
            safe_radius = np.where(valid, np.minimum(radius, self.radius_max), 0.0)
            angles, converged = self._invert_radius(safe_radius)
            valid &= converged
            directions = np.divide(
                np.where(valid[:, None], offsets, 0.0),
                radius[:, None],
                out=np.zeros_like(offsets),
                where=valid[:, None] & (radius[:, None] > 0.0),
            )
            rays = np.column_stack((directions * np.sin(angles)[:, None], np.cos(angles)))
            valid &= np.isfinite(rays).all(axis=1)
        rays[~valid] = np.nan
        return UnprojectionResult(
            rays=rays.reshape((*coordinates.shape[:-1], 3)),
            valid=valid.reshape(coordinates.shape[:-1]),
        )


__all__ = ["ProjectionResult", "RadialPolynomialFisheye", "UnprojectionResult"]
