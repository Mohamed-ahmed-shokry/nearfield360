"""Validated source-to-target rigid transforms without a SciPy dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

_ROTATION_TOLERANCE = 1e-6
_QUATERNION_NORM_TOLERANCE = 1e-3


def _finite_array(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return result


def _immutable_copy(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Detach both storage and array metadata using an immutable bytes buffer."""
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RigidTransform:
    """Map 3D source coordinates to target coordinates as ``R @ point + t``.

    Coordinate arrays use shape ``(..., 3)``; the equivalent batched operation
    is ``points @ R.T + t``. Translation uses the same units as the points
    (metres for WoodScape), and does not affect directions. The class does not
    attach frame names: callers must compose transforms with matching frames.

    Rotations must be finite, orthonormal, and right-handed within an absolute
    tolerance of ``1e-6`` to accommodate float32 roundoff. Matrices are never
    projected onto a rotation silently. Stored arrays cannot be mutated, and
    public array properties return independent read-only copies.
    """

    _rotation: npt.NDArray[np.float64] = field(repr=False)
    _translation: npt.NDArray[np.float64] = field(repr=False)

    def __init__(
        self,
        rotation: npt.ArrayLike,
        translation: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        rotation_array = _finite_array(rotation, "rotation")
        translation_array = _finite_array(translation, "translation")
        if rotation_array.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation_array.shape}")
        if translation_array.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {translation_array.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            gram_matrix = rotation_array.T @ rotation_array
        if not np.allclose(gram_matrix, np.eye(3), rtol=0.0, atol=_ROTATION_TOLERANCE):
            raise ValueError("rotation must be orthonormal within absolute tolerance 1e-6")
        if not math.isclose(
            float(np.linalg.det(rotation_array)),
            1.0,
            rel_tol=0.0,
            abs_tol=_ROTATION_TOLERANCE,
        ):
            raise ValueError("rotation must be right-handed with determinant +1")
        object.__setattr__(self, "_rotation", _immutable_copy(rotation_array))
        object.__setattr__(self, "_translation", _immutable_copy(translation_array))

    @property
    def rotation(self) -> npt.NDArray[np.float64]:
        """Return an independent read-only ``(3, 3)`` source-to-target rotation."""
        return _immutable_copy(self._rotation)

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        """Return an independent read-only ``(3,)`` translation in the target frame."""
        return _immutable_copy(self._translation)

    @classmethod
    def identity(cls) -> Self:
        """Return a transform that leaves points and directions unchanged."""
        return cls(np.eye(3))

    @classmethod
    def from_quaternion(
        cls,
        quaternion: npt.ArrayLike,
        translation: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> Self:
        """Construct from a right-handed active ``(x, y, z, w)`` quaternion.

        Norms within ``1e-3`` of one are normalized to accommodate calibration
        rounding, matching the WoodScape calibration parser. Zero quaternions
        and arbitrary scaled quaternions are rejected, not silently repaired.
        """
        values = _finite_array(quaternion, "quaternion")
        if values.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,), got {values.shape}")
        norm = math.hypot(*(float(component) for component in values))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
            raise ValueError(f"quaternion must have unit norm within 1e-3, got {norm:.8g}")
        x, y, z, w = values / norm
        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return cls(rotation, translation)

    def inverse(self) -> RigidTransform:
        """Return the target-to-source transform ``R.T @ point - R.T @ t``."""
        inverse_rotation = self._rotation.T
        with np.errstate(over="ignore", invalid="ignore"):
            inverse_translation = -(inverse_rotation @ self._translation)
        return RigidTransform(inverse_rotation, inverse_translation)

    def compose(self, other: RigidTransform) -> RigidTransform:
        """Apply ``other`` first, then ``self``, returning the combined transform.

        If ``other`` maps A to B and ``self`` maps B to C, the result maps A
        to C: ``R = self.R @ other.R`` and ``t = self.R @ other.t + self.t``.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            rotation = self._rotation @ other._rotation
            translation = self._rotation @ other._translation + self._translation
        return RigidTransform(rotation, translation)

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Rotate and translate finite real points, preserving shape ``(..., 3)``."""
        return self._transform_vectors(points, "points", translate=True)

    def transform_directions(self, directions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Rotate finite real directions without translation or normalization.

        Shape ``(..., 3)`` and vector lengths are preserved. Empty batches and
        zero vectors are accepted; a nonzero unit ray is the caller's contract.
        """
        return self._transform_vectors(directions, "directions", translate=False)

    def _transform_vectors(
        self, values: npt.ArrayLike, name: str, *, translate: bool
    ) -> npt.NDArray[np.float64]:
        vectors = _finite_array(values, name)
        if vectors.ndim == 0 or vectors.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (..., 3), got {vectors.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            result = vectors @ self._rotation.T
            if translate:
                result = result + self._translation
        if not np.all(np.isfinite(result)):
            raise ValueError(f"transformed {name} exceed finite float64 range")
        return result


__all__ = ["RigidTransform"]
