"""Strict parsing for official WoodScape per-image calibration JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nearfield360.data.woodscape import CameraId

MAX_CALIBRATION_BYTES = 1024 * 1024
FiniteValue = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteValue = Annotated[float, Field(gt=0, allow_inf_nan=False)]
ImageDimension = Annotated[int, Field(gt=0, le=100_000)]


class CalibrationError(ValueError):
    """Raised when a calibration file is missing, oversized, or invalid."""


class ExtrinsicParameters(BaseModel):
    """Camera-to-vehicle rigid transform using an ``(x, y, z, w)`` quaternion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quaternion: tuple[FiniteValue, FiniteValue, FiniteValue, FiniteValue]
    translation: tuple[FiniteValue, FiniteValue, FiniteValue]

    @model_validator(mode="after")
    def validate_unit_quaternion(self) -> Self:
        norm = math.sqrt(sum(component * component for component in self.quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"extrinsic quaternion must have unit norm, got {norm:.8g}")
        return self


class IntrinsicParameters(BaseModel):
    """WoodScape fourth-order radial-polynomial fisheye intrinsics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aspect_ratio: PositiveFiniteValue
    cx_offset: FiniteValue
    cy_offset: FiniteValue
    height: ImageDimension
    k1: PositiveFiniteValue
    k2: FiniteValue
    k3: FiniteValue
    k4: FiniteValue
    model: Literal["radial_poly"]
    poly_order: Literal[4]
    width: ImageDimension

    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        return (self.k1, self.k2, self.k3, self.k4)

    @property
    def principal_point(self) -> tuple[float, float]:
        """Return pixel-center coordinates using the official half-pixel convention."""
        return (
            self.width / 2.0 + self.cx_offset - 0.5,
            self.height / 2.0 + self.cy_offset - 0.5,
        )


class CameraCalibration(BaseModel):
    """Complete calibration record for one WoodScape camera image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extrinsic: ExtrinsicParameters
    intrinsic: IntrinsicParameters
    name: CameraId


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_calibration(path: Path) -> CameraCalibration:
    """Load one bounded UTF-8 calibration JSON document with contextual errors."""
    if not path.is_file():
        raise CalibrationError(f"Calibration file does not exist: {path}")
    try:
        size = path.stat().st_size
        if size > MAX_CALIBRATION_BYTES:
            raise CalibrationError(
                f"Calibration file exceeds the {MAX_CALIBRATION_BYTES}-byte safety limit: {path}"
            )
        document = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CalibrationError(f"Unable to read calibration file {path}: {exc}") from exc

    try:
        values = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_finite_constant,
        )
        return CameraCalibration.model_validate(values)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise CalibrationError(f"Invalid calibration file {path}: {exc}") from exc


__all__ = [
    "MAX_CALIBRATION_BYTES",
    "CalibrationError",
    "CameraCalibration",
    "ExtrinsicParameters",
    "IntrinsicParameters",
    "load_calibration",
]
