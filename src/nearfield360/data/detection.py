"""Bounded parsing of the official five-class WoodScape detection TXT format.

Each CSV row contains ``class_name,class_id,x_min,y_min,x_max,y_max`` in stored
image pixels. These class IDs are not semantic-segmentation IDs. Coordinates
use differences for width/height, without an inclusive-pixel ``+1`` convention.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import MappingProxyType

_INTEGER_FIELD = re.compile(r"[+-]?[0-9]+")
_NUMBER_FIELD = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class DetectionAnnotationError(ValueError):
    """Raised when detection annotations violate their file or value contract."""


@dataclass(frozen=True)
class DetectionClass:
    """One class from the official detection-specific five-class mapping."""

    class_id: int
    name: str


WOODSCAPE_DETECTION_CLASSES = (
    DetectionClass(0, "vehicles"),
    DetectionClass(1, "person"),
    DetectionClass(2, "bicycle"),
    DetectionClass(3, "traffic_light"),
    DetectionClass(4, "traffic_sign"),
)
_CLASS_BY_ID = MappingProxyType({item.class_id: item for item in WOODSCAPE_DETECTION_CLASSES})


def detection_class(class_id: int) -> DetectionClass:
    """Return detection class metadata, rejecting semantic IDs and non-integers."""
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        raise DetectionAnnotationError("detection class_id must be an integer")
    try:
        return _CLASS_BY_ID[class_id]
    except KeyError as exc:
        raise DetectionAnnotationError(f"Unknown WoodScape detection class_id: {class_id}") from exc


@dataclass(frozen=True)
class DetectionAnnotation:
    """A validated detection box in absolute stored-image XYXY coordinates."""

    class_id: int
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        expected = detection_class(self.class_id)
        if self.class_name != expected.name:
            raise DetectionAnnotationError(
                f"class_name {self.class_name!r} does not match class_id {self.class_id} "
                f"({expected.name!r})"
            )
        for name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise DetectionAnnotationError(f"{name} must be a finite numeric coordinate")
            try:
                coordinate = float(value)
            except OverflowError as exc:
                raise DetectionAnnotationError(f"{name} must be finite") from exc
            if not math.isfinite(coordinate) or coordinate < 0.0:
                raise DetectionAnnotationError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, coordinate)
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise DetectionAnnotationError("XYXY box must have positive width and height")
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise DetectionAnnotationError("XYXY box area must be finite and positive")

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class DetectionLimits:
    """Bound file bytes and annotation count independently."""

    max_file_bytes: int = 4 * 1024 * 1024
    max_objects: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_objects"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_DETECTION_LIMITS = DetectionLimits()


def _read_document(path: Path, limits: DetectionLimits) -> str:
    try:
        if not path.is_file():
            raise DetectionAnnotationError(f"Detection annotation file does not exist: {path}")
        with path.open("rb") as stream:
            payload = stream.read(limits.max_file_bytes + 1)
    except OSError as exc:
        raise DetectionAnnotationError(
            f"Unable to read detection annotation file {path}: {exc}"
        ) from exc
    if len(payload) > limits.max_file_bytes:
        raise DetectionAnnotationError(
            f"Detection annotation file exceeds the {limits.max_file_bytes}-byte safety limit: "
            f"{path}"
        )
    try:
        return payload.decode("utf-8").removeprefix("\ufeff")
    except UnicodeDecodeError as exc:
        line_number = (
            payload.count(b"\r", 0, exc.start)
            + payload.count(b"\n", 0, exc.start)
            - payload.count(b"\r\n", 0, exc.start)
            + 1
        )
        raise DetectionAnnotationError(f"{path}: line {line_number}: invalid UTF-8") from exc


def _validate_image_size(image_size: tuple[int, int] | None) -> None:
    if image_size is not None and (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in image_size
        )
    ):
        raise DetectionAnnotationError(
            "image_size must be a (height, width) pair of positive integers"
        )


def _parse_row(line: str, image_size: tuple[int, int] | None) -> DetectionAnnotation:
    fields = next(csv.reader([line], strict=True))
    if len(fields) != 6:
        raise DetectionAnnotationError(f"Expected six CSV fields, got {len(fields)}")
    class_name, class_id_text, *coordinate_fields = (field.strip() for field in fields)
    if _INTEGER_FIELD.fullmatch(class_id_text) is None:
        raise DetectionAnnotationError("class_id must be a base-10 integer")
    for name, value in zip(("x_min", "y_min", "x_max", "y_max"), coordinate_fields, strict=True):
        if _NUMBER_FIELD.fullmatch(value) is None:
            raise DetectionAnnotationError(f"{name} must be a finite numeric coordinate")
    annotation = DetectionAnnotation(
        class_id=int(class_id_text),
        class_name=class_name,
        x_min=float(coordinate_fields[0]),
        y_min=float(coordinate_fields[1]),
        x_max=float(coordinate_fields[2]),
        y_max=float(coordinate_fields[3]),
    )
    if image_size is not None:
        height, width = image_size
        if annotation.x_max > width or annotation.y_max > height:
            raise DetectionAnnotationError(
                f"XYXY box {annotation.xyxy} exceeds image size (height={height}, width={width})"
            )
    return annotation


def load_detection_annotations(
    path: Path,
    *,
    image_size: tuple[int, int] | None = None,
    limits: DetectionLimits = DEFAULT_DETECTION_LIMITS,
) -> tuple[DetectionAnnotation, ...]:
    """Read UTF-8 CSV rows with path/physical-line errors and optional image bounds.

    Empty files and blank lines are valid. A leading UTF-8 BOM is tolerated, but
    headers, comments, and multiline records are not part of this format. Bounds
    permit ``x_max == width`` and ``y_max == height``; boxes are never clipped.
    """
    _validate_image_size(image_size)
    document = _read_document(path, limits)
    annotations: list[DetectionAnnotation] = []
    for line_number, line in enumerate(StringIO(document, newline=None), start=1):
        if not line.strip():
            continue
        if len(annotations) >= limits.max_objects:
            raise DetectionAnnotationError(
                f"{path}: line {line_number}: exceeds the {limits.max_objects}-object safety limit"
            )
        try:
            annotations.append(_parse_row(line, image_size))
        except (ValueError, csv.Error) as exc:
            raise DetectionAnnotationError(f"{path}: line {line_number}: {exc}") from exc
    return tuple(annotations)


__all__ = [
    "DEFAULT_DETECTION_LIMITS",
    "WOODSCAPE_DETECTION_CLASSES",
    "DetectionAnnotation",
    "DetectionAnnotationError",
    "DetectionClass",
    "DetectionLimits",
    "detection_class",
    "load_detection_annotations",
]
