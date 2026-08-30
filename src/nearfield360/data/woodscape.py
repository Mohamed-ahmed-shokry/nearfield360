"""Deterministic discovery of a locally acquired WoodScape dataset."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})


class DatasetLayoutError(ValueError):
    """Raised when a dataset root is missing, ambiguous, or internally inconsistent."""


class CameraId(StrEnum):
    """Official WoodScape filename codes for the four surround-view cameras."""

    FRONT = "FV"
    REAR = "RV"
    MIRROR_LEFT = "MVL"
    MIRROR_RIGHT = "MVR"

    @property
    def description(self) -> str:
        return {
            CameraId.FRONT: "front",
            CameraId.REAR: "rear",
            CameraId.MIRROR_LEFT: "left mirror",
            CameraId.MIRROR_RIGHT: "right mirror",
        }[self]


@dataclass(frozen=True, order=True)
class SampleKey:
    """Stable identity shared by an image and all of its annotations."""

    frame_id: str
    camera: CameraId

    @property
    def stem(self) -> str:
        return f"{self.frame_id}_{self.camera.value}"


@dataclass(frozen=True)
class WoodScapeSample:
    """Paths associated with one camera frame; optional files are not fabricated."""

    key: SampleKey
    image_path: Path
    previous_image_path: Path | None = None
    semantic_mask_path: Path | None = None
    calibration_path: Path | None = None


def parse_sample_key(path: Path, *, previous: bool = False) -> SampleKey:
    """Parse ``<frame>_<camera>`` (and optional ``_prev``) without assuming numeric IDs."""
    stem = path.stem
    if previous:
        if not stem.endswith("_prev"):
            raise DatasetLayoutError(f"Previous image name must end with '_prev': {path.name}")
        stem = stem.removesuffix("_prev")

    try:
        frame_id, camera_code = stem.rsplit("_", maxsplit=1)
    except ValueError as exc:
        raise DatasetLayoutError(
            f"Expected '<frame>_<camera>' filename, received: {path.name}"
        ) from exc

    if not frame_id or any(character.isspace() for character in frame_id):
        raise DatasetLayoutError(f"Invalid WoodScape frame identifier: {path.name}")
    try:
        camera = CameraId(camera_code)
    except ValueError as exc:
        expected = ", ".join(camera.value for camera in CameraId)
        raise DatasetLayoutError(
            f"Unknown camera code '{camera_code}' in {path.name}; expected one of {expected}"
        ) from exc
    return SampleKey(frame_id=frame_id, camera=camera)


def locate_dataset_root(root: Path) -> Path:
    """Accept either the extracted data directory or its immediate archive parent."""
    expanded = root.expanduser()
    candidates = [
        candidate for candidate in (expanded, expanded / "data") if _has_rgb_dir(candidate)
    ]
    if not candidates:
        raise DatasetLayoutError(
            f"No 'rgb_images' directory found at {expanded} or {expanded / 'data'}"
        )
    if len(candidates) > 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise DatasetLayoutError(f"Ambiguous WoodScape roots containing rgb_images: {rendered}")
    return candidates[0].resolve()


def _has_rgb_dir(candidate: Path) -> bool:
    return candidate.is_dir() and (candidate / "rgb_images").is_dir()


def _index_files(
    directory: Path | None,
    *,
    suffixes: frozenset[str],
    previous: bool = False,
) -> dict[SampleKey, Path]:
    if directory is None:
        return {}

    indexed: dict[SampleKey, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = parse_sample_key(path, previous=previous)
        if key in indexed:
            raise DatasetLayoutError(
                f"Duplicate files for sample {key.stem}: {indexed[key].name}, {path.name}"
            )
        indexed[key] = path.resolve()
    return indexed


def _optional_directory(root: Path, candidates: Sequence[Path]) -> Path | None:
    matches = [root / candidate for candidate in candidates if (root / candidate).is_dir()]
    return matches[0] if matches else None


def _index_calibrations(root: Path) -> dict[SampleKey, Path]:
    """Accept either archive layout, but never silently discard populated alternatives."""
    indexed: dict[SampleKey, Path] = {}
    for directory in (root / "calibration_data", root / "calibration_data/calibration"):
        candidate = _index_files(
            directory if directory.is_dir() else None, suffixes=frozenset({".json"})
        )
        if indexed and candidate:
            raise DatasetLayoutError(
                "Ambiguous calibration layout: JSON files exist in both calibration_data "
                "and calibration_data/calibration"
            )
        indexed.update(candidate)
    return indexed


def _reject_orphans(
    primary: dict[SampleKey, Path],
    auxiliary: dict[SampleKey, Path],
    label: str,
) -> None:
    orphaned = sorted(set(auxiliary).difference(primary))
    if orphaned:
        examples = ", ".join(key.stem for key in orphaned[:3])
        remainder = " ..." if len(orphaned) > 3 else ""
        raise DatasetLayoutError(f"{label} files without RGB images: {examples}{remainder}")


class WoodScapeDataset:
    """Immutable, deterministic index over WoodScape camera samples."""

    __slots__ = ("_by_key", "_samples", "root")

    def __init__(self, root: Path, samples: Sequence[WoodScapeSample]) -> None:
        self.root = root
        self._samples = tuple(samples)
        self._by_key = {sample.key: sample for sample in self._samples}
        if len(self._by_key) != len(self._samples):
            raise DatasetLayoutError("Dataset samples must have unique sample keys")

    @classmethod
    def discover(cls, root: Path) -> WoodScapeDataset:
        dataset_root = locate_dataset_root(root)
        images = _index_files(dataset_root / "rgb_images", suffixes=IMAGE_SUFFIXES)
        if not images:
            raise DatasetLayoutError(
                f"No supported RGB images found in {dataset_root / 'rgb_images'}"
            )

        previous_directory = _optional_directory(dataset_root, (Path("previous_images"),))
        semantic_directory = _optional_directory(
            dataset_root, (Path("semantic_annotations/gtLabels"),)
        )

        previous_images = _index_files(
            previous_directory,
            suffixes=IMAGE_SUFFIXES,
            previous=True,
        )
        semantic_masks = _index_files(semantic_directory, suffixes=frozenset({".png"}))
        calibrations = _index_calibrations(dataset_root)

        _reject_orphans(images, previous_images, "Previous image")
        _reject_orphans(images, semantic_masks, "Semantic mask")
        _reject_orphans(images, calibrations, "Calibration")

        ordered_keys = sorted(images, key=lambda key: (key.frame_id, key.camera.value))
        samples = [
            WoodScapeSample(
                key=key,
                image_path=images[key],
                previous_image_path=previous_images.get(key),
                semantic_mask_path=semantic_masks.get(key),
                calibration_path=calibrations.get(key),
            )
            for key in ordered_keys
        ]
        return cls(root=dataset_root, samples=samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[WoodScapeSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> WoodScapeSample:
        return self._samples[index]

    def get(self, key: SampleKey) -> WoodScapeSample:
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"Unknown WoodScape sample: {key.stem}") from exc

    def for_camera(self, camera: CameraId) -> tuple[WoodScapeSample, ...]:
        return tuple(sample for sample in self._samples if sample.key.camera is camera)


__all__ = [
    "IMAGE_SUFFIXES",
    "CameraId",
    "DatasetLayoutError",
    "SampleKey",
    "WoodScapeDataset",
    "WoodScapeSample",
    "locate_dataset_root",
    "parse_sample_key",
]
