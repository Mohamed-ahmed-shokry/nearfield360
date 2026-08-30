"""Portable, versioned split manifests with identity and provenance checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nearfield360.data.splits import (
    DatasetSplit,
    DatasetSplits,
    SplitError,
    SplitRatios,
    create_splits,
)
from nearfield360.data.woodscape import CameraId, SampleKey, WoodScapeDataset
from nearfield360.utils.artifacts import read_json, write_json


class _Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str = Field(min_length=1, max_length=256, pattern=r"^[^\s\\/]+$")
    camera: CameraId
    split: DatasetSplit
    group: str = Field(min_length=1, max_length=256)

    @property
    def key(self) -> SampleKey:
        return SampleKey(self.frame_id, self.camera)


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(strict=True, ge=1, le=1)
    algorithm: Literal["sha256-group-v1"]
    seed: int = Field(strict=True, ge=0, le=2**64 - 1)
    ratios: SplitRatios
    grouping: Literal["filename_id", "explicit"]
    group_source: str | None
    sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignments: tuple[_Assignment, ...] = Field(max_length=100_000)


def sample_identity_digest(dataset: WoodScapeDataset) -> str:
    """Hash sorted sample identities, NOT image contents or machine-local paths."""
    return hashlib.sha256(
        "\n".join(sorted(sample.key.stem for sample in dataset)).encode()
    ).hexdigest()


def _validate_assignments(dataset: WoodScapeDataset, splits: DatasetSplits) -> None:
    if set(splits.assignments) != {sample.key for sample in dataset}:
        raise SplitError("Manifest assignments must exactly match dataset sample keys")
    expected = create_splits(
        dataset,
        ratios=splits.ratios,
        seed=splits.seed,
        groups=splits.groups if splits.grouping == "explicit" else None,
        group_source=splits.group_source,
    )
    if splits.assignments != expected.assignments:
        raise SplitError(
            "Manifest assignments do not match the declared algorithm, seed, and groups"
        )


def write_split_manifest(
    path: Path,
    dataset: WoodScapeDataset,
    splits: DatasetSplits,
    *,
    overwrite: bool = False,
) -> None:
    """Save a deterministic manifest; refuse to clobber an existing file by default."""
    _validate_assignments(dataset, splits)
    manifest = _Manifest(
        schema_version=1,
        algorithm="sha256-group-v1",
        seed=splits.seed,
        ratios=splits.ratios,
        grouping=splits.grouping,
        group_source=splits.group_source,
        sample_ids_sha256=sample_identity_digest(dataset),
        assignments=tuple(
            _Assignment(
                frame_id=key.frame_id,
                camera=key.camera,
                split=split,
                group=splits.groups[key],
            )
            for key, split in sorted(splits.assignments.items())
        ),
    )
    write_json(path, manifest.model_dump(mode="json"), overwrite=overwrite)


def load_split_manifest(path: Path, dataset: WoodScapeDataset) -> DatasetSplits:
    """Check schema, uniqueness, dataset identity, group isolation, and reproducibility."""
    try:
        manifest = _Manifest.model_validate(read_json(path))
        if manifest.sample_ids_sha256 != sample_identity_digest(dataset):
            raise SplitError("Manifest sample identity digest does not match this dataset")
        keys = [item.key for item in manifest.assignments]
        if len(set(keys)) != len(keys):
            raise SplitError("Manifest contains duplicate sample assignments")
        splits = DatasetSplits(
            seed=manifest.seed,
            ratios=manifest.ratios,
            assignments={item.key: item.split for item in manifest.assignments},
            groups={item.key: item.group for item in manifest.assignments},
            grouping=manifest.grouping,
            group_source=manifest.group_source,
        )
        _validate_assignments(dataset, splits)
        return splits
    except ValueError as exc:
        raise SplitError(f"Invalid split manifest {path}: {exc}") from exc


__all__ = ["load_split_manifest", "sample_identity_digest", "write_split_manifest"]
