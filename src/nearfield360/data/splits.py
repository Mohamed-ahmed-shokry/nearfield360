"""Stable train/validation/test assignment with explicit grouping provenance."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from nearfield360.data.woodscape import SampleKey, WoodScapeDataset, WoodScapeSample


class SplitError(ValueError):
    """Raised when split configuration or assignments are incomplete."""


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitRatios:
    """Dataset split proportions, which must be finite and sum to one."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise SplitError("split ratios must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SplitError(f"split ratios must sum to 1.0, got {sum(values):.12g}")


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable sample assignments plus their reproducibility inputs."""

    seed: int
    ratios: SplitRatios
    assignments: Mapping[SampleKey, DatasetSplit]
    groups: Mapping[SampleKey, str]
    grouping: Literal["filename_id", "explicit"]
    group_source: str | None = None

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        if set(self.assignments) != set(self.groups):
            raise SplitError("group keys must exactly match split assignment keys")
        _validate_groups(self.groups, self.grouping, self.group_source)
        group_splits: dict[str, DatasetSplit] = {}
        for key, split in self.assignments.items():
            if not isinstance(split, DatasetSplit):
                raise SplitError(f"Invalid split assignment for {key.stem}: {split}")
            group = self.groups[key]
            if group in group_splits and group_splits[group] is not split:
                raise SplitError(f"Group appears in more than one split: {group}")
            group_splits[group] = split
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    @property
    def counts(self) -> Mapping[DatasetSplit, int]:
        counts = Counter(self.assignments.values())
        return MappingProxyType({split: counts[split] for split in DatasetSplit})

    def split_for(self, key: SampleKey) -> DatasetSplit:
        try:
            return self.assignments[key]
        except KeyError as exc:
            raise SplitError(f"No split assignment for sample: {key.stem}") from exc

    def samples(
        self, dataset: WoodScapeDataset, split: DatasetSplit
    ) -> tuple[WoodScapeSample, ...]:
        return tuple(sample for sample in dataset if self.split_for(sample.key) is split)


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise SplitError("split seed must be an unsigned 64-bit integer")


def _validate_groups(
    groups: Mapping[SampleKey, str],
    grouping: Literal["filename_id", "explicit"],
    source: str | None,
) -> None:
    if grouping not in ("filename_id", "explicit"):
        raise SplitError(f"Unknown grouping policy: {grouping}")
    if grouping == "explicit" and (not isinstance(source, str) or not source.strip()):
        raise SplitError("Explicit groups require a non-empty group_source")
    if grouping == "filename_id" and source is not None:
        raise SplitError("group_source is only allowed with explicit groups")
    for key, group in groups.items():
        if not isinstance(group, str) or not group.strip() or len(group) > 256:
            raise SplitError(
                f"Group for {key.stem} must be a non-empty string of at most 256 chars"
            )
        if grouping == "filename_id" and group != key.frame_id:
            raise SplitError(f"Filename grouping must use the filename identifier: {key.stem}")


def _group_score(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"nearfield360:{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64


def create_splits(
    dataset: WoodScapeDataset,
    ratios: SplitRatios | None = None,
    *,
    seed: int = 42,
    groups: Mapping[SampleKey, str] | None = None,
    group_source: str | None = None,
) -> DatasetSplits:
    """Hash entire groups without reshuffling when unrelated samples are appended.

    The default only groups equal filename identifiers: it does NOT establish synchronized
    cameras or sequence-level isolation. Supply verified recording/sequence groups and their
    provenance for evaluation that requires those guarantees.
    """
    ratios = SplitRatios() if ratios is None else ratios
    _validate_seed(seed)
    grouping: Literal["filename_id", "explicit"] = "filename_id" if groups is None else "explicit"
    groups = {sample.key: sample.key.frame_id for sample in dataset} if groups is None else groups
    if set(groups) != {sample.key for sample in dataset}:
        raise SplitError("group keys must exactly match dataset sample keys")
    _validate_groups(groups, grouping, group_source)

    group_assignments: dict[str, DatasetSplit] = {}
    train_threshold = ratios.train
    validation_threshold = ratios.train + ratios.validation
    for group in sorted(set(groups.values())):
        score = _group_score(group, seed)
        if score < train_threshold:
            group_assignments[group] = DatasetSplit.TRAIN
        elif score < validation_threshold:
            group_assignments[group] = DatasetSplit.VALIDATION
        else:
            group_assignments[group] = DatasetSplit.TEST

    assignments = {sample.key: group_assignments[groups[sample.key]] for sample in dataset}
    return DatasetSplits(
        seed=seed,
        ratios=ratios,
        assignments=MappingProxyType(assignments),
        groups=groups,
        grouping=grouping,
        group_source=group_source,
    )


__all__ = [
    "DatasetSplit",
    "DatasetSplits",
    "SplitError",
    "SplitRatios",
    "create_splits",
]
