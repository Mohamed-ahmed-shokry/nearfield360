"""Stable, frame-grouped train/validation/test assignment."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

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


def _frame_score(frame_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"nearfield360:{seed}:{frame_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64


def create_splits(
    dataset: WoodScapeDataset,
    ratios: SplitRatios | None = None,
    *,
    seed: int = 42,
) -> DatasetSplits:
    """Assign entire frame groups using a stable score that is independent of dataset order."""
    ratios = SplitRatios() if ratios is None else ratios
    if not 0 <= seed <= 2**64 - 1:
        raise SplitError("split seed must be an unsigned 64-bit integer")

    frame_assignments: dict[str, DatasetSplit] = {}
    train_threshold = ratios.train
    validation_threshold = ratios.train + ratios.validation
    for frame_id in sorted({sample.key.frame_id for sample in dataset}):
        score = _frame_score(frame_id, seed)
        if score < train_threshold:
            frame_assignments[frame_id] = DatasetSplit.TRAIN
        elif score < validation_threshold:
            frame_assignments[frame_id] = DatasetSplit.VALIDATION
        else:
            frame_assignments[frame_id] = DatasetSplit.TEST

    assignments = {sample.key: frame_assignments[sample.key.frame_id] for sample in dataset}
    return DatasetSplits(
        seed=seed,
        ratios=ratios,
        assignments=MappingProxyType(assignments),
    )


__all__ = [
    "DatasetSplit",
    "DatasetSplits",
    "SplitError",
    "SplitRatios",
    "create_splits",
]
