from pathlib import Path

import pytest

from nearfield360.data import (
    CameraId,
    DatasetSplit,
    SampleKey,
    SplitError,
    SplitRatios,
    WoodScapeDataset,
    create_splits,
)


def _dataset(root: Path, frame_count: int = 100) -> WoodScapeDataset:
    image_dir = root / "rgb_images"
    image_dir.mkdir()
    for frame_index in range(frame_count):
        for camera in CameraId:
            (image_dir / f"{frame_index:05d}_{camera.value}.png").touch()
    return WoodScapeDataset.discover(root)


def test_create_splits_keeps_all_cameras_from_a_frame_together(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    splits = create_splits(dataset, seed=7)

    for frame_index in range(100):
        assignments = {
            splits.split_for(SampleKey(f"{frame_index:05d}", camera)) for camera in CameraId
        }
        assert len(assignments) == 1
    assert sum(splits.counts.values()) == len(dataset)
    assert all(splits.counts[split] > 0 for split in DatasetSplit)


def test_split_assignment_is_stable_and_seeded(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    first = create_splits(dataset, seed=42)
    repeated = create_splits(dataset, seed=42)
    different_seed = create_splits(dataset, seed=43)

    assert first.assignments == repeated.assignments
    assert first.assignments != different_seed.assignments


def test_existing_assignments_survive_appended_frames(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, frame_count=5)
    initial = create_splits(dataset, seed=9)
    image_dir = tmp_path / "rgb_images"
    for camera in CameraId:
        (image_dir / f"99999_{camera.value}.png").touch()

    expanded = create_splits(WoodScapeDataset.discover(tmp_path), seed=9)

    assert all(expanded.split_for(key) is split for key, split in initial.assignments.items())


def test_samples_returns_requested_partition(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, frame_count=20)
    splits = create_splits(dataset, ratios=SplitRatios(train=1.0, validation=0.0, test=0.0))

    assert splits.samples(dataset, DatasetSplit.TRAIN) == tuple(dataset)
    assert splits.samples(dataset, DatasetSplit.VALIDATION) == ()


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": -0.1, "validation": 0.5, "test": 0.6},
        {"train": float("nan"), "validation": 0.5, "test": 0.5},
        {"train": 0.5, "validation": 0.2, "test": 0.2},
    ],
)
def test_split_ratios_reject_invalid_values(ratios: dict[str, float]) -> None:
    with pytest.raises(SplitError, match="split ratios"):
        SplitRatios(**ratios)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_create_splits_rejects_invalid_seed(tmp_path: Path, seed: int) -> None:
    dataset = _dataset(tmp_path, frame_count=1)

    with pytest.raises(SplitError, match="unsigned 64-bit"):
        create_splits(dataset, seed=seed)


def test_split_for_rejects_unknown_sample(tmp_path: Path) -> None:
    splits = create_splits(_dataset(tmp_path, frame_count=1))

    with pytest.raises(SplitError, match="No split assignment"):
        splits.split_for(SampleKey("missing", CameraId.FRONT))
