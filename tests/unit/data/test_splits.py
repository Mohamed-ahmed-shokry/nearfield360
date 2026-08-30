from pathlib import Path

import pytest

from nearfield360.data import (
    CameraId,
    DatasetSplit,
    DatasetSplits,
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


def test_create_splits_keeps_equal_filename_identifiers_together(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("seed", [-1, 2**64, 42.0, True])
def test_create_splits_rejects_invalid_seed(tmp_path: Path, seed: int) -> None:
    dataset = _dataset(tmp_path, frame_count=1)

    with pytest.raises(SplitError, match="unsigned 64-bit"):
        create_splits(dataset, seed=seed)


def test_split_for_rejects_unknown_sample(tmp_path: Path) -> None:
    splits = create_splits(_dataset(tmp_path, frame_count=1))

    with pytest.raises(SplitError, match="No split assignment"):
        splits.split_for(SampleKey("missing", CameraId.FRONT))


def test_explicit_recording_groups_remain_together_and_are_copied(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, frame_count=10)
    groups = {sample.key: f"recording-{int(sample.key.frame_id) // 2}" for sample in dataset}

    splits = create_splits(dataset, groups=groups, group_source="verified recording manifest")

    assert splits.grouping == "explicit"
    assert splits.group_source == "verified recording manifest"
    for group in set(groups.values()):
        assert len({splits.split_for(key) for key, value in groups.items() if value == group}) == 1
    groups[dataset[0].key] = "changed"
    assert splits.groups[dataset[0].key] == "recording-0"


def test_explicit_group_mapping_must_be_complete_and_explain_its_source(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, frame_count=1)
    with pytest.raises(SplitError, match="exactly match"):
        create_splits(dataset, groups={}, group_source="recordings")
    groups = dict.fromkeys((sample.key for sample in dataset), "recording-1")
    with pytest.raises(SplitError, match="group_source"):
        create_splits(dataset, groups=groups)
    with pytest.raises(SplitError, match="only allowed with explicit"):
        create_splits(dataset, group_source="not supplied")
    groups[dataset[0].key] = " "
    with pytest.raises(SplitError, match="non-empty string"):
        create_splits(dataset, groups=groups, group_source="recordings")


def test_split_records_reject_group_leakage(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, frame_count=1)
    assignments = dict.fromkeys((sample.key for sample in dataset), DatasetSplit.TRAIN)
    assignments[dataset[0].key] = DatasetSplit.TEST

    with pytest.raises(SplitError, match="more than one split"):
        DatasetSplits(
            seed=42,
            ratios=SplitRatios(),
            assignments=assignments,
            groups=dict.fromkeys(assignments, "same-recording"),
            grouping="explicit",
            group_source="verified recording manifest",
        )
