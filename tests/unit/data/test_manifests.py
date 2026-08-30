import json
from pathlib import Path

import pytest

from nearfield360.data.manifests import (
    load_split_manifest,
    sample_identity_digest,
    write_split_manifest,
)
from nearfield360.data.splits import SplitError, create_splits
from nearfield360.data.woodscape import CameraId, SampleKey, WoodScapeDataset, WoodScapeSample
from nearfield360.utils.artifacts import ArtifactError


def _dataset(root: Path, count: int = 12) -> WoodScapeDataset:
    return WoodScapeDataset(
        root,
        [
            WoodScapeSample(SampleKey(str(i), CameraId.FRONT), root / f"{i}_FV.png")
            for i in range(count)
        ],
    )


def test_split_manifest_is_portable_deterministic_and_roundtrips(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    splits = create_splits(dataset)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_split_manifest(first, dataset, splits)
    write_split_manifest(second, dataset, splits)

    assert first.read_bytes() == second.read_bytes()
    relocated = _dataset(tmp_path / "other-machine")
    loaded = load_split_manifest(first, relocated)
    assert loaded == splits
    assert sample_identity_digest(relocated) == sample_identity_digest(dataset)
    assert str(tmp_path) not in first.read_text(encoding="utf-8")


def test_explicit_group_provenance_survives_roundtrip(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    groups = {sample.key: f"sequence-{int(sample.key.frame_id) // 3}" for sample in dataset}
    splits = create_splits(dataset, groups=groups, group_source="audited recording map")
    path = tmp_path / "split.json"
    write_split_manifest(path, dataset, splits)

    loaded = load_split_manifest(path, dataset)
    assert loaded.groups == groups
    assert loaded.group_source == "audited recording map"
    assert loaded.grouping == "explicit"


@pytest.mark.parametrize("mutation", ["version", "duplicate", "missing", "seed", "digest", "extra"])
def test_manifest_rejects_inconsistent_or_unsupported_metadata(
    tmp_path: Path, mutation: str
) -> None:
    dataset = _dataset(tmp_path)
    path = tmp_path / "split.json"
    write_split_manifest(path, dataset, create_splits(dataset))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "version":
        payload["schema_version"] = 2
    elif mutation == "duplicate":
        payload["assignments"].append(payload["assignments"][0])
    elif mutation == "missing":
        payload["assignments"].pop()
    elif mutation == "seed":
        payload["seed"] = 43
    elif mutation == "digest":
        payload["sample_ids_sha256"] = "0" * 64
    else:
        payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SplitError, match="Invalid split manifest"):
        load_split_manifest(path, dataset)


def test_manifest_rejects_changed_dataset_and_overwrite(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = tmp_path / "split.json"
    splits = create_splits(dataset)
    write_split_manifest(path, dataset, splits)
    with pytest.raises(SplitError, match="identity digest"):
        load_split_manifest(path, _dataset(tmp_path, count=13))
    with pytest.raises(ArtifactError, match="Unable to write"):
        write_split_manifest(path, dataset, splits)
    write_split_manifest(path, dataset, splits, overwrite=True)
    assert load_split_manifest(path, dataset) == splits
