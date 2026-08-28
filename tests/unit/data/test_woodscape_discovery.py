from pathlib import Path

import pytest

from nearfield360.data import (
    CameraId,
    DatasetLayoutError,
    SampleKey,
    WoodScapeDataset,
    parse_sample_key,
)


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_parse_sample_key_supports_all_cameras_and_underscored_frames() -> None:
    assert parse_sample_key(Path("00001_FV.png")) == SampleKey("00001", CameraId.FRONT)
    assert CameraId.FRONT.description == "front"
    assert parse_sample_key(Path("sequence_0042_MVL.jpg")) == SampleKey(
        "sequence_0042", CameraId.MIRROR_LEFT
    )
    assert parse_sample_key(Path("00001_RV_prev.png"), previous=True) == SampleKey(
        "00001", CameraId.REAR
    )


@pytest.mark.parametrize(
    "name",
    ["no-camera.png", "00001_UNKNOWN.png", "_FV.png", "bad frame_MVR.png"],
)
def test_parse_sample_key_rejects_invalid_names(name: str) -> None:
    with pytest.raises(DatasetLayoutError):
        parse_sample_key(Path(name))


def test_previous_file_requires_explicit_suffix() -> None:
    with pytest.raises(DatasetLayoutError, match="must end with '_prev'"):
        parse_sample_key(Path("00001_FV.png"), previous=True)


def test_discover_associates_optional_files_and_sorts_samples(tmp_path: Path) -> None:
    data_root = tmp_path / "extracted" / "data"
    _touch(data_root / "rgb_images", "00002_RV.png", "00001_FV.png", "notes.txt")
    _touch(data_root / "previous_images", "00001_FV_prev.png")
    _touch(data_root / "semantic_annotations" / "gtLabels", "00001_FV.png")
    _touch(data_root / "calibration_data" / "calibration", "00001_FV.json")

    dataset = WoodScapeDataset.discover(tmp_path / "extracted")

    assert dataset.root == data_root.resolve()
    assert len(dataset) == 2
    assert [sample.key.stem for sample in dataset] == ["00001_FV", "00002_RV"]
    assert dataset[0].previous_image_path == (data_root / "previous_images/00001_FV_prev.png")
    assert dataset[0].semantic_mask_path == (
        data_root / "semantic_annotations/gtLabels/00001_FV.png"
    )
    assert dataset[0].calibration_path == (data_root / "calibration_data/calibration/00001_FV.json")
    assert dataset[1].previous_image_path is None
    assert dataset.for_camera(CameraId.FRONT) == (dataset[0],)
    assert dataset.get(SampleKey("00002", CameraId.REAR)) == dataset[1]


def test_discover_rejects_missing_or_empty_rgb_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetLayoutError, match="No 'rgb_images'"):
        WoodScapeDataset.discover(tmp_path)

    (tmp_path / "rgb_images").mkdir()
    with pytest.raises(DatasetLayoutError, match="No supported RGB images"):
        WoodScapeDataset.discover(tmp_path)


def test_discover_rejects_ambiguous_nested_roots(tmp_path: Path) -> None:
    (tmp_path / "rgb_images").mkdir()
    (tmp_path / "data" / "rgb_images").mkdir(parents=True)

    with pytest.raises(DatasetLayoutError, match="Ambiguous WoodScape roots"):
        WoodScapeDataset.discover(tmp_path)


def test_discover_rejects_duplicate_sample_files(tmp_path: Path) -> None:
    _touch(tmp_path / "rgb_images", "00001_FV.jpg", "00001_FV.png")

    with pytest.raises(DatasetLayoutError, match="Duplicate files"):
        WoodScapeDataset.discover(tmp_path)


def test_discover_rejects_orphan_annotations(tmp_path: Path) -> None:
    _touch(tmp_path / "rgb_images", "00001_FV.png")
    _touch(tmp_path / "semantic_annotations" / "gtLabels", "00002_RV.png")

    with pytest.raises(DatasetLayoutError, match="Semantic mask files without RGB images"):
        WoodScapeDataset.discover(tmp_path)


def test_get_reports_unknown_sample(tmp_path: Path) -> None:
    _touch(tmp_path / "rgb_images", "00001_FV.png")
    dataset = WoodScapeDataset.discover(tmp_path)

    with pytest.raises(KeyError, match="Unknown WoodScape sample: missing_MVR"):
        dataset.get(SampleKey("missing", CameraId.MIRROR_RIGHT))
