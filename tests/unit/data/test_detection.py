from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from nearfield360.data.detection import (
    WOODSCAPE_DETECTION_CLASSES,
    DetectionAnnotation,
    DetectionAnnotationError,
    DetectionLimits,
    DetectionPrediction,
    detection_class,
    load_detection_annotations,
    load_detection_predictions,
)


def _write_annotations(tmp_path: Path, document: str) -> Path:
    path = tmp_path / "00001_FV.txt"
    path.write_text(document, encoding="utf-8")
    return path


def test_detection_classes_match_official_five_class_mapping() -> None:
    assert [(item.class_id, item.name) for item in WOODSCAPE_DETECTION_CLASSES] == [
        (0, "vehicles"),
        (1, "person"),
        (2, "bicycle"),
        (3, "traffic_light"),
        (4, "traffic_sign"),
    ]
    assert detection_class(0) is WOODSCAPE_DETECTION_CLASSES[0]
    # In the semantic taxonomy, vehicles are label 6; that ID is not a detector class.
    with pytest.raises(DetectionAnnotationError, match="Unknown WoodScape detection class_id"):
        detection_class(6)


@pytest.mark.parametrize("class_id", [-1, 5, True, 1.0, "1"])
def test_detection_class_rejects_invalid_ids(class_id: object) -> None:
    with pytest.raises(DetectionAnnotationError):
        detection_class(class_id)


def test_parser_reads_all_classes_and_absolute_xyxy_boxes(tmp_path: Path) -> None:
    document = "\n".join(
        f"{item.name},{item.class_id},10,20,30,50" for item in WOODSCAPE_DETECTION_CLASSES
    )
    annotations = load_detection_annotations(_write_annotations(tmp_path, document))

    assert isinstance(annotations, tuple)
    assert len(annotations) == 5
    assert [item.class_id for item in annotations] == list(range(5))
    assert annotations[0].xyxy == (10.0, 20.0, 30.0, 50.0)
    assert annotations[0].width == 20.0
    assert annotations[0].height == 30.0
    assert annotations[0].area == 600.0
    with pytest.raises(FrozenInstanceError):
        annotations[0].x_min = 11


def test_parser_supports_csv_quotes_spaces_and_finite_fractional_coordinates(
    tmp_path: Path,
) -> None:
    path = _write_annotations(tmp_path, '"person", 1, .5, 1.25, 2e1, 30.\r\n')

    assert load_detection_annotations(path)[0] == DetectionAnnotation(
        1, "person", 0.5, 1.25, 20.0, 30.0
    )


@pytest.mark.parametrize("document", ["", "\n\r\n \t\n", "\ufeff"])
def test_empty_or_blank_annotation_files_are_valid(tmp_path: Path, document: str) -> None:
    assert load_detection_annotations(_write_annotations(tmp_path, document)) == ()


def test_parser_accepts_utf8_bom_and_cr_line_endings(tmp_path: Path) -> None:
    path = _write_annotations(tmp_path, "\ufeffvehicles,0,0,0,1,2\rperson,1,1,2,3,4\r")
    assert len(load_detection_annotations(path)) == 2


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("vehicles,0,0,1,2", "six CSV fields"),
        ("vehicles,0,0,1,2,3,4", "six CSV fields"),
        ("vehicles,1,0,0,1,2", "does not match"),
        ("vehicle,0,0,0,1,2", "does not match"),
        ("VEHICLES,0,0,0,1,2", "does not match"),
        ("person,5,0,0,1,2", "Unknown WoodScape detection class_id"),
        ("person,-1,0,0,1,2", "Unknown WoodScape detection class_id"),
        ("person,1.0,0,0,1,2", "base-10 integer"),
        ("person,True,0,0,1,2", "base-10 integer"),
        ("person,,0,0,1,2", "base-10 integer"),
        ("person,1_0,0,0,1,2", "base-10 integer"),
        ("person,1,zero,0,1,2", "finite numeric coordinate"),
        ("person,1,,0,1,2", "finite numeric coordinate"),
        ("person,1,0,0,1_0,2", "finite numeric coordinate"),
        ("person,1,nan,0,1,2", "finite numeric coordinate"),
        ("person,1,0,0,inf,2", "finite numeric coordinate"),
        ("person,1,0,0,2,-Infinity", "finite numeric coordinate"),
        ("person,1,0,0,1e309,2", "finite and non-negative"),
        ("person,1,-1,0,1,2", "finite and non-negative"),
        ("person,1,0,-1,1,2", "finite and non-negative"),
        ("person,1,1,0,1,2", "positive width and height"),
        ("person,1,2,0,1,2", "positive width and height"),
        ("person,1,0,2,1,2", "positive width and height"),
        ("person,1,0,3,1,2", "positive width and height"),
        ("person,1,0,0,1e308,1e308", "area must be finite and positive"),
        ("person,1,0,0,1e-300,1e-300", "area must be finite and positive"),
        ('"vehicles,0,0,0,1,2', "unexpected end of data"),
        ("class_name,class_id,x_min,y_min,x_max,y_max", "base-10 integer"),
        ("# vehicles,0,0,0,1,2", "does not match"),
    ],
)
def test_parser_rejects_invalid_rows_with_path_and_line(
    tmp_path: Path, row: str, message: str
) -> None:
    path = _write_annotations(tmp_path, "\nvehicles,0,0,0,1,2\n" + row)

    with pytest.raises(DetectionAnnotationError, match=message) as caught:
        load_detection_annotations(path)

    assert str(path) in str(caught.value)
    assert "line 3:" in str(caught.value)


def test_parser_rejects_multiline_csv_records(tmp_path: Path) -> None:
    path = _write_annotations(tmp_path, '"vehicles\n",0,0,0,1,2')
    with pytest.raises(DetectionAnnotationError, match="line 1:"):
        load_detection_annotations(path)


def test_image_size_uses_height_width_and_accepts_right_bottom_edges(tmp_path: Path) -> None:
    path = _write_annotations(tmp_path, "vehicles,0,0,0,20,5")
    assert load_detection_annotations(path, image_size=(5, 20))[0].xyxy == (0, 0, 20, 5)


@pytest.mark.parametrize("row", ["vehicles,0,0,0,21,5", "vehicles,0,0,0,20,6"])
def test_parser_rejects_out_of_bounds_boxes_without_clipping(tmp_path: Path, row: str) -> None:
    path = _write_annotations(tmp_path, row)
    with pytest.raises(DetectionAnnotationError, match=r"line 1:.*exceeds image size"):
        load_detection_annotations(path, image_size=(5, 20))


@pytest.mark.parametrize("size", [(0, 3), (3, -1), (True, 3), (2.5, 3), (3,), [2, 3]])
def test_parser_rejects_invalid_image_size(tmp_path: Path, size: object) -> None:
    path = _write_annotations(tmp_path, "")
    with pytest.raises(DetectionAnnotationError, match="image_size"):
        load_detection_annotations(path, image_size=size)


def test_object_limit_counts_only_nonblank_rows(tmp_path: Path) -> None:
    limits = DetectionLimits(max_objects=1)
    path = _write_annotations(tmp_path, "\nvehicles,0,0,0,1,2\n\n")
    assert len(load_detection_annotations(path, limits=limits)) == 1
    path.write_text("\nvehicles,0,0,0,1,2\n\nperson,1,0,0,2,3", encoding="utf-8")
    with pytest.raises(DetectionAnnotationError, match="line 4: exceeds the 1-object"):
        load_detection_annotations(path, limits=limits)


def test_file_limit_allows_exact_size_and_rejects_one_byte_over(tmp_path: Path) -> None:
    document = "vehicles,0,0,0,1,2"
    path = _write_annotations(tmp_path, document)
    assert (
        len(load_detection_annotations(path, limits=DetectionLimits(max_file_bytes=len(document))))
        == 1
    )
    with pytest.raises(DetectionAnnotationError, match="byte safety limit"):
        load_detection_annotations(path, limits=DetectionLimits(max_file_bytes=len(document) - 1))


def test_parser_caps_snapshot_read() -> None:
    path = MagicMock(spec=Path)
    path.is_file.return_value = True
    reader = Mock(wraps=BytesIO(b"x" * 100))
    path.open.return_value.__enter__.return_value = reader
    with pytest.raises(DetectionAnnotationError, match="byte safety limit"):
        load_detection_annotations(path, limits=DetectionLimits(max_file_bytes=16))
    reader.read.assert_called_once_with(17)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_parser_rejects_invalid_utf8_with_physical_line(tmp_path: Path, newline: bytes) -> None:
    path = tmp_path / "bad-encoding.txt"
    path.write_bytes(b"vehicles,0,0,0,1,2" + newline + b"\xff")
    with pytest.raises(DetectionAnnotationError, match="line 2: invalid UTF-8"):
        load_detection_annotations(path)


def test_parser_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(DetectionAnnotationError, match="does not exist"):
        load_detection_annotations(tmp_path / "missing.txt")


def test_parser_wraps_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_annotations(tmp_path, "")
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("not readable")))
    with pytest.raises(DetectionAnnotationError, match="Unable to read detection annotation"):
        load_detection_annotations(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_file_bytes": 0},
        {"max_file_bytes": 1.5},
        {"max_objects": -1},
        {"max_objects": True},
        {"max_objects": float("nan")},
    ],
)
def test_limits_require_positive_integers(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DetectionLimits(**kwargs)


@pytest.mark.parametrize("coordinate", [True, "0", float("nan"), 10**400])
def test_direct_annotation_construction_validates_coordinates(coordinate: object) -> None:
    with pytest.raises(DetectionAnnotationError, match="finite"):
        DetectionAnnotation(0, "vehicles", coordinate, 0.0, 1.0, 2.0)


def test_prediction_parser_reads_scored_rows(tmp_path: Path) -> None:
    document = "\n".join(
        f"{item.name},{item.class_id},10,20,30,50,0.95" for item in WOODSCAPE_DETECTION_CLASSES
    )
    predictions = load_detection_predictions(_write_annotations(tmp_path, document))

    assert isinstance(predictions, tuple)
    assert len(predictions) == 5
    assert predictions[0] == DetectionPrediction(0, "vehicles", 10.0, 20.0, 30.0, 50.0, 0.95)
    assert predictions[0].area == 600.0
    with pytest.raises(FrozenInstanceError):
        predictions[0].score = 0.1


def test_prediction_parser_tolerates_scientific_scores_and_bom(tmp_path: Path) -> None:
    path = _write_annotations(tmp_path, "\ufeffperson,1,0,0,2,3,1e-1\rperson,1,0,0,2,3,.5")
    assert [item.score for item in load_detection_predictions(path)] == pytest.approx([0.1, 0.5])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("person,1,0,0,2,3", "seven CSV fields"),
        ("person,1,0,0,2,3,8,0.5", "seven CSV fields"),
        ("person,1,0,0,2,3,1.5", r"\[0, 1\]"),
        ("person,1,0,0,2,3,1..0", "finite numeric value"),
        ("person,1,0,0,2,3,nan", "finite numeric value"),
        ("person,1,0,0,2,3,inf", "finite numeric value"),
        ("person,1,0,0,2,3,0.5,", "seven CSV fields"),
        ("person,1,0,0,2,3,-0.1", r"\[0, 1\]"),
    ],
)
def test_prediction_parser_rejects_invalid_rows_with_line(
    tmp_path: Path, row: str, message: str
) -> None:
    path = _write_annotations(tmp_path, "\nperson,1,0,0,2,3,0.3\n" + row)

    with pytest.raises(DetectionAnnotationError, match=message) as caught:
        load_detection_predictions(path)

    assert str(path) in str(caught.value)
    assert "line 3:" in str(caught.value)


def test_prediction_parser_checks_image_bounds_and_limits(tmp_path: Path) -> None:
    path = _write_annotations(tmp_path, "vehicles,0,0,0,21,5,0.9")
    with pytest.raises(DetectionAnnotationError, match="exceeds image size"):
        load_detection_predictions(path, image_size=(5, 20))

    path.write_text("\nvehicles,0,0,0,1,2,0.5\n\nperson,1,0,0,2,3,0.4", encoding="utf-8")
    with pytest.raises(DetectionAnnotationError, match="line 4: exceeds the 1-object"):
        load_detection_predictions(path, limits=DetectionLimits(max_objects=1))
