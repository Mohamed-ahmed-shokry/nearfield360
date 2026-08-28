from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import pytest

from nearfield360.data import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticClass,
    SemanticMaskError,
    colorize_semantic_mask,
    load_semantic_mask,
    semantic_class,
    semantic_histogram,
    validate_semantic_mask,
)


def test_official_semantic_mapping_is_contiguous_and_stable() -> None:
    assert [item.label_id for item in WOODSCAPE_SEMANTIC_CLASSES] == list(range(10))
    assert [item.name for item in WOODSCAPE_SEMANTIC_CLASSES] == [
        "void",
        "road",
        "lanemarks",
        "curb",
        "person",
        "rider",
        "vehicles",
        "bicycle",
        "motorcycle",
        "traffic_sign",
    ]
    assert semantic_class(6).name == "vehicles"
    assert semantic_class(4).color_bgr == (0, 0, 255)


def test_semantic_class_validates_metadata() -> None:
    with pytest.raises(ValueError, match="label_id"):
        SemanticClass(-1, "invalid", (0, 0, 0))
    with pytest.raises(ValueError, match="name"):
        SemanticClass(1, "", (0, 0, 0))
    with pytest.raises(ValueError, match="color_bgr"):
        SemanticClass(1, "invalid", (0, 0, 256))


def test_semantic_class_rejects_unknown_label() -> None:
    with pytest.raises(SemanticMaskError, match="Unknown WoodScape semantic label: 10"):
        semantic_class(10)


def test_validate_semantic_mask_normalizes_valid_integer_dtype() -> None:
    mask = np.array([[0, 1], [8, 9]], dtype=np.uint16)

    validated = validate_semantic_mask(mask)

    np.testing.assert_array_equal(validated, mask)
    assert validated.dtype == np.uint8
    assert validated.flags.c_contiguous


@pytest.mark.parametrize(
    "mask",
    [
        np.array([], dtype=np.uint8),
        np.zeros((1, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_validate_semantic_mask_rejects_shape_and_dtype(mask: np.ndarray) -> None:
    with pytest.raises(SemanticMaskError, match="Semantic mask must"):
        validate_semantic_mask(mask)


def test_validate_semantic_mask_reports_unknown_labels() -> None:
    mask = np.array([[0, 10], [255, 1]], dtype=np.uint8)

    with pytest.raises(SemanticMaskError, match="10, 255"):
        validate_semantic_mask(mask)


def test_colorize_semantic_mask_supports_rgb_and_bgr() -> None:
    mask = np.array([[2, 4]], dtype=np.uint8)

    bgr = colorize_semantic_mask(mask, color_order="bgr")
    rgb = colorize_semantic_mask(mask, color_order="rgb")

    np.testing.assert_array_equal(bgr, np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8))
    np.testing.assert_array_equal(rgb, np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8))


def test_colorize_semantic_mask_rejects_unknown_order() -> None:
    with pytest.raises(ValueError, match="Unsupported color order"):
        colorize_semantic_mask(
            np.zeros((1, 1), dtype=np.uint8),
            color_order=cast(Literal["rgb", "bgr"], "xyz"),
        )


def test_semantic_histogram_includes_absent_classes() -> None:
    histogram = semantic_histogram(np.array([[0, 1], [1, 9]], dtype=np.uint8))

    assert histogram["void"] == 1
    assert histogram["road"] == 2
    assert histogram["traffic_sign"] == 1
    assert histogram["curb"] == 0
    assert sum(histogram.values()) == 4


def test_load_semantic_mask_decodes_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    expected = np.array([[0, 3], [6, 9]], dtype=np.uint8)
    assert cv2.imwrite(str(path), expected)

    np.testing.assert_array_equal(load_semantic_mask(path), expected)
