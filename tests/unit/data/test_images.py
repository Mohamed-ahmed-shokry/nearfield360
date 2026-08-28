from pathlib import Path

import cv2
import numpy as np
import pytest

from nearfield360.data import ImageLimits, ImageReadError, load_label_image, load_rgb_image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_load_rgb_image_converts_bgr_and_returns_contiguous_array(tmp_path: Path) -> None:
    path = tmp_path / "color.png"
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    _write_image(path, bgr)

    rgb = load_rgb_image(path)

    np.testing.assert_array_equal(rgb, np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8))
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_load_label_image_preserves_integer_values(tmp_path: Path, dtype: np.dtype) -> None:
    path = tmp_path / f"labels-{np.dtype(dtype).name}.png"
    expected = np.array([[0, 1], [8, 9]], dtype=dtype)
    _write_image(path, expected)

    actual = load_label_image(path)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_load_label_image_rejects_color_input(tmp_path: Path) -> None:
    path = tmp_path / "color-labels.png"
    _write_image(path, np.zeros((2, 3, 3), dtype=np.uint8))

    with pytest.raises(ImageReadError, match="single-channel"):
        load_label_image(path)


def test_image_loader_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ImageReadError, match="does not exist"):
        load_rgb_image(tmp_path / "missing.png")

    malformed = tmp_path / "malformed.png"
    malformed.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageReadError, match="could not decode"):
        load_rgb_image(malformed)


def test_image_loader_enforces_file_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "oversized.png"
    path.write_bytes(b"x" * 17)

    with pytest.raises(ImageReadError, match="byte safety limit"):
        load_rgb_image(path, limits=ImageLimits(max_file_bytes=16))


def test_image_loader_enforces_decoded_pixel_limit(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_image(path, np.zeros((3, 4, 3), dtype=np.uint8))

    with pytest.raises(ImageReadError, match="12 pixels"):
        load_rgb_image(path, limits=ImageLimits(max_pixels=11))


@pytest.mark.parametrize(
    "kwargs",
    [{"max_file_bytes": 0}, {"max_pixels": 0}],
)
def test_image_limits_must_be_positive(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ImageLimits(**kwargs)
