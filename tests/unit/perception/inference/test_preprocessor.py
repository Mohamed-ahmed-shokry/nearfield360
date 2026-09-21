"""Tests for FisheyeImagePreprocessor and spatial unscaling functions."""

from __future__ import annotations

import numpy as np
import pytest

from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
    PreprocessorError,
    PreprocessTransform,
)


def test_preprocessor_validation_errors() -> None:
    # Invalid target_size
    with pytest.raises(PreprocessorError, match="target_size must be positive"):
        FisheyeImagePreprocessor(target_size=(0, 640))
    with pytest.raises(PreprocessorError, match="target_size must be positive"):
        FisheyeImagePreprocessor(target_size=(-10, 640))

    # Invalid std
    with pytest.raises(PreprocessorError, match="std must be strictly positive"):
        FisheyeImagePreprocessor(std=(0.2, 0.0, 0.2))

    preprocessor = FisheyeImagePreprocessor(target_size=(480, 640))

    # Non-array input
    with pytest.raises(PreprocessorError, match="image must be a numpy ndarray"):
        preprocessor.preprocess([[1, 2], [3, 4]])  # type: ignore[arg-type]

    # Invalid ndim
    with pytest.raises(PreprocessorError, match="image must have shape"):
        preprocessor.preprocess(np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(PreprocessorError, match="image must have shape"):
        preprocessor.preprocess(np.zeros((100, 100, 4), dtype=np.uint8))

    # Invalid dtype
    with pytest.raises(PreprocessorError, match="image must have dtype uint8"):
        preprocessor.preprocess(np.zeros((100, 100, 3), dtype=np.float32))

    # Empty batch
    with pytest.raises(PreprocessorError, match="images sequence must not be empty"):
        preprocessor.preprocess_batch([])

    # Invalid mask shape in unscale_mask
    dummy_transform = PreprocessTransform(100, 100, 100, 100, 1.0, 0, 0)
    with pytest.raises(PreprocessorError, match="mask must have shape"):
        FisheyeImagePreprocessor.unscale_mask(
            np.zeros((10, 10, 1), dtype=np.int32), dummy_transform
        )


def test_preprocessor_direct_resize() -> None:
    target_h, target_w = 240, 320
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    preprocessor = FisheyeImagePreprocessor(
        target_size=(target_h, target_w),
        mean=mean,
        std=std,
        preserve_aspect_ratio=False,
    )
    assert preprocessor.target_size == (target_h, target_w)

    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    blob, transform = preprocessor.preprocess(img)

    assert blob.shape == (1, 3, target_h, target_w)
    assert blob.dtype == np.float32
    # Pixel value 255 -> 255/255 = 1.0 -> (1.0 - 0.5) / 0.5 = 1.0
    np.testing.assert_allclose(blob, 1.0, atol=1e-5)

    assert transform.orig_height == 400
    assert transform.orig_width == 600
    assert transform.target_height == target_h
    assert transform.target_width == target_w
    assert transform.scale == 1.0
    assert transform.pad_left == 0
    assert transform.pad_top == 0


def test_preprocessor_letterbox_aspect_ratio() -> None:
    target_h, target_w = 480, 640
    preprocessor = FisheyeImagePreprocessor(
        target_size=(target_h, target_w),
        preserve_aspect_ratio=True,
        pad_value=114,
    )

    # Wide image: 400x1200 -> scale = min(480/400, 640/1200) = min(1.2, 0.5333) = ~0.5333
    # new_w = 640, new_h = round(400 * (640/1200)) = round(213.33) = 213
    # pad_top = (480 - 213) // 2 = 133
    orig_img = np.zeros((400, 1200, 3), dtype=np.uint8)
    blob, transform = preprocessor.preprocess(orig_img)

    assert blob.shape == (1, 3, target_h, target_w)
    assert transform.orig_height == 400
    assert transform.orig_width == 1200
    assert transform.pad_left == 0
    assert transform.pad_top == (480 - round(400 * transform.scale)) // 2
    assert transform.scale == pytest.approx(640 / 1200, abs=1e-3)


def test_preprocessor_batch() -> None:
    preprocessor = FisheyeImagePreprocessor(target_size=(100, 100))
    imgs = [
        np.zeros((50, 50, 3), dtype=np.uint8),
        np.ones((60, 60, 3), dtype=np.uint8) * 100,
    ]
    batch_blob, transforms = preprocessor.preprocess_batch(imgs)

    assert batch_blob.shape == (2, 3, 100, 100)
    assert len(transforms) == 2
    assert transforms[0].orig_height == 50
    assert transforms[1].orig_height == 60


def test_unscale_boxes_direct_and_letterbox() -> None:
    # 1. Empty boxes
    dummy_transform = PreprocessTransform(400, 600, 200, 300, 1.0, 0, 0)
    empty_res = FisheyeImagePreprocessor.unscale_boxes(np.empty((0, 4)), dummy_transform)
    assert empty_res.shape == (0, 4)

    # 2. Direct resize unscale: 400x600 -> 200x300 (scale_x=2, scale_y=2)
    boxes = np.array([[10, 20, 50, 60]], dtype=np.float32)
    unscaled = FisheyeImagePreprocessor.unscale_boxes(boxes, dummy_transform)
    expected = np.array([[20, 40, 100, 120]], dtype=np.float32)
    np.testing.assert_allclose(unscaled, expected)

    # 3. Letterbox unscale with padding:
    # target 400x400, orig 200x400 -> scale=1.0, pad_top=100, pad_left=0
    # box in target space at [50, 120, 150, 250]
    # after unscale: ymin = (120 - 100)/1 = 20, ymax = (250 - 100)/1 = 150
    lb_transform = PreprocessTransform(
        orig_height=200,
        orig_width=400,
        target_height=400,
        target_width=400,
        scale=1.0,
        pad_left=0,
        pad_top=100,
    )
    lb_boxes = np.array([[50, 120, 150, 250]], dtype=np.float32)
    unscaled_lb = FisheyeImagePreprocessor.unscale_boxes(lb_boxes, lb_transform)
    np.testing.assert_allclose(unscaled_lb, [[50, 20, 150, 150]], atol=1e-4)

    # 4. Box clipping at boundaries
    out_boxes = np.array([[-10, 50, 500, 350]], dtype=np.float32)
    clipped = FisheyeImagePreprocessor.unscale_boxes(out_boxes, lb_transform)
    assert clipped[0, 0] == 0.0  # clipped to 0
    assert clipped[0, 2] == 400.0  # clipped to orig_width 400
    assert clipped[0, 3] == 200.0  # (350 - 100) = 250 -> clipped to orig_height 200


def test_unscale_mask_direct_and_letterbox() -> None:
    # Direct resize
    direct_transform = PreprocessTransform(
        orig_height=20,
        orig_width=30,
        target_height=10,
        target_width=15,
        scale=1.0,
        pad_left=0,
        pad_top=0,
    )
    mask = np.full((10, 15), 3, dtype=np.int32)
    unscaled_mask = FisheyeImagePreprocessor.unscale_mask(mask, direct_transform)
    assert unscaled_mask.shape == (20, 30)
    assert unscaled_mask.dtype == np.int32
    assert (unscaled_mask == 3).all()

    # Letterbox unscale
    # orig: 20x40, target: 40x40 -> scale = 1.0, pad_top = 10, pad_left = 0
    lb_transform = PreprocessTransform(
        orig_height=20,
        orig_width=40,
        target_height=40,
        target_width=40,
        scale=1.0,
        pad_left=0,
        pad_top=10,
    )
    # Mask has padding with label 0, inner 20x40 has label 5
    canvas_mask = np.zeros((40, 40), dtype=np.uint8)
    canvas_mask[10:30, :] = 5
    unscaled_lb = FisheyeImagePreprocessor.unscale_mask(canvas_mask, lb_transform)
    assert unscaled_lb.shape == (20, 40)
    assert (unscaled_lb == 5).all()
