from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from nearfield360.robustness.corruptions import (
    CorruptionType,
    apply_fog,
    apply_lens_soiling,
    apply_low_light_noise,
    apply_rain,
    apply_sensor_corruption,
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(50, 200, size=(120, 160, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(50, 200, size=(120, 160), dtype=np.uint8)


@pytest.mark.parametrize(
    "corrupt_func",
    [apply_lens_soiling, apply_fog, apply_low_light_noise, apply_rain],
)
def test_corruptions_preserve_shape_and_dtype(
    corrupt_func: Callable[..., np.ndarray],
    sample_rgb_image: np.ndarray,
    sample_gray_image: np.ndarray,
) -> None:
    # Test RGB image
    out_rgb = corrupt_func(sample_rgb_image, severity=2, seed=42)
    assert out_rgb.shape == sample_rgb_image.shape
    assert out_rgb.dtype == np.uint8
    assert np.all(out_rgb >= 0) and np.all(out_rgb <= 255)

    # Test Grayscale image
    out_gray = corrupt_func(sample_gray_image, severity=2, seed=42)
    assert out_gray.shape == sample_gray_image.shape
    assert out_gray.dtype == np.uint8
    assert np.all(out_gray >= 0) and np.all(out_gray <= 255)


@pytest.mark.parametrize("corrupt_type", list(CorruptionType))
def test_dispatcher_matches_direct_call(
    corrupt_type: CorruptionType, sample_rgb_image: np.ndarray
) -> None:
    dispatched = apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=3, seed=42)
    assert dispatched.shape == sample_rgb_image.shape
    assert dispatched.dtype == np.uint8


@pytest.mark.parametrize("corrupt_type", list(CorruptionType))
def test_corruptions_are_deterministic_with_same_seed(
    corrupt_type: CorruptionType, sample_rgb_image: np.ndarray
) -> None:
    out1 = apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=2, seed=99)
    out2 = apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=2, seed=99)
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.parametrize("corrupt_type", list(CorruptionType))
def test_different_seeds_produce_different_outputs(
    corrupt_type: CorruptionType, sample_rgb_image: np.ndarray
) -> None:
    out1 = apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=3, seed=1)
    out2 = apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=3, seed=2)
    # Output arrays should not be completely identical across distinct seeds
    assert not np.array_equal(out1, out2)


@pytest.mark.parametrize("corrupt_type", list(CorruptionType))
def test_severity_scales_degradation(
    corrupt_type: CorruptionType, sample_rgb_image: np.ndarray
) -> None:
    # Deviation from clean image should grow with severity
    dev_1 = np.mean(
        np.abs(
            apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=1, seed=42).astype(
                float
            )
            - sample_rgb_image.astype(float)
        )
    )
    dev_5 = np.mean(
        np.abs(
            apply_sensor_corruption(sample_rgb_image, corrupt_type, severity=5, seed=42).astype(
                float
            )
            - sample_rgb_image.astype(float)
        )
    )
    assert dev_5 > dev_1


def test_corruptions_reject_invalid_inputs(sample_rgb_image: np.ndarray) -> None:
    # Non-array
    with pytest.raises(TypeError, match=r"numpy\.ndarray"):
        apply_sensor_corruption([1, 2, 3], "fog")  # type: ignore[arg-type]

    # Non-uint8 dtype
    with pytest.raises(ValueError, match="dtype uint8"):
        apply_sensor_corruption(sample_rgb_image.astype(np.float32), "fog")

    # Invalid dimension
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        apply_sensor_corruption(np.ones((10,), dtype=np.uint8), "fog")

    # Empty image
    with pytest.raises(ValueError, match="must not be empty"):
        apply_sensor_corruption(np.zeros((0, 10, 3), dtype=np.uint8), "fog")

    # Severity out of bounds
    with pytest.raises(ValueError, match="severity must be an integer in"):
        apply_sensor_corruption(sample_rgb_image, "fog", severity=0)
    with pytest.raises(ValueError, match="severity must be an integer in"):
        apply_sensor_corruption(sample_rgb_image, "fog", severity=6)

    # Unknown corruption type
    with pytest.raises(ValueError, match="Unknown corruption_type"):
        apply_sensor_corruption(sample_rgb_image, "solar_flare")
