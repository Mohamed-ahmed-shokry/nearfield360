"""Unit tests for camera health, lens soiling, blur, and blockage assessment."""

from __future__ import annotations

import cv2
import numpy as np

from nearfield360.config import CameraHealthConfig
from nearfield360.health.detector import (
    assess_camera_health,
    calculate_blur_score,
    calculate_photometric_properties,
    detect_blockage,
    detect_lens_soiling,
)
from nearfield360.health.models import CameraHealthStatus, HealthAnomaly
from nearfield360.robustness.corruptions import apply_lens_soiling


def _create_textured_scene(h: int = 128, w: int = 128, seed: int = 42) -> np.ndarray:
    """Generate synthetic textured scene representing structured automotive imagery."""
    rng = np.random.default_rng(seed)
    # Background gradient
    y, x = np.mgrid[0:h, 0:w]
    base = (x * 0.5 + y * 0.5).astype(np.float64)
    # Add high-contrast lines and features (curbs, road edges, patterns)
    base[h // 4 : h // 4 + 10, :] += 80.0
    base[:, w // 3 : w // 3 + 12] += 60.0
    base += rng.normal(0.0, 15.0, (h, w))
    img = np.clip(base, 20.0, 230.0).astype(np.uint8)
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), dtype=np.uint8)


def test_calculate_blur_score_distinguishes_sharp_and_defocused() -> None:
    sharp = _create_textured_scene()
    blurred = cv2.GaussianBlur(sharp, (21, 21), sigmaX=8.0)

    sharp_score = calculate_blur_score(sharp)
    blur_score = calculate_blur_score(blurred)

    assert sharp_score > 200.0
    assert blur_score < 50.0
    assert sharp_score > blur_score * 5.0


def test_calculate_photometric_properties() -> None:
    flat = np.full((64, 64, 3), 120, dtype=np.uint8)
    mean_val, contrast_val = calculate_photometric_properties(flat)
    assert mean_val == 120.0
    assert contrast_val == 0.0

    scene = _create_textured_scene()
    m, c = calculate_photometric_properties(scene)
    assert 50.0 < m < 200.0
    assert c > 10.0


def test_detect_lens_soiling_on_clean_and_soiled() -> None:
    clean = _create_textured_scene(h=192, w=192)
    clean_soiling = detect_lens_soiling(clean, block_size=16)
    assert clean_soiling < 0.05

    soiled = apply_lens_soiling(clean, severity=5, seed=123)
    soiled_score = detect_lens_soiling(soiled, block_size=16)
    assert soiled_score > 0.1
    assert soiled_score > clean_soiling * 2.0


def test_detect_lens_soiling_small_image() -> None:
    tiny = np.zeros((16, 16, 3), dtype=np.uint8)
    assert detect_lens_soiling(tiny, block_size=32) == 0.0


def test_detect_blockage_identifies_occluded_sensors() -> None:
    blackout = np.zeros((128, 128, 3), dtype=np.uint8)
    assert detect_blockage(blackout, block_size=32) == 1.0

    flat_gray = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert detect_blockage(flat_gray, block_size=32) == 1.0

    textured = _create_textured_scene()
    assert detect_blockage(textured, block_size=32) == 0.0


def test_assess_camera_health_healthy_scene() -> None:
    clean = _create_textured_scene()
    report = assess_camera_health(clean, camera="FV")

    assert report.camera == "FV"
    assert report.status == CameraHealthStatus.HEALTHY
    assert len(report.anomalies) == 0
    assert report.discount_weight == 1.0
    assert report.metrics.confidence > 0.8


def test_assess_camera_health_soiled_scene() -> None:
    clean = _create_textured_scene(h=256, w=256)
    soiled = apply_lens_soiling(clean, severity=5, seed=777)
    cfg = CameraHealthConfig(soiling_threshold=0.1)
    report = assess_camera_health(soiled, camera="RV", config=cfg)

    assert report.camera == "RV"
    assert report.status == CameraHealthStatus.DEGRADED
    assert HealthAnomaly.LENS_SOILING in report.anomalies
    assert report.discount_weight < 1.0
    assert report.discount_weight > 0.0


def test_assess_camera_health_blurred_scene() -> None:
    clean = _create_textured_scene()
    blurred = cv2.GaussianBlur(clean, (25, 25), 10.0)
    cfg = CameraHealthConfig(blur_threshold=150.0)
    report = assess_camera_health(blurred, camera="MVL", config=cfg)

    assert report.camera == "MVL"
    assert report.status == CameraHealthStatus.DEGRADED
    assert HealthAnomaly.BLUR in report.anomalies


def test_assess_camera_health_blackout_and_glare() -> None:
    blackout = np.zeros((128, 128, 3), dtype=np.uint8)
    rep_black = assess_camera_health(blackout, camera="MVR")
    assert rep_black.status == CameraHealthStatus.BLOCKED
    assert rep_black.discount_weight == 0.0
    assert HealthAnomaly.BLOCKAGE in rep_black.anomalies
    assert HealthAnomaly.UNDEREXPOSURE in rep_black.anomalies

    glare = np.full((128, 128, 3), 255, dtype=np.uint8)
    rep_glare = assess_camera_health(glare, camera="MVR")
    assert rep_glare.status == CameraHealthStatus.BLOCKED
    assert rep_glare.discount_weight == 0.0
    assert HealthAnomaly.OVEREXPOSURE in rep_glare.anomalies
