"""Fisheye camera health, lens soiling, blur, and blockage assessment engine."""

from __future__ import annotations

import cv2
import numpy as np

from nearfield360.config import CameraHealthConfig
from nearfield360.health.models import (
    CameraHealthMetrics,
    CameraHealthReport,
    CameraHealthStatus,
    HealthAnomaly,
)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate the spatial Laplacian variance as an optical sharpness/blur metric.

    Lower scores indicate blur or defocus; higher scores indicate sharp, high-frequency edges.
    """
    gray = _to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def calculate_photometric_properties(image: np.ndarray) -> tuple[float, float]:
    """Return mean luminance and standard deviation (contrast) of an image."""
    gray = _to_grayscale(image)
    mean_val = float(np.mean(gray))
    contrast_val = float(np.std(gray))
    return mean_val, contrast_val


def detect_lens_soiling(
    image: np.ndarray,
    *,
    block_size: int = 16,
) -> float:
    """Estimate the fraction of lens surface afflicted by soiling, dirt, or deposits.

    Decomposes the image into spatial blocks and identifies regions with localized
    texture attenuation, low high-frequency Laplacian energy, and contrast loss.

    Returns:
        Estimated soiled fraction in [0.0, 1.0].
    """
    gray = _to_grayscale(image)

    h, w = gray.shape
    if h < block_size or w < block_size:
        return 0.0

    n_blocks_y = h // block_size
    n_blocks_x = w // block_size
    total_blocks = n_blocks_y * n_blocks_x
    if total_blocks == 0:
        return 0.0

    energies: list[float] = []
    stds: list[float] = []
    for by in range(n_blocks_y):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(n_blocks_x):
            x0 = bx * block_size
            x1 = x0 + block_size

            block_gray = gray[y0:y1, x0:x1]
            lap = cv2.Laplacian(block_gray, cv2.CV_64F)
            energies.append(float(np.var(lap)))
            stds.append(float(np.std(block_gray)))

    e_arr = np.asarray(energies, dtype=np.float64)
    s_arr = np.asarray(stds, dtype=np.float64)

    max_energy = float(np.percentile(e_arr, 85))
    if max_energy < 1.0:
        return 0.0

    soiled_mask = (e_arr < 0.35 * max_energy) & (s_arr < 12.0)
    return float(np.clip(np.mean(soiled_mask), 0.0, 1.0))


def detect_blockage(
    image: np.ndarray,
    *,
    block_size: int = 32,
    min_variance: float = 3.0,
) -> float:
    """Detect dead, untextured, or fully occluded optical blocks.

    Returns:
        Fraction of completely untextured blocks in [0.0, 1.0].
    """
    gray = _to_grayscale(image)

    h, w = gray.shape
    if h < block_size or w < block_size:
        return 0.0

    n_blocks_y = h // block_size
    n_blocks_x = w // block_size
    total_blocks = n_blocks_y * n_blocks_x
    if total_blocks == 0:
        return 0.0

    blocked_blocks = 0
    for by in range(n_blocks_y):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(n_blocks_x):
            x0 = bx * block_size
            x1 = x0 + block_size
            var_val = float(np.var(gray[y0:y1, x0:x1]))
            if var_val < min_variance:
                blocked_blocks += 1

    return float(np.clip(blocked_blocks / total_blocks, 0.0, 1.0))


def assess_camera_health(
    image: np.ndarray,
    *,
    camera: str = "FV",
    config: CameraHealthConfig | None = None,
) -> CameraHealthReport:
    """Assess overall operational health, soiling, blur, and exposure for a camera image.

    Args:
        image: RGB or grayscale image array.
        camera: Camera identifier string (e.g. 'FV', 'RV', 'MVL', 'MVR').
        config: Optional CameraHealthConfig thresholds (uses defaults if None).

    Returns:
        CameraHealthReport detailing metrics, detected anomalies, status, and discount weight.
    """
    if config is None:
        config = CameraHealthConfig()

    blur = calculate_blur_score(image)
    mean_bright, contrast = calculate_photometric_properties(image)
    soiling = detect_lens_soiling(image)
    blockage = detect_blockage(image)

    anomalies: list[HealthAnomaly] = []

    if soiling >= config.soiling_threshold:
        anomalies.append(HealthAnomaly.LENS_SOILING)
    if blur < config.blur_threshold:
        anomalies.append(HealthAnomaly.BLUR)
    if mean_bright < config.min_brightness:
        anomalies.append(HealthAnomaly.UNDEREXPOSURE)
    elif mean_bright > config.max_brightness:
        anomalies.append(HealthAnomaly.OVEREXPOSURE)
    if blockage >= config.blockage_ratio_threshold:
        anomalies.append(HealthAnomaly.BLOCKAGE)

    # Health confidence computation [0.0, 1.0]
    # Penalize based on soiling, blur deficiency, and blockage
    blur_factor = min(1.0, blur / max(config.blur_threshold, 1e-6))
    soiling_factor = max(0.0, 1.0 - soiling)
    blockage_factor = max(0.0, 1.0 - blockage)

    # Exposure factor
    if mean_bright < config.min_brightness:
        exposure_factor = max(0.0, mean_bright / max(config.min_brightness, 1e-6))
    elif mean_bright > config.max_brightness:
        exposure_factor = max(0.0, (255.0 - mean_bright) / max(255.0 - config.max_brightness, 1e-6))
    else:
        exposure_factor = 1.0

    raw_conf = (
        0.35 * soiling_factor + 0.25 * blur_factor + 0.25 * blockage_factor + 0.15 * exposure_factor
    )
    confidence = float(np.clip(raw_conf, 0.0, 1.0))

    # Determine discrete status
    if blockage >= config.blockage_ratio_threshold or mean_bright < 5.0 or mean_bright > 252.0:
        status = CameraHealthStatus.BLOCKED
        discount_weight = 0.0
    elif anomalies or confidence < 0.8:
        status = CameraHealthStatus.DEGRADED
        discount_weight = float(np.clip(confidence * config.discount_factor, 0.05, 0.95))
    else:
        status = CameraHealthStatus.HEALTHY
        discount_weight = 1.0

    metrics = CameraHealthMetrics(
        soiling_score=round(soiling, 4),
        blur_score=round(blur, 2),
        mean_brightness=round(mean_bright, 2),
        contrast=round(contrast, 2),
        blockage_ratio=round(blockage, 4),
        confidence=round(confidence, 4),
    )

    return CameraHealthReport(
        camera=camera,
        status=status,
        anomalies=tuple(anomalies),
        metrics=metrics,
        discount_weight=round(discount_weight, 4),
    )


__all__ = [
    "assess_camera_health",
    "calculate_blur_score",
    "calculate_photometric_properties",
    "detect_blockage",
    "detect_lens_soiling",
]
