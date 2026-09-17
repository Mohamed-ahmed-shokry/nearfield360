"""Deterministic synthetic sensor corruptions for automotive fisheye cameras.

Provides reproducible corruption models parameterized by severity levels (1 to 5)
and explicit pseudo-random seeds. Models include lens soiling (mud/water splatter),
atmospheric fog, low-light Poisson-Gaussian noise, and directional rain streaks.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt


class CorruptionType(StrEnum):
    """Supported synthetic sensor corruption models."""

    LENS_SOILING = "lens_soiling"
    FOG = "fog"
    LOW_LIGHT_NOISE = "low_light_noise"
    RAIN = "rain"


def _validate_image(image: Any) -> npt.NDArray[np.uint8]:
    """Validate image array type, dimension, and non-empty bounds."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must have 2 or 3 dimensions, got {image.ndim}")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")
    return image


def _validate_severity(severity: int) -> int:
    """Ensure corruption severity is an integer in [1, 5]."""
    if not isinstance(severity, int) or severity < 1 or severity > 5:
        raise ValueError(f"severity must be an integer in [1, 5], got {severity}")
    return severity


def _as_uint8(arr: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Clip float values into [0, 255] and return a uint8 array."""
    clipped = np.clip(arr, 0.0, 255.0)
    return np.asarray(clipped, dtype=np.uint8)


def apply_lens_soiling(
    image: npt.NDArray[np.uint8],
    severity: int = 1,
    seed: int = 42,
) -> npt.NDArray[np.uint8]:
    """Simulate mud and water droplet deposits on the fisheye lens surface.

    Parameters
    ----------
    image:
        Input image array of shape ``(H, W)`` or ``(H, W, 3)`` and dtype ``uint8``.
    severity:
        Severity level from 1 (minor droplets) to 5 (dense mud obscuration).
    seed:
        Random seed ensuring deterministic splatter placement and color.

    Returns
    -------
    npt.NDArray[np.uint8]
        Corrupted image with identical shape and dtype.
    """
    img = _validate_image(image)
    _validate_severity(severity)

    height, width = img.shape[:2]
    rng = np.random.default_rng(seed)

    # Scale splatter count and sizes by severity
    blob_counts = {1: 5, 2: 12, 3: 25, 4: 45, 5: 75}
    radius_ranges = {
        1: (max(4, width // 100), max(10, width // 40)),
        2: (max(8, width // 60), max(18, width // 25)),
        3: (max(12, width // 40), max(28, width // 16)),
        4: (max(18, width // 25), max(45, width // 10)),
        5: (max(25, width // 18), max(70, width // 6)),
    }

    n_blobs = blob_counts[severity]
    min_r, max_r = radius_ranges[severity]

    alpha_mask = np.zeros((height, width), dtype=np.float32)
    mud_canvas = np.zeros(
        (height, width, 3) if img.ndim == 3 else (height, width), dtype=np.float32
    )

    for _ in range(n_blobs):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(min_r, max_r + 1))
        ry = int(rng.integers(min_r, max_r + 1))
        angle = float(rng.uniform(0.0, 360.0))
        opacity = float(rng.uniform(0.40, 0.85))

        # Mud/dirt tones (brownish-gray RGB/BGR or grayscale)
        if img.ndim == 3:
            mud_b = float(rng.uniform(35.0, 75.0))
            mud_g = float(rng.uniform(50.0, 100.0))
            mud_r = float(rng.uniform(65.0, 125.0))
            color: tuple[float, ...] = (mud_b, mud_g, mud_r)
        else:
            color = (float(rng.uniform(50.0, 100.0)),)

        cv2.ellipse(
            alpha_mask,
            (cx, cy),
            (rx, ry),
            angle,
            0.0,
            360.0,
            opacity,
            thickness=-1,
        )
        cv2.ellipse(
            mud_canvas,
            (cx, cy),
            (rx, ry),
            angle,
            0.0,
            360.0,
            color,
            thickness=-1,
        )

    # Feather splatter edges for physical optical realism on camera dome
    ksize = max(3, (min_r // 2) * 2 + 1)
    blurred_mask = cv2.GaussianBlur(alpha_mask, (ksize, ksize), 0)
    alpha_mask = np.asarray(blurred_mask, dtype=np.float32)
    alpha_mask = np.clip(alpha_mask, 0.0, 0.95)

    if img.ndim == 3:
        alpha_3d = alpha_mask[:, :, np.newaxis]
        img_f = img.astype(np.float32)
        blended = (1.0 - alpha_3d) * img_f + alpha_3d * mud_canvas
    else:
        img_f = img.astype(np.float32)
        blended = (1.0 - alpha_mask) * img_f + alpha_mask * mud_canvas

    return _as_uint8(blended)


def apply_fog(
    image: npt.NDArray[np.uint8],
    severity: int = 1,
    seed: int = 42,
) -> npt.NDArray[np.uint8]:
    """Simulate atmospheric fog and haze via physical optical transmission decay.

    Follows the atmospheric scattering model:
    ``I_out = I * t(x) + A * (1 - t(x))``
    where ``t(x) = exp(-beta * d(x))`` is the optical transmission map.

    Parameters
    ----------
    image:
        Input image array of shape ``(H, W)`` or ``(H, W, 3)`` and dtype ``uint8``.
    severity:
        Severity level from 1 (light haze) to 5 (dense whiteout fog).
    seed:
        Random seed for volumetric haze turbulence.

    Returns
    -------
    npt.NDArray[np.uint8]
        Corrupted image with identical shape and dtype.
    """
    img = _validate_image(image)
    _validate_severity(severity)

    height, width = img.shape[:2]
    rng = np.random.default_rng(seed)

    # Beta attenuation coefficient scales with severity
    betas = {1: 0.20, 2: 0.40, 3: 0.65, 4: 0.95, 5: 1.40}
    beta = betas[severity]
    atmospheric_light = 225.0

    # Depth gradient from near ground (bottom) to horizon/periphery
    y_coords = np.linspace(0.2, 1.0, height, dtype=np.float32)
    depth_map = np.tile(y_coords[:, np.newaxis], (1, width))

    # Add gentle spatial turbulence
    low_res = rng.uniform(0.9, 1.1, size=(max(2, height // 16), max(2, width // 16))).astype(
        np.float32
    )
    turbulence = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_LINEAR)
    effective_depth = np.clip(depth_map * turbulence, 0.0, 1.5)

    transmission = np.exp(-beta * effective_depth)
    transmission = np.clip(transmission, 0.05, 0.98)

    img_f = img.astype(np.float32)
    if img.ndim == 3:
        t_3d = transmission[:, :, np.newaxis]
        fogged = img_f * t_3d + atmospheric_light * (1.0 - t_3d)
    else:
        fogged = img_f * transmission + atmospheric_light * (1.0 - transmission)

    return _as_uint8(fogged)


def apply_low_light_noise(
    image: npt.NDArray[np.uint8],
    severity: int = 1,
    seed: int = 42,
) -> npt.NDArray[np.uint8]:
    """Simulate low-light nighttime illumination with sensor photon and read noise.

    Combines luminance gain/gamma compression with Poisson shot noise
    and zero-mean Gaussian electronic read noise.

    Parameters
    ----------
    image:
        Input image array of shape ``(H, W)`` or ``(H, W, 3)`` and dtype ``uint8``.
    severity:
        Severity level from 1 (dim twilight) to 5 (extreme dark with high noise).
    seed:
        Random seed for deterministic sensor noise sampling.

    Returns
    -------
    npt.NDArray[np.uint8]
        Corrupted image with identical shape and dtype.
    """
    img = _validate_image(image)
    _validate_severity(severity)

    rng = np.random.default_rng(seed)

    # Attenuation factor and Gaussian read noise sigma
    scale_factors = {1: 0.75, 2: 0.55, 3: 0.40, 4: 0.28, 5: 0.18}
    read_noise_sigmas = {1: 6.0, 2: 12.0, 3: 20.0, 4: 30.0, 5: 45.0}

    scale = scale_factors[severity]
    sigma = read_noise_sigmas[severity]

    # Attenuate brightness
    scaled = img.astype(np.float32) * scale

    # Poisson shot noise: variance proportional to signal intensity
    # Scaled Poisson approximated by Gaussian with variance = scaled / 2.0
    shot_variance = np.maximum(scaled * 0.5, 0.0)
    shot_noise = rng.normal(0.0, np.sqrt(shot_variance)).astype(np.float32)

    # Electronic read noise: signal-independent Gaussian
    read_noise = rng.normal(0.0, sigma, size=scaled.shape).astype(np.float32)

    degraded = scaled + shot_noise + read_noise
    return _as_uint8(degraded)


def apply_rain(
    image: npt.NDArray[np.uint8],
    severity: int = 1,
    seed: int = 42,
) -> npt.NDArray[np.uint8]:
    """Simulate rain precipitation with directional streak vectors and mild haze.

    Parameters
    ----------
    image:
        Input image array of shape ``(H, W)`` or ``(H, W, 3)`` and dtype ``uint8``.
    severity:
        Severity level from 1 (light drizzle) to 5 (torrential downpour).
    seed:
        Random seed for streak positions, lengths, and angles.

    Returns
    -------
    npt.NDArray[np.uint8]
        Corrupted image with identical shape and dtype.
    """
    img = _validate_image(image)
    _validate_severity(severity)

    height, width = img.shape[:2]
    rng = np.random.default_rng(seed)

    streak_counts = {1: 200, 2: 450, 3: 850, 4: 1400, 5: 2200}
    length_ranges = {
        1: (10, 20),
        2: (15, 30),
        3: (20, 42),
        4: (25, 55),
        5: (35, 75),
    }

    n_streaks = streak_counts[severity]
    min_len, max_len = length_ranges[severity]

    streak_canvas = np.zeros((height, width), dtype=np.float32)
    slant = float(rng.uniform(70.0, 85.0))  # Direction angle in degrees
    slant_rad = np.radians(slant)
    cos_slant = float(np.cos(slant_rad))
    sin_slant = float(np.sin(slant_rad))

    for _ in range(n_streaks):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        length = float(rng.uniform(min_len, max_len))
        x2 = int(np.clip(x1 + length * cos_slant, 0, width - 1))
        y2 = int(np.clip(y1 + length * sin_slant, 0, height - 1))
        brightness = float(rng.uniform(140.0, 230.0))

        cv2.line(streak_canvas, (x1, y1), (x2, y2), brightness, thickness=1)

    # Slight blur to soften streaks into realistic raindrops
    blurred_streaks = cv2.GaussianBlur(streak_canvas, (3, 3), 0)
    streak_canvas = np.asarray(blurred_streaks, dtype=np.float32)

    # Mild rain veil contrast reduction
    contrast_decay = {1: 0.96, 2: 0.91, 3: 0.85, 4: 0.78, 5: 0.70}[severity]

    img_f = img.astype(np.float32) * contrast_decay
    if img.ndim == 3:
        streaks_3d = streak_canvas[:, :, np.newaxis]
        blended = img_f + streaks_3d * 0.4
    else:
        blended = img_f + streak_canvas * 0.4

    return _as_uint8(blended)


def apply_sensor_corruption(
    image: npt.NDArray[np.uint8],
    corruption_type: CorruptionType | str,
    severity: int = 1,
    seed: int = 42,
) -> npt.NDArray[np.uint8]:
    """Unified dispatcher for synthetic sensor corruptions.

    Parameters
    ----------
    image:
        Input uint8 image array.
    corruption_type:
        One of ``CorruptionType`` or its string equivalent:
        ``"lens_soiling"``, ``"fog"``, ``"low_light_noise"``, or ``"rain"``.
    severity:
        Severity level from 1 to 5.
    seed:
        Random seed for deterministic behavior.

    Returns
    -------
    npt.NDArray[np.uint8]
        Corrupted image with identical dimensions and uint8 dtype.
    """
    try:
        corrupt_enum = CorruptionType(corruption_type)
    except ValueError:
        valid = ", ".join(repr(c.value) for c in CorruptionType)
        raise ValueError(
            f"Unknown corruption_type {corruption_type!r}. Must be one of: {valid}"
        ) from None

    match corrupt_enum:
        case CorruptionType.LENS_SOILING:
            return apply_lens_soiling(image, severity=severity, seed=seed)
        case CorruptionType.FOG:
            return apply_fog(image, severity=severity, seed=seed)
        case CorruptionType.LOW_LIGHT_NOISE:
            return apply_low_light_noise(image, severity=severity, seed=seed)
        case CorruptionType.RAIN:
            return apply_rain(image, severity=severity, seed=seed)


__all__ = [
    "CorruptionType",
    "apply_fog",
    "apply_lens_soiling",
    "apply_low_light_noise",
    "apply_rain",
    "apply_sensor_corruption",
]
