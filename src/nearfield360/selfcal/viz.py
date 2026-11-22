"""Tactical 1280x720 HUD dashboard for extrinsic self-calibration and camera drift."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.hdmap.viz import (
    COLOR_AMBER,
    COLOR_BG,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_RED,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_YELLOW,
)
from nearfield360.selfcal.models import CalibrationEstimate, DriftReport, DriftStatus

Color = tuple[int, int, int]

STATUS_COLORS: dict[DriftStatus, Color] = {
    DriftStatus.NOMINAL: COLOR_GREEN,
    DriftStatus.WARNING: COLOR_YELLOW,
    DriftStatus.RECALIBRATE: COLOR_AMBER,
    DriftStatus.FAULT: COLOR_RED,
    DriftStatus.UNOBSERVABLE: COLOR_TEXT_MUTED,
}
SEVERITY: dict[DriftStatus, int] = {
    DriftStatus.UNOBSERVABLE: 0,
    DriftStatus.NOMINAL: 1,
    DriftStatus.WARNING: 2,
    DriftStatus.RECALIBRATE: 3,
    DriftStatus.FAULT: 4,
}
CAMERA_ORDER = (CameraId.FRONT, CameraId.REAR, CameraId.MIRROR_LEFT, CameraId.MIRROR_RIGHT)
CAMERA_COLORS: dict[CameraId, Color] = {
    CameraId.FRONT: COLOR_CYAN,
    CameraId.REAR: COLOR_GREEN,
    CameraId.MIRROR_LEFT: COLOR_AMBER,
    CameraId.MIRROR_RIGHT: (220, 50, 220),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(
    canvas: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.42,
    color: Color = COLOR_TEXT_SECONDARY,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, origin, _FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(canvas: NDArray[np.uint8], rect: tuple[int, int, int, int], title: str) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 28), (35, 40, 52), -1)
    cv2.line(canvas, (x, y + 28), (x + w, y + 28), COLOR_PANEL_BORDER, 1)
    _text(canvas, title.upper(), (x + 12, y + 19), scale=0.5, color=COLOR_TEXT_PRIMARY)


def _scale_max(config: SelfCalConfig, reports: Mapping[CameraId, DriftReport]) -> float:
    largest = max((report.magnitude_deg for report in reports.values()), default=0.0)
    return max(config.fault_deg * 1.25, largest * 1.1)


def _draw_drift_bars(
    canvas: NDArray[np.uint8],
    rect: tuple[int, int, int, int],
    config: SelfCalConfig,
    reports: Mapping[CameraId, DriftReport],
) -> None:
    x, y, w, h = rect
    _panel(canvas, rect, "Per-camera misalignment vs thresholds")
    left, right = x + 90, x + w - 30
    top, bottom = y + 50, y + h - 40
    limit = _scale_max(config, reports)
    for threshold, color in (
        (config.warning_deg, COLOR_YELLOW),
        (config.recalibrate_deg, COLOR_AMBER),
        (config.fault_deg, COLOR_RED),
    ):
        px = left + int((right - left) * threshold / limit)
        cv2.line(canvas, (px, top - 6), (px, bottom), color, 1, cv2.LINE_AA)
        _text(canvas, f"{threshold:g}", (px - 10, bottom + 16), scale=0.38, color=color)
    row = (bottom - top) / len(CAMERA_ORDER)
    for index, camera in enumerate(CAMERA_ORDER):
        cy = int(top + row * index + row / 2)
        _text(canvas, camera.value, (x + 20, cy + 5), scale=0.5, color=COLOR_TEXT_PRIMARY)
        report = reports.get(camera)
        if report is None:
            _text(canvas, "no data", (left + 6, cy + 5), color=COLOR_TEXT_MUTED)
            continue
        length = int((right - left) * report.magnitude_deg / limit)
        color = STATUS_COLORS[report.status]
        cv2.rectangle(canvas, (left, cy - 12), (left + max(length, 2), cy + 12), color, -1)
        _text(canvas, f"{report.magnitude_deg:.2f} deg", (left + max(length, 2) + 8, cy + 5))


def _draw_history(
    canvas: NDArray[np.uint8],
    rect: tuple[int, int, int, int],
    config: SelfCalConfig,
    history: Mapping[CameraId, Sequence[float]],
    limit: float,
) -> None:
    x, y, w, h = rect
    _panel(canvas, rect, "Smoothed drift history (deg)")
    left, right = x + 50, x + w - 20
    top, bottom = y + 44, y + h - 30
    cv2.rectangle(canvas, (left, top), (right, bottom), COLOR_PANEL_BORDER, 1)
    for threshold, color in (
        (config.warning_deg, COLOR_YELLOW),
        (config.recalibrate_deg, COLOR_AMBER),
        (config.fault_deg, COLOR_RED),
    ):
        py = bottom - int((bottom - top) * min(threshold / limit, 1.0))
        cv2.line(canvas, (left, py), (right, py), color, 1, cv2.LINE_AA)
        _text(canvas, f"{threshold:g}", (x + 12, py + 4), scale=0.38, color=color)
    longest = max((len(values) for values in history.values()), default=0)
    for camera in CAMERA_ORDER:
        values = history.get(camera, ())
        if len(values) < 1:
            continue
        step = (right - left) / max(longest - 1, 1)
        points = np.array(
            [
                [left + int(step * i), bottom - int((bottom - top) * min(v / limit, 1.0))]
                for i, v in enumerate(values)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, CAMERA_COLORS[camera], 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(points[-1][0]), int(points[-1][1])), 4, CAMERA_COLORS[camera], -1)
    for index, camera in enumerate(CAMERA_ORDER):
        lx = left + 10 + index * 90
        cv2.rectangle(canvas, (lx, bottom + 8), (lx + 10, bottom + 18), CAMERA_COLORS[camera], -1)
        _text(canvas, camera.value, (lx + 16, bottom + 18), scale=0.38)


def _draw_table(
    canvas: NDArray[np.uint8],
    rect: tuple[int, int, int, int],
    estimates: Mapping[CameraId, CalibrationEstimate],
    reports: Mapping[CameraId, DriftReport],
) -> None:
    x, y = rect[0], rect[1]
    _panel(canvas, rect, "Estimated residual rotation")
    columns = [("CAM", 14), ("STATUS", 60), ("ROLL", 190), ("PITCH", 260), ("YAW", 330)]
    columns += [("INLIER", 400), ("RMS B>A (CM)", 470)]
    for label, offset in columns:
        _text(canvas, label, (x + offset, y + 54), scale=0.38, color=COLOR_TEXT_MUTED)
    for index, camera in enumerate(CAMERA_ORDER):
        ry = y + 84 + index * 34
        estimate, report = estimates.get(camera), reports.get(camera)
        _text(canvas, camera.value, (x + 14, ry), scale=0.45, color=COLOR_TEXT_PRIMARY)
        if estimate is None or report is None:
            _text(canvas, "no data", (x + 60, ry), color=COLOR_TEXT_MUTED)
            continue
        _text(canvas, report.status.value, (x + 60, ry), color=STATUS_COLORS[report.status])
        for offset, value in zip(
            (190, 260, 330),
            (estimate.delta.roll_rad, estimate.delta.pitch_rad, estimate.delta.yaw_rad),
            strict=True,
        ):
            _text(canvas, f"{math.degrees(value):+.2f}", (x + offset, ry))
        _text(canvas, f"{estimate.inlier_ratio * 100:.0f}%", (x + 400, ry))
        _text(
            canvas,
            f"{estimate.rms_before_m * 100:.1f} > {estimate.rms_after_m * 100:.2f}",
            (x + 470, ry),
        )
        if estimate.reason:
            _text(canvas, estimate.reason[:78], (x + 14, ry + 15), scale=0.34, color=COLOR_AMBER)


def _draw_rms(
    canvas: NDArray[np.uint8],
    rect: tuple[int, int, int, int],
    estimates: Mapping[CameraId, CalibrationEstimate],
) -> None:
    x, y, w, h = rect
    _panel(canvas, rect, "Ground reprojection RMS before / after (cm)")
    left, right = x + 70, x + w - 70
    peak = max((e.rms_before_m for e in estimates.values()), default=0.0) or 1.0
    row = (h - 60) / len(CAMERA_ORDER)
    for index, camera in enumerate(CAMERA_ORDER):
        cy = int(y + 48 + row * index + row / 2)
        _text(canvas, camera.value, (x + 20, cy + 5), scale=0.5, color=COLOR_TEXT_PRIMARY)
        estimate = estimates.get(camera)
        if estimate is None:
            continue
        before = int((right - left) * estimate.rms_before_m / peak)
        after = int((right - left) * estimate.rms_after_m / peak)
        cv2.rectangle(canvas, (left, cy - 14), (left + max(before, 2), cy - 2), COLOR_RED, -1)
        cv2.rectangle(canvas, (left, cy + 2), (left + max(after, 2), cy + 14), COLOR_GREEN, -1)
        _text(canvas, f"{estimate.rms_before_m * 100:.1f}", (left + before + 6, cy - 4), scale=0.38)
        _text(canvas, f"{estimate.rms_after_m * 100:.2f}", (left + after + 6, cy + 12), scale=0.38)


def render_selfcal_dashboard(
    estimates: Sequence[CalibrationEstimate],
    reports: Mapping[CameraId, DriftReport],
    history: Mapping[CameraId, Sequence[float]] | None = None,
    config: SelfCalConfig | None = None,
    output_path: Path | str | None = None,
) -> NDArray[np.uint8]:
    """Render the 1280x720 self-calibration HUD and optionally write it as PNG."""
    cfg = config or SelfCalConfig()
    by_camera = {estimate.camera_id: estimate for estimate in estimates}
    canvas = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1280, 65), (24, 28, 36), -1)
    cv2.line(canvas, (0, 65), (1280, 65), COLOR_PANEL_BORDER, 2)
    _text(
        canvas,
        "NEARFIELD360 ONLINE EXTRINSIC SELF-CALIBRATION",
        (24, 28),
        scale=0.7,
        color=COLOR_TEXT_PRIMARY,
        thickness=2,
    )
    _text(canvas, "HD-MAP-ANCHORED CAMERA MISALIGNMENT ESTIMATION & DRIFT GATING", (24, 50))

    worst = max(
        (r.status for r in reports.values()),
        key=SEVERITY.__getitem__,
        default=DriftStatus.UNOBSERVABLE,
    )
    badge_x = 1280 - 200 - 24
    cv2.rectangle(canvas, (badge_x, 15), (badge_x + 200, 51), STATUS_COLORS[worst], -1)
    _text(canvas, worst.value, (badge_x + 14, 39), scale=0.6, color=(15, 15, 15), thickness=2)

    limit = _scale_max(cfg, reports)
    _draw_drift_bars(canvas, (24, 80, 620, 300), cfg, reports)
    _draw_history(canvas, (656, 80, 600, 300), cfg, history or {}, limit)
    _draw_table(canvas, (24, 392, 620, 310), by_camera, reports)
    _draw_rms(canvas, (656, 392, 600, 310), by_camera)

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), canvas):
            raise OSError(f"failed to write dashboard PNG to {target}")
    return canvas
