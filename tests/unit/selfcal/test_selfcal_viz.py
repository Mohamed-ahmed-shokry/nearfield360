import cv2
import numpy as np

from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.models import DriftStatus, RotationDelta
from nearfield360.selfcal.monitor import DriftMonitor
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations
from nearfield360.selfcal.viz import render_selfcal_dashboard

TRUTH = {
    CameraId.FRONT: (0.1, 0.0, 0.0),
    CameraId.REAR: (0.0, 0.6, 0.0),
    CameraId.MIRROR_LEFT: (1.2, 0.0, 0.0),
    CameraId.MIRROR_RIGHT: (3.0, 2.0, 1.0),
}


def _scenario() -> tuple[list, DriftMonitor]:
    monitor = DriftMonitor()
    calibrator = ExtrinsicSelfCalibrator()
    estimates = []
    for step in range(3):
        estimates = []
        for camera, (roll, pitch, yaw) in TRUTH.items():
            observations = synthesize_observations(
                camera, RotationDelta.from_degrees(roll, pitch, yaw), num_points=40, seed=step
            )
            estimate = calibrator.estimate(DEFAULT_CAMERA_ORIGIN, observations)
            monitor.update(estimate)
            estimates.append(estimate)
    return estimates, monitor


def test_dashboard_is_1280x720_bgr_with_content() -> None:
    estimates, monitor = _scenario()

    canvas = render_selfcal_dashboard(
        estimates, monitor.reports(), {c: monitor.history_deg(c) for c in TRUTH}
    )

    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8
    assert len(np.unique(canvas.reshape(-1, 3), axis=0)) > 8


def test_dashboard_writes_decodable_png(tmp_path) -> None:
    estimates, monitor = _scenario()
    target = tmp_path / "nested" / "selfcal.png"

    render_selfcal_dashboard(estimates, monitor.reports(), output_path=target)

    decoded = cv2.imread(str(target))
    assert decoded is not None and decoded.shape == (720, 1280, 3)
    assert target.read_bytes().startswith(b"\x89PNG")


def test_dashboard_handles_missing_cameras_and_rejected_estimates() -> None:
    monitor = DriftMonitor(SelfCalConfig())
    observations = synthesize_observations(CameraId.FRONT, RotationDelta(), num_points=3)
    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)
    monitor.update(estimate)

    canvas = render_selfcal_dashboard([estimate], monitor.reports())

    assert canvas.shape == (720, 1280, 3)
    assert monitor.report(CameraId.FRONT).status is DriftStatus.UNOBSERVABLE  # type: ignore[union-attr]


def test_dashboard_header_color_reflects_worst_status() -> None:
    estimates, monitor = _scenario()
    canvas = render_selfcal_dashboard(estimates, monitor.reports())

    badge_pixel = canvas[20, 1280 - 24 - 10]

    assert tuple(int(v) for v in badge_pixel) == (50, 60, 235)
