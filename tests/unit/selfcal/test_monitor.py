import pytest

from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.models import CalibrationEstimate, DriftStatus, RotationDelta
from nearfield360.selfcal.monitor import DriftMonitor
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations


def _estimate(
    camera: CameraId, pitch_deg: float, *, observable: bool = True, converged: bool = True
) -> CalibrationEstimate:
    return CalibrationEstimate(
        camera_id=camera,
        delta=RotationDelta.from_degrees(pitch=pitch_deg),
        observable=observable,
        converged=converged,
        num_observations=40,
        inlier_ratio=0.9,
        rms_before_m=0.1,
        rms_after_m=0.01,
        iterations=5,
    )


@pytest.mark.parametrize(
    ("pitch_deg", "status"),
    [
        (0.0, DriftStatus.NOMINAL),
        (0.29, DriftStatus.NOMINAL),
        (0.3, DriftStatus.WARNING),
        (0.79, DriftStatus.WARNING),
        (0.8, DriftStatus.RECALIBRATE),
        (2.99, DriftStatus.RECALIBRATE),
        (3.0, DriftStatus.FAULT),
        (6.0, DriftStatus.FAULT),
    ],
)
def test_classification_boundaries(pitch_deg: float, status: DriftStatus) -> None:
    monitor = DriftMonitor()

    report = monitor.update(_estimate(CameraId.FRONT, pitch_deg))

    assert report.status is status


def test_unseen_camera_has_no_report_and_unobservable_first_rejection() -> None:
    monitor = DriftMonitor()
    assert monitor.report(CameraId.REAR) is None
    assert not monitor.is_trusted(CameraId.REAR)

    report = monitor.update(_estimate(CameraId.REAR, 0.0, observable=False))

    assert report.status is DriftStatus.UNOBSERVABLE
    assert (report.accepted_updates, report.rejected_updates) == (0, 1)
    assert not monitor.is_trusted(CameraId.REAR)


def test_ewma_smooths_toward_new_estimates() -> None:
    monitor = DriftMonitor(SelfCalConfig(ewma_alpha=0.5))
    monitor.update(_estimate(CameraId.FRONT, 0.0))

    report = monitor.update(_estimate(CameraId.FRONT, 1.0))

    assert report.magnitude_deg == pytest.approx(0.5)
    assert report.status is DriftStatus.WARNING


def test_rejected_estimates_leave_state_untouched() -> None:
    monitor = DriftMonitor()
    monitor.update(_estimate(CameraId.FRONT, 2.0))

    report = monitor.update(_estimate(CameraId.FRONT, 0.0, converged=False))
    after_unobservable = monitor.update(_estimate(CameraId.FRONT, 0.0, observable=False))

    assert report.magnitude_deg == pytest.approx(2.0)
    assert after_unobservable.status is DriftStatus.RECALIBRATE
    assert (after_unobservable.accepted_updates, after_unobservable.rejected_updates) == (1, 2)


def test_ramping_misalignment_passes_through_every_severity() -> None:
    monitor = DriftMonitor(SelfCalConfig(ewma_alpha=1.0))
    calibrator = ExtrinsicSelfCalibrator()
    seen: list[DriftStatus] = []
    for step, pitch in enumerate([0.0, 0.5, 1.2, 4.0]):
        observations = synthesize_observations(
            CameraId.FRONT, RotationDelta.from_degrees(pitch=pitch), num_points=50, seed=step
        )
        seen.append(monitor.update(calibrator.estimate(DEFAULT_CAMERA_ORIGIN, observations)).status)

    assert seen == [
        DriftStatus.NOMINAL,
        DriftStatus.WARNING,
        DriftStatus.RECALIBRATE,
        DriftStatus.FAULT,
    ]
    assert len(monitor.history_deg(CameraId.FRONT)) == 4


def test_trust_gate_and_reset() -> None:
    monitor = DriftMonitor()
    monitor.update(_estimate(CameraId.FRONT, 0.5))
    monitor.update(_estimate(CameraId.REAR, 1.5))

    assert monitor.is_trusted(CameraId.FRONT)
    assert not monitor.is_trusted(CameraId.REAR)
    assert set(monitor.reports()) == {CameraId.FRONT, CameraId.REAR}

    monitor.reset(CameraId.REAR)

    assert monitor.report(CameraId.REAR) is None
    assert monitor.history_deg(CameraId.REAR) == []
