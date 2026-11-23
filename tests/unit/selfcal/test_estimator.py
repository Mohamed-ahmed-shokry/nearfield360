import math

import numpy as np
import pytest

from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.estimator import ExtrinsicSelfCalibrator
from nearfield360.selfcal.models import GroundObservation, RotationDelta
from nearfield360.selfcal.synthetic import DEFAULT_CAMERA_ORIGIN, synthesize_observations

TRUE_DELTA = RotationDelta.from_degrees(roll=0.5, pitch=-0.3, yaw=0.4)


def _max_error_deg(estimate_vector: np.ndarray) -> float:
    return float(np.degrees(np.max(np.abs(estimate_vector - TRUE_DELTA.as_vector()))))


def test_noiseless_observations_recover_rotation_exactly() -> None:
    observations = synthesize_observations(CameraId.FRONT, TRUE_DELTA, noise_deg=0.0, seed=1)

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert estimate.observable and estimate.converged
    np.testing.assert_allclose(estimate.delta.as_vector(), TRUE_DELTA.as_vector(), atol=1e-6)
    assert estimate.rms_after_m < 1e-4
    assert estimate.rms_before_m > 0.1


@pytest.mark.parametrize("seed", range(5))
def test_noisy_observations_recover_rotation_within_five_hundredths_degree(seed: int) -> None:
    observations = synthesize_observations(
        CameraId.REAR, TRUE_DELTA, noise_deg=0.02, num_points=60, seed=seed
    )

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert estimate.observable
    assert _max_error_deg(estimate.delta.as_vector()) < 0.05
    assert estimate.rms_after_m < estimate.rms_before_m
    assert estimate.sigma_deg is not None and max(estimate.sigma_deg) < 0.1


@pytest.mark.parametrize("seed", range(5))
def test_twenty_percent_outliers_do_not_break_estimate(seed: int) -> None:
    observations = synthesize_observations(
        CameraId.FRONT, TRUE_DELTA, noise_deg=0.02, num_points=60, outlier_fraction=0.2, seed=seed
    )

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert estimate.observable
    assert _max_error_deg(estimate.delta.as_vector()) < 0.05
    assert 0.7 <= estimate.inlier_ratio <= 0.85


def test_too_few_correspondences_are_unobservable() -> None:
    observations = synthesize_observations(CameraId.FRONT, TRUE_DELTA, num_points=5)

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert not estimate.observable
    assert estimate.delta.magnitude_rad == 0.0
    assert "correspondences" in estimate.reason
    assert estimate.sigma_deg is None


def test_degenerate_identical_observations_are_flagged() -> None:
    template = synthesize_observations(CameraId.FRONT, TRUE_DELTA, noise_deg=0.0, num_points=1)[0]
    observations = [template] * 20

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert not estimate.observable
    assert "ill-conditioned" in estimate.reason


def test_rays_pointing_above_the_horizon_are_ignored() -> None:
    observations = [
        GroundObservation(
            camera_id=CameraId.FRONT,
            direction=(1.0, 0.0, 0.5),
            reference_x_m=5.0,
            reference_y_m=0.0,
        )
        for _ in range(30)
    ]

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert not estimate.observable
    assert estimate.num_observations == 30
    assert "usable correspondences" in estimate.reason


def test_exhausted_iteration_budget_reports_non_convergence() -> None:
    config = SelfCalConfig(max_iterations=1, convergence_tolerance_rad=1e-12)
    observations = synthesize_observations(CameraId.FRONT, TRUE_DELTA, noise_deg=0.02, seed=2)

    estimate = ExtrinsicSelfCalibrator(config).estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert not estimate.converged
    assert not estimate.observable
    assert "converge" in estimate.reason


def test_mostly_wrong_matches_fail_inlier_gate() -> None:
    observations = synthesize_observations(
        CameraId.FRONT, TRUE_DELTA, noise_deg=0.02, outlier_fraction=0.9, num_points=60, seed=4
    )

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert not estimate.observable
    assert estimate.inlier_ratio < 0.5


def test_input_validation() -> None:
    calibrator = ExtrinsicSelfCalibrator()
    with pytest.raises(ValueError, match="empty"):
        calibrator.estimate(DEFAULT_CAMERA_ORIGIN, [])
    mixed = synthesize_observations(CameraId.FRONT, TRUE_DELTA, num_points=3)
    mixed += synthesize_observations(CameraId.REAR, TRUE_DELTA, num_points=3)
    with pytest.raises(ValueError, match="one camera"):
        calibrator.estimate(DEFAULT_CAMERA_ORIGIN, mixed)


def test_ground_height_offset_is_respected() -> None:
    observations = synthesize_observations(
        CameraId.FRONT, TRUE_DELTA, noise_deg=0.0, ground_z=0.1, seed=7
    )

    estimate = ExtrinsicSelfCalibrator(ground_z=0.1).estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert math.isclose(estimate.delta.roll_rad, TRUE_DELTA.roll_rad, abs_tol=1e-6)
    assert estimate.observable


@pytest.mark.parametrize("pitch_deg", [0.1, 0.5, 1.2, 4.0])
def test_converges_across_misalignment_magnitudes(pitch_deg: float) -> None:
    truth = RotationDelta.from_degrees(pitch=pitch_deg)
    observations = synthesize_observations(
        CameraId.FRONT, truth, noise_deg=0.02, num_points=50, seed=2
    )

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert estimate.converged and estimate.observable
    assert estimate.iterations < 15
    assert math.degrees(abs(estimate.delta.pitch_rad - truth.pitch_rad)) < 0.05


def _line_observations(
    truth: RotationDelta, *, with_normals: bool, shift_m: float = 1.0
) -> list[GroundObservation]:
    """Observations of perpendicular lines whose references slide along each line."""
    rng = np.random.default_rng(11)
    exact = synthesize_observations(CameraId.FRONT, truth, noise_deg=0.0, num_points=60, seed=5)
    observations = []
    for index, obs in enumerate(exact):
        across_x = index % 2 == 0  # line x=const has normal (1, 0); line y=const has (0, 1)
        slide = float(rng.uniform(-shift_m, shift_m))
        reference = (
            (obs.reference_x_m, obs.reference_y_m + slide)
            if across_x
            else (obs.reference_x_m + slide, obs.reference_y_m)
        )
        observations.append(
            GroundObservation(
                camera_id=obs.camera_id,
                direction=obs.direction,
                reference_x_m=reference[0],
                reference_y_m=reference[1],
                reference_normal=((1.0, 0.0) if across_x else (0.0, 1.0)) if with_normals else None,
            )
        )
    return observations


def test_point_to_line_residuals_ignore_along_line_displacement() -> None:
    observations = _line_observations(TRUE_DELTA, with_normals=True)

    estimate = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, observations)

    assert estimate.observable and estimate.converged
    np.testing.assert_allclose(estimate.delta.as_vector(), TRUE_DELTA.as_vector(), atol=1e-5)
    assert estimate.rms_after_m < 1e-4


def test_point_to_point_is_biased_by_the_same_along_line_displacement() -> None:
    observations = _line_observations(TRUE_DELTA, with_normals=False)

    estimate = ExtrinsicSelfCalibrator(SelfCalConfig(min_inlier_ratio=0.1)).estimate(
        DEFAULT_CAMERA_ORIGIN, observations
    )

    assert _max_error_deg(estimate.delta.as_vector()) > 0.05


def test_unnormalized_reference_normal_is_normalized() -> None:
    unit = _line_observations(TRUE_DELTA, with_normals=True)
    scaled = [
        o.model_copy(
            update={"reference_normal": (3.0 * o.reference_normal[0], 3.0 * o.reference_normal[1])}
        )  # type: ignore[index]
        for o in unit
    ]

    first = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, unit)
    second = ExtrinsicSelfCalibrator().estimate(DEFAULT_CAMERA_ORIGIN, scaled)

    np.testing.assert_allclose(first.delta.as_vector(), second.delta.as_vector(), atol=1e-9)
