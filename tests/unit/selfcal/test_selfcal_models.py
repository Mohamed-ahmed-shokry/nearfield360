import math

import numpy as np
import pytest
from pydantic import ValidationError

from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.models import (
    CalibrationEstimate,
    DriftReport,
    DriftStatus,
    GroundObservation,
    RotationDelta,
    rotation_matrix,
)


def test_rotation_delta_defaults_to_identity() -> None:
    delta = RotationDelta()

    assert delta.magnitude_rad == 0.0
    np.testing.assert_allclose(delta.matrix(), np.eye(3))


def test_rotation_delta_matrix_is_orthonormal_with_unit_determinant() -> None:
    matrix = RotationDelta.from_degrees(roll=0.5, pitch=-0.2, yaw=1.0).matrix()

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-12)


def test_rotation_delta_yaw_rotates_x_axis_toward_y() -> None:
    rotated = RotationDelta.from_degrees(yaw=30.0).matrix() @ np.array([1.0, 0.0, 0.0])

    np.testing.assert_allclose(rotated, [math.cos(math.pi / 6), 0.5, 0.0], atol=1e-12)


def test_rotation_delta_magnitude_matches_vector_norm() -> None:
    delta = RotationDelta.from_degrees(roll=3.0, pitch=4.0)

    assert math.isclose(delta.magnitude_deg, 5.0, rel_tol=1e-12)
    np.testing.assert_allclose(
        RotationDelta.from_vector(delta.as_vector()).as_vector(), delta.as_vector()
    )


def test_rotation_delta_rejects_bad_vector_shape_and_range() -> None:
    with pytest.raises(ValueError, match="shape"):
        RotationDelta.from_vector([0.1, 0.2])
    with pytest.raises(ValidationError):
        RotationDelta(roll_rad=2.0)
    with pytest.raises(ValidationError):
        RotationDelta(yaw_rad=float("nan"))


def test_rotation_matrix_small_angle_branch_is_first_order() -> None:
    matrix = rotation_matrix([1e-14, 0.0, 0.0])

    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


def test_ground_observation_validates_direction() -> None:
    observation = GroundObservation(
        camera_id=CameraId.FRONT, direction=(1.0, 0.0, -0.5), reference_x_m=4.0, reference_y_m=0.0
    )
    assert observation.camera_id is CameraId.FRONT
    with pytest.raises(ValidationError, match="non-zero"):
        GroundObservation(
            camera_id=CameraId.FRONT,
            direction=(0.0, 0.0, 0.0),
            reference_x_m=0.0,
            reference_y_m=0.0,
        )
    with pytest.raises(ValidationError, match="finite"):
        GroundObservation(
            camera_id=CameraId.FRONT,
            direction=(float("inf"), 0.0, 1.0),
            reference_x_m=0.0,
            reference_y_m=0.0,
        )


def test_estimate_and_report_round_trip_through_json() -> None:
    estimate = CalibrationEstimate(
        camera_id=CameraId.REAR,
        delta=RotationDelta.from_degrees(pitch=0.5),
        observable=True,
        converged=True,
        num_observations=40,
        inlier_ratio=0.95,
        rms_before_m=0.2,
        rms_after_m=0.01,
        iterations=6,
        sigma_deg=(0.02, 0.02, 0.03),
        condition_number=12.5,
    )
    report = DriftReport(
        camera_id=CameraId.REAR,
        status=DriftStatus.WARNING,
        smoothed_delta=estimate.delta,
        magnitude_deg=0.5,
        accepted_updates=3,
        rejected_updates=1,
    )

    assert CalibrationEstimate.model_validate_json(estimate.model_dump_json()) == estimate
    assert DriftReport.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValidationError):
        CalibrationEstimate(**{**estimate.model_dump(), "inlier_ratio": 1.5})


def test_ground_observation_reference_normal_validation() -> None:
    base = {
        "camera_id": CameraId.FRONT,
        "direction": (1.0, 0.0, -0.5),
        "reference_x_m": 4.0,
        "reference_y_m": 0.0,
    }
    assert GroundObservation(**base).reference_normal is None
    assert GroundObservation(**base, reference_normal=(0.0, 2.0)).reference_normal == (0.0, 2.0)
    with pytest.raises(ValidationError, match="non-zero"):
        GroundObservation(**base, reference_normal=(0.0, 0.0))
    with pytest.raises(ValidationError, match="finite"):
        GroundObservation(**base, reference_normal=(float("nan"), 1.0))
