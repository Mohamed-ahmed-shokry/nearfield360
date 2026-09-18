from __future__ import annotations

import math

import numpy as np
import pytest

from nearfield360.data.calibration import (
    CameraCalibration,
    ExtrinsicParameters,
    IntrinsicParameters,
)
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.fisheye import RadialPolynomialFisheye
from nearfield360.geometry.transforms import RigidTransform
from nearfield360.robustness.calibration import (
    euler_to_rotation_matrix,
    perturb_calibrated_camera,
    perturb_camera_calibration,
    perturb_rigid_transform,
    rotation_matrix_to_quaternion,
)


@pytest.fixture
def sample_camera() -> CalibratedCamera:
    # Front camera roughly pointed along vehicle +X
    intrinsic = IntrinsicParameters(
        aspect_ratio=1.0,
        cx_offset=0.0,
        cy_offset=0.0,
        height=480,
        k1=200.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model="radial_poly",
        poly_order=4,
        width=640,
    )
    # Camera frame: X right (-Y veh), Y down (-Z veh), Z forward (+X veh)
    rot = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    trans = np.array([2.5, 0.0, 0.7], dtype=np.float64)
    transform = RigidTransform(rot, trans)
    return CalibratedCamera(
        name=CameraId.FRONT,
        model=RadialPolynomialFisheye(intrinsic, theta_max=2.2),
        camera_to_vehicle=transform,
    )


def test_euler_to_rotation_matrix_identity() -> None:
    rot = euler_to_rotation_matrix(0.0, 0.0, 0.0)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-7)


@pytest.mark.parametrize(
    ("roll", "pitch", "yaw"),
    [
        (15.0, 0.0, 0.0),
        (0.0, -25.0, 0.0),
        (0.0, 0.0, 45.0),
        (12.5, -8.3, 19.1),
    ],
)
def test_euler_to_rotation_matrix_orthonormal(roll: float, pitch: float, yaw: float) -> None:
    rot = euler_to_rotation_matrix(roll, pitch, yaw)
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-6)
    assert math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=1e-6)


def test_quaternion_roundtrip() -> None:
    # Identity
    quat = rotation_matrix_to_quaternion(np.eye(3))
    assert quat == (0.0, 0.0, 0.0, 1.0)

    # Random rotations
    for roll, pitch, yaw in [(10.0, 5.0, -15.0), (45.0, -30.0, 60.0), (-80.0, 10.0, -5.0)]:
        orig_rot = euler_to_rotation_matrix(roll, pitch, yaw)
        quat = rotation_matrix_to_quaternion(orig_rot)
        # Verify unit norm
        norm = math.hypot(*quat)
        assert math.isclose(norm, 1.0, abs_tol=1e-6)

        # Convert back using RigidTransform.from_quaternion
        reconstructed = RigidTransform.from_quaternion(quat)
        np.testing.assert_allclose(reconstructed.rotation, orig_rot, atol=1e-6)


def test_rotation_matrix_to_quaternion_branch_coverage() -> None:
    # 180 deg around X: diag(1, -1, -1) -> r00 max
    rot_x = np.diag([1.0, -1.0, -1.0])
    quat_x = rotation_matrix_to_quaternion(rot_x)
    assert math.isclose(math.hypot(*quat_x), 1.0, abs_tol=1e-6)
    np.testing.assert_allclose(RigidTransform.from_quaternion(quat_x).rotation, rot_x, atol=1e-6)

    # 180 deg around Y: diag(-1, 1, -1) -> r11 max
    rot_y = np.diag([-1.0, 1.0, -1.0])
    quat_y = rotation_matrix_to_quaternion(rot_y)
    assert math.isclose(math.hypot(*quat_y), 1.0, abs_tol=1e-6)
    np.testing.assert_allclose(RigidTransform.from_quaternion(quat_y).rotation, rot_y, atol=1e-6)

    # 180 deg around Z: diag(-1, -1, 1) -> r22 max
    rot_z = np.diag([-1.0, -1.0, 1.0])
    quat_z = rotation_matrix_to_quaternion(rot_z)
    assert math.isclose(math.hypot(*quat_z), 1.0, abs_tol=1e-6)
    np.testing.assert_allclose(RigidTransform.from_quaternion(quat_z).rotation, rot_z, atol=1e-6)


def test_perturb_rigid_transform() -> None:
    base = RigidTransform(np.eye(3), (1.0, 2.0, 3.0))

    # Zero perturbation returns identical transform
    unperturbed = perturb_rigid_transform(base)
    np.testing.assert_allclose(unperturbed.rotation, base.rotation, atol=1e-7)
    np.testing.assert_allclose(unperturbed.translation, base.translation, atol=1e-7)

    # Translation only
    trans_perturbed = perturb_rigid_transform(base, translation_m=(0.1, -0.2, 0.05))
    np.testing.assert_allclose(trans_perturbed.translation, [1.1, 1.8, 3.05], atol=1e-7)

    # Rotation perturbation
    rot_perturbed = perturb_rigid_transform(base, pitch_deg=5.0)
    assert not np.allclose(rot_perturbed.rotation, base.rotation)


def test_perturb_calibrated_camera(sample_camera: CalibratedCamera) -> None:
    perturbed = perturb_calibrated_camera(
        sample_camera,
        roll_deg=1.0,
        pitch_deg=-2.0,
        yaw_deg=0.5,
        translation_m=(0.02, -0.01, 0.03),
    )

    assert perturbed.name == sample_camera.name
    assert perturbed.model == sample_camera.model

    # Center displaced by translation perturbation
    expected_translation = sample_camera.camera_to_vehicle.translation + np.array(
        [0.02, -0.01, 0.03]
    )
    np.testing.assert_allclose(
        perturbed.camera_to_vehicle.translation, expected_translation, atol=1e-6
    )

    # Ray directions deflected
    pixels = np.array([[320.0, 240.0]])  # optical center
    orig_ray = sample_camera.pixel_rays_vehicle(pixels)
    pert_ray = perturbed.pixel_rays_vehicle(pixels)

    assert not np.allclose(orig_ray.directions, pert_ray.directions)


def test_perturb_camera_calibration() -> None:
    intrinsic = IntrinsicParameters(
        aspect_ratio=1.0,
        cx_offset=0.0,
        cy_offset=0.0,
        height=480,
        k1=200.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model="radial_poly",
        poly_order=4,
        width=640,
    )
    extrinsic = ExtrinsicParameters(
        quaternion=(0.0, 0.0, 0.0, 1.0),
        translation=(2.0, 0.0, 0.5),
    )
    calib = CameraCalibration(
        name=CameraId.FRONT,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )

    perturbed = perturb_camera_calibration(
        calib,
        roll_deg=2.0,
        pitch_deg=1.0,
        yaw_deg=-1.5,
        translation_m=(0.05, 0.02, -0.03),
    )

    assert perturbed.name == calib.name
    assert perturbed.intrinsic == calib.intrinsic
    assert math.isclose(math.hypot(*perturbed.extrinsic.quaternion), 1.0, abs_tol=1e-3)
    np.testing.assert_allclose(
        perturbed.extrinsic.translation,
        [2.05, 0.02, 0.47],
        atol=1e-6,
    )
