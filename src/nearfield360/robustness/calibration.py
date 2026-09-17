"""Deterministic calibration perturbation engine for automotive fisheye cameras.

Enables sensitivity and robustness analysis by systematically perturbing camera
extrinsic parameters (roll, pitch, yaw rotation and 3D translation) and evaluating
geometric deviations in ray projection and ground intersection.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from nearfield360.data.calibration import CameraCalibration, ExtrinsicParameters
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.transforms import RigidTransform


def euler_to_rotation_matrix(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    *,
    degrees: bool = True,
) -> npt.NDArray[np.float64]:
    """Build an orthonormal (3, 3) rotation matrix from roll, pitch, yaw angles.

    In the camera frame (X right, Y down, Z forward):
    - Pitch (theta): rotation around the horizontal X axis
    - Yaw (psi): rotation around the vertical Y axis
    - Roll (phi): rotation around the optical Z axis

    Composite rotation is: ``R = R_z(roll) @ R_y(yaw) @ R_x(pitch)``.

    Parameters
    ----------
    roll:
        Roll angle around Z axis (degrees by default).
    pitch:
        Pitch angle around X axis (degrees by default).
    yaw:
        Yaw angle around Y axis (degrees by default).
    degrees:
        If True, interpret angles in degrees; otherwise radians.

    Returns
    -------
    npt.NDArray[np.float64]
        Orthonormal 3x3 rotation matrix with determinant +1.
    """
    if degrees:
        phi = math.radians(roll)
        theta = math.radians(pitch)
        psi = math.radians(yaw)
    else:
        phi = float(roll)
        theta = float(pitch)
        psi = float(yaw)

    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    cos_psi, sin_psi = math.cos(psi), math.sin(psi)

    # R_x(pitch)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )

    # R_y(yaw)
    ry = np.array(
        [
            [cos_psi, 0.0, sin_psi],
            [0.0, 1.0, 0.0],
            [-sin_psi, 0.0, cos_psi],
        ],
        dtype=np.float64,
    )

    # R_z(roll)
    rz = np.array(
        [
            [cos_phi, -sin_phi, 0.0],
            [sin_phi, cos_phi, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rot = rz @ (ry @ rx)
    # Ensure exact orthonormality against floating point roundoff
    u, _, vt = np.linalg.svd(rot)
    rot_ortho = u @ vt
    if np.linalg.det(rot_ortho) < 0.0:
        rot_ortho[:, -1] *= -1.0
    return np.asarray(rot_ortho, dtype=np.float64)


def rotation_matrix_to_quaternion(
    rotation: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Convert an orthonormal 3x3 rotation matrix to an ``(x, y, z, w)`` unit quaternion.

    Matches WoodScape's right-handed active quaternion convention.
    """
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {r.shape}")

    trace = float(np.trace(r))

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    norm = math.hypot(x, y, z, w)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(x / norm), float(y / norm), float(z / norm), float(w / norm))


def perturb_rigid_transform(
    transform: RigidTransform,
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidTransform:
    """Return a new RigidTransform with angular and positional perturbations.

    The rotation perturbation is composed in the camera frame:
    ``R_perturbed = R_original @ delta_R``.
    The translation perturbation is added in vehicle coordinates:
    ``t_perturbed = t_original + delta_t``.
    """
    delta_r = euler_to_rotation_matrix(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg, degrees=True)
    new_rotation = transform.rotation @ delta_r
    new_translation = transform.translation + np.asarray(translation_m, dtype=np.float64)

    # Clean slight numerical roundoff for strict RigidTransform checks
    u, _, vt = np.linalg.svd(new_rotation)
    ortho_rot = u @ vt
    if np.linalg.det(ortho_rot) < 0.0:
        ortho_rot[:, -1] *= -1.0

    return RigidTransform(ortho_rot, new_translation)


def perturb_calibrated_camera(
    camera: CalibratedCamera,
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CalibratedCamera:
    """Return a CalibratedCamera instance with perturbed extrinsics."""
    perturbed_transform = perturb_rigid_transform(
        camera.camera_to_vehicle,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        translation_m=translation_m,
    )
    return CalibratedCamera(
        name=camera.name,
        model=camera.model,
        camera_to_vehicle=perturbed_transform,
    )


def perturb_camera_calibration(
    calibration: CameraCalibration,
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraCalibration:
    """Return a CameraCalibration record with perturbed extrinsics."""
    orig_transform = RigidTransform.from_quaternion(
        calibration.extrinsic.quaternion, calibration.extrinsic.translation
    )
    perturbed_transform = perturb_rigid_transform(
        orig_transform,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        translation_m=translation_m,
    )
    new_quat = rotation_matrix_to_quaternion(perturbed_transform.rotation)
    new_trans_tuple = (
        float(perturbed_transform.translation[0]),
        float(perturbed_transform.translation[1]),
        float(perturbed_transform.translation[2]),
    )
    new_extrinsic = ExtrinsicParameters(
        quaternion=new_quat,
        translation=new_trans_tuple,
    )
    return CameraCalibration(
        name=calibration.name,
        extrinsic=new_extrinsic,
        intrinsic=calibration.intrinsic,
    )


__all__ = [
    "euler_to_rotation_matrix",
    "perturb_calibrated_camera",
    "perturb_camera_calibration",
    "perturb_rigid_transform",
    "rotation_matrix_to_quaternion",
]
