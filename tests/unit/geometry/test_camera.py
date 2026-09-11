import math

import numpy as np
import pytest
from numpy.typing import ArrayLike

from nearfield360.data.calibration import (
    CameraCalibration,
    ExtrinsicParameters,
    IntrinsicParameters,
)
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.camera import CalibratedCamera


def _calibration(
    *,
    quaternion: tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5),
    translation: tuple[float, float, float] = (3.75, 0.0, 0.66),
    intrinsics: IntrinsicParameters | None = None,
) -> CameraCalibration:
    return CameraCalibration(
        name=CameraId.FRONT,
        extrinsic=ExtrinsicParameters(quaternion=quaternion, translation=translation),
        intrinsic=intrinsics
        or IntrinsicParameters(
            aspect_ratio=1.0,
            cx_offset=0.0,
            cy_offset=0.0,
            height=480,
            width=640,
            k1=100.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            model="radial_poly",
            poly_order=4,
        ),
    )


def test_from_calibration_preserves_camera_identity_and_transform_conventions() -> None:
    calibration = _calibration()

    camera = CalibratedCamera.from_calibration(calibration, theta_max=2.2)

    assert camera.name is CameraId.FRONT
    assert camera.model.intrinsics == calibration.intrinsic
    assert camera.model.theta_max == 2.2
    np.testing.assert_array_equal(
        camera.camera_to_vehicle.rotation, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    )
    np.testing.assert_array_equal(camera.camera_to_vehicle.translation, (3.75, 0.0, 0.66))
    np.testing.assert_allclose(
        camera.vehicle_to_camera.transform_points((4.75, 0.0, 0.66)), (0, 0, 1), atol=1e-15
    )


def test_angular_limit_must_be_explicit_and_valid() -> None:
    calibration = _calibration()
    with pytest.raises(TypeError, match="theta_max"):
        CalibratedCamera.from_calibration(calibration)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="theta_max"):
        CalibratedCamera.from_calibration(calibration, theta_max=math.pi)


def test_project_vehicle_uses_camera_center_and_analytic_front_camera_axes() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)
    center = camera.camera_to_vehicle.translation
    points = center + np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    principal = np.array(camera.model.intrinsics.principal_point)
    quarter_turn_radius = 100.0 * math.pi / 2.0

    result = camera.project_vehicle(points)

    assert result.valid.all()
    np.testing.assert_allclose(
        result.pixels,
        principal + np.array([[0, 0], [quarter_turn_radius, 0], [0, quarter_turn_radius]]),
        atol=1e-12,
    )


def test_pixel_rays_have_metric_origins_and_rotated_unit_directions() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)
    principal = np.array(camera.model.intrinsics.principal_point)
    quarter_turn_radius = 100.0 * math.pi / 2.0
    pixels = principal + np.array([[0, 0], [quarter_turn_radius, 0], [0, quarter_turn_radius]])

    rays = camera.pixel_rays_vehicle(pixels)

    assert rays.valid.all()
    np.testing.assert_array_equal(rays.origins, np.tile((3.75, 0.0, 0.66), (3, 1)))
    np.testing.assert_allclose(rays.directions, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(rays.directions, axis=-1), np.ones(3), atol=1e-15)


def test_camera_center_does_not_define_a_ray_even_with_general_rotation() -> None:
    raw_quaternion = np.array([1.0, 2.0, 3.0, 4.0]) / math.sqrt(30.0)
    quaternion = tuple(float(value) for value in raw_quaternion)
    calibration = _calibration(
        quaternion=(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    )
    camera = CalibratedCamera.from_calibration(calibration, theta_max=2.2)

    result = camera.project_vehicle(camera.camera_to_vehicle.translation)

    assert not result.valid
    assert np.isnan(result.pixels).all()


def test_pixel_ray_point_roundtrip_preserves_pixels_at_caller_supplied_distances() -> None:
    intrinsics = _calibration().intrinsic.model_copy(
        update={"aspect_ratio": 1.25, "cx_offset": 3.0, "cy_offset": -2.0, "k2": -0.4, "k3": 0.05}
    )
    camera = CalibratedCamera.from_calibration(_calibration(intrinsics=intrinsics), theta_max=2.2)
    pixels = np.array(intrinsics.principal_point) + np.array(
        [[[-60, -40], [0, 0], [50, 25]], [[-80, 70], [30, -50], [90, 30]]]
    )
    distances = np.array([[0.25, 1.0, 1000.0], [3.4, 12.0, 0.5]])

    rays = camera.pixel_rays_vehicle(pixels)
    supplied_points = rays.origins + distances[..., None] * rays.directions
    projected = camera.project_vehicle(supplied_points)

    assert rays.valid.all()
    assert projected.valid.all()
    np.testing.assert_allclose(projected.pixels, pixels, atol=1e-10)


def test_invalid_pixels_keep_nan_origins_and_directions_without_poisoning_valid_rays() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)
    principal = np.array(camera.model.intrinsics.principal_point)
    pixels = np.array(
        [
            principal,
            [math.nan, 20.0],
            [10.0, math.inf],
            principal + np.array([camera.model.radius_max + 1, 0]),
        ]
    )
    original = pixels.copy()

    rays = camera.pixel_rays_vehicle(pixels)

    np.testing.assert_array_equal(rays.valid, [True, False, False, False])
    np.testing.assert_allclose(rays.directions[0], (1, 0, 0), atol=1e-15)
    assert np.isnan(rays.directions[~rays.valid]).all()
    assert np.isnan(rays.origins[~rays.valid]).all()
    np.testing.assert_array_equal(pixels, original)


def test_invalid_vehicle_points_do_not_poison_valid_projection_elements() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)
    center = camera.camera_to_vehicle.translation
    points = np.array(
        [
            center + np.array([1, 0, 0]),
            [math.nan, 0, 0],
            [0, math.inf, 0],
            center,
            center - np.array([1, 0, 0]),
        ]
    )
    original = points.copy()

    projected = camera.project_vehicle(points)

    np.testing.assert_array_equal(projected.valid, [True, False, False, False, False])
    np.testing.assert_allclose(projected.pixels[0], camera.model.intrinsics.principal_point)
    assert np.isnan(projected.pixels[~projected.valid]).all()
    np.testing.assert_array_equal(points, original)


def test_image_bounds_can_be_disabled_in_both_vehicle_operations() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.8)
    outside_pixel = np.array(camera.model.intrinsics.principal_point) + np.array((0.0, -250.0))

    bounded_rays = camera.pixel_rays_vehicle(outside_pixel)
    unbounded_rays = camera.pixel_rays_vehicle(outside_pixel, check_image_bounds=False)
    supplied_point = unbounded_rays.origins + 2.0 * unbounded_rays.directions
    bounded_projection = camera.project_vehicle(supplied_point)
    unbounded_projection = camera.project_vehicle(supplied_point, check_image_bounds=False)

    assert not bounded_rays.valid
    assert np.isnan(bounded_rays.origins).all()
    assert unbounded_rays.valid
    assert not bounded_projection.valid
    assert unbounded_projection.valid
    np.testing.assert_allclose(unbounded_projection.pixels, outside_pixel, atol=1e-12)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3), (0,), (2, 0)])
def test_vehicle_camera_operations_preserve_scalar_and_batch_shapes(
    batch_shape: tuple[int, ...],
) -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)
    pixels = np.broadcast_to(camera.model.intrinsics.principal_point, (*batch_shape, 2))
    points = np.broadcast_to(
        camera.camera_to_vehicle.translation + np.array((1, 0, 0)), (*batch_shape, 3)
    )

    rays = camera.pixel_rays_vehicle(pixels)
    projected = camera.project_vehicle(points)

    assert rays.origins.shape == (*batch_shape, 3)
    assert rays.directions.shape == (*batch_shape, 3)
    assert rays.valid.shape == batch_shape
    assert projected.pixels.shape == (*batch_shape, 2)
    assert projected.valid.shape == batch_shape
    assert rays.valid.all()
    assert projected.valid.all()


@pytest.mark.parametrize(
    "points", [1.0, [1, 2], np.ones((3, 1)), [1j, 0, 0], [True, False, True], [[1, 2, 3], [4, 5]]]
)
def test_project_vehicle_rejects_malformed_coordinate_arrays(points: ArrayLike) -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)

    with pytest.raises(ValueError):
        camera.project_vehicle(points)


def test_all_invalid_arrays_return_masks_without_finite_only_transform_errors() -> None:
    camera = CalibratedCamera.from_calibration(_calibration(), theta_max=2.2)

    rays = camera.pixel_rays_vehicle([[math.nan, math.nan], [math.inf, math.inf]])
    projected = camera.project_vehicle([[math.nan, math.nan, math.nan]])
    centers = camera.project_vehicle([camera.camera_to_vehicle.translation])

    assert not rays.valid.any()
    assert np.isnan(rays.origins).all()
    assert np.isnan(rays.directions).all()
    assert not projected.valid.any()
    assert np.isnan(projected.pixels).all()
    assert not centers.valid.any()


def test_projection_avoids_rotation_overflow_and_preserves_subnormal_displacements() -> None:
    calibration = _calibration(quaternion=(0.0, 0.0, 0.0, 1.0), translation=(0.0, 0.0, 0.0))
    camera = CalibratedCamera.from_calibration(calibration, theta_max=2.2)
    points = [[1.7e308, 1.7e308, 1.7e308], [0.0, 0.0, np.nextafter(0.0, 1.0)]]

    projected = camera.project_vehicle(points)
    expected = camera.model.project([[1, 1, 1], [0, 0, 1]])

    assert projected.valid.all()
    np.testing.assert_allclose(projected.pixels, expected.pixels, atol=1e-12)


def test_projection_handles_overflowing_displacement_without_losing_its_direction() -> None:
    calibration = _calibration(quaternion=(0.0, 0.0, 0.0, 1.0), translation=(-1.7e308, 0.0, 0.0))
    camera = CalibratedCamera.from_calibration(calibration, theta_max=2.2)

    projected = camera.project_vehicle([[1.7e308, 0.0, 0.0], [-1.7e308, 0.0, 0.0]])
    expected = camera.model.project([1, 0, 0])

    np.testing.assert_array_equal(projected.valid, [True, False])
    np.testing.assert_allclose(projected.pixels[0], expected.pixels, atol=1e-12)
    assert np.isnan(projected.pixels[1]).all()
