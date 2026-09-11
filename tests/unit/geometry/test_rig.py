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
from nearfield360.geometry.rig import SurroundRig
from nearfield360.geometry.transforms import RigidTransform


def _intrinsics() -> IntrinsicParameters:
    return IntrinsicParameters(
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
    )


def _calibration(
    name: CameraId,
    translation: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CameraCalibration:
    return CameraCalibration(
        name=name,
        extrinsic=ExtrinsicParameters(quaternion=(0.0, 0.0, 0.0, 1.0), translation=translation),
        intrinsic=_intrinsics(),
    )


def _rig() -> SurroundRig:
    return SurroundRig.from_calibrations(
        (
            _calibration(CameraId.FRONT, (3.0, 0.0, 1.0)),
            _calibration(CameraId.REAR, (-3.0, 0.0, 1.0)),
        ),
        theta_max=2.2,
    )


def test_from_calibrations_preserves_identities_and_order() -> None:
    rig = _rig()

    assert len(rig) == 2
    assert rig.names == (CameraId.FRONT, CameraId.REAR)
    assert [camera.name for camera in rig] == [CameraId.FRONT, CameraId.REAR]
    np.testing.assert_array_equal(
        rig.get(CameraId.FRONT).camera_to_vehicle.translation, (3.0, 0.0, 1.0)
    )


def test_from_calibrations_supports_mapping_and_per_camera_limits() -> None:
    rig = SurroundRig.from_calibrations(
        {
            CameraId.FRONT: _calibration(CameraId.FRONT),
            CameraId.REAR: _calibration(CameraId.REAR),
        },
        theta_max={CameraId.FRONT: 2.0, CameraId.REAR: 2.2},
    )

    assert rig.get(CameraId.FRONT).model.theta_max == 2.0
    assert rig.get(CameraId.REAR).model.theta_max == 2.2


def test_constructor_rejects_empty_duplicate_and_mistyped_cameras() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SurroundRig([])
    camera = CalibratedCamera.from_calibration(_calibration(CameraId.FRONT), theta_max=2.0)
    with pytest.raises(ValueError, match="unique names"):
        SurroundRig([camera, camera])
    with pytest.raises(ValueError, match="CalibratedCamera"):
        SurroundRig(["camera"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        SurroundRig.from_calibrations([], theta_max=2.0)
    with pytest.raises(ValueError, match="theta_max"):
        SurroundRig.from_calibrations(
            [_calibration(CameraId.FRONT)],
            theta_max={CameraId.REAR: 2.0},
        )


def test_get_reports_unknown_camera() -> None:
    rig = _rig()

    with pytest.raises(KeyError, match="Unknown rig camera"):
        rig.get(CameraId.MIRROR_LEFT)


def test_project_vehicle_covers_every_camera() -> None:
    rig = _rig()
    results = rig.project_vehicle([10.0, 0.0, 0.0])

    assert set(results) == {CameraId.FRONT, CameraId.REAR}
    assert bool(results[CameraId.FRONT].valid)


def test_pixel_batches_must_cover_every_camera_exactly_once() -> None:
    rig = _rig()
    principal = rig.get(CameraId.FRONT).model.intrinsics.principal_point

    with pytest.raises(ValueError, match="every rig camera"):
        rig.pixel_rays_vehicle({CameraId.FRONT: [principal]})
    with pytest.raises(ValueError, match="every rig camera"):
        rig.pixel_rays_vehicle(
            {
                CameraId.FRONT: [principal],
                CameraId.REAR: [principal],
                CameraId.MIRROR_LEFT: [principal],
            }
        )


def test_ground_footprints_combine_ray_and_plane_validity() -> None:
    rig = _rig()
    front_principal = np.array(rig.get(CameraId.FRONT).model.intrinsics.principal_point)
    rear_principal = np.array(rig.get(CameraId.REAR).model.intrinsics.principal_point)
    # Identity rotation maps the principal ray to vehicle +Z (up), which never
    # reaches the ground. A 200-pixel offset selects theta=2.0 rad, whose ray
    # points partly down (cos(2.0) < 0) and must intersect z=0.
    footprints = rig.ground_footprints(
        {
            CameraId.FRONT: np.array([front_principal, [math.nan, 0.0]]),
            CameraId.REAR: np.array([rear_principal, rear_principal + np.array([200.0, 0.0])]),
        }
    )

    front = footprints[CameraId.FRONT]
    assert front.valid.tolist() == [False, False]
    assert np.isnan(front.points).all()
    assert np.isnan(front.distances).all()

    rear = footprints[CameraId.REAR]
    assert rear.valid.tolist() == [False, True]
    assert np.isnan(rear.points[0]).all()
    assert math.isnan(float(rear.distances[0]))
    assert math.isclose(float(rear.points[1, 2]), 0.0, abs_tol=1e-12)
    expected_distance = 1.0 / -math.cos(2.0)
    assert float(rear.distances[1]) == pytest.approx(expected_distance)


def test_downward_camera_produces_footprint_below_its_center() -> None:
    # Camera X right / Y down / Z forward mapped so +Z camera points to -Z
    # vehicle (straight down) from one metre above the ground.
    down = CalibratedCamera(
        name=CameraId.FRONT,
        model=rig_down_model(),
        camera_to_vehicle=RigidTransform(
            np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
            (2.0, -1.0, 1.0),
        ),
    )
    rig = SurroundRig([down])
    principal = np.array(down.model.intrinsics.principal_point)

    footprints = rig.ground_footprints({CameraId.FRONT: principal}, max_distance=20.0)
    footprint = footprints[CameraId.FRONT]

    assert bool(footprint.valid)
    np.testing.assert_allclose(footprint.points, [2.0, -1.0, 0.0], atol=1e-12)
    assert float(footprint.distances) == pytest.approx(1.0)


def rig_down_model():  # type: ignore[no-untyped-def]
    from nearfield360.geometry.fisheye import RadialPolynomialFisheye

    return RadialPolynomialFisheye(_intrinsics(), theta_max=2.5)


def test_max_distance_filters_far_footprints() -> None:
    rig = _rig()
    rear_principal = np.array(rig.get(CameraId.REAR).model.intrinsics.principal_point)
    pixels = {
        CameraId.FRONT: np.array(rig.get(CameraId.FRONT).model.intrinsics.principal_point),
        CameraId.REAR: rear_principal + np.array([200.0, 0.0]),
    }

    near = rig.ground_footprints(pixels, max_distance=100.0)
    far = rig.ground_footprints(pixels, max_distance=0.5)

    assert bool(near[CameraId.REAR].valid)
    assert not bool(far[CameraId.REAR].valid)


def test_origins_array_returns_detached_metric_centers() -> None:
    rig = _rig()
    origins = rig.origins_array()

    np.testing.assert_array_equal(origins[CameraId.FRONT], (3.0, 0.0, 1.0))
    np.testing.assert_array_equal(origins[CameraId.REAR], (-3.0, 0.0, 1.0))
    origins[CameraId.FRONT][0] = 99.0
    np.testing.assert_array_equal(
        rig.get(CameraId.FRONT).camera_to_vehicle.translation, (3.0, 0.0, 1.0)
    )
