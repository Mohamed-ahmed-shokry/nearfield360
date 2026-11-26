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
from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    MapTile,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
)
from nearfield360.selfcal.association import PolylineMap, associate_pixel_detections
from nearfield360.selfcal.models import RotationDelta


def _front_camera() -> CalibratedCamera:
    calibration = CameraCalibration(
        name=CameraId.FRONT,
        extrinsic=ExtrinsicParameters(
            quaternion=(-0.5, 0.5, -0.5, 0.5),
            translation=(3.6, 0.0, 0.8),
        ),
        intrinsic=IntrinsicParameters(
            aspect_ratio=1.0,
            cx_offset=0.0,
            cy_offset=0.0,
            height=480,
            width=640,
            k1=180.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            model="radial_poly",
            poly_order=4,
        ),
    )
    return CalibratedCamera.from_calibration(calibration, theta_max=2.2)


def test_nearest_returns_exact_points_distances_and_normals() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (10.0, 0.0)], [(5.0, -5.0), (5.0, 5.0)]])

    result = polyline_map.nearest([[2.0, 1.0], [5.5, 3.0], [20.0, 0.0]])

    np.testing.assert_allclose(result.points, [[2.0, 0.0], [5.0, 3.0], [10.0, 0.0]])
    np.testing.assert_allclose(result.distances, [1.0, 0.5, 10.0])
    np.testing.assert_allclose(
        np.abs(result.normals), [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(result.normals, axis=1), 1.0)


def test_nearest_clamps_to_segment_endpoints() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (1.0, 0.0)]])

    result = polyline_map.nearest([[-3.0, 4.0]])

    np.testing.assert_allclose(result.points, [[0.0, 0.0]])
    np.testing.assert_allclose(result.distances, [5.0])


def test_multi_vertex_polylines_and_degenerate_segments() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]])

    assert polyline_map.num_segments == 2
    np.testing.assert_allclose(polyline_map.nearest([[4.5, 1.0]]).points, [[4.0, 1.0]])


def test_map_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        PolylineMap([[(0.0, 0.0)]])
    with pytest.raises(ValueError, match="non-finite"):
        PolylineMap([[(0.0, 0.0), (float("nan"), 1.0)]])
    with pytest.raises(ValueError, match="non-degenerate"):
        PolylineMap([[(1.0, 1.0), (1.0, 1.0)]])
    with pytest.raises(ValueError, match="non-degenerate"):
        PolylineMap([])
    with pytest.raises(ValueError, match="shape"):
        PolylineMap([[(0.0, 0.0), (1.0, 0.0)]]).nearest([1.0, 2.0])


def test_from_hdmap_extracts_lanelet_and_parking_boundaries() -> None:
    lanelet = Lanelet(
        lanelet_id="L1",
        left_boundary=Polyline2D(points=[Point2D(x=4.0, y=2.0), Point2D(x=12.0, y=2.0)]),
        right_boundary=Polyline2D(points=[Point2D(x=4.0, y=-2.0), Point2D(x=12.0, y=-2.0)]),
        centerline=Polyline2D(points=[Point2D(x=4.0, y=0.0), Point2D(x=12.0, y=0.0)]),
    )
    slot = ParkingSpaceFeature(
        slot_id="S1",
        boundary=Polyline2D(
            points=[
                Point2D(x=6.0, y=2.0),
                Point2D(x=8.5, y=2.0),
                Point2D(x=8.5, y=4.5),
                Point2D(x=6.0, y=4.5),
            ],
            is_closed=True,
        ),
        center=Point2D(x=7.25, y=3.25),
    )
    hd_map = HDVectorMap(
        map_id="M1",
        tiles=[
            MapTile(
                tile_id="T0",
                x_min=0.0,
                y_min=-10.0,
                x_max=20.0,
                y_max=10.0,
                lanelets=[lanelet],
                parking_spaces=[slot],
            )
        ],
    )

    pmap = PolylineMap.from_hdmap(hd_map)

    assert pmap.num_segments == 6
    assert float(pmap.nearest([[7.0, 2.0]]).distances[0]) == pytest.approx(0.0, abs=1e-12)


def test_associate_pixel_detections_emits_unit_normals_and_gates_outliers() -> None:
    camera = _front_camera()
    pmap = PolylineMap([[(4.0, 2.0), (12.0, 2.0)], [(7.0, -3.0), (7.0, 3.0)]])
    ground_pts = np.array(
        [
            [6.0, 2.0, 0.0],  # on line y=2
            [7.0, -1.0, 0.0],  # on line x=7
            [9.0, 0.0, 0.0],  # 2.0 m from both lines -> outside gate
        ]
    )
    proj = camera.project_vehicle(ground_pts)
    assert bool(np.all(proj.valid))

    observations = associate_pixel_detections(camera, proj.pixels, pmap, gate_m=0.5)

    assert len(observations) == 2
    for obs in observations:
        assert obs.camera_id is CameraId.FRONT
        assert obs.reference_normal is not None
        assert math.hypot(*obs.reference_normal) == pytest.approx(1.0, abs=1e-12)
    assert observations[0].reference_y_m == pytest.approx(2.0, abs=1e-4)
    assert observations[1].reference_x_m == pytest.approx(7.0, abs=1e-4)


def test_associate_handles_empty_invalid_and_bad_inputs() -> None:
    camera = _front_camera()
    pmap = PolylineMap([[(4.0, 2.0), (12.0, 2.0)]])

    assert associate_pixel_detections(camera, np.empty((0, 2)), pmap) == []
    # Sky pixel (above optical center) should miss ground plane
    assert associate_pixel_detections(camera, [[320.0, 10.0]], pmap) == []
    # Out-of-bounds pixel should be filtered
    assert associate_pixel_detections(camera, [[-50.0, -50.0]], pmap) == []

    with pytest.raises(ValueError, match="gate_m"):
        associate_pixel_detections(camera, [[320.0, 300.0]], pmap, gate_m=0.0)
    with pytest.raises(ValueError, match="shape"):
        associate_pixel_detections(camera, [320.0, 300.0], pmap)


def test_associate_applies_current_delta_hypothesis() -> None:
    camera = _front_camera()
    pmap = PolylineMap([[(4.0, 0.0), (12.0, 0.0)]])
    delta = RotationDelta.from_degrees(yaw=3.0)
    # Ray that after +3 deg yaw hits (8.0, 0.0, 0.0)
    true_ray = np.array([8.0, 0.0, 0.0]) - camera.camera_to_vehicle.translation
    nominal_ray = delta.matrix().T @ true_ray
    pixel = camera.project_vehicle(camera.camera_to_vehicle.translation + nominal_ray).pixels

    # Without delta it's ~0.23m off, so a tight 0.05m gate rejects it
    assert associate_pixel_detections(camera, [pixel], pmap, gate_m=0.05) == []
    # With delta it lands on the line
    matched = associate_pixel_detections(camera, [pixel], pmap, current_delta=delta, gate_m=0.05)
    assert len(matched) == 1
    assert matched[0].reference_y_m == pytest.approx(0.0, abs=1e-4)
