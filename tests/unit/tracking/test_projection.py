"""Unit tests for ground ray projection engine."""

from __future__ import annotations

from nearfield360.data.calibration import (
    CameraCalibration,
    ExtrinsicParameters,
    IntrinsicParameters,
)
from nearfield360.data.detection import DetectionAnnotation
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.tracking.projection import (
    DEFAULT_CLASS_DIMENSIONS,
    project_detection_to_ground,
    project_detections,
)


def _create_front_camera() -> CalibratedCamera:
    calib = CameraCalibration(
        name=CameraId.FRONT,
        extrinsic=ExtrinsicParameters(
            quaternion=(-0.5, 0.5, -0.5, 0.5),
            translation=(3.75, 0.0, 0.66),
        ),
        intrinsic=IntrinsicParameters(
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
    return CalibratedCamera.from_calibration(calib, theta_max=2.2)


def test_project_detection_to_ground_valid_vehicle() -> None:
    cam = _create_front_camera()
    # A box in the lower half of the image (pointing towards ground ahead of front camera)
    # Principal point is (320, 240); y > 240 is down in camera coords, which is towards the ground
    box = (250.0, 280.0, 390.0, 380.0)

    footprint = project_detection_to_ground(
        box,
        class_id=0,
        class_name="vehicles",
        camera=cam,
        camera_name="FV",
        ground_z=0.0,
        max_distance=20.0,
    )

    assert footprint is not None
    assert footprint.class_id == 0
    assert footprint.class_name == "vehicles"
    assert footprint.camera == "FV"
    # Front camera is mounted at x=3.75; ground point must be ahead (x > 3.75)
    assert footprint.center_x > 3.75
    # Center y should be close to 0 (lateral center)
    assert abs(footprint.center_y) < 1.0
    # Check default dimensions for vehicles
    assert (footprint.width, footprint.length) == DEFAULT_CLASS_DIMENSIONS[0]


def test_project_detection_custom_dimensions() -> None:
    cam = _create_front_camera()
    box = (250.0, 280.0, 390.0, 380.0)

    footprint = project_detection_to_ground(
        box,
        class_id=1,
        class_name="person",
        camera=cam,
        custom_dimensions=(0.8, 0.8),
    )

    assert footprint is not None
    assert footprint.width == 0.8
    assert footprint.length == 0.8


def test_project_detection_points_above_ground() -> None:
    cam = _create_front_camera()
    # Upper half of image (y < 240) points up into the sky, never intersects ground
    sky_box = (100.0, 50.0, 200.0, 100.0)

    footprint = project_detection_to_ground(
        sky_box,
        class_id=0,
        class_name="vehicles",
        camera=cam,
        ground_z=0.0,
    )

    assert footprint is None


def test_project_detections_batch() -> None:
    cam = _create_front_camera()
    dets = [
        DetectionAnnotation(
            class_id=0,
            class_name="vehicles",
            x_min=280.0,
            y_min=270.0,
            x_max=360.0,
            y_max=350.0,
        ),
        DetectionAnnotation(
            class_id=1,
            class_name="person",
            x_min=100.0,
            y_min=50.0,
            x_max=200.0,
            y_max=100.0,
        ),  # Above horizon
    ]

    footprints = project_detections(dets, camera=cam, camera_name="FV")
    assert len(footprints) == 1
    assert footprints[0].class_name == "vehicles"
