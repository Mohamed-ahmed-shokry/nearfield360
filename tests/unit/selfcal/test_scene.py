"""Tests for the synthetic fisheye camera, parking-aisle map, and pixel detections."""

from __future__ import annotations

import numpy as np
import pytest

from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.association import associate_pixel_detections
from nearfield360.selfcal.models import RotationDelta
from nearfield360.selfcal.scene import (
    demo_fisheye_camera,
    demo_parking_aisle_map,
    demo_parking_aisle_polylines,
    synthesize_pixel_detections,
)


def test_demo_scene_geometry_is_well_formed() -> None:
    camera = demo_fisheye_camera(CameraId.REAR)
    pmap = demo_parking_aisle_map()

    assert camera.name is CameraId.REAR
    assert camera.model.intrinsics.width == 1280
    assert camera.model.intrinsics.height == 966
    assert pmap.num_segments == 9


def test_noiseless_detections_round_trip_to_map_under_true_delta() -> None:
    camera = demo_fisheye_camera()
    polylines = demo_parking_aisle_polylines()
    pmap = demo_parking_aisle_map()
    truth = RotationDelta.from_degrees(roll=0.6, pitch=-0.4, yaw=0.5)

    pixels = synthesize_pixel_detections(
        camera, polylines, truth, points_per_segment=10, pixel_noise_px=0.0, seed=3
    )

    assert len(pixels) == 90
    matched = associate_pixel_detections(camera, pixels, pmap, current_delta=truth, gate_m=1e-3)
    assert len(matched) == len(pixels)


def test_clutter_fraction_appends_extra_pixels_and_is_seeded() -> None:
    camera = demo_fisheye_camera()
    polylines = demo_parking_aisle_polylines()
    truth = RotationDelta.from_degrees(pitch=0.3)

    clean = synthesize_pixel_detections(
        camera, polylines, truth, points_per_segment=10, clutter_fraction=0.0, seed=1
    )
    cluttered_a = synthesize_pixel_detections(
        camera, polylines, truth, points_per_segment=10, clutter_fraction=0.2, seed=1
    )
    cluttered_b = synthesize_pixel_detections(
        camera, polylines, truth, points_per_segment=10, clutter_fraction=0.2, seed=1
    )

    assert len(cluttered_a) > len(clean)
    np.testing.assert_array_equal(cluttered_a, cluttered_b)


def test_synthesize_pixel_detections_validates_inputs() -> None:
    camera = demo_fisheye_camera()
    polylines = demo_parking_aisle_polylines()

    assert synthesize_pixel_detections(camera, [], RotationDelta()).shape == (0, 2)
    with pytest.raises(ValueError, match="points_per_segment"):
        synthesize_pixel_detections(camera, polylines, RotationDelta(), points_per_segment=0)
    with pytest.raises(ValueError, match="pixel_noise_px"):
        synthesize_pixel_detections(camera, polylines, RotationDelta(), pixel_noise_px=-1.0)
    with pytest.raises(ValueError, match="clutter_fraction"):
        synthesize_pixel_detections(camera, polylines, RotationDelta(), clutter_fraction=1.0)
