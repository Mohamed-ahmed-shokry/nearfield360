"""Tests for the coarse-to-fine point-to-line ICP image-landmark calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from nearfield360.config import SelfCalConfig
from nearfield360.selfcal.association import PolylineMap
from nearfield360.selfcal.icp import ImageCalibrationResult, ImageLandmarkCalibrator
from nearfield360.selfcal.models import RotationDelta
from nearfield360.selfcal.scene import (
    demo_fisheye_camera,
    demo_parking_aisle_map,
    demo_parking_aisle_polylines,
    synthesize_pixel_detections,
)

TRUE_DELTA = RotationDelta.from_degrees(roll=0.5, pitch=-0.3, yaw=0.4)


def _max_axis_error_deg(estimate: RotationDelta, truth: RotationDelta) -> float:
    return float(np.degrees(np.max(np.abs(estimate.as_vector() - truth.as_vector()))))


def test_noiseless_icp_recovers_rotation_accurately() -> None:
    camera = demo_fisheye_camera()
    pixels = synthesize_pixel_detections(
        camera,
        demo_parking_aisle_polylines(),
        TRUE_DELTA,
        points_per_segment=20,
        pixel_noise_px=0.0,
        clutter_fraction=0.0,
        seed=0,
    )

    result = ImageLandmarkCalibrator().calibrate(camera, pixels, demo_parking_aisle_map())

    assert result.converged and result.estimate.observable
    assert _max_axis_error_deg(result.estimate.delta, TRUE_DELTA) < 0.01
    assert len(result.rounds) == 8
    assert result.rounds[0].gate_m > result.rounds[-1].gate_m
    assert ImageCalibrationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("seed", range(5))
def test_icp_recovers_half_degree_misalignment_with_noise_and_clutter(seed: int) -> None:
    camera = demo_fisheye_camera()
    pixels = synthesize_pixel_detections(
        camera,
        demo_parking_aisle_polylines(),
        TRUE_DELTA,
        points_per_segment=40,
        pixel_noise_px=1.0,
        clutter_fraction=0.2,
        seed=seed,
    )

    result = ImageLandmarkCalibrator().calibrate(camera, pixels, demo_parking_aisle_map())

    assert result.converged and result.estimate.observable
    assert _max_axis_error_deg(result.estimate.delta, TRUE_DELTA) < 0.05


def test_single_collinear_line_scene_is_flagged_unobservable() -> None:
    camera = demo_fisheye_camera()
    single_line = [[(4.5, 0.0), (12.5, 0.0)]]
    pixels = synthesize_pixel_detections(
        camera, single_line, TRUE_DELTA, points_per_segment=30, pixel_noise_px=0.5, seed=0
    )

    result = ImageLandmarkCalibrator().calibrate(camera, pixels, PolylineMap(single_line))

    assert not result.converged
    assert not result.estimate.observable
    assert result.estimate.reason != ""


def test_insufficient_detections_stops_on_first_round() -> None:
    camera = demo_fisheye_camera()

    result = ImageLandmarkCalibrator().calibrate(camera, np.empty((0, 2)), demo_parking_aisle_map())

    assert not result.converged
    assert not result.estimate.observable
    assert len(result.rounds) == 1
    assert "detections associated" in result.estimate.reason


@pytest.mark.parametrize("rounds", [1, 2])
def test_short_gate_schedules_run_cleanly(rounds: int) -> None:
    camera = demo_fisheye_camera()
    cfg = SelfCalConfig(max_association_rounds=rounds)
    pixels = synthesize_pixel_detections(
        camera,
        demo_parking_aisle_polylines(),
        RotationDelta.from_degrees(pitch=0.2),
        points_per_segment=20,
        pixel_noise_px=0.0,
        seed=1,
    )

    result = ImageLandmarkCalibrator(cfg).calibrate(camera, pixels, demo_parking_aisle_map())

    assert len(result.rounds) == rounds
    assert result.estimate.observable
