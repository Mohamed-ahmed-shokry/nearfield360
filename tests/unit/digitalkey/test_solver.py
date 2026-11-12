"""Unit tests for 3D UWB multilateration solver and AoA spatial tracking."""

import math

from nearfield360.digitalkey.models import ProximityZone, UWBRangeMeasurement
from nearfield360.digitalkey.solver import MultilaterationSolver, get_default_uwb_anchors


def test_default_uwb_anchors() -> None:
    anchors = get_default_uwb_anchors()
    assert len(anchors) == 4
    anchor_ids = {a.anchor_id for a in anchors}
    assert anchor_ids == {"FL", "FR", "RL", "RR"}


def test_cabin_interior_bounds() -> None:
    solver = MultilaterationSolver()
    # Inside center of vehicle
    assert solver.is_inside_cabin(0.0, 0.0, 1.0) is True
    assert solver.is_inside_cabin(0.5, -0.4, 0.8) is True
    # Outside vehicle
    assert solver.is_inside_cabin(3.0, 0.0, 1.0) is False
    assert solver.is_inside_cabin(0.0, 2.0, 1.0) is False
    assert solver.is_inside_cabin(0.0, 0.0, 2.5) is False


def test_proximity_zone_classification() -> None:
    solver = MultilaterationSolver()
    assert solver.classify_zone(0.5, is_interior=True) == ProximityZone.INTERIOR
    assert solver.classify_zone(1.2, is_interior=False) == ProximityZone.TOUCH
    assert solver.classify_zone(2.5, is_interior=False) == ProximityZone.IMMEDIATE
    assert solver.classify_zone(5.0, is_interior=False) == ProximityZone.NEAR
    assert solver.classify_zone(12.0, is_interior=False) == ProximityZone.FAR
    assert solver.classify_zone(35.0, is_interior=False) == ProximityZone.OUT_OF_RANGE


def test_multilateration_solver_accuracy() -> None:
    solver = MultilaterationSolver()
    anchors = solver.anchors

    # Target test point outside front-left of vehicle
    tx, ty, tz = 2.5, 1.8, 1.2
    measurements: list[UWBRangeMeasurement] = []
    ts = 1700000000000000

    for a_id, a in anchors.items():
        dist = math.sqrt((tx - a.x_m) ** 2 + (ty - a.y_m) ** 2 + (tz - a.z_m) ** 2)
        measurements.append(
            UWBRangeMeasurement(
                anchor_id=a_id,
                distance_m=dist,
                timestamp_ns=ts,
                sts_valid=True,
                tof_ns=dist / 0.299792458,
            )
        )

    pos = solver.solve(measurements)

    assert abs(pos.x_m - tx) < 0.15
    assert abs(pos.y_m - ty) < 0.15
    assert pos.residual_error_m < 0.1
    assert pos.confidence >= 0.85
    assert pos.is_interior is False
    assert pos.zone in (ProximityZone.IMMEDIATE, ProximityZone.NEAR)


def test_multilateration_solver_interior_position() -> None:
    solver = MultilaterationSolver()
    anchors = solver.anchors

    # Target inside cabin (driver seat area)
    tx, ty, tz = 0.2, 0.35, 1.0
    measurements: list[UWBRangeMeasurement] = []
    ts = 1700000000000000

    for a_id, a in anchors.items():
        dist = math.sqrt((tx - a.x_m) ** 2 + (ty - a.y_m) ** 2 + (tz - a.z_m) ** 2)
        measurements.append(
            UWBRangeMeasurement(
                anchor_id=a_id,
                distance_m=dist,
                timestamp_ns=ts,
                sts_valid=True,
                tof_ns=dist / 0.299792458,
            )
        )

    pos = solver.solve(measurements)

    assert pos.is_interior is True
    assert pos.zone == ProximityZone.INTERIOR
    assert abs(pos.x_m - tx) < 0.15
    assert abs(pos.y_m - ty) < 0.15


def test_multilateration_fallback_insufficient_anchors() -> None:
    solver = MultilaterationSolver()
    # Only 1 anchor measurement
    single_meas = [
        UWBRangeMeasurement(
            anchor_id="FL",
            distance_m=4.0,
            timestamp_ns=1700000000000000,
            sts_valid=True,
            tof_ns=13.34,
        )
    ]
    pos = solver.solve(single_meas)
    assert pos.confidence < 0.5
    assert pos.residual_error_m >= 0.5


def test_aoa_computation() -> None:
    solver = MultilaterationSolver()
    # Create position directly to the left (+y) of FL anchor
    fl = solver.anchors["FL"]
    target = solver.solve(
        [
            UWBRangeMeasurement(
                anchor_id="FL",
                distance_m=2.0,
                timestamp_ns=100,
                sts_valid=True,
                tof_ns=6.67,
            )
        ]
    )
    # Mock position to exact offset: dx = 0.0, dy = 2.0, dz = 0.0
    from nearfield360.digitalkey.models import DevicePosition3D

    pos = DevicePosition3D(
        x_m=fl.x_m,
        y_m=fl.y_m + 2.0,
        z_m=fl.z_m,
        distance_m=2.0,
        zone=ProximityZone.IMMEDIATE,
        is_interior=False,
        residual_error_m=0.0,
        confidence=1.0,
        timestamp_ns=100,
    )
    azimuth, elevation = solver.compute_aoa(pos, anchor_id="FL")
    assert abs(azimuth - 90.0) < 1.0
    assert abs(elevation - 0.0) < 1.0

    # Non-existent anchor returns 0.0, 0.0
    az, el = solver.compute_aoa(target, anchor_id="UNKNOWN")
    assert az == 0.0 and el == 0.0
