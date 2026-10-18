"""Unit tests for Roadside Perception Unit (RPU) manager and CPM dispatcher."""

from __future__ import annotations

import pytest

from nearfield360.coop.models import (
    CPMPerceivedObject,
    CPMStationType,
    PerceivedObjectClass,
    RPUSensorState,
    RPUSensorType,
)
from nearfield360.coop.rpu import RPUSensorManager


def test_rpu_registration_and_lookup() -> None:
    mgr = RPUSensorManager()
    station = RPUSensorState(
        station_id="RPU-TEST",
        sensor_type=RPUSensorType.LIDAR,
        position_m=(10.0, 10.0, 3.0),
        yaw_deg=90.0,
        range_max_m=20.0,
        fov_azimuth_deg=90.0,
    )
    mgr.register_station(station)

    retrieved = mgr.get_station("RPU-TEST")
    assert retrieved == station
    assert len(mgr.list_stations()) == 1

    with pytest.raises(KeyError, match="not registered"):
        mgr.get_station("RPU-NONEXISTENT")

    assert mgr.remove_station("RPU-TEST") is True
    assert mgr.remove_station("RPU-TEST") is False
    assert len(mgr.list_stations()) == 0


def test_rpu_fov_containment() -> None:
    mgr = RPUSensorManager()
    # Station at (0, 0) facing +X (yaw = 0 deg), range = 20 m, FOV = 90 deg (+/- 45 deg)
    station = RPUSensorState(
        station_id="RPU-ORIGIN",
        sensor_type=RPUSensorType.LIDAR,
        position_m=(0.0, 0.0, 3.0),
        yaw_deg=0.0,
        range_max_m=20.0,
        fov_azimuth_deg=90.0,
        is_operational=True,
    )
    mgr.register_station(station)

    # Point directly in front at 10m
    assert mgr.is_point_in_fov("RPU-ORIGIN", (10.0, 0.0)) is True

    # Point at angle +30 deg, distance 10m
    assert mgr.is_point_in_fov("RPU-ORIGIN", (8.66, 5.0)) is True

    # Point at angle +60 deg (outside 45 deg half-FOV)
    assert mgr.is_point_in_fov("RPU-ORIGIN", (5.0, 8.66)) is False

    # Point behind sensor
    assert mgr.is_point_in_fov("RPU-ORIGIN", (-5.0, 0.0)) is False

    # Point directly in front but beyond range (25m)
    assert mgr.is_point_in_fov("RPU-ORIGIN", (25.0, 0.0)) is False

    # Disabled sensor
    mgr.register_station(
        RPUSensorState(
            station_id="RPU-OFFLINE",
            sensor_type=RPUSensorType.LIDAR,
            position_m=(0.0, 0.0, 3.0),
            yaw_deg=0.0,
            range_max_m=20.0,
            fov_azimuth_deg=90.0,
            is_operational=False,
        )
    )
    assert mgr.is_point_in_fov("RPU-OFFLINE", (10.0, 0.0)) is False


def test_rpu_fov_polygon_geometry() -> None:
    mgr = RPUSensorManager()
    station = RPUSensorState(
        station_id="RPU-POLYGON",
        sensor_type=RPUSensorType.RADAR,
        position_m=(5.0, 5.0, 3.0),
        yaw_deg=0.0,
        range_max_m=10.0,
        fov_azimuth_deg=60.0,
    )
    mgr.register_station(station)

    poly = mgr.get_fov_polygon("RPU-POLYGON", num_arc_points=8)
    assert len(poly) == 10  # origin + 8 arc points + 1
    assert poly[0] == (5.0, 5.0)


def test_rpu_perceptions_and_cpm_generation() -> None:
    mgr = RPUSensorManager()
    station = RPUSensorState(
        station_id="RPU-CPM",
        sensor_type=RPUSensorType.FISHEYE_CAMERA,
        position_m=(12.0, 8.0, 4.0),
        yaw_deg=180.0,
        range_max_m=30.0,
        fov_azimuth_deg=120.0,
    )
    mgr.register_station(station)

    obj = CPMPerceivedObject(
        object_id="CAR-1",
        station_id="RPU-CPM",
        timestamp_s=50.0,
        category=PerceivedObjectClass.VEHICLE,
        x_m=5.0,
        y_m=8.0,
        vx_mps=-2.0,
    )
    mgr.add_perceived_object("RPU-CPM", obj)
    assert len(mgr.get_perceived_objects("RPU-CPM")) == 1

    # Mismatched station should raise ValueError
    wrong_obj = CPMPerceivedObject(
        object_id="CAR-2",
        station_id="RPU-OTHER",
        timestamp_s=50.0,
        x_m=0.0,
        y_m=0.0,
    )
    with pytest.raises(ValueError, match="does not match target station"):
        mgr.add_perceived_object("RPU-CPM", wrong_obj)

    # Generate first CPM
    cpm1 = mgr.generate_cpm("RPU-CPM", timestamp_s=50.1)
    assert cpm1.station_id == "RPU-CPM"
    assert cpm1.sequence_number == 0
    assert cpm1.station_type == CPMStationType.INFRASTRUCTURE_RPU
    assert len(cpm1.perceived_objects) == 1

    # Second CPM increments sequence number
    cpm2 = mgr.generate_cpm("RPU-CPM", timestamp_s=50.2)
    assert cpm2.sequence_number == 1

    # Clear perceptions
    mgr.clear_perceptions("RPU-CPM")
    assert len(mgr.get_perceived_objects("RPU-CPM")) == 0


def test_rpu_cpm_fov_filtering() -> None:
    mgr = RPUSensorManager()
    station = RPUSensorState(
        station_id="RPU-FILTER",
        sensor_type=RPUSensorType.LIDAR,
        position_m=(0.0, 0.0, 3.0),
        yaw_deg=0.0,
        range_max_m=20.0,
        fov_azimuth_deg=90.0,
    )
    mgr.register_station(station)

    obj_inside = CPMPerceivedObject(
        object_id="IN",
        station_id="RPU-FILTER",
        timestamp_s=10.0,
        x_m=10.0,
        y_m=0.0,
    )
    obj_outside = CPMPerceivedObject(
        object_id="OUT",
        station_id="RPU-FILTER",
        timestamp_s=10.0,
        x_m=-10.0,
        y_m=0.0,
    )
    mgr.add_perceived_object("RPU-FILTER", obj_inside)
    mgr.add_perceived_object("RPU-FILTER", obj_outside)

    cpm_unfiltered = mgr.generate_cpm("RPU-FILTER", timestamp_s=10.0, filter_fov=False)
    assert len(cpm_unfiltered.perceived_objects) == 2

    cpm_filtered = mgr.generate_cpm("RPU-FILTER", timestamp_s=10.0, filter_fov=True)
    assert len(cpm_filtered.perceived_objects) == 1
    assert cpm_filtered.perceived_objects[0].object_id == "IN"


def test_default_demonstrator_network() -> None:
    network = RPUSensorManager.create_default_demonstrator_network()
    stations = network.list_stations()
    assert len(stations) == 3
    station_ids = [s.station_id for s in stations]
    assert "RPU-BLIND-CORNER-1" in station_ids
    assert "RPU-RAMP-CREST-1" in station_ids
    assert "RPU-GATE-OVERHEAD-1" in station_ids
