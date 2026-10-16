"""Unit tests for FacilityEvacuationPlanner and prioritized egress routing."""

from __future__ import annotations

import pytest

from nearfield360.config import EmergencyConfig
from nearfield360.emergency.containment import HazardContainmentEngine
from nearfield360.emergency.evacuation import FacilityEvacuationPlanner
from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    HazardZone,
)


def test_plan_evacuation_route_clear() -> None:
    config = EmergencyConfig(evacuation_speed_limit_mps=4.0)
    engine = HazardContainmentEngine(config)
    planner = FacilityEvacuationPlanner(engine, config)

    route = planner.plan_evacuation_route(
        vehicle_id="v-101",
        origin_bay_id="bay-01",
        start_pose=(10.0, 10.0),
        exit_portal_id="exit-north",
        exit_pose=(10.0, 50.0),
    )

    assert route.vehicle_id == "v-101"
    assert route.total_length_m == pytest.approx(40.0)
    assert route.estimated_transit_time_s == pytest.approx(10.0)
    assert route.is_cleared is True
    assert len(route.waypoints) >= 2


def test_plan_evacuation_route_blocked_by_hazard() -> None:
    config = EmergencyConfig(evacuation_speed_limit_mps=4.0)
    engine = HazardContainmentEngine(config)

    # Place hazard directly in middle of path from (10, 10) to (10, 50) at (10, 30)
    hazard = HazardZone(
        hazard_id="hz-block",
        emergency_type=EmergencyType.FIRE_ALARM,
        severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
        center_x=10.0,
        center_y=30.0,
        radius_m=8.0,
        floor_id="L1",
        active=True,
    )
    engine.add_hazard(hazard)
    planner = FacilityEvacuationPlanner(engine, config)

    route = planner.plan_evacuation_route(
        vehicle_id="v-blocked",
        origin_bay_id="bay-02",
        start_pose=(10.0, 10.0),
        exit_portal_id="exit-north",
        exit_pose=(10.0, 50.0),
    )

    assert route.is_cleared is False
    assert route.min_hazard_distance_m <= 8.0


def test_select_best_exit_prefers_uncompromised_portal() -> None:
    engine = HazardContainmentEngine()
    # Place hazard at (0, 100)
    engine.add_hazard(
        HazardZone(
            hazard_id="hz-exit",
            emergency_type=EmergencyType.GAS_LEAK,
            severity=EmergencySeverity.PARTIAL_EVACUATION,
            center_x=0.0,
            center_y=100.0,
            radius_m=10.0,
            floor_id="L1",
            active=True,
        )
    )
    planner = FacilityEvacuationPlanner(engine)

    portals = {
        "portal-north": (0.0, 100.0),  # Closer to (0, 70), but inside hazard!
        "portal-south": (0.0, 0.0),  # Farther, but safe
    }

    best_portal, best_pose = planner.select_best_exit((0.0, 70.0), portals)
    assert best_portal == "portal-south"
    assert best_pose == (0.0, 0.0)


def test_schedule_evacuation_waves_staggering_and_urgency() -> None:
    config = EmergencyConfig(stagger_interval_s=4.0)
    engine = HazardContainmentEngine(config)

    # Hazard at (0, 0)
    engine.add_hazard(
        HazardZone(
            hazard_id="fire-1",
            emergency_type=EmergencyType.FIRE_ALARM,
            severity=EmergencySeverity.FULL_FACILITY_EVACUATION,
            center_x=0.0,
            center_y=0.0,
            radius_m=10.0,
            floor_id="L1",
            active=True,
        )
    )
    planner = FacilityEvacuationPlanner(engine, config)

    # Vehicles at distances 15m, 80m, 30m, 120m
    vehicles = {
        "v-far-2": ("bay-4", (120.0, 0.0)),
        "v-close-1": ("bay-1", (15.0, 0.0)),
        "v-close-2": ("bay-2", (30.0, 0.0)),
        "v-far-1": ("bay-3", (80.0, 0.0)),
    }
    portals = {"exit-main": (0.0, 150.0)}

    waves = planner.schedule_evacuation_waves(vehicles, portals, batch_size=2)
    assert len(waves) == 2

    # Wave 0 (t=0s) should contain closest vehicles
    assert waves[0].wave_index == 0
    assert waves[0].dispatch_time_s == 0.0
    assert "v-close-1" in waves[0].vehicle_ids
    assert "v-close-2" in waves[0].vehicle_ids

    # Wave 1 (t=4s) should contain farther vehicles
    assert waves[1].wave_index == 1
    assert waves[1].dispatch_time_s == 4.0
    assert "v-far-1" in waves[1].vehicle_ids
    assert "v-far-2" in waves[1].vehicle_ids


def test_plan_facility_evacuation_end_to_end() -> None:
    planner = FacilityEvacuationPlanner()
    vehicles = {
        "v1": ("b1", (10.0, 20.0)),
        "v2": ("b2", (10.0, 40.0)),
    }
    portals = {"portal-exit": (0.0, 0.0)}

    waves, routes = planner.plan_facility_evacuation(vehicles, portals)
    assert len(waves) == 1
    assert len(routes) == 2
    assert "v1" in routes
    assert "v2" in routes
    assert routes["v1"].is_cleared is True
