"""Unit tests for V2X communications and fleet domain data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.fleet.models import (
    ConflictPoint,
    ConflictResolutionAction,
    DispatchAssignment,
    FacilityOccupancySummary,
    FleetDispatchReport,
    SpatiotemporalWaypoint,
    TrajectoryProfile,
    V2XMessage,
    V2XMessageType,
    VehicleFleetStatus,
    VehicleHeartbeat,
)
from nearfield360.mapping.models import PoseEstimate


def test_v2x_message_envelope_creation() -> None:
    msg = V2XMessage(
        message_type=V2XMessageType.HEARTBEAT,
        sender_id="veh_01",
        timestamp_s=10.5,
        payload={"speed": 1.2},
    )
    assert msg.protocol_version == "1.0"
    assert msg.message_type == V2XMessageType.HEARTBEAT
    assert msg.sender_id == "veh_01"
    assert msg.recipient_id == "cloud_dispatcher"
    assert msg.payload["speed"] == 1.2


def test_vehicle_heartbeat_validation() -> None:
    pose = PoseEstimate(
        x=10.0, y=5.0, heading_rad=1.57, covariance=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.005]]
    )
    hb = VehicleHeartbeat(
        vehicle_id="veh_alpha",
        timestamp_s=1.0,
        pose=pose,
        speed_mps=1.8,
        steering_angle_rad=0.05,
        status=VehicleFleetStatus.EN_ROUTE_SLOT,
        target_slot_id="SLOT_01",
        battery_pct=95.5,
    )
    assert hb.vehicle_id == "veh_alpha"
    assert hb.status == VehicleFleetStatus.EN_ROUTE_SLOT
    assert hb.battery_pct == 95.5

    with pytest.raises(ValidationError):
        VehicleHeartbeat(
            vehicle_id="veh_bad",
            timestamp_s=1.0,
            pose=pose,
            speed_mps=-0.5,  # must be ge 0.0
        )


def test_spatiotemporal_waypoint_time_window_validation() -> None:
    wp = SpatiotemporalWaypoint(
        x=5.0,
        y=10.0,
        heading_rad=0.0,
        t_expected_s=5.0,
        t_min_s=4.5,
        t_max_s=5.5,
        speed_target_mps=1.5,
    )
    assert wp.t_expected_s == 5.0

    with pytest.raises(ValidationError, match="cannot be earlier"):
        SpatiotemporalWaypoint(
            x=5.0,
            y=10.0,
            heading_rad=0.0,
            t_expected_s=5.0,
            t_min_s=5.0,
            t_max_s=4.0,  # t_max < t_min
        )


def test_trajectory_profile_and_conflict_point() -> None:
    wp1 = SpatiotemporalWaypoint(x=0.0, y=0.0, t_expected_s=0.0, t_min_s=0.0, t_max_s=0.5)
    wp2 = SpatiotemporalWaypoint(x=5.0, y=0.0, t_expected_s=5.0, t_min_s=4.5, t_max_s=5.5)
    profile = TrajectoryProfile(
        vehicle_id="v1",
        waypoints=[wp1, wp2],
        total_duration_s=5.0,
    )
    assert len(profile.waypoints) == 2

    conflict = ConflictPoint(
        conflict_id="conf_01",
        primary_vehicle_id="v1",
        secondary_vehicle_id="v2",
        conflict_location=(5.0, 5.0),
        earliest_arrival_s=4.0,
        latest_departure_s=6.0,
        distance_between_trajectories_m=0.8,
        action_primary=ConflictResolutionAction.PROCEED,
        action_secondary=ConflictResolutionAction.YIELD_HOLD,
    )
    assert conflict.conflict_id == "conf_01"
    assert conflict.action_secondary == ConflictResolutionAction.YIELD_HOLD
    assert not conflict.resolved


def test_fleet_dispatch_report_serialization() -> None:
    occ = FacilityOccupancySummary(
        timestamp_s=12.0,
        total_slots=20,
        vacant_slots=15,
        reserved_slots=3,
        occupied_slots=2,
        blocked_slots=0,
        slot_allocations={"SLOT_01": "v1"},
    )
    assignment = DispatchAssignment(
        assignment_id="asg_01",
        vehicle_id="v1",
        assigned_slot_id="SLOT_01",
        route_waypoint_ids=["W1", "W2"],
        issued_at_s=10.0,
        lease_expires_at_s=70.0,
    )
    report = FleetDispatchReport(
        facility_id="facility_garage_alpha",
        timestamp_s=12.0,
        total_vehicles_active=1,
        occupancy_summary=occ,
        active_assignments=[assignment],
        active_conflicts=[],
        resolved_conflicts_count=2,
        average_wait_time_s=1.5,
    )

    json_str = report.to_json()
    assert "facility_garage_alpha" in json_str

    deserialized = FleetDispatchReport.from_json(json_str)
    assert deserialized.facility_id == report.facility_id
    assert deserialized.occupancy_summary.vacant_slots == 15
    assert len(deserialized.active_assignments) == 1
    assert deserialized.active_assignments[0].assigned_slot_id == "SLOT_01"
