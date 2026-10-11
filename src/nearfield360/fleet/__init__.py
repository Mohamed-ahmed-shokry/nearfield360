"""Cloud fleet coordination, V2X dispatch, and multi-agent conflict management."""

from __future__ import annotations

from nearfield360.fleet.coordinator import FleetCoordinator, SimulatedVehicle
from nearfield360.fleet.deconfliction import DeconflictionEngine, build_trajectory_profile
from nearfield360.fleet.dispatcher import SlotAllocationError, SlotDispatcher, SlotReservationLease
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
from nearfield360.fleet.viz import render_fleet_coordination_dashboard

__all__ = [
    "ConflictPoint",
    "ConflictResolutionAction",
    "DeconflictionEngine",
    "DispatchAssignment",
    "FacilityOccupancySummary",
    "FleetCoordinator",
    "FleetDispatchReport",
    "SimulatedVehicle",
    "SlotAllocationError",
    "SlotDispatcher",
    "SlotReservationLease",
    "SpatiotemporalWaypoint",
    "TrajectoryProfile",
    "V2XMessage",
    "V2XMessageType",
    "VehicleFleetStatus",
    "VehicleHeartbeat",
    "build_trajectory_profile",
    "render_fleet_coordination_dashboard",
]
