"""Emergency Vehicle Preemption (EVP), Facility Fire Evacuation, and Life Safety."""

from __future__ import annotations

from nearfield360.emergency.containment import HazardContainmentEngine
from nearfield360.emergency.evacuation import FacilityEvacuationPlanner
from nearfield360.emergency.models import (
    EmergencySeverity,
    EmergencyType,
    EmergencyVehicle,
    EvacuationRoute,
    EvacuationWave,
    EvacuationWaypoint,
    FireDoorInterlock,
    FireDoorStatus,
    HazardZone,
    LifeSafetyReport,
    ResponderType,
    VehicleEvacuationStatus,
)
from nearfield360.emergency.preemption import (
    EmergencyPreemptionArbiter,
    point_to_segment_distance,
)
from nearfield360.emergency.viz import render_emergency_dashboard

__all__ = [
    "EmergencyPreemptionArbiter",
    "EmergencySeverity",
    "EmergencyType",
    "EmergencyVehicle",
    "EvacuationRoute",
    "EvacuationWave",
    "EvacuationWaypoint",
    "FacilityEvacuationPlanner",
    "FireDoorInterlock",
    "FireDoorStatus",
    "HazardContainmentEngine",
    "HazardZone",
    "LifeSafetyReport",
    "ResponderType",
    "VehicleEvacuationStatus",
    "point_to_segment_distance",
    "render_emergency_dashboard",
]
