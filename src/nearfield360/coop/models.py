"""Pydantic domain models and ETSI CPM protocol schemas for cooperative perception."""

from __future__ import annotations

from enum import StrEnum


class RPUSensorType(StrEnum):
    """Sensing modality deployed at a roadside perception unit station."""

    LIDAR = "lidar"
    FISHEYE_CAMERA = "fisheye_camera"
    RADAR = "radar"
    FUSED_STATION = "fused_station"


class CPMStationType(StrEnum):
    """ETSI ITS station type transmitting or receiving perception messages."""

    INFRASTRUCTURE_RPU = "infrastructure_rpu"
    VEHICLE_ONBOARD = "vehicle_onboard"
    SPECIAL_EQUIPMENT = "special_equipment"


class CoopTrackSource(StrEnum):
    """Origin of a perceptual track within the facility fusion supervisor."""

    ONBOARD_ONLY = "onboard_only"
    INFRASTRUCTURE_ONLY = "infrastructure_only"
    FUSED_COOPERATIVE = "fused_cooperative"


class BlindSpotRiskLevel(StrEnum):
    """Categorized collision risk for an occluded hazard approaching conflict."""

    SAFE = "safe"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL_BRAKE = "critical_brake"


class PerceivedObjectClass(StrEnum):
    """Classification of an object perceived by infrastructure or onboard sensors."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    SHOPPING_CART = "shopping_cart"
    STATIC_OBSTACLE = "static_obstacle"
    UNKNOWN = "unknown"


class OcclusionState(StrEnum):
    """Line-of-sight visibility of an object from a given observer vantage."""

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    UNKNOWN = "unknown"


class WarningAction(StrEnum):
    """Action recommended to automated vehicle executive upon blind-spot hazard detection."""

    MAINTAIN_SPEED = "maintain_speed"
    DECELERATE = "decelerate"
    STOP_AND_YIELD = "stop_and_yield"
    EMERGENCY_BRAKE = "emergency_brake"
