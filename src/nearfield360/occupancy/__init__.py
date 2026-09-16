"""Camera-based occupancy evidence and spatial risk zones on the BEV grid."""

from nearfield360.occupancy.evidence import (
    NAME_BY_LABEL_ID,
    OccupancyEvidence,
    OccupancyPolicy,
    OccupancyPolicyError,
    distance_weights,
    fuse_occupancy,
    rasterize_occupancy,
)
from nearfield360.occupancy.risk import (
    RiskZone,
    ZoneRisk,
    circular_zone,
    corridor_zone,
    lateral_clearance_zone,
    rear_corridor_zone,
    risk_report,
    surround_parking_zones,
    validate_zone_mask,
)

__all__ = [
    "NAME_BY_LABEL_ID",
    "OccupancyEvidence",
    "OccupancyPolicy",
    "OccupancyPolicyError",
    "RiskZone",
    "ZoneRisk",
    "circular_zone",
    "corridor_zone",
    "distance_weights",
    "fuse_occupancy",
    "lateral_clearance_zone",
    "rasterize_occupancy",
    "rear_corridor_zone",
    "risk_report",
    "surround_parking_zones",
    "validate_zone_mask",
]
