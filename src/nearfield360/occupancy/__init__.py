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
    risk_report,
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
    "rasterize_occupancy",
    "risk_report",
    "validate_zone_mask",
]
