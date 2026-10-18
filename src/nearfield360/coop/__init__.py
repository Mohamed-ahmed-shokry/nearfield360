"""Infrastructure Cooperative Perception, ETSI CPM & Blind-Spot Occlusion Fusion."""

from __future__ import annotations

from nearfield360.coop.alignment import Covariance2D, Pose2D, SpatiotemporalAligner
from nearfield360.coop.blind_spot import (
    DEFAULT_FACILITY_PILLARS,
    BlindSpotAssessor,
    compute_pillar_shadow_polygon,
    is_occluded_from_observer,
    segment_intersects_circle,
)
from nearfield360.coop.fusion import (
    CooperativeTrackManager,
    MatchResult,
    associate_gated_greedy,
    compute_mahalanobis_distance,
    fuse_covariance_intersection,
    fuse_velocity_ci,
)
from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    BlindSpotWarning,
    CollectivePerceptionMessage,
    CooperativeTrack,
    CoopTelemetryReport,
    CoopTrackSource,
    CPMPerceivedObject,
    CPMStationType,
    PerceivedObjectClass,
    RPUSensorState,
    RPUSensorType,
    WarningAction,
)
from nearfield360.coop.rpu import RPUSensorManager
from nearfield360.coop.viz import render_coop_dashboard

__all__ = [
    "DEFAULT_FACILITY_PILLARS",
    "BlindSpotAssessor",
    "BlindSpotRiskLevel",
    "BlindSpotWarning",
    "CPMPerceivedObject",
    "CPMStationType",
    "CollectivePerceptionMessage",
    "CoopTelemetryReport",
    "CoopTrackSource",
    "CooperativeTrack",
    "CooperativeTrackManager",
    "Covariance2D",
    "MatchResult",
    "PerceivedObjectClass",
    "Pose2D",
    "RPUSensorManager",
    "RPUSensorState",
    "RPUSensorType",
    "SpatiotemporalAligner",
    "WarningAction",
    "associate_gated_greedy",
    "compute_mahalanobis_distance",
    "compute_pillar_shadow_polygon",
    "fuse_covariance_intersection",
    "fuse_velocity_ci",
    "is_occluded_from_observer",
    "render_coop_dashboard",
    "segment_intersects_circle",
]
