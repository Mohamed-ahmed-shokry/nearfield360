"""HD vector map generation, spatial tiling, format export, and particle filter localization."""

from __future__ import annotations

from nearfield360.hdmap.formats import MapFormatExporter
from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    LaneletType,
    LocalizationConvergence,
    LocalizationState,
    MapTile,
    ObstacleZone,
    ParkingSlotType,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
    PoseHypothesis,
    TurnDirection,
)
from nearfield360.hdmap.particle_filter import FisheyeParticleFilter
from nearfield360.hdmap.tiler import MapTiler
from nearfield360.hdmap.vectorizer import BEVVectorizer, douglas_peucker
from nearfield360.hdmap.viz import render_hdmap_dashboard

__all__ = [
    "BEVVectorizer",
    "FisheyeParticleFilter",
    "HDVectorMap",
    "Lanelet",
    "LaneletType",
    "LocalizationConvergence",
    "LocalizationState",
    "MapFormatExporter",
    "MapTile",
    "MapTiler",
    "ObstacleZone",
    "ParkingSlotType",
    "ParkingSpaceFeature",
    "Point2D",
    "Polyline2D",
    "PoseHypothesis",
    "TurnDirection",
    "douglas_peucker",
    "render_hdmap_dashboard",
]
