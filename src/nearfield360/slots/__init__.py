"""3D metric parking slot detection, occupancy classification, and corridor feasibility."""

from __future__ import annotations

from nearfield360.slots.classifier import (
    SlotOccupancyClassifier,
    rasterize_slot_mask,
)
from nearfield360.slots.corridor import (
    ApproachCorridorEvaluator,
    compute_approach_path,
    point_to_segment_distance,
)
from nearfield360.slots.detector import (
    ParkingSlotDetector,
    classify_slot_type,
    filter_and_suppress_slots,
    order_slot_corners,
    polygon_iou,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotApproachPath,
    SlotDetectionReport,
    SlotDetectionSummary,
    SlotOccupancyStatus,
)
from nearfield360.slots.viz import render_slots_bev_overlay

__all__ = [
    "ApproachCorridorEvaluator",
    "ParkingSlot",
    "ParkingSlotCorner",
    "ParkingSlotDetector",
    "ParkingSlotType",
    "SlotApproachPath",
    "SlotDetectionReport",
    "SlotDetectionSummary",
    "SlotOccupancyClassifier",
    "SlotOccupancyStatus",
    "classify_slot_type",
    "compute_approach_path",
    "filter_and_suppress_slots",
    "order_slot_corners",
    "point_to_segment_distance",
    "polygon_iou",
    "rasterize_slot_mask",
    "render_slots_bev_overlay",
]
