"""Curbside drop-off/pick-up staging, dwell queue management, and pedestrian safety interlocks."""

from __future__ import annotations

from nearfield360.curbside.manager import CurbsideCapacityExceededError, CurbsideManager
from nearfield360.curbside.models import (
    BayStatus,
    CurbsideBay,
    CurbsideMission,
    CurbsideReport,
    HandoverType,
    PedestrianDetection,
    SafetyInterlockStatus,
)
from nearfield360.curbside.safety import InterlockViolationError, SafetyInterlockEngine
from nearfield360.curbside.viz import render_curbside_dashboard

__all__ = [
    "BayStatus",
    "CurbsideBay",
    "CurbsideCapacityExceededError",
    "CurbsideManager",
    "CurbsideMission",
    "CurbsideReport",
    "HandoverType",
    "InterlockViolationError",
    "PedestrianDetection",
    "SafetyInterlockEngine",
    "SafetyInterlockStatus",
    "render_curbside_dashboard",
]
