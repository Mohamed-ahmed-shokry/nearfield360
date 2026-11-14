"""SAE J2954 Wireless Power Transfer (WPT) and autonomous inductive docking subsystem."""

from nearfield360.wpt.beacon import MagneticBeaconEstimator
from nearfield360.wpt.controller import WPTInchingController
from nearfield360.wpt.efficiency import WPTEfficiencyModel
from nearfield360.wpt.fod import FODDetector
from nearfield360.wpt.models import (
    CoilGeometry,
    CoilOffset3D,
    FODStatus,
    WPTAlignmentStatus,
    WPTDockingStep,
    WPTEfficiencyReport,
    WPTInterlockState,
    WPTPowerClass,
    WPTReport,
    WPTZClass,
)
from nearfield360.wpt.viz import render_wpt_dashboard

__all__ = [
    "CoilGeometry",
    "CoilOffset3D",
    "FODDetector",
    "FODStatus",
    "MagneticBeaconEstimator",
    "WPTAlignmentStatus",
    "WPTDockingStep",
    "WPTEfficiencyModel",
    "WPTEfficiencyReport",
    "WPTInchingController",
    "WPTInterlockState",
    "WPTPowerClass",
    "WPTReport",
    "WPTZClass",
    "render_wpt_dashboard",
]
