"""Automated vehicle condition inspection, 360-degree damage assessment, and liability audit."""

from nearfield360.inspection.certificate import LiabilityCertificateManager
from nearfield360.inspection.glass import GlassIntegrityAnalyzer
from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    GlassInspectionReport,
    InspectionPhase,
    InspectionReport,
    LiabilityCertificate,
    LiabilityVerdict,
    OpticalAnomaly,
    TirePosition,
    TireTelemetry,
    TireWearStatus,
    VehicleConditionSnapshot,
)
from nearfield360.inspection.optical import OpticalDamageDetector
from nearfield360.inspection.tires import TireProfiler
from nearfield360.inspection.viz import render_inspection_dashboard

__all__ = [
    "DamageSeverity",
    "DamageType",
    "GlassInspectionReport",
    "GlassIntegrityAnalyzer",
    "InspectionPhase",
    "InspectionReport",
    "LiabilityCertificate",
    "LiabilityCertificateManager",
    "LiabilityVerdict",
    "OpticalAnomaly",
    "OpticalDamageDetector",
    "TirePosition",
    "TireProfiler",
    "TireTelemetry",
    "TireWearStatus",
    "VehicleConditionSnapshot",
    "render_inspection_dashboard",
]
