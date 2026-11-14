"""Domain models, damage classifications, and certificate schemas for vehicle inspection."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DamageSeverity(StrEnum):
    """Severity tier for vehicle exterior and structural surface damage."""

    NEGLIGIBLE = "NEGLIGIBLE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


class DamageType(StrEnum):
    """Classification of surface and structural anomalies detected on the vehicle."""

    SCRATCH = "SCRATCH"
    DENT = "DENT"
    PAINT_CHIP = "PAINT_CHIP"
    GLASS_CRACK = "GLASS_CRACK"
    PANEL_MISALIGNMENT = "PANEL_MISALIGNMENT"
    SOILING_BLOCKAGE = "SOILING_BLOCKAGE"


class InspectionPhase(StrEnum):
    """Operational lifecycle phase during autonomous valet parking handover."""

    PRE_DROPOFF = "PRE_DROPOFF"
    POST_DROPOFF = "POST_DROPOFF"
    PERIODIC_AUDIT = "PERIODIC_AUDIT"
    PRE_PICKUP = "PRE_PICKUP"
    POST_PICKUP = "POST_PICKUP"


class TirePosition(StrEnum):
    """Mounting location of the tire on the vehicle chassis."""

    FRONT_LEFT = "FRONT_LEFT"
    FRONT_RIGHT = "FRONT_RIGHT"
    REAR_LEFT = "REAR_LEFT"
    REAR_RIGHT = "REAR_RIGHT"


class TireWearStatus(StrEnum):
    """Wear condition and safety rating of a vehicle tire."""

    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    WARNING_LOW_TREAD = "WARNING_LOW_TREAD"
    CRITICAL_REPLACE = "CRITICAL_REPLACE"
    PRESSURE_FAULT = "PRESSURE_FAULT"


class LiabilityVerdict(StrEnum):
    """Legal liability verdict comparing pre-dropoff versus post-pickup inspection states."""

    NO_NEW_DAMAGE = "NO_NEW_DAMAGE"
    PRE_EXISTING_DOCUMENTED = "PRE_EXISTING_DOCUMENTED"
    NEW_DAMAGE_DETECTED = "NEW_DAMAGE_DETECTED"
    DISPUTE_REQUIRES_REVIEW = "DISPUTE_REQUIRES_REVIEW"


class OpticalAnomaly(BaseModel):
    """Localized surface or optical anomaly identified by 360-degree surround cameras."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anomaly_id: str
    camera_id: str
    surface_location: str
    damage_type: DamageType
    severity: DamageSeverity
    bounding_box_norm: tuple[float, float, float, float]
    estimated_area_cm2: float = Field(ge=0.0, allow_inf_nan=False)
    depth_estimate_mm: float = Field(ge=0.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    description: str


class TireTelemetry(BaseModel):
    """Comprehensive tread depth profile and TPMS cold pressure telemetry for a single wheel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: TirePosition
    inner_tread_depth_mm: float = Field(ge=0.0, le=15.0, allow_inf_nan=False)
    center_tread_depth_mm: float = Field(ge=0.0, le=15.0, allow_inf_nan=False)
    outer_tread_depth_mm: float = Field(ge=0.0, le=15.0, allow_inf_nan=False)
    mean_tread_depth_mm: float = Field(ge=0.0, le=15.0, allow_inf_nan=False)
    pressure_bar: float = Field(ge=0.0, le=6.0, allow_inf_nan=False)
    temperature_c: float = Field(allow_inf_nan=False)
    status: TireWearStatus
    uneven_wear_detected: bool


class GlassInspectionReport(BaseModel):
    """ECE R43 automotive glass integrity, transparency, and crack assessment report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    windshield_clear: bool = True
    cracks_detected: list[OpticalAnomaly] = Field(default_factory=list)
    chip_count: int = Field(default=0, ge=0)
    optical_transmittance_pct: float = Field(default=85.0, ge=0.0, le=100.0, allow_inf_nan=False)
    ece_r43_compliant: bool = True


class VehicleConditionSnapshot(BaseModel):
    """Complete multi-sensor visual and physical health snapshot of a vehicle at an epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vin: str
    inspection_phase: InspectionPhase
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    mileage_km: float = Field(ge=0.0, allow_inf_nan=False)
    anomalies: list[OpticalAnomaly] = Field(default_factory=list)
    tires: list[TireTelemetry] = Field(default_factory=list)
    glass: GlassInspectionReport = Field(default_factory=GlassInspectionReport)
    overall_condition_score: float = Field(default=100.0, ge=0.0, le=100.0, allow_inf_nan=False)


class LiabilityCertificate(BaseModel):
    """Cryptographically signed liability certificate binding vehicle condition to an epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    certificate_id: str
    vin: str
    session_id: str
    phase: InspectionPhase
    payload_sha256: str
    signature_hmac: str
    issuer_id: str
    operator_id: str
    verdict: LiabilityVerdict
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class InspectionReport(BaseModel):
    """Top-level telemetry and diagnostic container for vehicle condition inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str
    facility_id: str
    bay_id: str
    phase: InspectionPhase
    snapshot: VehicleConditionSnapshot
    certificate: LiabilityCertificate | None = None
    historical_comparison: dict[str, Any] | None = None
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert inspection report to JSON-serializable dictionary."""
        return self.model_dump(mode="json")
