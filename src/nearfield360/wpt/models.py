"""Domain models and schemas for SAE J2954 Wireless Power Transfer (WPT)."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WPTPowerClass(StrEnum):
    """SAE J2954 standard inductive power classes for light-duty electric vehicles."""

    WPT1 = "WPT1"  # 3.7 kW
    WPT2 = "WPT2"  # 7.7 kW
    WPT3 = "WPT3"  # 11.0 kW
    WPT4 = "WPT4"  # 22.0 kW


class WPTZClass(StrEnum):
    """SAE J2954 vertical ground-to-vehicle air gap classification."""

    Z1 = "Z1"  # 100 mm - 150 mm
    Z2 = "Z2"  # 140 mm - 210 mm
    Z3 = "Z3"  # 170 mm - 250 mm


class WPTAlignmentStatus(StrEnum):
    """Operational alignment status between primary (GA) and secondary (VA) coils."""

    OUT_OF_RANGE = "OUT_OF_RANGE"
    APPROACHING = "APPROACHING"
    ALIGNED = "ALIGNED"
    MISALIGNED = "MISALIGNED"
    DOCKED = "DOCKED"


class WPTInterlockState(StrEnum):
    """Safety interlock states for magnetic field energization and power transfer."""

    NORMAL = "NORMAL"
    FOD_FAULT = "FOD_FAULT"
    LOD_FAULT = "LOD_FAULT"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


class CoilGeometry(BaseModel):
    """Geometric and electrical specifications of an inductive resonant coil assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    length_mm: float = Field(default=650.0, gt=100.0, le=1500.0, allow_inf_nan=False)
    width_mm: float = Field(default=500.0, gt=100.0, le=1500.0, allow_inf_nan=False)
    num_turns: int = Field(default=12, ge=4, le=50)
    self_inductance_uh: float = Field(default=78.5, gt=10.0, le=500.0, allow_inf_nan=False)


class CoilOffset3D(BaseModel):
    """Relative spatial displacement and orientation between ground pad and vehicle pad."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_x_mm: float = Field(allow_inf_nan=False)
    delta_y_mm: float = Field(allow_inf_nan=False)
    delta_z_mm: float = Field(gt=0.0, allow_inf_nan=False)
    delta_theta_deg: float = Field(allow_inf_nan=False)
    radial_offset_mm: float = Field(ge=0.0, allow_inf_nan=False)
    alignment_status: WPTAlignmentStatus
    is_within_tolerance: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    timestamp_ns: int = Field(default=0, ge=0)


class FODStatus(BaseModel):
    """Foreign Object Debris (FOD) and Living Object Detection (LOD) status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metal_object_detected: bool
    frequency_shift_khz: float = Field(allow_inf_nan=False)
    q_drop_percent: float = Field(allow_inf_nan=False)
    living_object_detected: bool
    interlock_state: WPTInterlockState
    power_inhibited: bool
    reason: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class WPTEfficiencyReport(BaseModel):
    """Electrical resonance coupling, transfer efficiency, and power governor report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coupling_factor_k: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    transfer_efficiency: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    commanded_power_kw: float = Field(ge=0.0, allow_inf_nan=False)
    delivered_power_kw: float = Field(ge=0.0, allow_inf_nan=False)
    thermal_loss_kw: float = Field(ge=0.0, allow_inf_nan=False)
    power_class: WPTPowerClass = WPTPowerClass.WPT3
    z_class: WPTZClass = WPTZClass.Z2


class WPTDockingStep(BaseModel):
    """Single discrete step in an autonomous micro-docking inching trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    vehicle_x_m: float = Field(allow_inf_nan=False)
    vehicle_y_m: float = Field(allow_inf_nan=False)
    heading_deg: float = Field(allow_inf_nan=False)
    velocity_mps: float = Field(ge=0.0, allow_inf_nan=False)
    steering_angle_deg: float = Field(allow_inf_nan=False)
    coil_offset: CoilOffset3D
    docked: bool


class WPTReport(BaseModel):
    """Comprehensive diagnostic and operational telemetry report for WPT subsystem."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str
    pad_id: str
    power_class: WPTPowerClass
    z_class: WPTZClass
    offset: CoilOffset3D
    fod: FODStatus
    efficiency: WPTEfficiencyReport
    docking_history: list[WPTDockingStep] = Field(default_factory=list)
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert model to JSON-serializable dictionary."""
        return self.model_dump(mode="json")
