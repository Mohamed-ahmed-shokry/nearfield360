"""Domain data models and schemas for vehicle CAN bus and HIL test harness (ISO 11898)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class CANFrameType(StrEnum):
    """CAN frame architecture variants conforming to ISO 11898-1."""

    STANDARD = "standard"  # 11-bit identifier, max 8 data bytes (CAN 2.0A)
    EXTENDED = "extended"  # 29-bit identifier, max 8 data bytes (CAN 2.0B)
    CAN_FD_STANDARD = "can_fd_standard"  # 11-bit identifier, up to 64 data bytes
    CAN_FD_EXTENDED = "can_fd_extended"  # 29-bit identifier, up to 64 data bytes


class CANFaultType(StrEnum):
    """Simulated hardware bus and ECU faults injected into CAN transmissions."""

    NONE = "none"
    CORRUPTED_CRC = "corrupted_crc"
    FRAME_DROP = "frame_drop"
    TIMING_JITTER = "timing_jitter"
    BIT_FLIP = "bit_flip"
    DLC_MISMATCH = "dlc_mismatch"


class CANFrame(BaseModel):
    """Raw CAN data link layer frame representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arbitration_id: int = Field(..., ge=0, le=0x1FFFFFFF)
    is_extended_id: bool = False
    is_fd: bool = False
    payload: bytes = Field(default=b"", max_length=64)
    dlc: int = Field(..., ge=0, le=64)
    timestamp_us: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    crc: int = Field(default=0, ge=0, le=0xFFFF)
    fault_type: CANFaultType = CANFaultType.NONE


class VehicleControlSignal(BaseModel):
    """Drive-by-wire vehicle control actuation command transmitted to ECU."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steering_angle_rad: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    throttle_pct: float = Field(default=0.0, ge=0.0, le=100.0, allow_inf_nan=False)
    brake_pct: float = Field(default=0.0, ge=0.0, le=100.0, allow_inf_nan=False)
    gear: Annotated[str, Field(pattern=r"^[PRND]$")] = "P"
    speed_mps: float = Field(default=0.0, ge=-10.0, le=50.0, allow_inf_nan=False)
    heartbeat_counter: int = Field(default=0, ge=0, le=15)


class VehicleFeedbackSignal(BaseModel):
    """Low-level chassis and sensor feedback received from vehicle gateway."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wheel_speed_fl_mps: float = Field(default=0.0, ge=0.0, le=60.0, allow_inf_nan=False)
    wheel_speed_fr_mps: float = Field(default=0.0, ge=0.0, le=60.0, allow_inf_nan=False)
    wheel_speed_rl_mps: float = Field(default=0.0, ge=0.0, le=60.0, allow_inf_nan=False)
    wheel_speed_rr_mps: float = Field(default=0.0, ge=0.0, le=60.0, allow_inf_nan=False)
    yaw_rate_rads: float = Field(default=0.0, ge=-2.0, le=2.0, allow_inf_nan=False)
    longitudinal_accel_mps2: float = Field(default=0.0, ge=-15.0, le=15.0, allow_inf_nan=False)
    ultrasound_distances_m: list[float] = Field(default_factory=list)
    heartbeat_counter: int = Field(default=0, ge=0, le=15)


class CANBusTelemetry(BaseModel):
    """Aggregated physical bus performance metrics and error rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frames_sent: int = Field(default=0, ge=0)
    frames_received: int = Field(default=0, ge=0)
    dropped_frames: int = Field(default=0, ge=0)
    corrupted_frames: int = Field(default=0, ge=0)
    bus_load_percent: float = Field(default=0.0, ge=0.0, le=100.0, allow_inf_nan=False)
    mean_jitter_us: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    peak_jitter_us: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    is_overloaded: bool = False


class HILSimulationReport(BaseModel):
    """Hardware-in-the-Loop simulation execution summary and fault log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_s: float = Field(..., gt=0.0, allow_inf_nan=False)
    target_ecus: list[str] = Field(default_factory=list)
    telemetry: CANBusTelemetry
    fault_injection_log: list[dict[str, str | float | int]] = Field(default_factory=list)
    ecu_health_status: dict[str, str] = Field(default_factory=dict)
