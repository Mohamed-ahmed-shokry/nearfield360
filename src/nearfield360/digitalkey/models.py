"""Domain models and schemas for CCC Digital Key 3.0, UWB ranging, and proximity interlock."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProximityZone(StrEnum):
    """Spatial proximity zones relative to the vehicle perimeter and interior."""

    OUT_OF_RANGE = "OUT_OF_RANGE"
    FAR = "FAR"
    NEAR = "NEAR"
    IMMEDIATE = "IMMEDIATE"
    TOUCH = "TOUCH"
    INTERIOR = "INTERIOR"


class HandoverState(StrEnum):
    """Lifecycle states for vehicle-to-device session delegation and human takeover."""

    DISCONNECTED = "DISCONNECTED"
    PAIRED = "PAIRED"
    AUTHENTICATED = "AUTHENTICATED"
    APPROACHING = "APPROACHING"
    AVP_DELEGATED = "AVP_DELEGATED"
    PICKUP_READY = "PICKUP_READY"
    TAKEOVER_IN_PROGRESS = "TAKEOVER_IN_PROGRESS"
    MANUAL_CONTROL = "MANUAL_CONTROL"


class DelegationScope(StrEnum):
    """Granular permission scopes granted by CCC Digital Key delegation tokens."""

    AVP_OPERATE = "AVP_OPERATE"
    CABIN_LOCK = "CABIN_LOCK"
    TRUNK_ACCESS = "TRUNK_ACCESS"
    IMMOBILIZER_OVERRIDE = "IMMOBILIZER_OVERRIDE"
    TELEOP_ASSIST = "TELEOP_ASSIST"


class UWBAnchor(BaseModel):
    """Configuration and 3D vehicle coordinate position of a physical UWB transceiver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_id: str = Field(min_length=1)
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(default=0.8, allow_inf_nan=False)
    channel: int = Field(default=5, ge=1, le=15)
    is_aoa_capable: bool = True


class UWBRangeMeasurement(BaseModel):
    """Physical layer IEEE 802.15.4z UWB STS two-way ranging packet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_id: str = Field(min_length=1)
    distance_m: float = Field(ge=0.0, allow_inf_nan=False)
    timestamp_ns: int = Field(ge=0)
    sts_valid: bool = True
    tof_ns: float = Field(ge=0.0, allow_inf_nan=False)
    snr_db: float = Field(default=24.0, allow_inf_nan=False)
    first_path_power_dbm: float = Field(default=-75.0, allow_inf_nan=False)
    aoa_azimuth_deg: float | None = Field(default=None, allow_inf_nan=False)
    aoa_elevation_deg: float | None = Field(default=None, allow_inf_nan=False)


class DevicePosition3D(BaseModel):
    """Resolved 3D spatial position of the mobile device relative to the vehicle center."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(default=1.0, allow_inf_nan=False)
    distance_m: float = Field(ge=0.0, allow_inf_nan=False)
    zone: ProximityZone
    is_interior: bool
    residual_error_m: float = Field(ge=0.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    timestamp_ns: int = Field(ge=0)


class CCCDelegationToken(BaseModel):
    """Cryptographic delegation token authorizing autonomous valet parking operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_id: str = Field(min_length=4)
    vin: str = Field(min_length=8)
    user_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    scopes: list[DelegationScope] = Field(min_length=1)
    issued_at_utc: str
    expires_at_utc: str
    nonce: str = Field(min_length=8)
    signature: str = Field(min_length=16)
    is_revoked: bool = False


class ProximityInterlockDecision(BaseModel):
    """Real-time safety interlock decision preventing collisions during proximity transit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interlock_active: bool
    standstill_command: bool
    detected_distance_m: float = Field(ge=0.0, allow_inf_nan=False)
    proximity_zone: ProximityZone
    handover_state: HandoverState
    reason: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DigitalKeyReport(BaseModel):
    """Comprehensive diagnostic and audit telemetry report for CCC Digital Key subsystem."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: str
    vin: str
    handover_state: HandoverState
    proximity_zone: ProximityZone
    device_position: DevicePosition3D
    interlock: ProximityInterlockDecision
    uwb_measurements: list[UWBRangeMeasurement]
    delegation_token: CCCDelegationToken | None = None
    relay_attack_detected: bool = False
    relay_discrepancy_ns: float = Field(default=0.0, allow_inf_nan=False)
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert model to JSON-serializable dictionary."""
        return self.model_dump(mode="json")
