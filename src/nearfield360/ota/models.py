"""Domain data models and schemas for UN R156 SUMS and firmware verification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PartitionSlot(StrEnum):
    """Dual-slot A/B partition identifier."""

    SLOT_A = "slot_a"
    SLOT_B = "slot_b"

    def opposite(self) -> PartitionSlot:
        """Return the opposite inactive slot."""
        return PartitionSlot.SLOT_B if self == PartitionSlot.SLOT_A else PartitionSlot.SLOT_A


class UpdateState(StrEnum):
    """Lifecycle phase of an over-the-air firmware update campaign."""

    IDLE = "idle"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    STAGED = "staged"
    PREFLIGHT = "preflight"
    APPLYING = "applying"
    TESTING = "testing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class PreflightStatus(StrEnum):
    """Validation outcome of vehicle pre-update safety interlocks."""

    PASSED = "passed"
    FAILED = "failed"


class FirmwareManifest(BaseModel):
    """Cryptographically signed UN R156 firmware release package manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(..., min_length=2)
    version: str = Field(..., min_length=2)
    monotonic_revision: int = Field(..., ge=1)
    rxswin: str = Field(..., min_length=6)
    target_ecu: str = Field(..., min_length=2)
    payload_size_bytes: int = Field(..., gt=0)
    sha256_hash: str = Field(..., min_length=64, max_length=64)
    signature_hex: str = Field(..., min_length=32)
    created_at_iso: str = Field(default="2026-11-03T00:00:00Z")


class PreflightCheck(BaseModel):
    """Vehicle operational safety interlock check results prior to firmware staging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gear_is_park: bool = Field(...)
    speed_mps: float = Field(..., ge=0.0, allow_inf_nan=False)
    battery_soc_pct: float = Field(..., ge=0.0, le=100.0, allow_inf_nan=False)
    safety_supervisor_ready: bool = Field(...)
    status: PreflightStatus = Field(...)
    reasons: list[str] = Field(default_factory=list)


class SlotMetadata(BaseModel):
    """Status and firmware content installed in an A/B partition slot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: PartitionSlot = Field(...)
    is_active: bool = Field(default=False)
    is_bootable: bool = Field(default=True)
    version: str = Field(default="v1.0.0")
    monotonic_revision: int = Field(default=0, ge=0)
    rxswin: str = Field(default="RXSWIN-DEFAULT-0001")
    sha256_hash: str = Field(default="0" * 64, min_length=64, max_length=64)
    boot_count: int = Field(default=0, ge=0)


class ECUPartitionState(BaseModel):
    """Full dual-slot state of an electronic control unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ecu_id: str = Field(..., min_length=2)
    active_slot: PartitionSlot = Field(default=PartitionSlot.SLOT_A)
    next_boot_slot: PartitionSlot = Field(default=PartitionSlot.SLOT_A)
    slots: dict[str, SlotMetadata] = Field(default_factory=dict)
    rollback_count: int = Field(default=0, ge=0)


class OTACampaignStepRecord(BaseModel):
    """Audit log record of an atomic state step in the OTA campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_state: UpdateState = Field(...)
    to_state: UpdateState = Field(...)
    message: str = Field(...)
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)


class OTACampaignReport(BaseModel):
    """Consolidated UN R156 compliance report for a firmware update campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(..., min_length=2)
    target_ecu: str = Field(..., min_length=2)
    active_state: UpdateState = Field(default=UpdateState.IDLE)
    manifest: FirmwareManifest = Field(...)
    preflight: PreflightCheck = Field(...)
    partition_state: ECUPartitionState = Field(...)
    audit_trail: list[OTACampaignStepRecord] = Field(default_factory=list)
    success: bool = Field(default=False)
    duration_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
