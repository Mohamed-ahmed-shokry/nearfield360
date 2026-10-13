"""Domain data models for multi-tenant facility partitioning and VIP express corridors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TenantTier(StrEnum):
    """Classification of tenant operational profile and priority class."""

    PUBLIC_TRANSIENT = "public_transient"
    COMMERCIAL_FLEET = "commercial_fleet"
    CAR_SHARING = "car_sharing"
    VIP_EXECUTIVE = "vip_executive"


class AccessPermission(StrEnum):
    """Access clearance level for a facility zone or corridor."""

    STANDARD_ACCESS = "standard_access"
    PRIORITY_ACCESS = "priority_access"
    RESTRICTED_ACCESS = "restricted_access"
    EXCLUSIVE_VIP = "exclusive_vip"


class TenantPartition(BaseModel):
    """Designated spatial zone and bay quota allocation for a specific operator tenant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    tenant_name: str = Field(min_length=1)
    tier: TenantTier = TenantTier.COMMERCIAL_FLEET
    assigned_slot_ids: list[str] = Field(default_factory=list)
    dedicated_bay_quota: int = Field(ge=0)
    current_occupancy: int = Field(default=0, ge=0)
    active_vehicle_ids: list[str] = Field(default_factory=list)
    access_permission: AccessPermission = AccessPermission.PRIORITY_ACCESS

    @property
    def utilization_ratio(self) -> float:
        """Compute occupancy ratio against assigned quota."""
        if self.dedicated_bay_quota <= 0:
            return 0.0
        return round(self.current_occupancy / self.dedicated_bay_quota, 3)

    @property
    def is_quota_exceeded(self) -> bool:
        """Check whether partition current occupancy exceeds dedicated quota."""
        return self.current_occupancy > self.dedicated_bay_quota


class TenantSLAContract(BaseModel):
    """Service Level Agreement contract terms and performance guarantees for a tenant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    max_dispatch_latency_s: float = Field(default=180.0, gt=0.0, allow_inf_nan=False)
    target_availability_ratio: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    breach_penalty_per_min: float = Field(default=2.50, ge=0.0, allow_inf_nan=False)
    recorded_dispatches_count: int = Field(default=0, ge=0)
    breach_count: int = Field(default=0, ge=0)
    total_penalty_incurred: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @property
    def compliance_rate(self) -> float:
        """Compute fraction of dispatches that met SLA latency."""
        if self.recorded_dispatches_count <= 0:
            return 1.0
        successful = max(0, self.recorded_dispatches_count - self.breach_count)
        return round(successful / self.recorded_dispatches_count, 3)


class VIPExpressCorridor(BaseModel):
    """Reserved express travel corridor through driving lanes for priority vehicle preemption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corridor_id: str = Field(min_length=1)
    lane_ids: list[str] = Field(min_length=1)
    active_vehicle_id: str | None = None
    reservation_start_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    duration_s: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    is_preempting: bool = False
    priority_level: int = Field(default=1, ge=1, le=10)

    def is_active_at(self, current_time_s: float) -> bool:
        """Check if corridor reservation is active at a given timestamp."""
        if self.active_vehicle_id is None:
            return False
        return (
            self.reservation_start_s <= current_time_s < self.reservation_start_s + self.duration_s
        )


class TenantPartitionReport(BaseModel):
    """Comprehensive multi-tenant facility partitioning and SLA compliance audit report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    timestamp_s: float = Field(ge=0.0, allow_inf_nan=False)
    total_slots_count: int = Field(ge=0)
    total_partitions_count: int = Field(ge=0)
    partitions: list[TenantPartition] = Field(default_factory=list)
    sla_contracts: list[TenantSLAContract] = Field(default_factory=list)
    express_corridors: list[VIPExpressCorridor] = Field(default_factory=list)
    overall_sla_compliance: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    total_breach_penalties: float = Field(ge=0.0, allow_inf_nan=False)

    @field_validator("overall_sla_compliance", "total_breach_penalties")
    @classmethod
    def round_kpis(cls, value: float) -> float:
        return round(value, 2)
