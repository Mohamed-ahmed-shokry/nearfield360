"""Unit tests for tenant domain models, quotas, SLA calculations, and corridors."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.tenant.models import (
    AccessPermission,
    TenantPartition,
    TenantPartitionReport,
    TenantSLAContract,
    TenantTier,
    VIPExpressCorridor,
)


def test_tenant_tier_and_access_permission_enums() -> None:
    assert TenantTier.PUBLIC_TRANSIENT == "public_transient"
    assert TenantTier.COMMERCIAL_FLEET == "commercial_fleet"
    assert TenantTier.CAR_SHARING == "car_sharing"
    assert TenantTier.VIP_EXECUTIVE == "vip_executive"

    assert AccessPermission.STANDARD_ACCESS == "standard_access"
    assert AccessPermission.PRIORITY_ACCESS == "priority_access"
    assert AccessPermission.RESTRICTED_ACCESS == "restricted_access"
    assert AccessPermission.EXCLUSIVE_VIP == "exclusive_vip"


def test_tenant_partition_quota_and_utilization() -> None:
    partition = TenantPartition(
        tenant_id="TENANT-01",
        tenant_name="FleetCo",
        tier=TenantTier.COMMERCIAL_FLEET,
        assigned_slot_ids=["slot-1", "slot-2", "slot-3", "slot-4"],
        dedicated_bay_quota=4,
        current_occupancy=3,
        active_vehicle_ids=["veh-1", "veh-2", "veh-3"],
    )

    assert partition.utilization_ratio == 0.75
    assert partition.is_quota_exceeded is False

    over_partition = partition.model_copy(update={"current_occupancy": 5})
    assert over_partition.utilization_ratio == 1.25
    assert over_partition.is_quota_exceeded is True


def test_tenant_sla_contract_compliance() -> None:
    contract = TenantSLAContract(
        contract_id="SLA-01",
        tenant_id="TENANT-01",
        max_dispatch_latency_s=120.0,
        target_availability_ratio=0.95,
        breach_penalty_per_min=5.0,
        recorded_dispatches_count=10,
        breach_count=2,
        total_penalty_incurred=25.0,
    )

    assert contract.compliance_rate == 0.80

    zero_contract = TenantSLAContract(
        contract_id="SLA-02",
        tenant_id="TENANT-02",
        recorded_dispatches_count=0,
    )
    assert zero_contract.compliance_rate == 1.0


def test_vip_express_corridor_active_window() -> None:
    corridor = VIPExpressCorridor(
        corridor_id="CORRIDOR-VIP",
        lane_ids=["lane-1", "lane-2"],
        active_vehicle_id="VIP-007",
        reservation_start_s=100.0,
        duration_s=30.0,
        is_preempting=True,
    )

    assert corridor.is_active_at(99.0) is False
    assert corridor.is_active_at(100.0) is True
    assert corridor.is_active_at(129.9) is True
    assert corridor.is_active_at(130.0) is False

    inactive_corridor = corridor.model_copy(update={"active_vehicle_id": None})
    assert inactive_corridor.is_active_at(110.0) is False


def test_tenant_partition_report() -> None:
    report = TenantPartitionReport(
        facility_id="AVP-FACILITY-01",
        timestamp_s=1000.0,
        total_slots_count=20,
        total_partitions_count=2,
        overall_sla_compliance=0.96,
        total_breach_penalties=50.0,
    )

    assert report.total_slots_count == 20
    assert report.overall_sla_compliance == 0.96
