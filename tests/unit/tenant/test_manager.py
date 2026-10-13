"""Unit tests for TenantManager partition allocation, quotas, and access control."""

from __future__ import annotations

import pytest

from nearfield360.config import TenantConfig
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.models import FacilityMap
from nearfield360.tenant.manager import (
    QuotaExceededError,
    TenantManager,
    UnauthorizedAccessError,
)
from nearfield360.tenant.models import AccessPermission, TenantTier


@pytest.fixture
def empty_facility() -> FacilityMap:
    return build_benchmark_garage(aisle_count=2, slots_per_aisle=4)


def test_tenant_manager_create_partition(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    part = mgr.create_partition(
        tenant_id="tenant-alpha",
        tenant_name="Alpha Logistics",
        tier=TenantTier.VIP_EXECUTIVE,
        slot_ids=["slot-1", "slot-2"],
        quota=2,
    )
    assert part.tenant_id == "tenant-alpha"
    assert part.dedicated_bay_quota == 2
    assert len(mgr.partitions) == 1


def test_tenant_manager_max_partitions_limit(empty_facility: FacilityMap) -> None:
    cfg = TenantConfig(max_tenant_partitions=1)
    mgr = TenantManager(facility_map=empty_facility, config=cfg)
    mgr.create_partition(tenant_id="t1", tenant_name="Tenant 1")
    with pytest.raises(ValueError, match="Maximum tenant partitions limit"):
        mgr.create_partition(tenant_id="t2", tenant_name="Tenant 2")


def test_tenant_manager_slot_collision(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    mgr.create_partition(tenant_id="t1", tenant_name="Tenant 1", slot_ids=["slot-A"])
    with pytest.raises(ValueError, match="already assigned"):
        mgr.create_partition(tenant_id="t2", tenant_name="Tenant 2", slot_ids=["slot-A"])


def test_tenant_manager_access_control(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    mgr.create_partition(
        tenant_id="t1",
        tenant_name="Tenant 1",
        slot_ids=["slot-A"],
        access_permission=AccessPermission.EXCLUSIVE_VIP,
    )
    mgr.register_vehicle("veh-auth", "t1")

    # Authorized vehicle
    assert mgr.check_access("veh-auth", "slot-A") is True

    # Unauthorized vehicle
    assert mgr.check_access("veh-intruder", "slot-A") is False

    # Unpartitioned slot
    assert mgr.check_access("veh-intruder", "slot-public") is True

    # Unknown tenant for vehicle registration
    with pytest.raises(KeyError):
        mgr.register_vehicle("veh-unknown", "t-nonexistent")


def test_tenant_manager_quota_enforcement(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    mgr.create_partition(
        tenant_id="t1",
        tenant_name="Tenant 1",
        slot_ids=["slot-1", "slot-2"],
        quota=1,
    )
    mgr.register_vehicle("veh-1", "t1")
    mgr.register_vehicle("veh-2", "t1")

    # First assignment succeeds
    mgr.assign_vehicle_to_slot("veh-1", "slot-1")
    assert mgr.partitions[0].current_occupancy == 1
    assert "veh-1" in mgr.partitions[0].active_vehicle_ids

    # Second assignment exceeds quota of 1
    with pytest.raises(QuotaExceededError, match="quota of 1 bays exceeded"):
        mgr.assign_vehicle_to_slot("veh-2", "slot-2")


def test_tenant_manager_unauthorized_assignment(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    mgr.create_partition(
        tenant_id="t1",
        tenant_name="Tenant 1",
        slot_ids=["slot-1"],
        access_permission=AccessPermission.EXCLUSIVE_VIP,
    )
    with pytest.raises(UnauthorizedAccessError, match="unauthorized for restricted tenant"):
        mgr.assign_vehicle_to_slot("veh-rogue", "slot-1")


def test_tenant_manager_release_slot(empty_facility: FacilityMap) -> None:
    mgr = TenantManager(facility_map=empty_facility)
    mgr.create_partition(
        tenant_id="t1",
        tenant_name="Tenant 1",
        slot_ids=["slot-1"],
        quota=2,
    )
    mgr.register_vehicle("veh-1", "t1")
    mgr.assign_vehicle_to_slot("veh-1", "slot-1")
    assert mgr.partitions[0].current_occupancy == 1

    mgr.release_slot("veh-1", "slot-1")
    assert mgr.partitions[0].current_occupancy == 0
    assert "veh-1" not in mgr.partitions[0].active_vehicle_ids

    # Extra release does not go below 0
    mgr.release_slot("veh-1", "slot-1")
    assert mgr.partitions[0].current_occupancy == 0
