"""Multi-tenant facility partition manager and quota enforcement engine."""

from __future__ import annotations

from nearfield360.config import TenantConfig
from nearfield360.mapping.models import FacilityMap
from nearfield360.tenant.models import (
    AccessPermission,
    TenantPartition,
    TenantTier,
)


class QuotaExceededError(ValueError):
    """Raised when a tenant partition has reached or exceeded its allocated bay quota."""


class UnauthorizedAccessError(PermissionError):
    """Raised when an unassigned vehicle attempts to access a restricted tenant partition."""


class TenantManager:
    """Manages facility spatial partitions, operator quotas, and access control."""

    def __init__(
        self,
        facility_map: FacilityMap,
        config: TenantConfig | None = None,
    ) -> None:
        self.facility_map = facility_map
        self.config = config or TenantConfig()
        self._partitions: dict[str, TenantPartition] = {}
        self._slot_to_tenant: dict[str, str] = {}
        self._vehicle_to_tenant: dict[str, str] = {}

    @property
    def partitions(self) -> list[TenantPartition]:
        """Return all registered tenant partitions."""
        return list(self._partitions.values())

    def create_partition(
        self,
        tenant_id: str,
        tenant_name: str,
        tier: TenantTier = TenantTier.COMMERCIAL_FLEET,
        slot_ids: list[str] | None = None,
        quota: int | None = None,
        access_permission: AccessPermission = AccessPermission.PRIORITY_ACCESS,
    ) -> TenantPartition:
        """Create and register a new tenant partition."""
        if len(self._partitions) >= self.config.max_tenant_partitions:
            raise ValueError(
                f"Maximum tenant partitions limit ({self.config.max_tenant_partitions}) reached."
            )

        assigned_slots = slot_ids or []
        for sid in assigned_slots:
            if sid in self._slot_to_tenant and self._slot_to_tenant[sid] != tenant_id:
                existing = self._slot_to_tenant[sid]
                raise ValueError(f"Slot '{sid}' is already assigned to tenant '{existing}'.")

        allocated_quota = quota if quota is not None else len(assigned_slots)

        partition = TenantPartition(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier,
            assigned_slot_ids=assigned_slots,
            dedicated_bay_quota=allocated_quota,
            current_occupancy=0,
            active_vehicle_ids=[],
            access_permission=access_permission,
        )

        self._partitions[tenant_id] = partition
        for sid in assigned_slots:
            self._slot_to_tenant[sid] = tenant_id

        return partition

    def register_vehicle(self, vehicle_id: str, tenant_id: str) -> None:
        """Associate a vehicle with a tenant operator."""
        if tenant_id not in self._partitions:
            raise KeyError(f"Unknown tenant ID: {tenant_id}")
        self._vehicle_to_tenant[vehicle_id] = tenant_id

    def check_access(self, vehicle_id: str, slot_id: str) -> bool:
        """Verify whether vehicle is permitted to park in target slot."""
        tenant_id = self._slot_to_tenant.get(slot_id)
        if tenant_id is None:
            # Slot is public/unpartitioned
            return True

        partition = self._partitions[tenant_id]
        if partition.access_permission == AccessPermission.STANDARD_ACCESS:
            return True

        assigned_tenant = self._vehicle_to_tenant.get(vehicle_id)
        return assigned_tenant == tenant_id

    def assign_vehicle_to_slot(self, vehicle_id: str, slot_id: str) -> None:
        """Assign vehicle to slot, validating quota and access permissions."""
        if not self.check_access(vehicle_id, slot_id):
            tenant_id = self._slot_to_tenant.get(slot_id, "unknown")
            raise UnauthorizedAccessError(
                f"Vehicle '{vehicle_id}' unauthorized for restricted tenant '{tenant_id}'."
            )

        target_tenant = self._slot_to_tenant.get(slot_id)
        if target_tenant is not None:
            partition = self._partitions[target_tenant]
            if partition.current_occupancy >= partition.dedicated_bay_quota:
                raise QuotaExceededError(
                    f"Tenant '{target_tenant}' quota of "
                    f"{partition.dedicated_bay_quota} bays exceeded."
                )

            updated = partition.model_copy(
                update={
                    "current_occupancy": partition.current_occupancy + 1,
                    "active_vehicle_ids": [*partition.active_vehicle_ids, vehicle_id],
                }
            )
            self._partitions[target_tenant] = updated

    def release_slot(self, vehicle_id: str, slot_id: str) -> None:
        """Release parked vehicle from slot and decrement partition occupancy."""
        tenant_id = self._slot_to_tenant.get(slot_id)
        if tenant_id is not None and tenant_id in self._partitions:
            partition = self._partitions[tenant_id]
            new_vids = [v for v in partition.active_vehicle_ids if v != vehicle_id]
            new_occ = max(0, partition.current_occupancy - 1)
            updated = partition.model_copy(
                update={
                    "current_occupancy": new_occ,
                    "active_vehicle_ids": new_vids,
                }
            )
            self._partitions[tenant_id] = updated
