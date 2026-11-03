"""Dual-slot A/B partition manager and fail-safe rollback guard (ISO 24089 / UN R156)."""

from __future__ import annotations

import logging

from nearfield360.ota.models import (
    ECUPartitionState,
    FirmwareManifest,
    PartitionSlot,
    SlotMetadata,
)

logger = logging.getLogger(__name__)


class PartitionManager:
    """Manages dual-slot A/B flash partitions, staging, atomic boot pointers, and rollbacks."""

    def __init__(
        self,
        ecu_id: str = "primary_ecu",
        initial_slot: PartitionSlot = PartitionSlot.SLOT_A,
        initial_version: str = "v1.0.0",
        initial_revision: int = 1,
        initial_rxswin: str = "RXSWIN-DEFAULT-0001",
    ) -> None:
        self.ecu_id = ecu_id
        other_slot = initial_slot.opposite()

        slot_init = SlotMetadata(
            slot=initial_slot,
            is_active=True,
            is_bootable=True,
            version=initial_version,
            monotonic_revision=initial_revision,
            rxswin=initial_rxswin,
            sha256_hash="0" * 64,
            boot_count=1,
        )

        slot_standby = SlotMetadata(
            slot=other_slot,
            is_active=False,
            is_bootable=False,
            version="v0.0.0",
            monotonic_revision=0,
            rxswin="RXSWIN-NONE-0000",
            sha256_hash="0" * 64,
            boot_count=0,
        )

        slots_dict = {
            initial_slot.value: slot_init,
            other_slot.value: slot_standby,
        }

        self._state = ECUPartitionState(
            ecu_id=ecu_id,
            active_slot=initial_slot,
            next_boot_slot=initial_slot,
            slots=slots_dict,
            rollback_count=0,
        )

    @property
    def state(self) -> ECUPartitionState:
        """Current ECU partition state."""
        return self._state

    def get_active_slot(self) -> PartitionSlot:
        """Return the currently executing active partition slot."""
        return self._state.active_slot

    def get_standby_slot(self) -> PartitionSlot:
        """Return the alternate inactive partition slot."""
        return self._state.active_slot.opposite()

    def get_active_metadata(self) -> SlotMetadata:
        """Retrieve metadata for currently running active partition."""
        return self._state.slots[self.get_active_slot().value]

    def get_standby_metadata(self) -> SlotMetadata:
        """Retrieve metadata for standby staging partition."""
        return self._state.slots[self.get_standby_slot().value]

    def stage_update(
        self,
        manifest: FirmwareManifest,
        payload_bytes: bytes | None = None,
    ) -> SlotMetadata:
        """Stage verified firmware manifest into the standby partition slot."""
        del payload_bytes
        standby_slot = self.get_standby_slot()

        staged_slot = SlotMetadata(
            slot=standby_slot,
            is_active=False,
            is_bootable=True,
            version=manifest.version,
            monotonic_revision=manifest.monotonic_revision,
            rxswin=manifest.rxswin,
            sha256_hash=manifest.sha256_hash,
            boot_count=0,
        )

        new_slots = dict(self._state.slots)
        new_slots[standby_slot.value] = staged_slot

        self._state = ECUPartitionState(
            ecu_id=self._state.ecu_id,
            active_slot=self._state.active_slot,
            next_boot_slot=self._state.next_boot_slot,
            slots=new_slots,
            rollback_count=self._state.rollback_count,
        )

        logger.info(
            "Staged firmware %s (rev %d, %s) into standby partition %s",
            manifest.version,
            manifest.monotonic_revision,
            manifest.rxswin,
            standby_slot.value,
        )
        return staged_slot

    def activate_standby_slot(self) -> PartitionSlot:
        """Set the standby slot as the target for the next vehicle boot cycle."""
        standby_slot = self.get_standby_slot()

        self._state = ECUPartitionState(
            ecu_id=self._state.ecu_id,
            active_slot=self._state.active_slot,
            next_boot_slot=standby_slot,
            slots=dict(self._state.slots),
            rollback_count=self._state.rollback_count,
        )

        logger.info(
            "Target next boot slot updated to %s",
            standby_slot.value,
        )
        return standby_slot

    def confirm_boot_success(self) -> ECUPartitionState:
        """Confirm successful boot and post-update health check, committing new slot."""
        target_slot = self._state.next_boot_slot
        previous_slot = target_slot.opposite()

        new_slots = dict(self._state.slots)
        active_meta = new_slots[target_slot.value]
        new_slots[target_slot.value] = SlotMetadata(
            slot=target_slot,
            is_active=True,
            is_bootable=True,
            version=active_meta.version,
            monotonic_revision=active_meta.monotonic_revision,
            rxswin=active_meta.rxswin,
            sha256_hash=active_meta.sha256_hash,
            boot_count=active_meta.boot_count + 1,
        )

        prev_meta = new_slots[previous_slot.value]
        new_slots[previous_slot.value] = SlotMetadata(
            slot=previous_slot,
            is_active=False,
            is_bootable=True,
            version=prev_meta.version,
            monotonic_revision=prev_meta.monotonic_revision,
            rxswin=prev_meta.rxswin,
            sha256_hash=prev_meta.sha256_hash,
            boot_count=prev_meta.boot_count,
        )

        self._state = ECUPartitionState(
            ecu_id=self._state.ecu_id,
            active_slot=target_slot,
            next_boot_slot=target_slot,
            slots=new_slots,
            rollback_count=self._state.rollback_count,
        )

        logger.info(
            "Boot confirmed successful: committed active slot %s (version %s)",
            target_slot.value,
            active_meta.version,
        )
        return self._state

    def trigger_rollback(self, reason: str = "Self-test failure") -> ECUPartitionState:
        """Abort new slot boot and immediately roll back to previous active partition."""
        failed_slot = self._state.next_boot_slot
        fallback_slot = self._state.active_slot

        new_slots = dict(self._state.slots)
        failed_meta = new_slots[failed_slot.value]
        new_slots[failed_slot.value] = SlotMetadata(
            slot=failed_slot,
            is_active=False,
            is_bootable=False,  # Invalidate failed slot
            version=failed_meta.version,
            monotonic_revision=failed_meta.monotonic_revision,
            rxswin=failed_meta.rxswin,
            sha256_hash=failed_meta.sha256_hash,
            boot_count=failed_meta.boot_count,
        )

        self._state = ECUPartitionState(
            ecu_id=self._state.ecu_id,
            active_slot=fallback_slot,
            next_boot_slot=fallback_slot,
            slots=new_slots,
            rollback_count=self._state.rollback_count + 1,
        )

        logger.warning(
            "Rollback triggered (%s): reverted next boot slot to %s, invalidated %s",
            reason,
            fallback_slot.value,
            failed_slot.value,
        )
        return self._state
