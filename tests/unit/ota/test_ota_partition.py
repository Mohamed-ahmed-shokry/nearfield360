"""Unit tests for dual-slot A/B partition manager and automated rollback guard."""

from __future__ import annotations

from nearfield360.ota.models import FirmwareManifest, PartitionSlot
from nearfield360.ota.partition import PartitionManager


def test_partition_manager_initialization() -> None:
    pm = PartitionManager(
        ecu_id="trajectory_planner",
        initial_slot=PartitionSlot.SLOT_A,
        initial_version="v1.2.0",
        initial_revision=3,
        initial_rxswin="RXSWIN-E4-156R00-NF360-0012",
    )

    assert pm.get_active_slot() == PartitionSlot.SLOT_A
    assert pm.get_standby_slot() == PartitionSlot.SLOT_B
    assert pm.state.rollback_count == 0

    active_meta = pm.get_active_metadata()
    assert active_meta.version == "v1.2.0"
    assert active_meta.monotonic_revision == 3
    assert active_meta.is_active is True
    assert active_meta.is_bootable is True

    standby_meta = pm.get_standby_metadata()
    assert standby_meta.is_active is False
    assert standby_meta.is_bootable is False


def test_partition_staging_and_successful_commit() -> None:
    pm = PartitionManager(initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0")

    manifest = FirmwareManifest(
        component_id="ecu_planner",
        version="v1.1.0",
        monotonic_revision=2,
        rxswin="RXSWIN-E4-156R00-NF360-0013",
        target_ecu="trajectory_planner",
        payload_size_bytes=1024,
        sha256_hash="e" * 64,
        signature_hex="f" * 64,
    )

    # 1. Stage update into standby (slot B)
    staged = pm.stage_update(manifest)
    assert staged.slot == PartitionSlot.SLOT_B
    assert staged.version == "v1.1.0"
    assert staged.monotonic_revision == 2
    assert staged.is_bootable is True
    assert staged.is_active is False
    # Active slot remains Slot A
    assert pm.get_active_slot() == PartitionSlot.SLOT_A

    # 2. Activate standby slot for next boot
    next_slot = pm.activate_standby_slot()
    assert next_slot == PartitionSlot.SLOT_B
    assert pm.state.next_boot_slot == PartitionSlot.SLOT_B
    assert pm.get_active_slot() == PartitionSlot.SLOT_A

    # 3. Confirm boot success
    state = pm.confirm_boot_success()
    assert state.active_slot == PartitionSlot.SLOT_B
    assert state.next_boot_slot == PartitionSlot.SLOT_B
    assert pm.get_active_slot() == PartitionSlot.SLOT_B
    assert pm.get_active_metadata().version == "v1.1.0"
    assert pm.get_active_metadata().boot_count >= 1
    assert pm.get_standby_slot() == PartitionSlot.SLOT_A
    assert pm.get_standby_metadata().is_active is False


def test_partition_rollback_on_failed_boot() -> None:
    pm = PartitionManager(initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0")

    manifest = FirmwareManifest(
        component_id="ecu_planner",
        version="v2.0.0-broken",
        monotonic_revision=4,
        rxswin="RXSWIN-E4-156R00-NF360-0014",
        target_ecu="trajectory_planner",
        payload_size_bytes=2048,
        sha256_hash="1" * 64,
        signature_hex="2" * 64,
    )

    pm.stage_update(manifest)
    pm.activate_standby_slot()
    assert pm.state.next_boot_slot == PartitionSlot.SLOT_B

    # Trigger rollback due to self-test health check failure
    rb_state = pm.trigger_rollback(reason="Watchdog timeout after reboot")
    assert rb_state.active_slot == PartitionSlot.SLOT_A
    assert rb_state.next_boot_slot == PartitionSlot.SLOT_A
    assert rb_state.rollback_count == 1

    # Standby slot (Slot B) marked invalid / not bootable
    standby_meta = pm.get_standby_metadata()
    assert standby_meta.slot == PartitionSlot.SLOT_B
    assert standby_meta.is_bootable is False
    assert standby_meta.is_active is False
