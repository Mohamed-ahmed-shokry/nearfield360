"""Unit tests for OTA update executive, safety preflight interlocks, and lifecycle orchestrator."""

from __future__ import annotations

from nearfield360.config import OTAConfig
from nearfield360.ota.executive import OTAExecutive
from nearfield360.ota.models import (
    FirmwareManifest,
    PartitionSlot,
    PreflightStatus,
    UpdateState,
)
from nearfield360.ota.partition import PartitionManager
from nearfield360.ota.verifier import FirmwareVerifier


def _create_test_package(
    version: str = "v1.1.0",
    revision: int = 2,
    ecu: str = "trajectory_planner",
) -> tuple[FirmwareManifest, bytes]:
    verifier = FirmwareVerifier()
    payload = f"BINARY-PAYLOAD-FOR-{ecu}-{version}".encode()
    sha = verifier.compute_payload_sha256(payload)
    sig = verifier.sign_payload(payload)

    manifest = FirmwareManifest(
        component_id=f"ecu_{ecu}",
        version=version,
        monotonic_revision=revision,
        rxswin=f"RXSWIN-E4-156R00-NF360-00{revision:02d}",
        target_ecu=ecu,
        payload_size_bytes=len(payload),
        sha256_hash=sha,
        signature_hex=sig,
    )
    return manifest, payload


def test_preflight_interlocks_validation() -> None:
    exec_engine = OTAExecutive(config=OTAConfig(min_battery_soc_pct=30.0))

    # 1. Nominal case
    res_nom = exec_engine.check_preflight_interlocks(
        gear="PARK", speed_mps=0.0, battery_soc_pct=75.0, safety_supervisor_ready=True
    )
    assert res_nom.status == PreflightStatus.PASSED
    assert len(res_nom.reasons) == 1

    # 2. Gear in DRIVE
    res_gear = exec_engine.check_preflight_interlocks(
        gear="D", speed_mps=0.0, battery_soc_pct=75.0, safety_supervisor_ready=True
    )
    assert res_gear.status == PreflightStatus.FAILED
    assert any("gear 'D'" in r for r in res_gear.reasons)

    # 3. Vehicle moving
    res_speed = exec_engine.check_preflight_interlocks(
        gear="PARK", speed_mps=1.2, battery_soc_pct=75.0, safety_supervisor_ready=True
    )
    assert res_speed.status == PreflightStatus.FAILED
    assert any("in motion" in r for r in res_speed.reasons)

    # 4. Low battery
    res_soc = exec_engine.check_preflight_interlocks(
        gear="PARK", speed_mps=0.0, battery_soc_pct=20.0, safety_supervisor_ready=True
    )
    assert res_soc.status == PreflightStatus.FAILED
    assert any("State of Charge" in r for r in res_soc.reasons)

    # 5. Safety supervisor degraded / not ready
    res_safety = exec_engine.check_preflight_interlocks(
        gear="PARK", speed_mps=0.0, battery_soc_pct=80.0, safety_supervisor_ready=False
    )
    assert res_safety.status == PreflightStatus.FAILED
    assert any("Functional safety supervisor" in r for r in res_safety.reasons)


def test_execute_campaign_nominal_success() -> None:
    pm = PartitionManager(
        initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0", initial_revision=1
    )
    executive = OTAExecutive(partition_manager=pm)
    manifest, payload = _create_test_package(version="v1.1.0", revision=2)

    report = executive.execute_campaign(
        campaign_id="CAMP-2026-TEST-NOMINAL",
        manifest=manifest,
        payload_bytes=payload,
        vehicle_state={"gear": "P", "speed_mps": 0.0, "battery_soc_pct": 85.0},
        simulate_reboot_health=True,
    )

    assert report.success is True
    assert report.active_state == UpdateState.COMMITTED
    assert report.partition_state.active_slot == PartitionSlot.SLOT_B
    assert pm.get_active_slot() == PartitionSlot.SLOT_B
    assert pm.get_active_metadata().version == "v1.1.0"
    assert len(report.audit_trail) >= 5


def test_execute_campaign_preflight_failure() -> None:
    pm = PartitionManager(
        initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0", initial_revision=1
    )
    executive = OTAExecutive(partition_manager=pm)
    manifest, payload = _create_test_package(version="v1.1.0", revision=2)

    # Vehicle in motion during update attempt
    report = executive.execute_campaign(
        campaign_id="CAMP-2026-FAIL-PREFLIGHT",
        manifest=manifest,
        payload_bytes=payload,
        vehicle_state={"gear": "D", "speed_mps": 2.5, "battery_soc_pct": 15.0},
    )

    assert report.success is False
    assert report.active_state == UpdateState.FAILED
    assert pm.get_active_slot() == PartitionSlot.SLOT_A


def test_execute_campaign_tampered_payload_failure() -> None:
    pm = PartitionManager(
        initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0", initial_revision=1
    )
    executive = OTAExecutive(partition_manager=pm)
    manifest, payload = _create_test_package(version="v1.1.0", revision=2)

    # Tamper with binary payload
    tampered_payload = payload + b"-CORRUPTED-BITS"

    report = executive.execute_campaign(
        campaign_id="CAMP-2026-FAIL-TAMPER",
        manifest=manifest,
        payload_bytes=tampered_payload,
    )

    assert report.success is False
    assert report.active_state == UpdateState.FAILED
    assert pm.get_active_slot() == PartitionSlot.SLOT_A


def test_execute_campaign_reboot_failure_triggers_rollback() -> None:
    pm = PartitionManager(
        initial_slot=PartitionSlot.SLOT_A, initial_version="v1.0.0", initial_revision=1
    )
    executive = OTAExecutive(partition_manager=pm)
    manifest, payload = _create_test_package(version="v1.1.0", revision=2)

    # Simulate post-reboot self-test health check failure
    report = executive.execute_campaign(
        campaign_id="CAMP-2026-TEST-ROLLBACK",
        manifest=manifest,
        payload_bytes=payload,
        simulate_reboot_health=False,
    )

    assert report.success is False
    assert report.active_state == UpdateState.ROLLED_BACK
    assert pm.get_active_slot() == PartitionSlot.SLOT_A
    assert pm.state.rollback_count == 1
    assert pm.get_standby_metadata().is_bootable is False
