"""OTA update executive, safety preflight interlocks, and lifecycle orchestrator (UN R156)."""

from __future__ import annotations

import logging
from typing import Any

from nearfield360.config import OTAConfig
from nearfield360.ota.models import (
    FirmwareManifest,
    OTACampaignReport,
    OTACampaignStepRecord,
    PreflightCheck,
    PreflightStatus,
    UpdateState,
)
from nearfield360.ota.partition import PartitionManager
from nearfield360.ota.verifier import FirmwareVerifier

logger = logging.getLogger(__name__)


class OTAExecutive:
    """Orchestrates UN R156 software update campaigns and vehicle safety interlocks."""

    def __init__(
        self,
        config: OTAConfig | None = None,
        verifier: FirmwareVerifier | None = None,
        partition_manager: PartitionManager | None = None,
    ) -> None:
        self.config = config or OTAConfig()
        self.verifier = verifier or FirmwareVerifier(self.config)
        self.partition = partition_manager or PartitionManager()
        self._audit_trail: list[OTACampaignStepRecord] = []
        self._state: UpdateState = UpdateState.IDLE

    @property
    def current_state(self) -> UpdateState:
        """Active lifecycle state of the OTA campaign."""
        return self._state

    def check_preflight_interlocks(
        self,
        gear: str,
        speed_mps: float,
        battery_soc_pct: float,
        safety_supervisor_ready: bool = True,
    ) -> PreflightCheck:
        """Validate vehicle operating state satisfies strict pre-update safety prerequisites."""
        reasons: list[str] = []
        is_parked = gear.upper() in ("P", "PARK")

        if self.config.require_charging_or_parked and not is_parked:
            reasons.append(f"Vehicle transmission is in gear '{gear}', expected PARK (P)")

        if abs(speed_mps) > 0.05:
            reasons.append(f"Vehicle is in motion ({speed_mps:.2f} m/s > 0.05 m/s threshold)")

        if battery_soc_pct < self.config.min_battery_soc_pct:
            reasons.append(
                f"Battery State of Charge ({battery_soc_pct:.1f}%) is below minimum "
                f"required threshold ({self.config.min_battery_soc_pct:.1f}%)"
            )

        if not safety_supervisor_ready:
            reasons.append("Functional safety supervisor is not in nominal operating state")

        status = PreflightStatus.PASSED if not reasons else PreflightStatus.FAILED
        if not reasons:
            reasons.append("All safety preflight interlocks verified successfully")

        return PreflightCheck(
            gear_is_park=is_parked,
            speed_mps=round(speed_mps, 3),
            battery_soc_pct=round(battery_soc_pct, 2),
            safety_supervisor_ready=safety_supervisor_ready,
            status=status,
            reasons=reasons,
        )

    def _record_step(
        self,
        from_state: UpdateState,
        to_state: UpdateState,
        message: str,
        timestamp_s: float,
    ) -> None:
        record = OTACampaignStepRecord(
            from_state=from_state,
            to_state=to_state,
            message=message,
            timestamp_s=timestamp_s,
        )
        self._audit_trail.append(record)
        self._state = to_state
        logger.info("OTA Lifecycle Step: %s -> %s: %s", from_state.value, to_state.value, message)

    def execute_campaign(
        self,
        campaign_id: str,
        manifest: FirmwareManifest,
        payload_bytes: bytes,
        vehicle_state: dict[str, Any] | None = None,
        simulate_reboot_health: bool = True,
        start_time_s: float = 0.0,
    ) -> OTACampaignReport:
        """Execute complete end-to-end UN R156 software update campaign lifecycle."""
        t_curr = start_time_s
        self._audit_trail.clear()
        self._state = UpdateState.IDLE

        # Step 1: IDLE -> DOWNLOADING
        self._record_step(
            from_state=UpdateState.IDLE,
            to_state=UpdateState.DOWNLOADING,
            message=f"Beginning download of package {manifest.version} for {manifest.target_ecu}",
            timestamp_s=t_curr,
        )
        t_curr += 1.0

        # Step 2: DOWNLOADING -> VERIFYING
        self._record_step(
            from_state=UpdateState.DOWNLOADING,
            to_state=UpdateState.VERIFYING,
            message="Validating payload SHA-256 digest, signature, and anti-rollback counter",
            timestamp_s=t_curr,
        )
        t_curr += 1.0

        # Cryptographic check
        manifest_ok, manifest_msg = self.verifier.verify_manifest(manifest, payload_bytes)
        if not manifest_ok:
            self._record_step(
                from_state=UpdateState.VERIFYING,
                to_state=UpdateState.FAILED,
                message=f"Firmware verification rejected: {manifest_msg}",
                timestamp_s=t_curr,
            )
            preflight_dummy = self.check_preflight_interlocks(
                gear="PARK", speed_mps=0.0, battery_soc_pct=100.0
            )
            return OTACampaignReport(
                campaign_id=campaign_id,
                target_ecu=manifest.target_ecu,
                active_state=UpdateState.FAILED,
                manifest=manifest,
                preflight=preflight_dummy,
                partition_state=self.partition.state,
                audit_trail=list(self._audit_trail),
                success=False,
                duration_s=t_curr - start_time_s,
            )

        # Anti-rollback check
        current_rev = self.partition.get_active_metadata().monotonic_revision
        arb_ok, arb_msg = self.verifier.verify_anti_rollback(
            candidate_revision=manifest.monotonic_revision,
            current_installed_revision=current_rev,
        )
        if not arb_ok:
            self._record_step(
                from_state=UpdateState.VERIFYING,
                to_state=UpdateState.FAILED,
                message=f"Anti-rollback rejected: {arb_msg}",
                timestamp_s=t_curr,
            )
            preflight_dummy = self.check_preflight_interlocks(
                gear="PARK", speed_mps=0.0, battery_soc_pct=100.0
            )
            return OTACampaignReport(
                campaign_id=campaign_id,
                target_ecu=manifest.target_ecu,
                active_state=UpdateState.FAILED,
                manifest=manifest,
                preflight=preflight_dummy,
                partition_state=self.partition.state,
                audit_trail=list(self._audit_trail),
                success=False,
                duration_s=t_curr - start_time_s,
            )

        # Step 3: VERIFYING -> STAGED
        self.partition.stage_update(manifest)
        standby_id = self.partition.get_standby_slot().value
        self._record_step(
            from_state=UpdateState.VERIFYING,
            to_state=UpdateState.STAGED,
            message=f"Firmware staged into standby partition slot {standby_id}",
            timestamp_s=t_curr,
        )
        t_curr += 1.0

        # Step 4: STAGED -> PREFLIGHT
        v_state = vehicle_state or {
            "gear": "PARK",
            "speed_mps": 0.0,
            "battery_soc_pct": 80.0,
            "safety_supervisor_ready": True,
        }
        preflight = self.check_preflight_interlocks(
            gear=str(v_state.get("gear", "PARK")),
            speed_mps=float(v_state.get("speed_mps", 0.0)),
            battery_soc_pct=float(v_state.get("battery_soc_pct", 80.0)),
            safety_supervisor_ready=bool(v_state.get("safety_supervisor_ready", True)),
        )

        self._record_step(
            from_state=UpdateState.STAGED,
            to_state=UpdateState.PREFLIGHT,
            message=f"Preflight status: {preflight.status.value}",
            timestamp_s=t_curr,
        )
        t_curr += 1.0

        if preflight.status != PreflightStatus.PASSED:
            reasons_str = "; ".join(preflight.reasons)
            self._record_step(
                from_state=UpdateState.PREFLIGHT,
                to_state=UpdateState.FAILED,
                message=f"Preflight safety interlock check failed: {reasons_str}",
                timestamp_s=t_curr,
            )
            return OTACampaignReport(
                campaign_id=campaign_id,
                target_ecu=manifest.target_ecu,
                active_state=UpdateState.FAILED,
                manifest=manifest,
                preflight=preflight,
                partition_state=self.partition.state,
                audit_trail=list(self._audit_trail),
                success=False,
                duration_s=t_curr - start_time_s,
            )

        # Step 5: PREFLIGHT -> APPLYING
        next_slot = self.partition.activate_standby_slot()
        self._record_step(
            from_state=UpdateState.PREFLIGHT,
            to_state=UpdateState.APPLYING,
            message=f"Activating standby partition {next_slot.value} as next boot target",
            timestamp_s=t_curr,
        )
        t_curr += self.config.activation_dwell_s

        # Step 6: APPLYING -> TESTING
        self._record_step(
            from_state=UpdateState.APPLYING,
            to_state=UpdateState.TESTING,
            message="Rebooted into new partition slot; verifying ECU self-test health status",
            timestamp_s=t_curr,
        )
        t_curr += 2.0

        # Step 7: TESTING -> COMMITTED or ROLLED_BACK
        if simulate_reboot_health:
            self.partition.confirm_boot_success()
            msg_commit = (
                f"Self-test healthy: committed active slot {next_slot.value} "
                f"(RXSWIN: {manifest.rxswin})"
            )
            self._record_step(
                from_state=UpdateState.TESTING,
                to_state=UpdateState.COMMITTED,
                message=msg_commit,
                timestamp_s=t_curr,
            )
            return OTACampaignReport(
                campaign_id=campaign_id,
                target_ecu=manifest.target_ecu,
                active_state=UpdateState.COMMITTED,
                manifest=manifest,
                preflight=preflight,
                partition_state=self.partition.state,
                audit_trail=list(self._audit_trail),
                success=True,
                duration_s=round(t_curr - start_time_s, 2),
            )
        else:
            self.partition.trigger_rollback(reason="Reboot self-test health check failed")
            self._record_step(
                from_state=UpdateState.TESTING,
                to_state=UpdateState.ROLLED_BACK,
                message="Post-update self-test failed; rolled back to previous active slot",
                timestamp_s=t_curr,
            )
            return OTACampaignReport(
                campaign_id=campaign_id,
                target_ecu=manifest.target_ecu,
                active_state=UpdateState.ROLLED_BACK,
                manifest=manifest,
                preflight=preflight,
                partition_state=self.partition.state,
                audit_trail=list(self._audit_trail),
                success=False,
                duration_s=round(t_curr - start_time_s, 2),
            )
