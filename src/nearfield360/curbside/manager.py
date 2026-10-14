"""Curbside drop-off/pick-up staging, dwell queue management, and overstay auditing."""

from __future__ import annotations

from nearfield360.config import CurbsideConfig
from nearfield360.curbside.models import (
    BayStatus,
    CurbsideBay,
    CurbsideMission,
    CurbsideReport,
    HandoverType,
    SafetyInterlockStatus,
)
from nearfield360.curbside.safety import SafetyInterlockEngine


class CurbsideCapacityExceededError(ValueError):
    """Raised when curbside staging capacity and queue limits are exceeded."""


class CurbsideManager:
    """Coordinates curbside staging bay allocation, handover lifecycles, and dwell auditing."""

    def __init__(
        self,
        facility_id: str = "benchmark_garage_avp",
        config: CurbsideConfig | None = None,
        initial_bays: list[CurbsideBay] | None = None,
    ) -> None:
        self.facility_id = facility_id
        self.config = config or CurbsideConfig()
        self.interlock_engine = SafetyInterlockEngine(self.config)
        self._bays: dict[str, CurbsideBay] = {}
        self._queue: list[CurbsideMission] = []
        self._active_missions: dict[str, CurbsideMission] = {}
        self._completed_missions: list[CurbsideMission] = []
        self._mission_counter = 0

        if initial_bays:
            for b in initial_bays:
                self._bays[b.bay_id] = b
        else:
            self._initialize_default_bays()

    def _initialize_default_bays(self) -> None:
        cap = self.config.staging_capacity
        dropoff_count = max(1, cap // 2)
        pickup_count = cap - dropoff_count

        for i in range(dropoff_count):
            bid = f"bay-dropoff-{i + 1:02d}"
            self._bays[bid] = CurbsideBay(bay_id=bid, bay_type=HandoverType.DROPOFF)

        for i in range(pickup_count):
            bid = f"bay-pickup-{i + 1:02d}"
            self._bays[bid] = CurbsideBay(bay_id=bid, bay_type=HandoverType.PICKUP)

    @property
    def bays(self) -> list[CurbsideBay]:
        """Return all managed curbside staging bays."""
        return list(self._bays.values())

    @property
    def queue(self) -> list[CurbsideMission]:
        """Return pending queue of unassigned handover requests."""
        return list(self._queue)

    @property
    def active_missions(self) -> list[CurbsideMission]:
        """Return currently active staging bay handover missions."""
        return list(self._active_missions.values())

    @property
    def completed_missions(self) -> list[CurbsideMission]:
        """Return historical completed handover missions."""
        return list(self._completed_missions)

    def _next_mission_id(self) -> str:
        self._mission_counter += 1
        return f"mission-{self._mission_counter:04d}"

    def _find_vacant_bay(self, handover_type: HandoverType) -> CurbsideBay | None:
        for b in self._bays.values():
            if b.bay_type == handover_type and not b.is_occupied:
                return b
        return None

    def request_dropoff(
        self, vehicle_id: str, current_time_s: float
    ) -> tuple[CurbsideMission, CurbsideBay | None]:
        """Request drop-off handover bay allocation for arriving vehicle."""
        mid = self._next_mission_id()
        bay = self._find_vacant_bay(HandoverType.DROPOFF)

        if bay is not None:
            updated_bay = bay.model_copy(
                update={
                    "status": BayStatus.OCCUPIED_UNLOADING,
                    "occupied_vehicle_id": vehicle_id,
                    "occupancy_start_s": current_time_s,
                    "interlock_status": SafetyInterlockStatus.DISENGAGED_CLEAR,
                }
            )
            self._bays[bay.bay_id] = updated_bay
            mission = CurbsideMission(
                mission_id=mid,
                vehicle_id=vehicle_id,
                handover_type=HandoverType.DROPOFF,
                bay_id=bay.bay_id,
                requested_at_s=current_time_s,
            )
            self._active_missions[vehicle_id] = mission
            return mission, updated_bay

        mission = CurbsideMission(
            mission_id=mid,
            vehicle_id=vehicle_id,
            handover_type=HandoverType.DROPOFF,
            bay_id="QUEUED",
            requested_at_s=current_time_s,
        )
        self._queue.append(mission)
        return mission, None

    def complete_dropoff(self, vehicle_id: str, current_time_s: float) -> CurbsideMission:
        """Complete passenger egress, verify interlocks, and release drop-off bay."""
        self.interlock_engine.assert_motion_permitted(vehicle_id, current_time_s)

        mission = self._active_missions.get(vehicle_id)
        if mission is None:
            raise KeyError(f"No active mission found for vehicle '{vehicle_id}'.")

        bay = self._bays.get(mission.bay_id)
        start_s = bay.occupancy_start_s if bay and bay.occupancy_start_s else mission.requested_at_s
        dwell_s = max(0.0, current_time_s - start_s)

        overstay_penalty = 0.0
        if dwell_s > self.config.max_dropoff_dwell_s:
            excess_min = (dwell_s - self.config.max_dropoff_dwell_s) / 60.0
            overstay_penalty = round(excess_min * self.config.overstay_penalty_per_min, 2)

        completed_mission = mission.model_copy(
            update={
                "completed_at_s": current_time_s,
                "dwell_duration_s": dwell_s,
                "overstay_penalty_usd": overstay_penalty,
                "is_completed": True,
            }
        )

        del self._active_missions[vehicle_id]
        self._completed_missions.append(completed_mission)

        # Free bay
        if bay is not None:
            freed_bay = bay.model_copy(
                update={
                    "status": BayStatus.VACANT,
                    "occupied_vehicle_id": None,
                    "occupancy_start_s": None,
                    "current_dwell_s": 0.0,
                    "interlock_status": SafetyInterlockStatus.DISENGAGED_CLEAR,
                }
            )
            self._bays[bay.bay_id] = freed_bay

            # Process queued dropoff requests if any
            queued_idx = next(
                (i for i, qm in enumerate(self._queue) if qm.handover_type == HandoverType.DROPOFF),
                None,
            )
            if queued_idx is not None:
                next_mission = self._queue.pop(queued_idx)
                next_assigned_bay = freed_bay.model_copy(
                    update={
                        "status": BayStatus.OCCUPIED_UNLOADING,
                        "occupied_vehicle_id": next_mission.vehicle_id,
                        "occupancy_start_s": current_time_s,
                    }
                )
                self._bays[freed_bay.bay_id] = next_assigned_bay
                activated_mission = next_mission.model_copy(update={"bay_id": freed_bay.bay_id})
                self._active_missions[next_mission.vehicle_id] = activated_mission

        return completed_mission

    def request_pickup(
        self, vehicle_id: str, current_time_s: float
    ) -> tuple[CurbsideMission, CurbsideBay | None]:
        """Request pick-up summon bay allocation for passenger retrieval."""
        mid = self._next_mission_id()
        bay = self._find_vacant_bay(HandoverType.PICKUP)

        if bay is not None:
            updated_bay = bay.model_copy(
                update={
                    "status": BayStatus.OCCUPIED_BOARDING,
                    "occupied_vehicle_id": vehicle_id,
                    "occupancy_start_s": current_time_s,
                    "interlock_status": SafetyInterlockStatus.DISENGAGED_CLEAR,
                }
            )
            self._bays[bay.bay_id] = updated_bay
            mission = CurbsideMission(
                mission_id=mid,
                vehicle_id=vehicle_id,
                handover_type=HandoverType.PICKUP,
                bay_id=bay.bay_id,
                requested_at_s=current_time_s,
            )
            self._active_missions[vehicle_id] = mission
            return mission, updated_bay

        mission = CurbsideMission(
            mission_id=mid,
            vehicle_id=vehicle_id,
            handover_type=HandoverType.PICKUP,
            bay_id="QUEUED",
            requested_at_s=current_time_s,
        )
        self._queue.append(mission)
        return mission, None

    def complete_pickup(self, vehicle_id: str, current_time_s: float) -> CurbsideMission:
        """Complete passenger boarding, verify interlocks, and release pick-up bay."""
        self.interlock_engine.assert_motion_permitted(vehicle_id, current_time_s)

        mission = self._active_missions.get(vehicle_id)
        if mission is None:
            raise KeyError(f"No active mission found for vehicle '{vehicle_id}'.")

        bay = self._bays.get(mission.bay_id)
        start_s = bay.occupancy_start_s if bay and bay.occupancy_start_s else mission.requested_at_s
        dwell_s = max(0.0, current_time_s - start_s)

        overstay_penalty = 0.0
        if dwell_s > self.config.max_pickup_dwell_s:
            excess_min = (dwell_s - self.config.max_pickup_dwell_s) / 60.0
            overstay_penalty = round(excess_min * self.config.overstay_penalty_per_min, 2)

        completed_mission = mission.model_copy(
            update={
                "completed_at_s": current_time_s,
                "dwell_duration_s": dwell_s,
                "overstay_penalty_usd": overstay_penalty,
                "is_completed": True,
            }
        )

        del self._active_missions[vehicle_id]
        self._completed_missions.append(completed_mission)

        # Free bay
        if bay is not None:
            freed_bay = bay.model_copy(
                update={
                    "status": BayStatus.VACANT,
                    "occupied_vehicle_id": None,
                    "occupancy_start_s": None,
                    "current_dwell_s": 0.0,
                    "interlock_status": SafetyInterlockStatus.DISENGAGED_CLEAR,
                }
            )
            self._bays[bay.bay_id] = freed_bay

            # Process queued pickup requests if any
            queued_idx = next(
                (i for i, qm in enumerate(self._queue) if qm.handover_type == HandoverType.PICKUP),
                None,
            )
            if queued_idx is not None:
                next_mission = self._queue.pop(queued_idx)
                next_assigned_bay = freed_bay.model_copy(
                    update={
                        "status": BayStatus.OCCUPIED_BOARDING,
                        "occupied_vehicle_id": next_mission.vehicle_id,
                        "occupancy_start_s": current_time_s,
                    }
                )
                self._bays[freed_bay.bay_id] = next_assigned_bay
                activated_mission = next_mission.model_copy(update={"bay_id": freed_bay.bay_id})
                self._active_missions[next_mission.vehicle_id] = activated_mission

        return completed_mission

    def generate_report(self, current_time_s: float) -> CurbsideReport:
        """Compile comprehensive curbside facility staging and safety audit report."""
        vacant_count = sum(1 for b in self._bays.values() if not b.is_occupied)
        active_interlocks = 0

        # Update live dwell on active bays
        synced_bays: list[CurbsideBay] = []
        for b in self._bays.values():
            dwell = 0.0
            interlock = SafetyInterlockStatus.DISENGAGED_CLEAR
            if b.occupied_vehicle_id and b.occupancy_start_s:
                dwell = max(0.0, current_time_s - b.occupancy_start_s)
                permitted, interlock = self.interlock_engine.evaluate_interlock(
                    b.occupied_vehicle_id, current_time_s
                )
                if not permitted:
                    active_interlocks += 1
            synced_bays.append(
                b.model_copy(update={"current_dwell_s": dwell, "interlock_status": interlock})
            )

        total_penalties = sum(m.overstay_penalty_usd for m in self._completed_missions)
        avg_dwell = 0.0
        if self._completed_missions:
            avg_dwell = sum(m.dwell_duration_s for m in self._completed_missions) / len(
                self._completed_missions
            )

        return CurbsideReport(
            facility_id=self.facility_id,
            timestamp_s=current_time_s,
            total_bays_count=len(self._bays),
            vacant_bays_count=vacant_count,
            bays=synced_bays,
            active_missions=list(self._active_missions.values()),
            completed_missions_count=len(self._completed_missions),
            active_interlocks_count=active_interlocks,
            total_overstay_penalties_usd=total_penalties,
            average_dwell_duration_s=avg_dwell,
        )
