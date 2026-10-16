"""Automated vehicle elevator dispatching, hoisting dynamics, and leveling safety interlocks."""

from __future__ import annotations

from typing import Any, NamedTuple

from nearfield360.multilevel.models import ElevatorCabin, ElevatorState, FloorLevel


class ElevatorCall(NamedTuple):
    """Pending transit request for an automated vehicle elevator."""

    call_id: str
    vehicle_id: str
    origin_floor_id: str
    destination_floor_id: str
    vehicle_weight_kg: float
    requested_at_s: float


class VehicleElevatorDispatcher:
    """Manages scheduling, hoisting velocity, and door/leveling interlocks for car hoists."""

    def __init__(
        self,
        door_dwell_s: float = 5.0,
        hoist_speed_mps: float = 1.2,
        leveling_tolerance_m: float = 0.015,
        max_cabin_weight_kg: float = 3500.0,
    ) -> None:
        self.door_dwell_s = door_dwell_s
        self.hoist_speed_mps = hoist_speed_mps
        self.leveling_tolerance_m = leveling_tolerance_m
        self.max_cabin_weight_kg = max_cabin_weight_kg

        self._floors: dict[str, FloorLevel] = {}
        self._cabins: dict[str, ElevatorCabin] = {}
        self._pending_calls: list[ElevatorCall] = []
        self._active_calls: dict[str, ElevatorCall] = {}  # elevator_id -> call
        self._dwell_timers: dict[str, float] = {}  # elevator_id -> seconds remaining in dwell
        self._sim_time_s: float = 0.0
        self._total_cycles_completed: int = 0

    @property
    def total_cycles_completed(self) -> int:
        """Total completed elevator transport cycles."""
        return self._total_cycles_completed

    def register_floor(self, floor: FloorLevel) -> None:
        """Register a facility floor level and its elevation."""
        self._floors[floor.floor_id] = floor

    def register_elevator(
        self,
        elevator_id: str,
        initial_floor_id: str,
        max_weight_kg: float | None = None,
    ) -> None:
        """Register a car hoist cabin at a specified initial floor."""
        if initial_floor_id not in self._floors:
            raise KeyError(f"Initial floor {initial_floor_id} not registered.")
        floor = self._floors[initial_floor_id]
        capacity = max_weight_kg or self.max_cabin_weight_kg

        self._cabins[elevator_id] = ElevatorCabin(
            elevator_id=elevator_id,
            current_floor_id=initial_floor_id,
            target_floor_id=None,
            state=ElevatorState.IDLE,
            current_elevation_m=floor.elevation_m,
            max_weight_kg=capacity,
            occupied_vehicle_id=None,
            leveling_offset_m=0.0,
        )
        self._dwell_timers[elevator_id] = 0.0

    def request_elevator(
        self,
        vehicle_id: str,
        origin_floor_id: str,
        destination_floor_id: str,
        vehicle_weight_kg: float = 1800.0,
    ) -> str:
        """Submit a vehicle transfer call between floors."""
        if origin_floor_id not in self._floors or destination_floor_id not in self._floors:
            raise KeyError(
                f"Both origin ({origin_floor_id}) and destination ({destination_floor_id}) "
                f"must be registered floors."
            )
        if origin_floor_id == destination_floor_id:
            raise ValueError("Origin and destination floors must be distinct.")
        if vehicle_weight_kg > self.max_cabin_weight_kg:
            raise ValueError(
                f"Vehicle weight ({vehicle_weight_kg} kg) exceeds elevator "
                f"capacity ({self.max_cabin_weight_kg} kg)."
            )

        call_id = f"call-{vehicle_id}-{origin_floor_id}-to-{destination_floor_id}"
        call = ElevatorCall(
            call_id=call_id,
            vehicle_id=vehicle_id,
            origin_floor_id=origin_floor_id,
            destination_floor_id=destination_floor_id,
            vehicle_weight_kg=vehicle_weight_kg,
            requested_at_s=self._sim_time_s,
        )
        self._pending_calls.append(call)
        self._dispatch_pending_calls()
        return call_id

    def is_drive_in_permitted(self, elevator_id: str) -> bool:
        """Verify whether cabin doors are open and leveling alignment allows vehicle ingress."""
        cabin = self._cabins.get(elevator_id)
        if cabin is None:
            return False
        return (
            cabin.state == ElevatorState.LOADING
            and abs(cabin.leveling_offset_m) <= self.leveling_tolerance_m
            and cabin.occupied_vehicle_id is None
        )

    def is_drive_out_permitted(self, elevator_id: str) -> bool:
        """Verify whether destination doors are open and leveling alignment allows egress."""
        cabin = self._cabins.get(elevator_id)
        if cabin is None:
            return False
        return (
            cabin.state == ElevatorState.UNLOADING
            and abs(cabin.leveling_offset_m) <= self.leveling_tolerance_m
            and cabin.occupied_vehicle_id is not None
        )

    def step(self, dt_s: float) -> list[dict[str, Any]]:
        """Advance discrete-time simulation of all cabins, doors, and hoisting mechanisms."""
        self._sim_time_s += dt_s
        events: list[dict[str, Any]] = []

        self._dispatch_pending_calls()

        for elevator_id, cabin in list(self._cabins.items()):
            active_call = self._active_calls.get(elevator_id)
            timer = self._dwell_timers.get(elevator_id, 0.0)

            if cabin.state == ElevatorState.IDLE:
                continue

            elif cabin.state == ElevatorState.TRANSITING:
                target_floor = self._floors[cabin.target_floor_id or cabin.current_floor_id]
                target_z = target_floor.elevation_m
                diff = target_z - cabin.current_elevation_m
                step_dist = self.hoist_speed_mps * dt_s

                if abs(diff) <= step_dist:
                    new_z = target_z
                    self._dwell_timers[elevator_id] = self.door_dwell_s
                    self._update_cabin(
                        elevator_id,
                        state=ElevatorState.DOORS_OPENING,
                        current_elevation_m=new_z,
                        current_floor_id=target_floor.floor_id,
                        leveling_offset_m=0.005,
                    )
                    events.append(
                        {
                            "event": "arrived_at_pickup",
                            "elevator_id": elevator_id,
                            "floor_id": target_floor.floor_id,
                        }
                    )
                else:
                    direction = 1.0 if diff > 0 else -1.0
                    new_z = cabin.current_elevation_m + direction * step_dist
                    self._update_cabin(elevator_id, current_elevation_m=new_z)

            elif cabin.state == ElevatorState.DOORS_OPENING:
                timer -= dt_s
                if timer <= 0:
                    self._dwell_timers[elevator_id] = self.door_dwell_s
                    if cabin.occupied_vehicle_id is not None:
                        self._update_cabin(
                            elevator_id,
                            state=ElevatorState.UNLOADING,
                            leveling_offset_m=0.002,
                        )
                        events.append(
                            {"event": "doors_opened_unloading", "elevator_id": elevator_id}
                        )
                    else:
                        self._update_cabin(
                            elevator_id,
                            state=ElevatorState.LOADING,
                            leveling_offset_m=0.002,
                        )
                        events.append({"event": "doors_opened_loading", "elevator_id": elevator_id})
                else:
                    self._dwell_timers[elevator_id] = timer

            elif cabin.state == ElevatorState.LOADING:
                timer -= dt_s
                if timer <= 0:
                    veh_id = active_call.vehicle_id if active_call else "veh-default"
                    self._dwell_timers[elevator_id] = self.door_dwell_s
                    target_dest = (
                        active_call.destination_floor_id if active_call else cabin.current_floor_id
                    )
                    self._update_cabin(
                        elevator_id,
                        state=ElevatorState.HOISTING,
                        target_floor_id=target_dest,
                        occupied_vehicle_id=veh_id,
                    )
                    events.append(
                        {
                            "event": "vehicle_boarded",
                            "elevator_id": elevator_id,
                            "vehicle_id": veh_id,
                        }
                    )
                else:
                    self._dwell_timers[elevator_id] = timer

            elif cabin.state == ElevatorState.HOISTING:
                target_floor = self._floors[cabin.target_floor_id or cabin.current_floor_id]
                target_z = target_floor.elevation_m
                diff = target_z - cabin.current_elevation_m
                step_dist = self.hoist_speed_mps * dt_s

                if abs(diff) <= step_dist:
                    new_z = target_z
                    self._dwell_timers[elevator_id] = self.door_dwell_s
                    self._update_cabin(
                        elevator_id,
                        state=ElevatorState.DOORS_OPENING,
                        current_elevation_m=new_z,
                        current_floor_id=target_floor.floor_id,
                        leveling_offset_m=0.003,
                    )
                    events.append(
                        {
                            "event": "arrived_at_destination",
                            "elevator_id": elevator_id,
                            "floor_id": target_floor.floor_id,
                        }
                    )
                else:
                    direction = 1.0 if diff > 0 else -1.0
                    new_z = cabin.current_elevation_m + direction * step_dist
                    self._update_cabin(elevator_id, current_elevation_m=new_z)

            elif cabin.state == ElevatorState.UNLOADING:
                timer -= dt_s
                if timer <= 0:
                    self._update_cabin(
                        elevator_id,
                        state=ElevatorState.IDLE,
                        occupied_vehicle_id=None,
                        target_floor_id=None,
                        leveling_offset_m=0.0,
                    )
                    if elevator_id in self._active_calls:
                        del self._active_calls[elevator_id]
                    self._total_cycles_completed += 1
                    events.append({"event": "cycle_completed", "elevator_id": elevator_id})
                else:
                    self._dwell_timers[elevator_id] = timer

        return events

    def get_cabin_state(self, elevator_id: str) -> ElevatorCabin:
        """Retrieve current state of a registered car hoist cabin."""
        if elevator_id not in self._cabins:
            raise KeyError(f"Elevator {elevator_id} not registered.")
        return self._cabins[elevator_id]

    def _dispatch_pending_calls(self) -> None:
        if not self._pending_calls:
            return

        idle_elevators = [
            eid for eid, cabin in self._cabins.items() if cabin.state == ElevatorState.IDLE
        ]

        while idle_elevators and self._pending_calls:
            call = self._pending_calls.pop(0)
            origin_z = self._floors[call.origin_floor_id].elevation_m

            best_elevator = min(
                idle_elevators,
                key=lambda eid: abs(self._cabins[eid].current_elevation_m - origin_z),
            )
            idle_elevators.remove(best_elevator)
            self._active_calls[best_elevator] = call

            cabin = self._cabins[best_elevator]
            if cabin.current_floor_id == call.origin_floor_id:
                self._dwell_timers[best_elevator] = self.door_dwell_s
                self._update_cabin(
                    best_elevator,
                    state=ElevatorState.DOORS_OPENING,
                    target_floor_id=call.origin_floor_id,
                )
            else:
                self._update_cabin(
                    best_elevator,
                    state=ElevatorState.TRANSITING,
                    target_floor_id=call.origin_floor_id,
                )

    def _update_cabin(self, elevator_id: str, **kwargs: Any) -> None:
        current = self._cabins[elevator_id]
        data = current.model_dump()
        data.update(kwargs)
        self._cabins[elevator_id] = ElevatorCabin.model_validate(data)
