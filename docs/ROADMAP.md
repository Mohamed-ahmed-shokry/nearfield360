# Autonomous Valet Parking (AVP) Development Roadmap

## Phase 25: Cloud Fleet Coordination, V2X Facility Dispatch & Multi-Agent Conflict Deconfliction

### 1. Objective & Expected Outcome

Deliver an end-to-end Cloud Fleet Coordination, V2X Facility Dispatch, and Multi-Agent Conflict Deconfliction subsystem for NearField360. Building directly upon Phase 24's HD Vector Mapping, Topological Route Planning, and Multi-Sensor Pose Graph Localization, Phase 25 connects individual autonomous vehicles to an intelligent facility management cloud. The system coordinates multi-vehicle fleet assignments, dispatches target bays dynamically based on facility occupancy status, synchronizes vehicle telematics over standardized V2I/V2V message schemas, predicts multi-agent spatiotemporal trajectory intersections at narrow aisles and intersections, and executes reservation-based priority scheduling and cooperative yield protocols to ensure deadlock-free facility throughput.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `FleetConfig` specifying heartbeat interval (`heartbeat_interval_s`), telematics timeout (`telematics_timeout_s`), time-space conflict buffer radius (`conflict_radius_m`), temporal headway margin (`temporal_headway_s`), max speed in conflict zones, and yield priority policy (`FIRST_COME_FIRST_SERVED`, `DISTANCE_TO_CONFLICT`, `MISSION_PRIORITY`).
- **Domain Models & V2X Protocols (`src/nearfield360/fleet/models.py`)**:
  - `V2XMessage` envelope with protocol version, sender ID, timestamp, and message type.
  - V2I/V2V payload schemas:
    - `VehicleHeartbeat`: Vehicle ID, timestamp, estimated pose $(x, y, \theta)$, linear velocity, current mission state, battery/fuel status, target slot ID.
    - `DispatchAssignment`: Assigned vehicle ID, assigned parking slot ID, assigned topological route ID/waypoints, expected ETA, dynamic priority level.
    - `FacilityOccupancyState`: Real-time map of vacant, occupied, reserved, and obstructed parking bays.
    - `YieldRequest` / `YieldGrant`: Negotiated right-of-way handshake at intersecting conflict points.
  - `FleetVehicleState`: Server-side representation of registered vehicles in the facility.
  - `SpatiotemporalWaypoint`: Waypoint $(x, y, \theta, t)$ defining expected arrival time window $[t_{\min}, t_{\max}]$.
  - `ConflictPoint`: Identified spatial collision zone between two vehicle trajectories with intersection point, conflicting vehicle IDs, arrival times, and resolution status.
  - `FleetDispatchReport`: Comprehensive facility telemetry and coordination report.
- **Dynamic Slot Dispatcher & Fleet Allocator (`src/nearfield360/fleet/dispatcher.py`)**:
  - Optimal slot assignment matching incoming drop-off vehicles with available facility bays (proximity, aisle congestion, slot geometry compatibility).
  - Dynamic re-assignment and lock reservation engine preventing race conditions when multiple vehicles target the same aisle.
- **Multi-Agent Spatiotemporal Deconfliction Engine (`src/nearfield360/fleet/deconfliction.py`)**:
  - Continuous trajectory sweeping detecting overlapping swept-volumes across time horizons.
  - Reservation-based intersection and aisle access control (Virtual Traffic Signal / Priority Token).
  - Speed-profile adjustment and virtual hold waypoints instructing approaching vehicles to decelerate or wait at pre-intersection staging areas until the corridor is cleared.
- **Fleet Simulator & Cloud Management Server (`src/nearfield360/fleet/server.py`)**:
  - In-process facility dispatch server simulating cloud-to-vehicle (V2I) communication, message queues, latency jitter, and packet loss tolerance.
  - Multi-agent discrete-time simulation step coordinating multiple concurrent AVP vehicles navigating through the facility simultaneously.
- **BEV Fleet Visualization & Coordination Dashboard (`src/nearfield360/fleet/viz.py`)**:
  - Multi-vehicle BEV layout rendering: each vehicle footprint color-coded by mission state and ID, planned routes, active conflict zones marked with warning halos, and slot reservation status.
  - Telemetry HUD showing fleet throughput, active conflicts resolved, average wait time, and facility capacity utilization.
- **CLI Command Group & Pipeline Integration (`src/nearfield360/cli/fleet.py`, `src/nearfield360/cli/pipeline.py`)**:
  - Dedicated CLI command group: `nearfield360 fleet` with `info`, `dispatch`, `simulate`, and `deconflict` subcommands.
  - Update release readiness audit for 17 CLI command groups.
  - Four-camera pipeline integration: `nearfield360 pipeline run --fleet-server <config>`.
- **Comprehensive Verification Suite**:
  - Unit and integration tests covering fleet config, V2X models, dispatcher, deconfliction algorithm, multi-vehicle simulation, visualization, CLI, and release audit.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `FleetConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for V2X protocols, heartbeats, dispatch assignments, and conflict points | `uv run pytest tests/unit/fleet/test_fleet_models.py` |
| AC-3 | `SlotDispatcher` allocates optimal bays, prevents double-booking, and handles dynamic reservation releases | `uv run pytest tests/unit/fleet/test_dispatcher.py` |
| AC-4 | `DeconflictionEngine` detects multi-agent spatiotemporal trajectory intersections and generates speed-profile adjustments / yield holds | `uv run pytest tests/unit/fleet/test_deconfliction.py` |
| AC-5 | `FleetCoordinator` simulates multi-vehicle concurrent navigation without deadlocks or collisions | `uv run pytest tests/unit/fleet/test_coordinator.py` |
| AC-6 | Multi-vehicle BEV visualization renders fleet positions, routes, conflict zones, and reservation HUD to valid PNG | `uv run pytest tests/unit/fleet/test_fleet_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 fleet` executes `info`, `dispatch`, `simulate`, and `deconflict` | `uv run pytest tests/unit/test_fleet_cli.py` |
| AC-8 | Root CLI registers `fleet` command group, passing release audit with 17 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates fleet dispatch options and exports multi-vehicle telemetry | `uv run pytest tests/unit/test_pipeline_fleet.py` |
| AC-10 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, dispatcher, deconfliction, coordinator, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 fleet dispatch`, `fleet simulate`, `fleet deconflict` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Public cellular 5G-NR modem AT commands and physical SIM authentication.
- Payment billing gateways and credit card processing for commercial parking garages.
- High-frequency RF channel propagation ray-tracing (stochastic network packet loss model used).

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Fleet Config Foundation**
   - [x] 1.1 `docs: establish Phase 25 fleet coordination and V2X dispatch roadmap`
   - [x] 1.2 `feat(config): add FleetConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for FleetConfig validation`
2. **Task 2: V2X Protocols & Fleet Domain Models**
   - [x] 2.1 `feat(fleet): implement V2X message schemas, heartbeats, and dispatch models`
   - [x] 2.2 `feat(fleet): implement spatiotemporal waypoints and conflict point models`
   - [x] 2.3 `test(fleet): verify fleet domain models, serialization, and validation`
3. **Task 3: Dynamic Slot Dispatcher & Reservation Manager**
   - [x] 3.1 `feat(fleet): implement dynamic slot dispatcher with distance/congestion scoring`
   - [x] 3.2 `feat(fleet): implement bay reservation lock and lease expiry engine`
   - [x] 3.3 `test(fleet): verify dispatcher allocation fairness, collision avoidance, and slot release`
4. **Task 4: Multi-Agent Spatiotemporal Deconfliction Engine**
   - [x] 4.1 `feat(fleet): implement trajectory swept-volume intersection and headway analysis`
   - [x] 4.2 `feat(fleet): implement virtual token priority scheduling and yield velocity profiling`
   - [x] 4.3 `test(fleet): verify conflict detection, headway compliance, and deadlock prevention`
5. **Task 5: Fleet Coordinator & Multi-Vehicle Simulation**
   - [x] 5.1 `feat(fleet): implement centralized fleet coordinator and discrete-time simulation step`
   - [x] 5.2 `feat(fleet): implement V2I communication channel simulation with latency and packet drop`
   - [x] 5.3 `test(fleet): verify multi-vehicle simulation throughput, collision-free execution, and KPIs`
6. **Task 6: Multi-Vehicle BEV Layout & Coordination Visualizer**
   - [x] 6.1 `feat(fleet): implement multi-vehicle BEV layout and route conflict rendering`
   - [x] 6.2 `feat(fleet): implement fleet telemetry dashboard with throughput and utilization charts`
   - [x] 6.3 `test(fleet): verify fleet visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports & CLI Command Group**
   - [x] 7.1 `feat(fleet): export public fleet API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 fleet command group (info, dispatch, simulate, deconflict)`
   - [x] 7.3 `feat(cli): register fleet CLI group and update release audit expected count to 17`
   - [x] 7.4 `feat(pipeline): integrate --fleet options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify fleet CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline fleet integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 25 fleet coordination capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 25 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 26: Autonomous EV Charging Dispatch, Power Grid Management & Robot/Automated Bay Scheduling

### 1. Objective & Expected Outcome

Deliver an end-to-end Autonomous EV Charging Dispatch, Facility Grid Power Management, and Automated Charging Bay Scheduling subsystem for NearField360. Building on Phase 24's HD Vector Mapping and Phase 25's Cloud Fleet Coordination, Phase 26 empowers smart AVP facilities to autonomously schedule electric vehicles (EVs) for charging sessions, model dynamic battery state-of-charge (SoC) profiles, respect facility transformer power limits through peak-shaving and dynamic load balancing, orchestrate multi-phase parking (regular bay → charging bay → relocation to staging bay upon target SoC), and simulate automated robotic arm / conductive underbody charging hardware.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `ChargingConfig` specifying total facility grid power capacity (`facility_grid_power_kw`), peak-shaving threshold (`peak_shaving_threshold_ratio`), charger station default specs (Level 2 AC vs DC Fast), default target SoC (`default_target_soc`), critical low SoC threshold (`critical_soc_threshold`), idle penalty tariff (`idle_penalty_per_min`), and robotic plug docking dwell time (`plug_dock_dwell_s`).
- **Domain Models & Charging Protocol (`src/nearfield360/charging/models.py`)**:
  - `ChargerType` (`AC_LEVEL_2`, `DC_FAST_50KW`, `DC_FAST_150KW`, `ROBOTIC_UNDERBODY_PDU`).
  - `ChargerPortStatus` (`AVAILABLE`, `OCCUPIED_CHARGING`, `OCCUPIED_COMPLETE_WAITING_RELOCATION`, `OUT_OF_SERVICE`).
  - `ChargerBay`: ID, slot ID reference, charger type, max power rating (kW), current power deliverable (kW), hardware status, efficiency rating ($0.0 - 1.0$).
  - `EVBatteryState`: Vehicle ID, battery capacity (kWh), current SoC ($0.0 - 1.0$), max acceptance rate (kW), target SoC, degradation factor.
  - `ChargingSession`: Session ID, vehicle ID, charger bay ID, start timestamp, completion timestamp, energy delivered (kWh), peak power (kW), status (`PENDING_DOCKING`, `ACTIVE_CHARGING`, `TOPPING_OFF`, `COMPLETED`, `IDLE_OVERSTAY`), estimated time to target SoC (ETA).
  - `FacilityPowerSummary`: Total capacity, instantaneous draw, available headroom, active sessions count, throttled sessions count, load factor.
  - `ChargingDispatchReport`: Full telemetry report of all bays, sessions, grid load, and relocation queues.
- **Dynamic Power Balancer & Peak Shaving Engine (`src/nearfield360/charging/grid.py`)**:
  - `GridPowerManager`: Real-time allocation of power across active charging sessions ensuring the instantaneous aggregate draw never exceeds `facility_grid_power_kw`. Proportional and priority-weighted curtailment (e.g. low SoC vehicles receive higher power shares during grid peak demand).
- **EV Charging Scheduler & Relocation Queue Manager (`src/nearfield360/charging/scheduler.py`)**:
  - `ChargingScheduler`: Prioritizes charging queue based on arrival urgency, current SoC, departure deadlines, and charger hardware compatibility.
  - Generates multi-phase AVP missions: dispatching vehicle to a charging bay, monitoring charge completion, and triggering automated relocation dispatches to free the charging bay for the next vehicle once target SoC is reached.
- **Automated Charging Hardware Simulation (`src/nearfield360/charging/hardware.py`)**:
  - Simulates non-linear CC/CV (Constant Current / Constant Voltage) battery charging curves, robotic plug mating/unmating sequences, connector safety handshakes, and fault protection.
- **BEV Charging Facility Visualizer & Telemetry Dashboard (`src/nearfield360/charging/viz.py`)**:
  - Dual-panel BEV layout rendering: left panel shows facility map with highlighted charging bays (color-coded by charger status: green=available, amber=charging, cyan=completed/relocating, red=faulted), active cable/plug lines to vehicle footprints, SoC gauge badges; right panel displays real-time grid power consumption curves, capacity limit line, individual vehicle SoC progress bars, and queue status.
- **CLI Command Group (`nearfield360 charging`) & Release Audit (`src/nearfield360/cli/charging.py`, `src/nearfield360/cli/release.py`)**:
  - Subcommands: `info` (inspect facility charging stations and power capacity), `schedule` (schedule vehicles for charging sessions), `simulate` (step multi-vehicle charging simulation with dynamic power throttling), and `grid` (inspect grid power status).
  - Register `charging` CLI command group in `app.py` and update release audit expected count from 17 to 18 groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--charging` flag on `nearfield360 pipeline run` to integrate charging session telemetry into the multi-camera perception pipeline report.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, power balancer, scheduling & relocation queue, hardware simulation, visualization, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `ChargingConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for charger bays, battery states, charging sessions, and grid summaries | `uv run pytest tests/unit/charging/test_charging_models.py` |
| AC-3 | `GridPowerManager` enforces peak shaving and dynamically balances power across sessions without exceeding grid limit | `uv run pytest tests/unit/charging/test_grid.py` |
| AC-4 | `ChargingScheduler` prioritizes queue, assigns bays, and issues relocation dispatches upon reaching target SoC | `uv run pytest tests/unit/charging/test_scheduler.py` |
| AC-5 | Charging hardware simulator models CC/CV charging curves and robotic connector docking dwell | `uv run pytest tests/unit/charging/test_hardware.py` |
| AC-6 | Dual-panel BEV visualizer renders charging bays, vehicle SoC badges, and grid power telemetry HUD to valid PNG | `uv run pytest tests/unit/charging/test_charging_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 charging` executes `info`, `schedule`, `simulate`, and `grid` | `uv run pytest tests/unit/test_charging_cli.py` |
| AC-8 | Root CLI registers `charging` command group, passing release audit with 18 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates `--charging` option and exports charging telemetry in report | `uv run pytest tests/unit/test_pipeline_charging.py` |
| AC-10 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, grid power manager, scheduler, hardware, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 charging schedule`, `charging simulate`, `charging grid` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- High-voltage physical electrical wiring diagrams and inverter hardware firmware flashing.
- ISO 15118 TLS certificate cryptographic chain verification with external cloud payment clearinghouses.
- Three-phase AC waveform harmonics and reactive power (kVAR) modeling (active power kW modeled).

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Charging Config Foundation**
   - [ ] 1.1 `docs: establish Phase 26 EV charging dispatch and power management roadmap`
   - [ ] 1.2 `feat(config): add ChargingConfig and update default yaml configuration`
   - [ ] 1.3 `test(config): add unit tests for ChargingConfig validation and environment overrides`
2. **Task 2: Charging Domain Models & Session Protocols**
   - [ ] 2.1 `feat(charging): implement charger bays, battery state, and charging session models`
   - [ ] 2.2 `feat(charging): implement facility power summary and dispatch report models`
   - [ ] 2.3 `test(charging): verify charging domain models, serialization, and validation`
3. **Task 3: Grid Power Manager & Dynamic Load Balancing**
   - [ ] 3.1 `feat(charging): implement grid power manager with dynamic load balancing and peak shaving`
   - [ ] 3.2 `test(charging): verify power curtailment, priority allocation, and grid limit compliance`
4. **Task 4: EV Charging Scheduler & Relocation Queue Manager**
   - [ ] 4.1 `feat(charging): implement charging queue scheduler with priority scoring and bay allocation`
   - [ ] 4.2 `feat(charging): implement charge completion monitoring and relocation queue generation`
   - [ ] 4.3 `test(charging): verify scheduling fairness, bay assignments, and relocation triggers`
5. **Task 5: Charging Hardware & Non-Linear CC/CV Battery Simulation**
   - [ ] 5.1 `feat(charging): implement non-linear CC/CV battery curve and robotic docking simulation`
   - [ ] 5.2 `test(charging): verify battery charging curves, docking dwell, and energy calculations`
6. **Task 6: BEV Charging Visualizer & Telemetry Dashboard**
   - [ ] 6.1 `feat(charging): implement BEV charging layout rendering with SoC badges and cable connections`
   - [ ] 6.2 `feat(charging): implement grid power curve and vehicle SoC progress telemetry charts`
   - [ ] 6.3 `test(charging): verify charging visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [ ] 7.1 `feat(charging): export public charging API from package init`
   - [ ] 7.2 `feat(cli): implement nearfield360 charging command group (info, schedule, simulate, grid)`
   - [ ] 7.3 `feat(cli): register charging CLI group and update release audit expected count to 18`
   - [ ] 7.4 `feat(pipeline): integrate --charging options into surround pipeline runner`
   - [ ] 7.5 `test(cli): verify charging CLI commands, JSON payloads, and visual outputs`
   - [ ] 7.6 `test(pipeline): verify pipeline charging integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [ ] 8.1 `docs: update README with Phase 26 EV charging dispatch capabilities and CLI usage`
   - [ ] 8.2 `docs(progress): document completed Phase 26 milestone, Decision Log, and verification gates`
   - [ ] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

