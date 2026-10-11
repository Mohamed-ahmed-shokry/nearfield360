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
   - [ ] 8.1 `docs: update README with Phase 25 fleet coordination capabilities and CLI usage`
   - [ ] 8.2 `docs(progress): document completed Phase 25 milestone, Decision Log, and verification gates`
   - [ ] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
