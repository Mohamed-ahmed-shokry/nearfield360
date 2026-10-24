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
   - [x] 1.1 `docs: establish Phase 26 EV charging dispatch and power management roadmap`
   - [x] 1.2 `feat(config): add ChargingConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for ChargingConfig validation and environment overrides`
2. **Task 2: Charging Domain Models & Session Protocols**
   - [x] 2.1 `feat(charging): implement charger bays, battery state, and charging session models`
   - [x] 2.2 `feat(charging): implement facility power summary and dispatch report models`
   - [x] 2.3 `test(charging): verify charging domain models, serialization, and validation`
3. **Task 3: Grid Power Manager & Dynamic Load Balancing**
   - [x] 3.1 `feat(charging): implement grid power manager with dynamic load balancing and peak shaving`
   - [x] 3.2 `test(charging): verify power curtailment, priority allocation, and grid limit compliance`
4. **Task 4: EV Charging Scheduler & Relocation Queue Manager**
   - [x] 4.1 `feat(charging): implement charging queue scheduler with priority scoring and bay allocation`
   - [x] 4.2 `feat(charging): implement charge completion monitoring and relocation queue generation`
   - [x] 4.3 `test(charging): verify scheduling fairness, bay assignments, and relocation triggers`
5. **Task 5: Charging Hardware & Non-Linear CC/CV Battery Simulation**
   - [x] 5.1 `feat(charging): implement non-linear CC/CV battery curve and robotic docking simulation`
   - [x] 5.2 `test(charging): verify battery charging curves, docking dwell, and energy calculations`
6. **Task 6: BEV Charging Visualizer & Telemetry Dashboard**
   - [x] 6.1 `feat(charging): implement BEV charging layout rendering with SoC badges and cable connections`
   - [x] 6.2 `feat(charging): implement grid power curve and vehicle SoC progress telemetry charts`
   - [x] 6.3 `test(charging): verify charging visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(charging): export public charging API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 charging command group (info, schedule, simulate, grid)`
   - [x] 7.3 `feat(cli): register charging CLI group and update release audit expected count to 18`
   - [x] 7.4 `feat(pipeline): integrate --charging options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify charging CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline charging integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 26 EV charging dispatch capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 26 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 27: Autonomous Valet Parking Smart Billing, Dynamic Tariffs & Cryptographic Session Receipts

### 1. Objective & Expected Outcome

Deliver an end-to-end Autonomous Valet Parking Smart Billing, Dynamic Tariffs, and Cryptographically Signed Session Receipts subsystem for NearField360. Building on Phase 24's HD Vector Mapping, Phase 25's Cloud Fleet Coordination, and Phase 26's EV Charging Dispatch, Phase 27 provides facilities with automated parking fee calculation, dynamic demand tariffs (congestion pricing and peak-hour multipliers), multi-component billing (parking dwell time, EV energy consumed, charging idle overstay fines, and priority valet premium fees), cryptographic SHA-256 HMAC session receipt signing for tamper-evident auditing, and exportable financial settlement reports.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `BillingConfig` specifying base parking hourly rate (`base_hourly_rate`), peak-hour congestion multiplier (`peak_multiplier`), energy tariff per kWh (`kwh_tariff`), charging idle overstay penalty per minute (`idle_overstay_per_min`), valet service flat fee (`valet_service_fee`), free grace period in minutes (`grace_period_minutes`), and receipt HMAC secret key (`receipt_hmac_secret`).
- **Domain Models & Billing Protocol (`src/nearfield360/billing/models.py`)**:
  - `TariffTier` (`STANDARD`, `PEAK_CONGESTION`, `EV_PREMIUM`, `OVERNIGHT_DISCOUNT`).
  - `PaymentStatus` (`PENDING`, `SETTLED`, `WAIVED`, `DISPUTED`).
  - `BillingItem`: Line item description, category (`PARKING_TIME`, `EV_ENERGY`, `IDLE_OVERSTAY`, `VALET_SERVICE`), quantity, unit price, total amount.
  - `ParkingReceipt`: Receipt ID, vehicle ID, entry timestamp, exit timestamp, duration minutes, itemized charges list, subtotal, tax, total charged, tariff tier, payment status, cryptographic HMAC signature, and verification digest.
  - `FinancialSettlementReport`: Total revenue, parking revenue, energy revenue, penalty revenue, transactions count, average session dwell, and settlement timestamp.
- **Dynamic Tariff & Pricing Engine (`src/nearfield360/billing/tariffs.py`)**:
  - `TariffEngine`: Calculates dynamic hourly rate based on facility occupancy load factor (dynamic congestion pricing) and time-of-day multipliers. Applies grace period deductions (sessions shorter than grace period incur zero parking charge).
- **Billing Calculator & Session Auditor (`src/nearfield360/billing/calculator.py`)**:
  - `BillingCalculator`: Combines parking dwell durations, EV energy delivered, and charger overstay dwell to compute itemized line items.
  - Generates verifiable `ParkingReceipt` with cryptographic SHA-256 HMAC signature using facility secret key.
  - `ReceiptVerifier`: Validates receipt integrity and detects tampering or forged billing amounts.
- **Financial Settlement & Facility Revenue Reporter (`src/nearfield360/billing/settlement.py`)**:
  - Aggregates multi-session receipts into `FinancialSettlementReport`.
  - Calculates occupancy-revenue correlation, average revenue per bay, and EV charging gross profit.
- **BEV Billing & Revenue Visualizer (`src/nearfield360/billing/viz.py`)**:
  - Dual-panel BEV layout rendering: left panel shows facility map with bays color-coded by revenue generation status and active billing badges; right panel displays revenue breakdown chart (parking vs energy vs penalty), tariff curve, and financial KPI summary.
- **CLI Command Group (`nearfield360 billing`) & Release Audit (`src/nearfield360/cli/billing.py`, `src/nearfield360/cli/release.py`)**:
  - Subcommands: `info` (inspect active tariff schedule and pricing policies), `calculate` (compute bill for completed session), `receipt` (generate signed receipt and verify cryptographic signature), `settle` (generate facility settlement report).
  - Register `billing` CLI command group in `app.py` and update release audit expected count from 18 to 19 groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--billing` flag on `nearfield360 pipeline run` to integrate billing calculations and receipt generation into perception pipeline summary.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, tariff engine, billing calculator, cryptographic receipt verification, settlement aggregator, visualizer, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `BillingConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for billing items, parking receipts, tariff tiers, and settlement reports | `uv run pytest tests/unit/billing/test_billing_models.py` |
| AC-3 | `TariffEngine` calculates dynamic pricing based on occupancy load and enforces grace period rules | `uv run pytest tests/unit/billing/test_tariffs.py` |
| AC-4 | `BillingCalculator` computes itemized parking, EV charging, and overstay charges accurately | `uv run pytest tests/unit/billing/test_calculator.py` |
| AC-5 | Cryptographic SHA-256 HMAC receipt signing and verification detects tampered receipt payloads | `uv run pytest tests/unit/billing/test_receipt_crypto.py` |
| AC-6 | `SettlementAggregator` generates comprehensive financial reports with revenue breakdown | `uv run pytest tests/unit/billing/test_settlement.py` |
| AC-7 | Dual-panel BEV visualizer renders facility revenue map and financial telemetry charts to PNG | `uv run pytest tests/unit/billing/test_billing_viz.py` |
| AC-8 | Dedicated CLI command group `nearfield360 billing` executes `info`, `calculate`, `receipt`, and `settle` | `uv run pytest tests/unit/test_billing_cli.py` |
| AC-9 | Root CLI registers `billing` command group, passing release audit with 19 groups | `uv run nearfield360 release audit` |
| AC-10 | Surround pipeline runner integrates `--billing` option and exports billing telemetry in report | `uv run pytest tests/unit/test_pipeline_billing.py` |
| AC-11 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, tariffs, calculator, crypto verification, settlement, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 billing calculate`, `billing receipt`, `billing settle` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Integration with external bank clearinghouses, payment gateways (Stripe/PayPal), or credit card PCI-DSS terminal hardware.
- Real-time tax authority online fiscal register submission protocols.
- Cash collection / coin vault mechatronics.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Billing Config Foundation**
   - [x] 1.1 `docs: establish Phase 27 AVP smart billing and cryptographic receipts roadmap`
   - [x] 1.2 `feat(config): add BillingConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for BillingConfig validation and environment overrides`
2. **Task 2: Billing Domain Models & Protocol Schemas**
   - [x] 2.1 `feat(billing): implement tariff tiers, billing items, and receipt models`
   - [x] 2.2 `feat(billing): implement financial settlement report models`
   - [x] 2.3 `test(billing): verify billing domain models, serialization, and validation`
3. **Task 3: Dynamic Tariff Engine & Congestion Pricing**
   - [x] 3.1 `feat(billing): implement dynamic tariff engine with occupancy load multipliers and grace period`
   - [x] 3.2 `test(billing): verify dynamic pricing tiers, peak multipliers, and grace period deduction`
4. **Task 4: Billing Calculator & Cryptographic Receipt Signing**
   - [x] 4.1 `feat(billing): implement itemized billing calculator for dwell, energy, and overstay fines`
   - [x] 4.2 `feat(billing): implement cryptographic SHA-256 HMAC receipt signing and verification`
   - [x] 4.3 `test(billing): verify billing calculation accuracy, HMAC signature generation, and tamper detection`
5. **Task 5: Financial Settlement Aggregator**
   - [x] 5.1 `feat(billing): implement financial settlement aggregator and revenue metrics calculator`
   - [x] 5.2 `test(billing): verify settlement aggregation, revenue categorization, and KPIs`
6. **Task 6: BEV Billing Visualizer & Telemetry Dashboard**
   - [x] 6.1 `feat(billing): implement BEV revenue map and bay billing badge visualizer`
   - [x] 6.2 `feat(billing): implement financial breakdown telemetry charts and KPI rendering`
   - [x] 6.3 `test(billing): verify billing visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(billing): export public billing API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 billing command group (info, calculate, receipt, settle)`
   - [x] 7.3 `feat(cli): register billing CLI group and update release audit expected count to 19`
   - [x] 7.4 `feat(pipeline): integrate --billing options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify billing CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline billing integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 27 smart billing capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 27 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 28: Multi-Tenant Facility Partitioning, Fleet Operator SLA Management & Reserved VIP Corridor Access

### 1. Objective & Expected Outcome

Deliver an end-to-end Multi-Tenant Facility Partitioning, Fleet Operator SLA Management, and Reserved VIP Corridor Access subsystem for NearField360. Building on Phase 24's HD Vector Mapping, Phase 25's Cloud Fleet Coordination, Phase 26's EV Charging Dispatch, and Phase 27's Smart Billing, Phase 28 enables autonomous valet parking garages to partition facilities into dedicated tenant zones (e.g. ride-hail fleets, car-sharing services, hotel guest VIPs, and public transient parkers). The system enforces operator quota allocations, monitors Service Level Agreements (SLAs: wait times, turnaround guarantees, breach penalties), and schedules exclusive VIP express corridors to grant prioritized vehicles right-of-way through congested facility bottlenecks.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `TenantConfig` specifying default tenant zone quota ratio (`default_quota_ratio`), SLA max dispatch wait time in seconds (`sla_max_dispatch_wait_s`), SLA breach penalty rate per minute (`sla_breach_penalty_per_min`), VIP priority weight multiplier (`vip_priority_multiplier`), express corridor preemption time buffer in seconds (`express_preemption_buffer_s`), and max concurrent tenant partitions (`max_tenant_partitions`).
- **Domain Models & Multi-Tenant Protocol (`src/nearfield360/tenant/models.py`)**:
  - `TenantTier` (`PUBLIC_TRANSIENT`, `COMMERCIAL_FLEET`, `CAR_SHARING`, `VIP_EXECUTIVE`).
  - `AccessPermission` (`STANDARD_ACCESS`, `PRIORITY_ACCESS`, `RESTRICTED_ACCESS`, `EXCLUSIVE_VIP`).
  - `TenantPartition`: Partition ID, tenant name, tier, assigned slot IDs, dedicated bay quota, current bay occupancy, active vehicle count, and access permissions.
  - `TenantSLAContract`: Contract ID, tenant ID, max allowed dispatch latency seconds, target bay availability ratio, breach penalty rate, and breach count.
  - `VIPExpressCorridor`: Corridor ID, designated lane IDs, active reservation vehicle ID, reservation start timestamp, reservation duration seconds, and preemption status.
  - `TenantPartitionReport`: Summary of all partitions, quota utilization, SLA compliance rates, breach penalties, and active VIP corridors.
- **Tenant Partitioning & Quota Enforcement Engine (`src/nearfield360/tenant/manager.py`)**:
  - `TenantManager`: Allocates and validates slot assignments respecting tenant zone boundaries. Blocks unauthorized vehicles from entering restricted tenant partitions. Dynamically recalculates quota usage and triggers overflow policies.
- **Fleet Operator SLA Monitor & Breach Auditor (`src/nearfield360/tenant/sla.py`)**:
  - `SLAMonitor`: Tracks vehicle dispatch request-to-arrival dwell times. Evaluates SLA contract compliance and calculates monetary breach penalties incurred by the facility when wait thresholds are exceeded.
- **VIP Express Corridor Reservation & Preemption Engine (`src/nearfield360/tenant/corridor.py`)**:
  - `VIPCorridorManager`: Reserves dedicated express driving lanes for high-priority VIP/commercial vehicles. Issues virtual preemption hold tokens to standard fleet vehicles approaching designated corridor intersections, clearing the route before the VIP vehicle arrives.
- **BEV Multi-Tenant Partition Visualizer (`src/nearfield360/tenant/viz.py`)**:
  - Dual-panel BEV layout rendering: left panel shows facility map with parking bays color-coded by tenant partition (e.g., green=Public, blue=Ride-Hail Fleet, purple=Car-Sharing, gold=VIP Executive), with glowing express corridor routes; right panel displays tenant quota gauge charts, SLA compliance gauges, and active corridor reservations.
- **CLI Command Group (`nearfield360 tenant`) & Release Audit (`src/nearfield360/cli/tenant.py`, `src/nearfield360/cli/release.py`)**:
  - Subcommands: `info` (inspect tenant partitions, quotas, and SLAs), `partition` (configure/assign bays to a tenant), `corridor` (reserve/preempt an express corridor), `audit` (evaluate SLA compliance and generate tenant report).
  - Register `tenant` CLI command group in `app.py` and update release audit expected count from 19 to 20 groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--tenant` flag on `nearfield360 pipeline run` to integrate tenant zone partition status and SLA metrics into perception pipeline summary.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, partition manager, SLA monitor, corridor preemption, visualizer, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `TenantConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for tenant partitions, SLA contracts, express corridors, and reports | `uv run pytest tests/unit/tenant/test_tenant_models.py` |
| AC-3 | `TenantManager` partitions bays, enforces quota constraints, and restricts unauthorized access | `uv run pytest tests/unit/tenant/test_manager.py` |
| AC-4 | `SLAMonitor` measures dispatch latency, tracks SLA compliance, and calculates breach penalties | `uv run pytest tests/unit/tenant/test_sla.py` |
| AC-5 | `VIPCorridorManager` reserves express lanes and preempts standard traffic at conflict points | `uv run pytest tests/unit/tenant/test_corridor.py` |
| AC-6 | Dual-panel BEV visualizer renders tenant partitions, VIP lanes, and SLA dashboard to PNG | `uv run pytest tests/unit/tenant/test_tenant_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 tenant` executes `info`, `partition`, `corridor`, and `audit` | `uv run pytest tests/unit/test_tenant_cli.py` |
| AC-8 | Root CLI registers `tenant` command group, passing release audit with 20 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates `--tenant` option and exports tenant telemetry in report | `uv run pytest tests/unit/test_pipeline_tenant.py` |
| AC-10 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, partition manager, SLA monitor, corridor manager, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 tenant partition`, `tenant corridor`, `tenant audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Multi-facility cross-city roaming federated identity management (SAML/OAuth2).
- Integration with municipal Department of Transportation public traffic lights.
- Biometric facial recognition at driver pickup lounge.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Tenant Config Foundation**
   - [x] 1.1 `docs: establish Phase 28 multi-tenant partitioning and SLA corridors roadmap`
   - [x] 1.2 `feat(config): add TenantConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for TenantConfig validation and environment overrides`
2. **Task 2: Tenant Domain Models & Partition Schemas**
   - [x] 2.1 `feat(tenant): implement tenant partitions, SLA contracts, and corridor models`
   - [x] 2.2 `feat(tenant): implement tenant partition summary and audit report models`
   - [x] 2.3 `test(tenant): verify tenant domain models, serialization, and validation`
3. **Task 3: Tenant Partitioning & Quota Enforcement Engine**
   - [x] 3.1 `feat(tenant): implement tenant partition manager with quota enforcement and access control`
   - [x] 3.2 `test(tenant): verify partition slot allocation, quota limits, and access rejection`
4. **Task 4: Fleet Operator SLA Monitor & Breach Auditor**
   - [x] 4.1 `feat(tenant): implement SLA monitor with dispatch latency tracking and penalty calculation`
   - [x] 4.2 `test(tenant): verify SLA compliance scoring, latency measurements, and breach penalties`
5. **Task 5: VIP Express Corridor Reservation & Preemption Engine**
   - [x] 5.1 `feat(tenant): implement VIP express corridor reservation and traffic preemption`
   - [x] 5.2 `test(tenant): verify corridor reservations, conflict clearance, and preemption holds`
6. **Task 6: BEV Multi-Tenant Partition Visualizer**
   - [x] 6.1 `feat(tenant): implement BEV tenant partition zone map and VIP corridor rendering`
   - [x] 6.2 `feat(tenant): implement quota utilization gauges and SLA telemetry HUD`
   - [x] 6.3 `test(tenant): verify tenant visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(tenant): export public tenant API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 tenant command group (info, partition, corridor, audit)`
   - [x] 7.3 `feat(cli): register tenant CLI group and update release audit expected count to 20`
   - [x] 7.4 `feat(pipeline): integrate --tenant options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify tenant CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline tenant integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 28 multi-tenant partitioning capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 28 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 29: Curbside Drop-off/Pick-up Staging, Dwell Management & Pedestrian Safety Interlocks

### 1. Objective & Expected Outcome

Deliver an end-to-end Curbside Drop-off & Pick-up (DOPU) Staging, Dwell Management, and Pedestrian Safety Interlock subsystem for NearField360. Building on Phase 24's HD Vector Mapping, Phase 25's Fleet Coordination, Phase 26's Charging Dispatch, Phase 27's Smart Billing, and Phase 28's Multi-Tenant Partitioning, Phase 29 implements the critical Human-Machine Interface Handover Zone (ISO 23374 AVP Type 2). The system manages dedicated curbside drop-off and pick-up staging bays, tracks driver/passenger egress and ingress, enforces safety interlocks that inhibit autonomous motion while doors/trunks are open or pedestrians are detected in the active clearance envelope, optimizes curbside dwell times, queues vehicles awaiting available pick-up bays, and visualizes curbside telemetry and safety interlock statuses.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `CurbsideConfig` specifying max drop-off dwell time (`max_dropoff_dwell_s`), max pick-up boarding dwell time (`max_pickup_dwell_s`), pedestrian critical safety radius (`pedestrian_safety_radius_m`), interlock clearance hold debounce duration (`interlock_clearance_hold_s`), curbside staging bays capacity (`staging_capacity`), and overstay penalty tariff (`overstay_penalty_per_min`).
- **Domain Models & Handover Schemas (`src/nearfield360/curbside/models.py`)**:
  - `HandoverType`: Enum representing `DROPOFF` and `PICKUP`.
  - `BayStatus`: Enum representing `VACANT`, `OCCUPIED_UNLOADING`, `OCCUPIED_BOARDING`, `HANDOVER_READY`, and `CLEARING`.
  - `SafetyInterlockStatus`: Enum representing `ENGAGED_DOOR_OPEN`, `ENGAGED_PEDESTRIAN_PROXIMITY`, `ENGAGED_TRUNK_OPEN`, and `DISENGAGED_CLEAR`.
  - `CurbsideBay`: ID, bay type, status, occupied vehicle ID, occupancy start timestamp, current interlock state.
  - `CurbsideMission`: Mission ID, vehicle ID, handover type, assigned bay ID, requested at, completed at, dwell seconds, interlock events count, overstay penalty.
  - `PedestrianDetection`: ID, distance_m, relative_bearing_rad, in_critical_zone: bool.
  - `CurbsideReport`: Comprehensive facility curbside handover audit report.
- **Pedestrian Safety Interlock Engine (`src/nearfield360/curbside/safety.py`)**:
  - `SafetyInterlockEngine`: Enforces safety invariants inhibiting vehicle motion (`is_motion_permitted == False`) when doors/trunks are unlatched or pedestrians encroach within `pedestrian_safety_radius_m`.
  - Implements temporal debounce clearance holding ensuring uninterrupted clearance for `interlock_clearance_hold_s` before disengaging interlocks.
- **Curbside Staging & Dwell Queue Manager (`src/nearfield360/curbside/manager.py`)**:
  - `CurbsideManager`: Coordinates arrival staging, drop-off handover lifecycle, passenger pick-up summon queue, bay assignment, and overstay penalty calculations.
- **BEV Curbside & Safety HUD Visualizer (`src/nearfield360/curbside/viz.py`)**:
  - Dual-panel BEV layout rendering curbside staging bays, vehicle footprints, highlighted pedestrian safety envelope circles (green=clear, red=interlocked), passenger ingress/egress markers, and right-panel curbside queue and safety HUD.
- **CLI Command Group (`nearfield360 curbside`) & Release Audit (`src/nearfield360/cli/curbside.py`, `src/nearfield360/cli/release.py`)**:
  - Subcommands: `info` (inspect curbside thresholds and bay capacity), `dropoff` (simulate vehicle arrival, passenger egress, and autonomous handover), `summon` (simulate summon request, pick-up bay dispatch, boarding, and departure), and `audit` (facility curbside performance audit).
  - Register `curbside` in `app.py` and update release audit expected count from 20 to 21 groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--curbside` flag on `nearfield360 pipeline run` to integrate curbside handover telemetry into the surround perception report.
- **Comprehensive Verification Suite**:
  - Unit and integration tests covering curbside config, models, safety interlock engine, curbside manager, visualizer, CLI, and pipeline runner.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `CurbsideConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for curbside bays, missions, pedestrian detections, and audit reports | `uv run pytest tests/unit/curbside/test_curbside_models.py` |
| AC-3 | `SafetyInterlockEngine` inhibits motion when doors/trunks open or pedestrians encroach, requiring clearance hold | `uv run pytest tests/unit/curbside/test_safety.py` |
| AC-4 | `CurbsideManager` allocates staging bays, handles drop-off/pick-up lifecycle, and calculates dwell overstay | `uv run pytest tests/unit/curbside/test_manager.py` |
| AC-5 | Dual-panel BEV visualizer renders curbside bays, safety circles, and queue HUD to PNG | `uv run pytest tests/unit/curbside/test_curbside_viz.py` |
| AC-6 | Dedicated CLI command group `nearfield360 curbside` executes `info`, `dropoff`, `summon`, and `audit` | `uv run pytest tests/unit/test_curbside_cli.py` |
| AC-7 | Root CLI registers `curbside` command group, passing release audit with 21 groups | `uv run nearfield360 release audit` |
| AC-8 | Surround pipeline runner integrates `--curbside` option and exports curbside telemetry in report | `uv run pytest tests/unit/test_pipeline_curbside.py` |
| AC-9 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, safety interlock engine, curbside manager, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 curbside dropoff`, `curbside summon`, `curbside audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- High-voltage overhead robotic pantograph charging inside curbside passenger bays.
- Automated passenger luggage conveyor belt weighing integration.
- Public road municipal traffic ticketing integration.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Curbside Config Foundation**
   - [x] 1.1 `docs: establish Phase 29 curbside staging and pedestrian safety interlocks roadmap`
   - [x] 1.2 `feat(config): add CurbsideConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for CurbsideConfig validation and environment overrides`
2. **Task 2: Curbside Domain Models & Handover Schemas**
   - [x] 2.1 `feat(curbside): implement curbside bays, missions, pedestrian, and interlock models`
   - [x] 2.2 `feat(curbside): implement curbside audit report models`
   - [x] 2.3 `test(curbside): verify curbside domain models, serialization, and validation`
3. **Task 3: Pedestrian Safety Interlock Engine**
   - [x] 3.1 `feat(curbside): implement pedestrian safety interlock engine with clearance debounce`
   - [x] 3.2 `test(curbside): verify interlock inhibition on door/trunk open and pedestrian proximity`
4. **Task 4: Curbside Staging & Dwell Queue Manager**
   - [x] 4.1 `feat(curbside): implement curbside staging manager with dwell tracking and overstay penalties`
   - [x] 4.2 `test(curbside): verify dropoff/pickup allocation, staging queue, and overstay calculation`
5. **Task 5: BEV Curbside Staging & Safety HUD Visualizer**
   - [x] 5.1 `feat(curbside): implement BEV curbside bay map, pedestrian safety circles, and queue HUD`
   - [x] 5.2 `test(curbside): verify curbside visualizer exports valid PNG artifacts`
6. **Task 6: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 6.1 `feat(curbside): export public curbside API from package init`
   - [x] 6.2 `feat(cli): implement nearfield360 curbside command group (info, dropoff, summon, audit)`
   - [x] 6.3 `feat(cli): register curbside CLI group and update release audit expected count to 21`
   - [x] 6.4 `feat(pipeline): integrate --curbside options into surround pipeline runner`
   - [x] 6.5 `test(cli): verify curbside CLI commands, JSON payloads, and visual outputs`
   - [x] 6.6 `test(pipeline): verify pipeline curbside integration workflow`
7. **Task 7: End-to-End Documentation, Self-Review & Delivery**
   - [x] 7.1 `docs: update README with Phase 29 curbside staging capabilities and CLI usage`
   - [x] 7.2 `docs(progress): document completed Phase 29 milestone, Decision Log, and verification gates`
   - [x] 7.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 30: Remote Teleoperation Assistance, Video Fallback Streaming & Emergency Minimum Risk Maneuver (MRM) Execution

### 1. Objective & Expected Outcome

Deliver an end-to-end Remote Teleoperation Assistance, Latency-Adaptive Video Fallback Streaming, and Emergency Minimum Risk Maneuver (MRM) Execution subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3) and SAE J3016 operational safety concepts, Phase 30 provides the critical remote human fallback layer when autonomous vehicles encounter edge-case blockages, perception ambiguities, or communication degradation. The system coordinates remote operator assistance requests, modes of intervention (`TELE_ASSISTANCE`, `TELE_GUIDANCE`, `DIRECT_DRIVING`), adaptive video fallback encoding/frame-rate throttling based on real-time RTT jitter and packet loss, a fail-safe heartbeat watchdog, and deterministic autonomous execution of Minimum Risk Maneuvers (controlled deceleration, emergency safe stopping in lane or pulling to shoulder, hazard beacon actuation, and failsafe brake locking).

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `TeleopConfig` specifying `heartbeat_timeout_s`, `max_rtt_ms`, `video_target_fps`, `video_fallback_fps`, `mrm_deceleration_mps2`, `safe_stop_standstill_speed_mps`, and `auto_mrm_on_heartbeat_loss`.
- **Domain Models & Teleoperation Protocol (`src/nearfield360/teleop/models.py`)**:
  - `TeleopMode`: Enum representing `IDLE_AUTONOMOUS`, `TELE_ASSISTANCE`, `TELE_GUIDANCE`, `DIRECT_DRIVING`, and `MRM_SAFE_STOP`.
  - `MRMStage`: Enum representing `INACTIVE`, `DECELERATING`, `PULLING_TO_SHOULDER`, `STANDSTILL_LOCKED`, and `EMERGENCY_BEACON`.
  - `NetworkQuality`: Enum representing `OPTIMAL`, `DEGRADED`, `CRITICAL`, and `SEVERED`.
  - `TeleopCommand`: Operator steering angle, speed, emergency stop override, waypoint suggestions, timestamp.
  - `TeleopHeartbeat`: Vehicle ID, timestamp, sequence number, round trip time (RTT), network quality.
  - `TeleopSession`: Session ID, vehicle ID, operator ID, active mode, start timestamp, commands received count, current MRM state.
  - `TeleopReport`: Facility-wide teleoperation events, interventions count, average RTT, and MRM executions summary.
- **Bandwidth-Adaptive Video Fallback & Network Monitor (`src/nearfield360/teleop/network.py`)**:
  - `NetworkMonitor`: Tracks packet sequence numbers, RTT exponential moving average (EMA), jitter, and packet loss ratio. Classifies network quality (`OPTIMAL` < 100ms, `DEGRADED` 100-250ms, `CRITICAL` 250-500ms, `SEVERED` > 500ms or timeout).
  - Provides dynamic video throttling recommendations (target FPS, compression scale factor).
- **Minimum Risk Maneuver (MRM) Safe Stop Engine (`src/nearfield360/teleop/mrm.py`)**:
  - `MRMEngine`: Finite state machine orchestrating safe transitions:
    - Detects trigger: heartbeat timeout, network severance, critical sensor failure, or explicit operator E-stop.
    - Computes safe deceleration profile to zero velocity.
    - Engages parking brake lock, disables traction motor, activates emergency hazard beacon, and emits distress event.
- **Remote Assistance Executive & Session Coordinator (`src/nearfield360/teleop/manager.py`)**:
  - `TeleopManager`: Manages active teleoperation sessions across facility vehicles, processes incoming commands, enforces safety checks (rejects direct steering if RTT > `max_rtt_ms`), routes guidance waypoints, and evaluates session termination.
- **BEV Teleoperation Cockpit & MRM Dashboard (`src/nearfield360/teleop/viz.py`)**:
  - `render_teleop_dashboard`: Dual-panel cockpit view:
    - Left panel: BEV map with vehicle footprint, suggested remote path overlay, MRM safe stop trajectory, hazard envelope warning icons.
    - Right panel: Operator telemetry cockpit HUD with real-time RTT gauge, network quality indicator, current teleoperation mode badge, and MRM status.
- **Package Exports & CLI Command Group (`src/nearfield360/teleop/__init__.py`, `src/nearfield360/cli/teleop.py`)**:
  - Dedicated CLI command group: `nearfield360 teleop` with `info`, `session`, `fallback`, and `audit` subcommands.
  - Root CLI registration (`app.py`).
  - Update release readiness audit for 22 CLI command groups (`release.py`).
  - Four-camera pipeline integration: `nearfield360 pipeline run --teleop`.
- **Comprehensive Verification Suite**:
  - Unit and integration tests covering config, models, network monitor, MRM engine, session manager, visualization, CLI, release audit, and pipeline runner.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `TeleopConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for teleoperation sessions, heartbeats, network quality, commands, and MRM states | `uv run pytest tests/unit/teleop/test_teleop_models.py` |
| AC-3 | `NetworkMonitor` evaluates RTT, jitter, packet loss, and throttles streaming frame rate | `uv run pytest tests/unit/teleop/test_network.py` |
| AC-4 | `MRMEngine` executes deterministic safe stop upon heartbeat timeout or emergency trigger | `uv run pytest tests/unit/teleop/test_mrm.py` |
| AC-5 | `TeleopManager` handles remote assistance lifecycle, guidance waypoint routing, and safety bounds | `uv run pytest tests/unit/teleop/test_teleop_manager.py` |
| AC-6 | Dual-panel BEV cockpit visualizer renders remote path, MRM stop point, and HUD gauges to PNG | `uv run pytest tests/unit/teleop/test_teleop_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 teleop` executes `info`, `session`, `fallback`, and `audit` | `uv run pytest tests/unit/test_teleop_cli.py` |
| AC-8 | Root CLI registers `teleop` command group, passing release audit with 22 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates teleoperation options and exports fallback telemetry | `uv run pytest tests/unit/test_pipeline_teleop.py` |
| AC-10 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, network monitor, MRM engine, manager, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 teleop session`, `teleop fallback`, `teleop audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Real WebRTC / SIP media server infrastructure (discrete frame packet simulation is used).
- Physical CAN-bus steer-by-wire transceiver hardware protocols.
- Regulatory cellular spectrum licensing negotiations.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Teleoperation Config Foundation**
   - [x] 1.1 `docs: establish Phase 30 teleoperation assistance and MRM roadmap`
   - [x] 1.2 `feat(config): add TeleopConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for TeleopConfig validation and environment overrides`
2. **Task 2: Teleoperation Domain Models & Protocol Schemas**
   - [x] 2.1 `feat(teleop): implement teleoperation mode, network quality, heartbeat, and command models`
   - [x] 2.2 `feat(teleop): implement teleoperation session and facility audit report models`
   - [x] 2.3 `test(teleop): verify teleoperation domain models, serialization, and validation`
3. **Task 3: Bandwidth-Adaptive Network Monitor & Fallback Throttling**
   - [x] 3.1 `feat(teleop): implement network quality monitoring, RTT tracking, and video throttling`
   - [x] 3.2 `test(teleop): verify network quality classification and bitrate/fps degradation`
4. **Task 4: Minimum Risk Maneuver (MRM) Safe Stop Engine**
   - [x] 4.1 `feat(teleop): implement MRM state machine, deceleration profile, and brake locking`
   - [x] 4.2 `test(teleop): verify MRM execution on heartbeat timeout, network loss, and emergency stop`
5. **Task 5: Teleoperation Manager & Session Coordinator**
   - [x] 5.1 `feat(teleop): implement teleoperation session manager and guidance waypoint dispatcher`
   - [x] 5.2 `test(teleop): verify session lifecycle, direct driving latency safety rejection, and audit reporting`
6. **Task 6: BEV Teleoperation Cockpit & Telemetry HUD Visualizer**
   - [x] 6.1 `feat(teleop): implement BEV cockpit view, remote guidance path, and MRM HUD rendering`
   - [x] 6.2 `test(teleop): verify teleop visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(teleop): export public teleop API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 teleop command group (info, session, fallback, audit)`
   - [x] 7.3 `feat(cli): register teleop CLI group and update release audit expected count to 22`
   - [x] 7.4 `feat(pipeline): integrate --teleop options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify teleop CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline teleop integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 30 teleoperation capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 30 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 31: Multi-Floor Facility Topology, Spiral Ramp Kinematics, Steep Incline Pitch Dynamics & Automated Vehicle Elevator Dispatch (ISO 23374 Vertical Transit)

### 1. Objective & Expected Outcome

Deliver an end-to-end Multi-Floor Facility Topology, Spiral & Straight Ramp Kinematics, Steep Incline Pitch Dynamics, and Automated Vehicle Elevator Dispatch subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3) multi-level facility navigation standards, Phase 31 equips automated vehicles with the capability to transit vertically between multiple garage floors (subterranean basements, ground floors, multi-tier elevated decks, and rooftops). The system models 3D multi-floor topological connectivity (`FloorGraph`, `FloorLevel`, `VerticalPortal`), calculates longitudinal gravity resistance and reduced tire adhesion on steep inclines ($F_g = mg \sin\theta$), validates chassis approach/departure/ramp-breakover clearances, arbitrates bidirectional single-lane ramp traffic tokens to prevent mid-ramp deadlocks, orchestrates automated car elevator / vehicle hoist lift sequencing with leveling safety interlocks, renders multi-floor stacked BEV floorplans and isometric ramp elevation dashboards, and exposes dedicated CLI commands with full surround perception pipeline integration.

### 2. Scope

- **Configuration (`MultiFloorConfig` in `src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `default_floor_height_m`: Default vertical distance between facility floors (e.g. 3.5 m).
  - `max_ramp_grade_pct`: Maximum safe ramp grade allowed for autonomous driving (e.g. 15.0%).
  - `min_breakover_clearance_m`: Minimum underbody chassis breakover clearance buffer (e.g. 0.05 m).
  - `ramp_speed_limit_mps`: Speed limit enforced while traversing inclined/spiral ramps (e.g. 2.5 m/s).
  - `elevator_hoist_speed_mps`: Vertical elevator ascent/descent velocity (e.g. 1.2 m/s).
  - `elevator_door_dwell_s`: Elevator door opening and closing safety duration (e.g. 5.0 s).
  - `elevator_leveling_tolerance_m`: Platform vertical leveling tolerance threshold (e.g. 0.015 m).
  - `ramp_token_timeout_s`: Single-lane ramp reservation lease timeout (e.g. 30.0 s).
- **Domain Models & Schemas (`src/nearfield360/multilevel/models.py`)**:
  - `PortalType` Enum: `STRAIGHT_RAMP`, `SPIRAL_RAMP`, `VEHICLE_ELEVATOR`.
  - `ElevatorState` Enum: `IDLE`, `CALLED`, `TRANSITING`, `DOORS_OPENING`, `LOADING`, `LEVELING`, `HOISTING`, `UNLOADING`, `HOLD`.
  - `RampDirection` Enum: `ASCENDING`, `DESCENDING`, `BIDIRECTIONAL`.
  - `RampTokenPriority` Enum: `ASCENDING_FIRST`, `DESCENDING_FIRST`, `FIFO`.
  - `FloorLevel`: `floor_id`, `name`, `elevation_m`, `bays_count`, `available_bays`, `is_rooftop`, `is_basement`.
  - `VerticalPortal`: `portal_id`, `portal_type`, `from_floor_id`, `to_floor_id`, `length_m`, `grade_pct`, `curvature_radius_m`, `is_bidirectional`, `speed_limit_mps`.
  - `VehiclePitchState`: `vehicle_id`, `pitch_angle_deg`, `grade_pct`, `elevation_m`, `gravity_force_n`, `normal_force_n`, `traction_margin`, `breakover_clearance_m`, `is_clearance_safe`.
  - `ElevatorCabin`: `elevator_id`, `current_floor_id`, `target_floor_id`, `state`, `current_elevation_m`, `max_weight_kg`, `occupied_vehicle_id`, `leveling_offset_m`.
  - `MultiFloorRoute`: `route_id`, `vehicle_id`, `origin_floor_id`, `destination_floor_id`, `portals_sequence`, `waypoints_3d`, `total_distance_m`, `estimated_transit_time_s`.
  - `MultiFloorFacilityReport`: `facility_name`, `floors_count`, `active_vehicles_per_floor`, `ramp_token_events`, `elevator_cycles_count`, `average_vertical_transit_time_s`.
- **Multi-Floor Topological Router & Graph Builder (`src/nearfield360/multilevel/graph.py`)**:
  - Directed multi-layer topological graph connecting floor waypoints and vertical portal links.
  - Multi-floor routing engine calculating lowest-cost 3D vertical routes across ramps and elevators, factoring in transit times and grade penalties.
- **Ramp Kinematics, Pitch Dynamics & Chassis Breakover Checker (`src/nearfield360/multilevel/ramp.py`)**:
  - `RampDynamicsEngine`: Straight ramp slope dynamics, gravity forces, normal force reduction, and traction coefficients.
  - Spiral ramp helical kinematics: radius $R$, yaw rate $\omega = v / R$, centripetal acceleration, and vertical climb velocity.
  - Chassis breakover angle verification based on wheelbase ($L$), ground clearance ($h$), and ramp crest angle $\Delta \theta$ to prevent underbody grounding.
- **Single-Lane Bidirectional Ramp Token Arbiter (`src/nearfield360/multilevel/arbiter.py`)**:
  - `RampTokenArbiter`: Mutual exclusion locking for single-lane ramps.
  - Directional priority arbitration (`ASCENDING_FIRST` to prevent rollback/stall) and lease timeout auto-release.
- **Automated Vehicle Elevator Dispatcher & Leveling Interlocks (`src/nearfield360/multilevel/elevator.py`)**:
  - `VehicleElevatorDispatcher`: Manages call queues, floor scheduling, door cycles, weight limits, vertical hoisting velocity profiles, and platform leveling precision verification ($\le 15$ mm) before granting vehicle ingress/egress.
- **Multi-Floor BEV & Isometric Vertical Transit Visualizer (`src/nearfield360/multilevel/viz.py`)**:
  - `render_multilevel_dashboard`: Dual-panel / multi-floor visualizer:
    - Left panel: Multi-layer stacked BEV floorplans (e.g. Level 1 and Level 2) with floor transitions, ramp entries/exits, active vehicle positions with elevation tags ($z$), and elevator shafts.
    - Right panel: Isometric vertical elevation profile showing ramp slope cross-section, vehicle pitch angle, elevator hoist shaft with real-time cabin height badge, and token status HUD.
- **Package Exports & Dedicated CLI Command Group (`src/nearfield360/multilevel/__init__.py`, `src/nearfield360/cli/floor.py`)**:
  - Subcommands: `info`, `route`, `ramp`, `elevator`, `audit`.
  - Register `floor` CLI group in `src/nearfield360/cli/app.py`.
  - Update release audit in `src/nearfield360/cli/release.py` to expect 23 command groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--floor` flag on `nearfield360 pipeline run` to integrate vertical transit and multi-floor facility metrics into perception pipeline summary.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, graph router, ramp dynamics, arbiter, elevator dispatcher, viz, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `MultiFloorConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for floors, vertical portals, pitch states, elevator cabins, routes, and facility reports | `uv run pytest tests/unit/multilevel/test_multilevel_models.py` |
| AC-3 | `MultiFloorGraph` models 3D layer topology and routes optimal vertical paths across ramps and elevators | `uv run pytest tests/unit/multilevel/test_graph.py` |
| AC-4 | `RampDynamicsEngine` calculates incline pitch dynamics, gravity forces, traction requirements, and breakover clearance | `uv run pytest tests/unit/multilevel/test_ramp.py` |
| AC-5 | `RampTokenArbiter` enforces single-lane mutual exclusion, priority arbitration (ascending first), and lease timeouts | `uv run pytest tests/unit/multilevel/test_arbiter.py` |
| AC-6 | `VehicleElevatorDispatcher` coordinates call queues, door cycles, weight limits, hoist profiles, and leveling safety interlocks | `uv run pytest tests/unit/multilevel/test_elevator.py` |
| AC-7 | Dual-panel BEV visualizer renders stacked floorplans, vertical elevation cross-sections, and elevator HUD to valid PNG | `uv run pytest tests/unit/multilevel/test_multilevel_viz.py` |
| AC-8 | Dedicated CLI command group `nearfield360 floor` executes `info`, `route`, `ramp`, `elevator`, and `audit` | `uv run pytest tests/unit/test_floor_cli.py` |
| AC-9 | Root CLI registers `floor` command group, passing release audit with 23 groups | `uv run nearfield360 release audit` |
| AC-10 | Surround pipeline runner integrates `--floor` option and exports multi-floor vertical transit telemetry in report | `uv run pytest tests/unit/test_pipeline_floor.py` |
| AC-11 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, graph router, ramp dynamics, arbiter, elevator dispatcher, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 floor route`, `floor ramp`, `floor elevator`, `floor audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Hydraulic fluid pressure sensor calibration and physical hoist winch motor microcode.
- Building structural seismic damper finite element analysis.
- Public utility municipal 480V substation transformer power supply redundancy.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Multi-Floor Configuration Foundation**
   - [x] 1.1 `docs(roadmap): establish Phase 31 multi-floor topology, ramp kinematics and elevator roadmap`
   - [x] 1.2 `feat(config): add MultiFloorConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for MultiFloorConfig validation and environment overrides`
2. **Task 2: Multi-Floor Domain Models & Protocol Schemas**
   - [x] 2.1 `feat(multilevel): implement floor levels, vertical portals, and pitch state models`
   - [x] 2.2 `feat(multilevel): implement elevator cabins, routes, and facility report models`
   - [x] 2.3 `test(multilevel): verify multi-floor domain models, serialization, and validation`
3. **Task 3: Multi-Floor Topological Router & Graph Builder**
   - [x] 3.1 `feat(multilevel): implement multi-floor directed graph and 3D portal routing`
   - [x] 3.2 `test(multilevel): verify multi-floor graph construction, shortest path search, and transit times`
4. **Task 4: Ramp Kinematics, Pitch Dynamics & Chassis Breakover Checker**
   - [x] 4.1 `feat(multilevel): implement ramp grade dynamics, spiral kinematics, and breakover clearance`
   - [x] 4.2 `test(multilevel): verify steep incline pitch physics, traction limits, and chassis scraping checks`
5. **Task 5: Single-Lane Ramp Token Arbiter & Conflict Resolution**
   - [x] 5.1 `feat(multilevel): implement ramp token arbiter with ascending priority and deadlock prevention`
   - [x] 5.2 `test(multilevel): verify ramp mutual exclusion, priority token allocation, and timeout release`
6. **Task 6: Automated Vehicle Elevator Dispatcher & Leveling Interlocks**
   - [x] 6.1 `feat(multilevel): implement vehicle elevator state machine, dispatch queue, and leveling interlocks`
   - [x] 6.2 `test(multilevel): verify elevator scheduling, hoist velocity profiling, and safety interlock gates`
7. **Task 7: Multi-Floor BEV & Vertical Transit Dashboard Visualizer**
   - [x] 7.1 `feat(multilevel): implement multi-floor BEV floorplans, elevation profile, and telemetry HUD`
   - [x] 7.2 `test(multilevel): verify multi-floor visualizer exports valid PNG artifacts`
8. **Task 8: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 8.1 `feat(multilevel): export public multilevel API from package init`
   - [x] 8.2 `feat(cli): implement nearfield360 floor command group (info, route, ramp, elevator, audit)`
   - [x] 8.3 `feat(cli): register floor CLI group and update release audit expected count to 23`
   - [x] 8.4 `feat(pipeline): integrate --floor options into surround pipeline runner`
   - [x] 8.5 `test(cli): verify floor CLI commands, JSON payloads, and visual outputs`
   - [x] 8.6 `test(pipeline): verify pipeline floor integration workflow`
9. **Task 9: End-to-End Documentation, Self-Review & Delivery**
   - [x] 9.1 `docs: update README with Phase 31 multi-floor capabilities and CLI usage`
   - [x] 9.2 `docs(progress): document completed Phase 31 milestone, Decision Log, and verification gates`
   - [x] 9.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 32: Emergency Vehicle Preemption (EVP), Facility Fire Evacuation, Dynamic Hazard Containment & Life Safety Dispatch (ISO 23374 / NFPA 88A)

### 1. Objective & Expected Outcome

Deliver an end-to-end Emergency Vehicle Preemption (EVP), Facility Fire Evacuation, Dynamic Hazard Containment, and Life Safety Dispatch subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3), NFPA 88A (Standard for Parking Structures), and UL 3300 operational safety standards, Phase 32 equips autonomous parking facilities with life-safety response capabilities during facility emergencies. The system manages emergency vehicle preemption (granting immediate right-of-way and clearing corridors for arriving fire engines, ambulances, and police), dynamically geofences hazard containment perimeters (e.g., EV lithium-ion battery thermal runaway fires, gas leaks, structural blockages), enforces fire door / smoke shutter interlock clearances to prevent vehicles from becoming trapped, coordinates prioritized multi-wave evacuation routing to unobstructed egress exits, renders tactical first-responder BEV situational dashboards, and exposes dedicated CLI commands with full surround perception pipeline integration.

### 2. Scope

- **Configuration (`EmergencyConfig` in `src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `preemption_radius_m`: Spatial radius around approaching emergency vehicles where automated valet parking vehicles must yield or pull into refuge bays (e.g. 35.0 m).
  - `hazard_buffer_radius_m`: Minimum clearance buffer distance maintained around active hazard perimeters (e.g. 12.0 m).
  - `evacuation_speed_limit_mps`: Speed limit enforced during emergency evacuation transit (e.g. 4.0 m/s).
  - `fire_door_clearance_timeout_s`: Maximum allowable transit duration before descending fire curtains / rolling shutters seal a compartment (e.g. 15.0 s).
  - `stagger_interval_s`: Temporal delay between departing evacuation waves to prevent egress bottleneck gridlock (e.g. 3.0 s).
  - `emergency_heartbeat_timeout_s`: Failsafe watchdog heartbeat timeout during emergency operations (e.g. 1.0 s).
  - `thermal_runaway_temp_c`: Battery pack temperature threshold triggering automatic EV thermal runaway containment (e.g. 70.0 °C).
- **Domain Models & Life Safety Protocols (`src/nearfield360/emergency/models.py`)**:
  - `EmergencyType` Enum: `FIRE_ALARM`, `EV_THERMAL_RUNAWAY`, `GAS_LEAK`, `STRUCTURAL_HAZARD`, `EMERGENCY_VEHICLE_APPROACH`.
  - `EmergencySeverity` Enum: `ADVISORY`, `EVP_PREEMPTION`, `PARTIAL_EVACUATION`, `FULL_FACILITY_EVACUATION`.
  - `VehicleEvacuationStatus` Enum: `STATIONARY_SAFE`, `HOLD_YIELDING`, `EVACUATING`, `EVACUATED`, `TRAPPED_IN_HAZARD`.
  - `HazardZone`: `hazard_id`, `emergency_type`, `severity`, `center_x`, `center_y`, `center_z`, `radius_m`, `floor_id`, `active`, `reported_at`.
  - `FireDoorInterlock`: `door_id`, `floor_id`, `location_x`, `location_y`, `status` (`OPEN`, `CLOSING`, `LOCKED_SHUT`), `time_to_close_s`.
  - `EmergencyVehicle`: `vehicle_id`, `responder_type` (`FIRE_ENGINE`, `AMBULANCE`, `POLICE`), `x`, `y`, `z`, `yaw_rad`, `velocity_mps`, `target_x`, `target_y`, `active_siren`.
  - `EvacuationWave`: `wave_index`, `vehicle_ids`, `dispatch_time_s`, `exit_portal_id`.
  - `EvacuationRoute`: `route_id`, `vehicle_id`, `waypoints`, `exit_portal_id`, `hazard_clearance_m`, `estimated_transit_time_s`.
  - `LifeSafetyReport`: `facility_name`, `emergency_active`, `active_hazards`, `total_vehicles`, `evacuated_count`, `evacuating_count`, `yielding_count`, `trapped_count`, `evp_active`.
- **Emergency Vehicle Preemption (EVP) Arbiter (`src/nearfield360/emergency/preemption.py`)**:
  - Tracks incoming emergency vehicles, projects clearance corridors toward destination incident points.
  - Detects spatial conflicts with standard automated valet parking vehicles.
  - Issues mandatory yield holds (`HOLD_YIELDING`) and routes standard vehicles into refuge pockets to clear right-of-way.
- **Dynamic Hazard Containment & Fire Door Interlocks (`src/nearfield360/emergency/containment.py`)**:
  - Dynamic geofenced exclusion zones around hazard sources with safety buffers.
  - Threat level classification for all facility vehicles based on distance to danger boundary.
  - Fire door / smoke shutter transit validation: blocks routing through closing doors if transit time exceeds `time_to_close_s`.
- **Multi-Wave Facility Evacuation Planner (`src/nearfield360/emergency/evacuation.py`)**:
  - Computes hazard-avoiding egress paths to primary and secondary emergency exits/ramps.
  - Prioritizes evacuation waves based on proximity to hazard zones while staggering departures to avoid gate bottlenecks.
- **Life Safety BEV Tactical Map & Situational Visualizer (`src/nearfield360/emergency/viz.py`)**:
  - `render_emergency_dashboard`:
    - Left panel: BEV facility tactical map with danger halos around hazard zones, closed fire door barriers, emergency responder corridor with siren waves, and vehicle markers color-coded by evacuation status.
    - Right panel: Tactical Life Safety HUD showing active incident severity, responder ETA, evacuation progress progress-bars, trapped vehicle alert indicators, and battery temperature monitors.
- **Package Exports & Dedicated CLI Command Group (`src/nearfield360/emergency/__init__.py`, `src/nearfield360/cli/emergency.py`)**:
  - Subcommands: `info`, `preempt`, `evacuate`, `hazard`, `audit`.
  - Register `emergency` CLI group in `src/nearfield360/cli/app.py`.
  - Update release audit in `src/nearfield360/cli/release.py` to expect 24 command groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--emergency` flag on `nearfield360 pipeline run` to integrate life safety telemetry into perception pipeline summary.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, preemption arbiter, hazard containment, evacuation planner, visualizer, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `EmergencyConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for hazard zones, fire doors, emergency vehicles, evacuation routes, and life safety reports | `uv run pytest tests/unit/emergency/test_emergency_models.py` |
| AC-3 | `EmergencyPreemptionArbiter` detects emergency responder corridors and issues preemptive yield holds to standard vehicles | `uv run pytest tests/unit/emergency/test_preemption.py` |
| AC-4 | `HazardContainmentEngine` maintains dynamic exclusion perimeters and enforces fire door / shutter transit interlocks | `uv run pytest tests/unit/emergency/test_containment.py` |
| AC-5 | `FacilityEvacuationPlanner` computes hazard-free egress routes and schedules prioritized multi-wave departures | `uv run pytest tests/unit/emergency/test_evacuation.py` |
| AC-6 | Tactical BEV dashboard visualizer renders hazard halos, emergency responder paths, fire doors, and life safety HUD to valid PNG | `uv run pytest tests/unit/emergency/test_emergency_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 emergency` executes `info`, `preempt`, `evacuate`, `hazard`, and `audit` | `uv run pytest tests/unit/test_emergency_cli.py` |
| AC-8 | Root CLI registers `emergency` command group, passing release audit with 24 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates `--emergency` option and exports life safety telemetry in report | `uv run pytest tests/unit/test_pipeline_emergency.py` |
| AC-10 | Full test suite passes with zero failures, clean `mypy`, clean `ruff`, and successful build | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, preemption arbiter, hazard containment, evacuation planner, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 emergency preempt`, `emergency evacuate`, `emergency hazard`, `emergency audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Direct municipal 911 Computer Aided Dispatch (CAD) radio frequency telematics integration.
- Hardware actuator relay switching for building deluge sprinkler pumps.
- Structural seismic earthquake collapse modeling.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Emergency Configuration Foundation**
   - [x] 1.1 `docs(roadmap): establish Phase 32 emergency preemption, evacuation and life safety roadmap`
   - [x] 1.2 `feat(config): add EmergencyConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for EmergencyConfig validation and environment overrides`
2. **Task 2: Life Safety Domain Models & Protocol Schemas**
   - [x] 2.1 `feat(emergency): implement emergency type, severity, status, and hazard zone models`
   - [x] 2.2 `feat(emergency): implement fire doors, emergency vehicles, evacuation routes, and life safety report models`
   - [x] 2.3 `test(emergency): verify life safety domain models, serialization, and validation`
3. **Task 3: Emergency Vehicle Preemption (EVP) Arbiter**
   - [x] 3.1 `feat(emergency): implement emergency responder tracking and preemptive corridor clearance`
   - [x] 3.2 `test(emergency): verify preemption corridor detection, yield holds, and right-of-way clearance`
4. **Task 4: Dynamic Hazard Containment & Fire Door Interlocks**
   - [x] 4.1 `feat(emergency): implement hazard perimeter geofencing, threat classification, and fire door clearance interlocks`
   - [x] 4.2 `test(emergency): verify hazard exclusion buffer enforcement, EV thermal runaway handling, and fire door gate timeout`
5. **Task 5: Multi-Wave Facility Evacuation Planner**
   - [x] 5.1 `feat(emergency): implement hazard-avoiding evacuation routing and prioritized wave scheduling`
   - [x] 5.2 `test(emergency): verify evacuation route clearance, egress wave staggering, and bottleneck avoidance`
6. **Task 6: Life Safety BEV Tactical Map & Situational Visualizer**
   - [x] 6.1 `feat(emergency): implement BEV hazard zones, responder trajectory, fire doors, and life safety HUD rendering`
   - [x] 6.2 `test(emergency): verify emergency visualizer exports valid PNG artifacts`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(emergency): export public emergency API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 emergency command group (info, preempt, evacuate, hazard, audit)`
   - [x] 7.3 `feat(cli): register emergency CLI group and update release audit expected count to 24`
   - [x] 7.4 `feat(pipeline): integrate --emergency options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify emergency CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline emergency integration workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 32 emergency preemption capabilities and CLI usage`
   - [x] 8.2 `docs(progress): document completed Phase 32 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`

---

## Phase 33: Facility Access Control, Barrier Gate Interlocks, License Plate Recognition (ALPR) & Vehicle Clearance Enveloping (ISO 23374 Gateway Operations)

### 1. Objective & Expected Outcome

Deliver an end-to-end Automated Facility Access Control, Barrier Gate Interlock, Automated License Plate Recognition (ALPR), and Vehicle Clearance Enveloping subsystem for NearField360, conforming to ISO 23374 (AVP Type 1 / Type 2 / Type 3 Gateway Interface and Physical Access Architecture). Phase 33 equips autonomous parking facilities with automated perimeter security and gateway transit management. The system authenticates approaching vehicles via optical ALPR plate recognition and encrypted digital keys (UWB/RFID tokens), evaluates 3D vehicle clearance envelopes (height, width, ground clearance, and approach angles) against structural portal limits to prevent structural collisions, orchestrates barrier arm kinematics ($0^\circ \to 90^\circ$ trapezoidal angular trajectory) with inductive safety loops (approach, presence, safety reversal, and departure), enforces anti-tailgating interlocks to halt unauthorized ingress, provides tactical BEV access gateway dashboards with real-time HUD telemetry, exposes a dedicated `nearfield360 gate` CLI command group, and integrates access control telemetry directly into the surround camera perception pipeline.

### 2. Scope

- **Configuration (`GateConfig` in `src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `portal_clearance_height_m`: Maximum vertical gateway portal clearance (e.g. 2.10 m).
  - `portal_clearance_width_m`: Maximum horizontal gateway portal clearance (e.g. 2.60 m).
  - `ground_clearance_min_m`: Minimum underbody clearance required to clear speed humps and ramps (e.g. 0.12 m).
  - `barrier_arm_length_m`: Physical length of the barrier gate boom arm (e.g. 3.0 m).
  - `barrier_opening_time_s`: Duration for barrier arm to actuate from $0^\circ$ to $90^\circ$ (e.g. 1.8 s).
  - `barrier_holding_time_s`: Duration barrier arm remains raised before starting closure (e.g. 5.0 s).
  - `barrier_closing_time_s`: Duration for barrier arm to lower from $90^\circ$ to $0^\circ$ (e.g. 2.2 s).
  - `alpr_confidence_threshold`: Minimum optical character recognition confidence score for automated plate granting (e.g. 0.85).
  - `anti_tailgating_timeout_s`: Window after departure loop clearance before secondary vehicle can claim passage (e.g. 2.0 s).
  - `safety_loop_debounce_ms`: Inductive loop sensor debounce filter duration (e.g. 50.0 ms).
  - `uwb_rssi_threshold_dbm`: Minimum signal strength for UWB digital key ranging validation (e.g. -75.0 dBm).
- **Domain Models & Access Protocols (`src/nearfield360/gate/models.py`)**:
  - `GateDirection` Enum: `ENTRY`, `EXIT`, `BIDIRECTIONAL`.
  - `BarrierArmState` Enum: `LOWERED_CLOSED`, `OPENING`, `RAISED_OPEN`, `CLOSING`, `SAFETY_REVERSING`, `FAULT_LOCKED`.
  - `CredentialType` Enum: `ALPR_PLATE`, `UWB_DIGITAL_KEY`, `RFID_BADGE`, `QR_BARCODE`, `V2X_DIRECT_AUTH`.
  - `AccessDecisionStatus` Enum: `GRANTED`, `REJECTED_UNKNOWN_CREDENTIAL`, `REJECTED_OVER_HEIGHT`, `REJECTED_OVER_WIDTH`, `REJECTED_LOW_GROUND_CLEARANCE`, `REJECTED_EXPIRED_RESERVATION`, `TAILGATING_LOCKOUT`.
  - `InductiveLoopId` Enum: `LOOP_A_APPROACH`, `LOOP_B_PRESENCE`, `LOOP_C_SAFETY`, `LOOP_D_DEPARTURE`.
  - `VehicleDimensionEnvelope`: `length_m`, `width_m`, `height_m`, `ground_clearance_m`, `front_overhang_m`, `rear_overhang_m`.
  - `VehicleCredential`: `credential_id`, `credential_type`, `plate_number`, `token_hash`, `valid_from`, `valid_until`, `tenant_id`, `vip_status`.
  - `GateAccessRequest`: `request_id`, `gate_id`, `timestamp`, `vehicle_id`, `detected_plate`, `plate_confidence`, `credential`, `envelope`, `approach_speed_mps`.
  - `GateAccessDecision`: `decision_id`, `request_id`, `status`, `granted`, `message`, `authorized_at`, `headroom_margin_m`, `width_margin_m`, `issued_pass_id`.
  - `BarrierKinematicsState`: `barrier_angle_deg`, `angular_velocity_deg_s`, `state`, `safety_interlock_active`, `obstruction_detected`.
  - `GateTransitEvent`: `event_id`, `gate_id`, `vehicle_id`, `decision`, `duration_s`, `tailgating_detected`, `timestamp`.
  - `GateAuditReport`: `gate_id`, `total_transits`, `granted_count`, `rejected_count`, `tailgating_incidents`, `clearance_violations`, `average_transit_duration_s`.
- **Access Credential Validator & ALPR Optical Matcher (`src/nearfield360/gate/validator.py`)**:
  - Validates optical ALPR plate character strings against reservation registry with normalized distance scoring and confidence cutoff.
  - Verifies digital cryptographic tokens (UWB / RFID), checking validity windows and tenant permissions.
- **Dimensional Clearance & Envelope Profiler (`src/nearfield360/gate/envelope.py`)**:
  - Evaluates vehicle 3D bounding envelope against physical portal limits.
  - Computes exact safety margins (`headroom_margin_m`, `width_margin_m`) and rejects out-of-spec vehicles (e.g. over-height roof boxes or low ground clearance).
- **Inductive Loop & Anti-Tailgating Barrier Interlock Controller (`src/nearfield360/gate/controller.py`)**:
  - Sequentially manages four-loop sensor state transitions: Approach (Loop A), Presence/Scan (Loop B), Safety Reversal (Loop C), Departure (Loop D).
  - Trapping and safety reversal: If Loop C is obstructed during barrier descent, immediately interrupts and executes `SAFETY_REVERSING` to $90^\circ$.
  - Enforces anti-tailgating interlocks if a trailing vehicle enters Loop B/C without prior authorization, triggering alarm lockout.
  - Aggregates operational transit events and exports `GateAuditReport`.
- **Gateway BEV Tactical Map & Visualizer (`src/nearfield360/gate/viz.py`)**:
  - Dual-panel layout:
    - Left panel: BEV gateway map showing portal structural posts, lane boundaries, inductive loop pads color-coded by activation state, barrier boom arm rendered at instantaneous angle $\theta$, ALPR scanner cone, and vehicle bounding box with clearance callouts.
    - Right panel: Access Gateway HUD showing gate ID, barrier state (`RAISED_OPEN`, etc.), angle gauge, access authorization badge, dimensional headroom and width margin indicators, safety loop sensors, and anti-tailgating status.
- **Package Exports & Dedicated CLI Command Group (`src/nearfield360/gate/__init__.py`, `src/nearfield360/cli/gate.py`)**:
  - Subcommands: `info`, `verify`, `transit`, `render`, `audit`.
  - Register `gate` CLI group in `src/nearfield360/cli/app.py`.
  - Update release audit in `src/nearfield360/cli/release.py` to expect 25 command groups.
- **Pipeline Runner Integration (`src/nearfield360/cli/pipeline.py`)**:
  - Support `--gate` option on `nearfield360 pipeline run` to integrate access control and barrier transit telemetry into perception pipeline summary.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, credential validator, envelope profiler, controller, visualizer, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `GateConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for barrier states, credentials, vehicle envelopes, loop states, access requests/decisions, and audit reports | `uv run pytest tests/unit/gate/test_gate_models.py` |
| AC-3 | `AccessCredentialValidator` matches ALPR plates, verifies tokens, and enforces reservation time windows | `uv run pytest tests/unit/gate/test_validator.py` |
| AC-4 | `VehicleEnvelopeProfiler` evaluates 3D dimensions against portal limits, computing safety margins and rejecting out-of-spec vehicles | `uv run pytest tests/unit/gate/test_envelope.py` |
| AC-5 | `BarrierInterlockController` manages inductive loop sequencing, barrier arm kinematics ($0^\circ \to 90^\circ$), safety reversals, and anti-tailgating interlocks | `uv run pytest tests/unit/gate/test_controller.py` |
| AC-6 | Tactical BEV gateway visualizer renders portal structure, loop sensor pads, barrier arm angle, clearance profiles, and status HUD to valid PNG | `uv run pytest tests/unit/gate/test_gate_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 gate` executes `info`, `verify`, `transit`, `render`, and `audit` | `uv run pytest tests/unit/test_gate_cli.py` |
| AC-8 | Root CLI registers `gate` command group, passing release audit with 25 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates `--gate` option and exports access control telemetry in report | `uv run pytest tests/unit/test_pipeline_gate.py` |
| AC-10 | All local CI-parity checks and remote CI workflow runs pass 100% | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, validator, envelope profiler, controller, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 gate verify`, `gate transit`, `gate render`, `gate audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Direct physical relay RS-485/Wiegand wire protocol hardware interfacing.
- High-voltage barrier motor inverter thermal runaway and phase-angle PWM drive control.
- Toll tag microwave DSRC transponder RF frequency synthesis.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Gateway Configuration Foundation**
   - [x] 1.1 `docs(roadmap): establish Phase 33 facility access control and barrier gate roadmap`
   - [x] 1.2 `feat(config): add GateConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for GateConfig validation and environment overrides`
   - [x] 1.4 `docs(progress): mark Task 1 configuration foundation completed`
2. **Task 2: Access Control Domain Models & Gateway Schemas**
   - [x] 2.1 `feat(gate): implement gate directions, barrier states, credential types, and envelope models`
   - [x] 2.2 `feat(gate): implement inductive loop state, access request, decision, and transit event models`
   - [x] 2.3 `test(gate): verify gateway domain models, serialization, and validation constraints`
   - [x] 2.4 `docs(progress): mark Task 2 domain models completed`
3. **Task 3: Access Credential Validator & ALPR Optical Matcher**
   - [x] 3.1 `feat(gate): implement license plate string matching and credential token validation`
   - [x] 3.2 `test(gate): verify ALPR confidence filtering, reservation matching, and credential expiry`
   - [x] 3.3 `docs(progress): mark Task 3 credential validator completed`
4. **Task 4: Dimensional Clearance & Envelope Profiler**
   - [x] 4.1 `feat(gate): implement vehicle clearance envelope profiling and physical margin computation`
   - [x] 4.2 `test(gate): verify over-height, over-width, and ground clearance violation detection`
   - [x] 4.3 `docs(progress): mark Task 4 envelope profiler completed`
5. **Task 5: Inductive Loop & Anti-Tailgating Barrier Interlock Controller**
   - [x] 5.1 `feat(gate): implement inductive loop state sequencing and barrier arm kinematics`
   - [x] 5.2 `feat(gate): implement anti-tailgating interlocks, safety reversal, and transit audit aggregation`
   - [x] 5.3 `test(gate): verify loop transitions, barrier timing, anti-tailgating lockout, and safety reversals`
   - [x] 5.4 `docs(progress): mark Task 5 barrier controller completed`
6. **Task 6: Gateway Tactical BEV Map & Visualizer**
   - [x] 6.1 `feat(gate): implement BEV access portal, inductive loop pads, barrier arm angle, and HUD rendering`
   - [x] 6.2 `test(gate): verify gateway visualizer exports valid PNG artifacts`
   - [x] 6.3 `docs(progress): mark Task 6 gateway visualizer completed`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(gate): export public gate API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 gate command group (info, verify, transit, render, audit)`
   - [x] 7.3 `feat(cli): register gate CLI group and update release audit expected count to 25`
   - [x] 7.4 `feat(pipeline): integrate --gate options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify gate CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline gate integration workflow`
   - [x] 7.7 `docs(progress): mark Task 7 package exports and CLI integration completed`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 33 facility access and barrier gate usage examples`
   - [x] 8.2 `docs(progress): document completed Phase 33 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
   - [x] 8.4 `docs(roadmap): mark Phase 33 fully completed and all acceptance criteria met`

---

## Phase 34: Infrastructure Cooperative Perception, ETSI Collective Perception Service (CPM) & Blind-Spot Occlusion Fusion (ISO 23374 Type 2/3)

### 1. Objective & Expected Outcome

Deliver an end-to-end Infrastructure Cooperative Perception and Blind-Spot Occlusion Fusion subsystem for NearField360, conforming to ISO 23374 (AVP Type 2 / Type 3 Infrastructure-to-Vehicle / I2V Cooperative Perception) and ETSI TR 103 562 / SAE J3224 Collective Perception Message (CPM) standards. The system enables stationary roadside perception units (RPUs) — including ceiling-mounted LiDAR, overhead wide-angle cameras, and corner radar beacons — to continuously monitor facility blind spots (e.g. 90-degree aisle junctions, ramp crests, structural pillars) and broadcast standardized CPM telemetry. The vehicle fuses incoming infrastructure detections with its onboard 4-camera nearfield surround-view tracks via spatiotemporal coordinate transformation, latency compensation, and track-to-track covariance intersection. The engine calculates an occlusion horizon and generates preemptive blind-spot collision warnings and yield holds for concealed hazards before they emerge into direct onboard line-of-sight.

### 2. Scope

- **Configuration (`CoopConfig` in `src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `max_latency_compensation_s`: Maximum wireless age of CPM object eligible for kinematic extrapolation (e.g. 0.5 s).
  - `association_gate_mahalanobis`: Mahalanobis distance gating threshold for track-to-track association (e.g. 3.0).
  - `min_cpm_confidence`: Minimum infrastructure detection confidence score required for fusion (e.g. 0.60).
  - `blind_spot_warning_ttc_s`: Time-to-collision warning horizon threshold for blind-spot hazards (e.g. 2.5 s).
  - `occlusion_grid_resolution_m`: Spatial grid resolution for line-of-sight raycasting (e.g. 0.25 m).
  - `rpu_broadcast_frequency_hz`: Roadside perception unit transmission frequency (e.g. 10.0 Hz).
  - `track_covariance_inflation_factor`: Inflation factor applied during covariance intersection to account for unmodeled cross-correlation (e.g. 1.25).
- **Domain Models & CPM Protocols (`src/nearfield360/coop/models.py`)**:
  - `RPUSensorType` Enum: `LIDAR`, `FISHEYE_CAMERA`, `RADAR`, `FUSED_STATION`.
  - `CPMStationType` Enum: `INFRASTRUCTURE_RPU`, `VEHICLE_ONBOARD`, `SPECIAL_EQUIPMENT`.
  - `CoopTrackSource` Enum: `ONBOARD_ONLY`, `INFRASTRUCTURE_ONLY`, `FUSED_COOPERATIVE`.
  - `BlindSpotRiskLevel` Enum: `SAFE`, `ADVISORY`, `WARNING`, `CRITICAL_BRAKE`.
  - `RPUSensorState`: `station_id`, `sensor_type`, `position_m` $(x, y, z)$, `orientation_deg` $(\text{yaw}, \text{pitch}, \text{roll})$, `range_max_m`, `fov_azimuth_deg`, `is_operational`.
  - `CPMPerceivedObject`: `object_id`, `station_id`, `timestamp`, `category`, `position` $(x, y, z)$, `velocity` $(\dot{x}, \dot{y})$, `dimensions` $(l, w, h)$, `position_covariance`, `confidence`, `occluded`.
  - `CollectivePerceptionMessage`: `station_id`, `station_type`, `sequence_number`, `timestamp`, `reference_position`, `perceived_objects`.
  - `CooperativeTrack`: `track_id`, `source`, `position` $(x, y)$, `velocity` $(\dot{x}, \dot{y})$, `covariance_2x2`, `category`, `confidence`, `age_s`, `onboard_track_id`, `rpu_station_id`, `rpu_object_id`.
  - `BlindSpotWarning`: `target_vehicle_id`, `hazard_object_id`, `conflict_point` $(x, y)$, `ttc_s`, `risk_level`, `recommended_action`, `occlusion_source`.
  - `CoopTelemetryReport`: `active_rpus`, `cpm_message_count`, `onboard_track_count`, `fused_track_count`, `blind_spot_hazard_count`, `warnings`, `mean_fusion_latency_ms`.
- **Roadside Perception Unit (RPU) & CPM Dispatcher (`src/nearfield360/coop/rpu.py`)**:
  - Manages roadside perception stations with geometric FOV cones, visibility filtering, and compliant CPM payload encoding.
- **Spatiotemporal Alignment & Wireless Latency Extrapolation (`src/nearfield360/coop/alignment.py`)**:
  - Transforms RPU local coordinates into facility global coordinates and vehicle body frame using SE(2)/SE(3) rigid transformations.
  - Applies constant-velocity kinematic extrapolation to compensate for wireless transmission delays.
- **Track-to-Track Cooperative Fusion Engine (`src/nearfield360/coop/fusion.py`)**:
  - Gated Mahalanobis distance metric and Hungarian/greedy association between onboard fisheye tracks and received CPM objects.
  - Covariance Intersection (CI) state update fusing independent onboard and infrastructure estimates with provable bounds.
  - Maintains track lifecycle (initiation, confirmation, prediction, deletion).
- **Blind-Spot Occlusion Profiler & Preemptive Conflict Assessor (`src/nearfield360/coop/blind_spot.py`)**:
  - Raycasts onboard sensor FOV against structural pillars to construct line-of-sight visibility polygons and identify occluded shadow regions.
  - Assesses whether infrastructure-detected hazards reside in blind zones and computes TTC along projected vehicle trajectory.
- **Tactical Cooperative BEV Visualizer (`src/nearfield360/coop/viz.py`)**:
  - Dual-panel layout:
    - Left panel: Facility BEV map showing aisles, structural pillars, vehicle position, onboard 360-degree fisheye coverage polygon, RPU stations with sensor FOV cones, occluded blind shadow zones highlighted in translucent amber, onboard tracks (blue), infrastructure CPM detections (magenta), and fused cooperative tracks (bright green) with velocity vectors and uncertainty ellipses.
    - Right panel: Cooperative Perception HUD showing active RPU stations, CPM throughput and latency, track association matrix, blind-spot hazard alerts with TTC countdown gauges, and cooperative risk status badge.
- **Package Exports, CLI Command Group & Surround Pipeline Integration (`src/nearfield360/coop/__init__.py`, `src/nearfield360/cli/coop.py`, `src/nearfield360/cli/pipeline.py`)**:
  - Public symbols exported from `src/nearfield360/coop/__init__.py`.
  - Dedicated CLI group: `nearfield360 coop` with subcommands: `info`, `rpu`, `fuse`, `render`, `audit`.
  - Register `coop` CLI command group in `src/nearfield360/cli/app.py`.
  - Update release audit in `src/nearfield360/cli/release.py` to expect 26 command groups.
  - Pipeline runner integration: `nearfield360 pipeline run --coop` / `--no-coop`.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, RPU dispatcher, alignment/extrapolation, cooperative fusion, blind-spot warning, visualizer, CLI commands, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `CoopConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for RPU states, CPM perceived objects, CPM messages, cooperative tracks, and blind spot warnings | `uv run pytest tests/unit/coop/test_coop_models.py` |
| AC-3 | `RPUSensorManager` manages roadside perception stations, visibility geometry, and generates compliant CPM messages | `uv run pytest tests/unit/coop/test_rpu.py` |
| AC-4 | `SpatiotemporalAligner` executes coordinate transforms and latency compensation extrapolation | `uv run pytest tests/unit/coop/test_alignment.py` |
| AC-5 | `CooperativeTrackFusion` performs Mahalanobis gated association and Covariance Intersection track fusion | `uv run pytest tests/unit/coop/test_fusion.py` |
| AC-6 | `BlindSpotAssessor` computes line-of-sight occlusion polygons, detects hidden hazards, and generates preemptive collision warnings | `uv run pytest tests/unit/coop/test_blind_spot.py` |
| AC-7 | Tactical BEV cooperative visualizer renders facility layout, onboard FOV, RPU cones, blind zones, fused tracks, and HUD to valid PNG | `uv run pytest tests/unit/coop/test_coop_viz.py` |
| AC-8 | Dedicated CLI command group `nearfield360 coop` executes `info`, `rpu`, `fuse`, `render`, and `audit` | `uv run pytest tests/unit/test_coop_cli.py` |
| AC-9 | Root CLI registers `coop` command group, passing release audit with 26 groups | `uv run nearfield360 release audit` |
| AC-10 | Surround pipeline runner integrates `--coop` option and exports cooperative perception telemetry in report | `uv run pytest tests/unit/test_pipeline_coop.py` |
| AC-11 | All local CI-parity checks and remote CI workflow runs pass 100% | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, RPU manager, alignment/extrapolation, fusion, blind-spot assessor, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 coop info`, `coop rpu`, `coop fuse`, `coop render`, `coop audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- Direct physical IEEE 802.11p / C-V2X PC5 direct sidelink baseband radio physical layer decoding.
- Multi-building fiber optic distribution network PTP IEEE 1588 nanosecond clock synchronization.
- Overhead LiDAR point cloud raw voxel segmentation deep learning training loops.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Cooperative Perception Configuration Foundation**
   - [x] 1.1 `docs(roadmap): establish Phase 34 cooperative perception and infrastructure CPM roadmap`
   - [x] 1.2 `feat(config): add CoopConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for CoopConfig validation and environment overrides`
   - [x] 1.4 `docs(progress): mark Task 1 configuration foundation completed`
2. **Task 2: Cooperative Perception Domain Models & CPM Protocols**
   - [x] 2.1 `feat(coop): implement RPU sensor types, station types, and risk level enums`
   - [x] 2.2 `feat(coop): implement RPU state, CPM object, CPM message, and cooperative track models`
   - [x] 2.3 `feat(coop): implement blind spot warning and cooperative telemetry report models`
   - [x] 2.4 `test(coop): verify cooperative domain models, validation, and JSON serialization`
   - [x] 2.5 `docs(progress): mark Task 2 domain models completed`
3. **Task 3: Roadside Perception Unit (RPU) Manager & CPM Dispatcher**
   - [x] 3.1 `feat(coop): implement RPU sensor station management and FOV coverage geometry`
   - [x] 3.2 `feat(coop): implement CPM message generation and station payload dispatch`
   - [x] 3.3 `test(coop): verify RPU registration, FOV filtering, and CPM dispatch`
   - [x] 3.4 `docs(progress): mark Task 3 RPU manager completed`
4. **Task 4: Spatiotemporal Alignment & Wireless Latency Extrapolation**
   - [x] 4.1 `feat(coop): implement coordinate frame transformations between RPU, facility, and vehicle`
   - [x] 4.2 `feat(coop): implement kinematic extrapolation and latency compensation for CPM objects`
   - [x] 4.3 `test(coop): verify coordinate transformation accuracy and kinematic time-extrapolation`
   - [x] 4.4 `docs(progress): mark Task 4 alignment and extrapolation completed`
5. **Task 5: Track-to-Track Cooperative Fusion Engine**
   - [x] 5.1 `feat(coop): implement Mahalanobis distance association and gated track matching`
   - [x] 5.2 `feat(coop): implement Covariance Intersection (CI) track state and covariance update`
   - [x] 5.3 `feat(coop): implement cooperative track manager with track lifecycle management`
   - [x] 5.4 `test(coop): verify track association, Covariance Intersection fusion, and lifecycle`
   - [x] 5.5 `docs(progress): mark Task 5 cooperative fusion completed`
6. **Task 6: Blind-Spot Occlusion Profiler & Preemptive Conflict Assessor**
   - [x] 6.1 `feat(coop): implement line-of-sight raycasting and pillar occlusion shadow computation`
   - [x] 6.2 `feat(coop): implement blind-spot hazard detection and TTC warning generation`
   - [x] 6.3 `test(coop): verify occlusion polygon calculation, blind-spot hazard detection, and TTC warnings`
   - [x] 6.4 `docs(progress): mark Task 6 blind-spot assessor completed`
7. **Task 7: Tactical BEV Cooperative Map & Visualizer**
   - [x] 7.1 `feat(coop): implement dual-panel BEV layout with onboard FOV, RPU cones, and blind shadows`
   - [x] 7.2 `feat(coop): implement cooperative perception HUD with latency, tracks, and TTC risk gauges`
   - [x] 7.3 `test(coop): verify cooperative perception visualizer exports valid PNG artifacts`
   - [x] 7.4 `docs(progress): mark Task 7 cooperative visualizer completed`
8. **Task 8: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 8.1 `feat(coop): export public cooperative perception API from package init`
   - [x] 8.2 `feat(cli): implement nearfield360 coop command group (info, rpu, fuse, render, audit)`
   - [x] 8.3 `feat(cli): register coop CLI group and update release audit expected count to 26`
   - [x] 8.4 `feat(pipeline): integrate --coop options into surround pipeline runner`
   - [x] 8.5 `test(cli): verify coop CLI commands, JSON payloads, and visual outputs`
   - [x] 8.6 `test(pipeline): verify pipeline coop integration workflow`
   - [x] 8.7 `docs(progress): mark Task 8 package exports and CLI integration completed`
9. **Task 9: End-to-End Documentation, Self-Review & Delivery**
   - [x] 9.1 `docs: update README with Phase 34 cooperative perception usage examples`
   - [x] 9.2 `docs(progress): document completed Phase 34 milestone, Decision Log, and verification gates`
   - [x] 9.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
   - [x] 9.4 `docs(roadmap): mark Phase 34 fully completed and all acceptance criteria met`

---

## Phase 35: Vehicle-to-Grid (V2G) Bi-Directional Power Flow, Peak-Shaving Arbitrage & Automated Inverter Telemetry (ISO 15118-20 / OCPP 2.0.1)

### 1. Objective & Expected Outcome

Deliver an end-to-end Vehicle-to-Grid (V2G) Bi-Directional Energy Transfer, Dynamic Peak-Shaving Arbitrage, and Smart Inverter Microgrid Telemetry subsystem for NearField360, conforming to ISO 15118-20 (Bi-directional Power Transfer / BPT) and OCPP 2.0.1 (SmartCharging Profiles). In automated valet parking facilities hosting large fleets of plugged-in electric vehicles, this system aggregates vehicle battery capacities into a virtual energy storage system (VESS/BESS). The engine models dynamic time-of-use (ToU) electricity tariffs, grid stress events, facility transformer peak thresholds, battery degradation cycle costs, and strict driver departure state-of-charge (SoC) reserves. During peak grid tariff hours or facility demand spikes, vehicles discharge power back into the facility microgrid (peak shaving) or earn export feed-in credits. During surplus generation or off-peak periods, vehicles charge rapidly. In an emergency grid outage, the facility operates in an islanded microgrid mode powered by fleet batteries.

### 2. Scope

- **Configuration (`V2GConfig` in `src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `facility_transformer_limit_kw`: Maximum facility transformer import power (e.g. 500.0 kW).
  - `peak_shaving_target_kw`: Peak demand threshold above which discharging is triggered (e.g. 350.0 kW).
  - `min_departure_reserve_soc`: Minimum battery state of charge guaranteed to customer at pickup (e.g. 0.60).
  - `max_discharge_power_kw`: Maximum bi-directional discharge rate per AC/DC vehicle port (e.g. 22.0 kW).
  - `max_charge_power_kw`: Maximum charge rate per port (e.g. 50.0 kW).
  - `battery_degradation_penalty_per_kwh`: Cost per discharged kWh to account for Li-ion battery cycle aging (e.g. 0.04 $/kWh).
  - `grid_frequency_nominal_hz`: Nominal AC grid frequency (e.g. 50.0 Hz).
  - `frequency_droop_gain_kw_per_hz`: Droop response sensitivity for primary frequency support (e.g. 25.0 kW/Hz).
  - `arbitrage_interval_minutes`: Temporal resolution for energy dispatch scheduling (e.g. 15 min).
- **Domain Models & V2G Protocol Schemas (`src/nearfield360/v2g/models.py`)**:
  - `V2GPowerMode` Enum: `STANDBY`, `CHARGE`, `DISCHARGE_PEAK_SHAVE`, `DISCHARGE_FREQUENCY_SUPPORT`, `ISLANDED_BACKUP`.
  - `InverterPhase` Enum: `SINGLE_PHASE`, `THREE_PHASE`.
  - `GridTariffWindow`: `start_hour`, `end_hour`, `import_price_per_kwh`, `export_price_per_kwh`, `carbon_intensity_g_per_kwh`.
  - `VehicleBPTProfile`: `vehicle_id`, `battery_capacity_kwh`, `current_soc`, `min_departure_soc`, `max_discharge_kw`, `max_charge_kw`, `arrival_hour`, `departure_hour`, `degradation_cost_factor`.
  - `InverterTelemetry`: `inverter_id`, `phase`, `active_power_kw`, `reactive_power_kvar`, `grid_frequency_hz`, `voltage_rms`, `dc_bus_voltage_v`, `efficiency_percent`, `timestamp`.
  - `V2GTimeSlotDispatch`: `slot_index`, `hour`, `power_mode`, `active_power_kw`, `soc_end`, `import_tariff`, `export_tariff`, `cost_or_revenue_delta`.
  - `V2GArbitrageSchedule`: `vehicle_id`, `dispatches`, `total_energy_charged_kwh`, `total_energy_discharged_kwh`, `net_cost_dollars`, `net_revenue_dollars`, `net_profit_dollars`, `final_soc`, `min_departure_soc_met`.
  - `MicrogridFacilityReport`: `total_bays_connected`, `total_battery_capacity_kwh`, `available_flexible_energy_kwh`, `baseline_facility_peak_kw`, `peak_shaved_demand_kw`, `transformer_utilization_percent`, `net_arbitrage_profit_dollars`, `carbon_offset_kg`, `schedules`, `is_islanded`.
- **Smart Inverter Controller & Droop Response (`src/nearfield360/v2g/inverter.py`)**:
  - Bidirectional AC/DC power conversion efficiency curve with thermal losses.
  - Frequency-Watt ($P(f)$) droop curve modulating active power during grid frequency excursions.
  - Voltage-VAR ($Q(V)$) reactive power compensation maintaining facility power factor.
- **Dynamic Arbitrage & Degradation-Constrained Scheduler (`src/nearfield360/v2g/arbitrage.py`)**:
  - Multi-interval economic optimization over the vehicle dwell horizon.
  - Strict enforcement of driver departure reserve SoC and cycle degradation economic break-even.
- **Facility Microgrid Energy Aggregator & Peak Shaver (`src/nearfield360/v2g/microgrid.py`)**:
  - Aggregates multi-vehicle fleet batteries into a unified Virtual Power Plant (VPP).
  - Peak-shaving dispatch preventing facility transformer overload.
  - Islanded emergency backup mode during grid outages.
- **Tactical V2G Energy Dashboard & SLD Visualizer (`src/nearfield360/v2g/viz.py`)**:
  - Dual-panel layout:
    - Left panel: Facility electrical single-line diagram (SLD) showing transformer, AC bus, solar PV, and parking bays with directional power vectors (blue charging, amber discharging).
    - Right panel: Microgrid energy HUD with transformer load curve, ToU tariff schedule, fleet SoC distribution, cumulative arbitrage revenue, and grid stability gauges.
- **Package Exports, CLI Command Group & Surround Pipeline Integration (`src/nearfield360/v2g/__init__.py`, `src/nearfield360/cli/v2g.py`, `src/nearfield360/cli/pipeline.py`)**:
  - Public symbols exported in `src/nearfield360/v2g/__init__.py`.
  - Dedicated CLI group: `nearfield360 v2g` with subcommands: `info`, `schedule`, `arbitrage`, `render`, `audit`.
  - Register `v2g` CLI command group in `src/nearfield360/cli/app.py`.
  - Update release audit in `src/nearfield360/cli/release.py` to expect 27 command groups.
  - Pipeline runner integration: `nearfield360 pipeline run --v2g` / `--no-v2g`.
- **Comprehensive Verification Suite**:
  - Unit tests covering configuration, domain models, inverter controller, arbitrage scheduler, microgrid aggregator, visualizer, CLI commands, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `V2GConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for V2G power modes, tariff windows, vehicle BPT profiles, inverter telemetry, and arbitrage schedules | `uv run pytest tests/unit/v2g/test_v2g_models.py` |
| AC-3 | `SmartInverter` computes bidirectional conversion, frequency-watt droop response, and reactive power compensation | `uv run pytest tests/unit/v2g/test_inverter.py` |
| AC-4 | `ArbitrageScheduler` optimizes charging/discharging intervals, prevents unprofitable degradation, and guarantees departure reserve SoC | `uv run pytest tests/unit/v2g/test_arbitrage.py` |
| AC-5 | `FacilityMicrogridManager` aggregates fleet capacity, caps transformer peak load, and supports islanded emergency operation | `uv run pytest tests/unit/v2g/test_microgrid.py` |
| AC-6 | Tactical V2G energy visualizer renders facility electrical SLD, active bay power flows, and telemetry HUD to valid PNG | `uv run pytest tests/unit/v2g/test_v2g_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 v2g` executes `info`, `schedule`, `arbitrage`, `render`, and `audit` | `uv run pytest tests/unit/test_v2g_cli.py` |
| AC-8 | Root CLI registers `v2g` command group, passing release audit with 27 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline runner integrates `--v2g` option and exports bidirectional power flow telemetry in report | `uv run pytest tests/unit/test_pipeline_v2g.py` |
| AC-10 | All local CI-parity checks and remote CI workflow runs pass 100% | `uv run pytest`, `uv run mypy`, `uv run ruff check .`, `uv build` |

### 4. Validation Plan

1. Unit tests covering config, models, inverter, arbitrage, microgrid, viz, CLI, and pipeline.
2. CLI integration tests verifying `nearfield360 v2g info`, `v2g schedule`, `v2g arbitrage`, `v2g render`, `v2g audit` with JSON/PNG outputs.
3. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
4. Release audit: `uv run nearfield360 release audit`.
5. Full test suite: `uv run pytest -q`.

### 5. Exclusions

- High-voltage DC contactor physical arc flash plasma simulation.
- Real-time CAN bus ISO 15118-20 binary EXI packet deserialization on microcontroller hardware.
- Wholesale electricity market automated bidding financial exchange API integrations.

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & V2G Configuration Foundation**
   - [x] 1.1 `docs(roadmap): establish Phase 35 vehicle-to-grid bidirectional power flow roadmap`
   - [x] 1.2 `feat(config): add V2GConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for V2GConfig validation and environment overrides`
   - [x] 1.4 `docs(progress): mark Task 1 configuration foundation completed`
2. **Task 2: V2G Domain Models & Telemetry Schemas**
   - [x] 2.1 `feat(v2g): implement V2G power mode, inverter phase, and tariff window models`
   - [x] 2.2 `feat(v2g): implement vehicle BPT profile and smart inverter telemetry models`
   - [x] 2.3 `feat(v2g): implement arbitrage schedule and microgrid facility report models`
   - [x] 2.4 `test(v2g): verify V2G domain models, constraints, and JSON serialization`
   - [x] 2.5 `docs(progress): mark Task 2 domain models completed`
3. **Task 3: Smart Inverter Controller & Droop Response**
   - [x] 3.1 `feat(v2g): implement smart inverter AC/DC conversion and bidirectional power transfer`
   - [x] 3.2 `feat(v2g): implement frequency-watt droop response and voltage-VAR reactive compensation`
   - [x] 3.3 `test(v2g): verify inverter conversion efficiency, frequency droop, and reactive power`
   - [x] 3.4 `docs(progress): mark Task 3 smart inverter controller completed`
4. **Task 4: Dynamic Arbitrage & Degradation-Constrained Scheduler**
   - [x] 4.1 `feat(v2g): implement time-of-use economic evaluation and battery degradation penalty`
   - [x] 4.2 `feat(v2g): implement multi-interval arbitrage scheduler with departure SoC guarantee`
   - [x] 4.3 `test(v2g): verify arbitrage schedule optimization, degradation avoidance, and departure SoC`
   - [x] 4.4 `docs(progress): mark Task 4 arbitrage scheduler completed`
5. **Task 5: Facility Microgrid Energy Aggregator & Peak Shaver**
   - [x] 5.1 `feat(v2g): implement fleet battery capacity aggregation and facility base load modeling`
   - [x] 5.2 `feat(v2g): implement transformer peak-shaving dispatch and islanded emergency backup`
   - [x] 5.3 `test(v2g): verify microgrid peak shaving, transformer overload prevention, and islanding`
   - [x] 5.4 `docs(progress): mark Task 5 microgrid aggregator completed`
6. **Task 6: Tactical V2G Energy Dashboard & SLD Visualizer**
   - [x] 6.1 `feat(v2g): implement facility electrical single-line diagram and bay power flow rendering`
   - [x] 6.2 `feat(v2g): implement microgrid telemetry HUD with load curves, tariffs, and revenue gauges`
   - [x] 6.3 `test(v2g): verify V2G visualizer exports valid PNG artifacts`
   - [x] 6.4 `docs(progress): mark Task 6 V2G visualizer completed`
7. **Task 7: Package Exports, CLI Command Group & Pipeline Integration**
   - [x] 7.1 `feat(v2g): export public V2G API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 v2g command group (info, schedule, arbitrage, render, audit)`
   - [x] 7.3 `feat(cli): register v2g CLI group and update release audit expected count to 27`
   - [x] 7.4 `feat(pipeline): integrate --v2g options into surround pipeline runner`
   - [x] 7.5 `test(cli): verify v2g CLI commands, JSON payloads, and visual outputs`
   - [x] 7.6 `test(pipeline): verify pipeline v2g integration workflow`
   - [x] 7.7 `docs(progress): mark Task 7 package exports and CLI integration completed`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [x] 8.1 `docs: update README with Phase 35 vehicle-to-grid power flow usage examples`
   - [x] 8.2 `docs(progress): document completed Phase 35 milestone, Decision Log, and verification gates`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
   - [x] 8.4 `docs(roadmap): mark Phase 35 fully completed and all acceptance criteria met`


## Phase 36: Facility Operations Control Tower, Coordinated Day-Run Simulation & Cross-Subsystem KPI Aggregation

### 1. Objective & Expected Outcome

Deliver a facility operations control tower that runs a seeded, deterministic day-level simulation of an automated valet parking facility and aggregates cross-subsystem KPIs. A stochastic demand generator (double-peaked arrival profile) feeds a coordinated simulator that moves each vehicle through gate arrival, slot dispatch, parking, optional charging window with V2G export, unparking, and settlement billing. The simulator emits hourly facility snapshots and a daily KPI report (throughput, peak occupancy, energy balance, revenue streams, service-level compliance). A unified dashboard PNG and the `nearfield360 ops` CLI group expose the result. The release audit then expects 28 CLI groups.

### 2. Scope

- **Configuration**: `OperationsConfig` in `src/nearfield360/config.py` and `configs/default.yaml` (done).
- **Domain models (`src/nearfield360/operations/models.py`)**: `ServiceLevel`, `JourneyStage`, `VehicleArrival`, `VehicleJourneyRecord`, `HourlySnapshot`, `DailyKPIReport`, `OperationsDayReport`.
- **Demand generator (`demand.py`)**: seeded, non-homogeneous Poisson arrivals with morning/evening Gaussian peaks over a base rate.
- **Day-run simulator (`simulator.py`)**: capacity-constrained slot dispatch with queueing, blocked/degraded service levels, charging windows, V2G export energy, tariff-based energy fees and grace-period parking fees.
- **KPI aggregation (`kpi.py`)**: hourly snapshots and the daily KPI report.
- **Dashboard (`viz.py`)**: occupancy/arrival timeline, revenue and energy panels, service-level breakdown PNG.
- **Package exports, CLI group `nearfield360 ops` (`info`, `simulate`, `render`, `audit`)**, registration in `app.py`, release audit expecting 28 groups.
- **Verification suite**: unit tests for every module plus CLI tests.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `OperationsConfig` validated and present in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Domain models validate inputs and round-trip JSON | `uv run pytest tests/unit/operations/test_operations_models.py` |
| AC-3 | Demand generator is deterministic per seed and shows two arrival peaks | `uv run pytest tests/unit/operations/test_demand.py` |
| AC-4 | Simulator conserves vehicles, respects capacity, and is deterministic per seed | `uv run pytest tests/unit/operations/test_simulator.py` |
| AC-5 | KPI aggregation is consistent with journey records (revenue, energy, service levels) | `uv run pytest tests/unit/operations/test_kpi.py` |
| AC-6 | Dashboard renders a valid PNG | `uv run pytest tests/unit/operations/test_operations_viz.py` |
| AC-7 | `nearfield360 ops` executes `info`, `simulate`, `render`, `audit` | `uv run pytest tests/unit/test_ops_cli.py` |
| AC-8 | Release audit passes with 28 groups | `uv run nearfield360 release audit` |
| AC-9 | All local CI-parity checks and remote CI workflow runs pass 100% | pre-commit, `uv run pytest`, `uv build` |

### 4. Validation & CI Plan

Local (mirrors `.github/workflows/ci.yml`): `uv lock --check`, `uv sync --locked --group dev`, `uv run pre-commit run --all-files --show-diff-on-failure`, `uv run pytest -q --cov=nearfield360 --cov-report=term-missing --cov-report=xml`, `uv build`, `uv run twine check dist/*`. End-to-end: `nearfield360 ops simulate --seed 7 -o day.json --png day.png`.

### 5. Exclusions

- Surround pipeline integration (`pipeline run --ops`): deferred; the day-run is a facility-level simulation, not a per-frame perception concern.
- Deep re-use of the fleet/charging/V2G engines inside the simulator: the simulator uses a compact analytic model of each subsystem (documented in the Decision Log) to stay deterministic and fast; engine-level co-simulation is a follow-up.

### 6. Micro-Task Breakdown

1. [x] `docs(roadmap): plan Phase 36 facility operations control tower (item 27)` (README)
2. [x] `feat(config): add OperationsConfig for day-level facility operations`
3. [x] Roadmap Phase 36 section, domain models completion and tests
4. [x] Demand generator and tests
5. [x] Simulator and tests
6. [x] KPI aggregation and tests
7. [x] Dashboard visualizer and tests
8. [x] Package exports, CLI group, registration, audit 28, tests
9. [x] README/PROGRESS docs, full CI-parity verification, push, draft PR
