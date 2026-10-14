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
   - [ ] 1.1 `docs: establish Phase 29 curbside staging and pedestrian safety interlocks roadmap`
   - [ ] 1.2 `feat(config): add CurbsideConfig and update default yaml configuration`
   - [ ] 1.3 `test(config): add unit tests for CurbsideConfig validation and environment overrides`
2. **Task 2: Curbside Domain Models & Handover Schemas**
   - [ ] 2.1 `feat(curbside): implement curbside bays, missions, pedestrian, and interlock models`
   - [ ] 2.2 `feat(curbside): implement curbside audit report models`
   - [ ] 2.3 `test(curbside): verify curbside domain models, serialization, and validation`
3. **Task 3: Pedestrian Safety Interlock Engine**
   - [ ] 3.1 `feat(curbside): implement pedestrian safety interlock engine with clearance debounce`
   - [ ] 3.2 `test(curbside): verify interlock inhibition on door/trunk open and pedestrian proximity`
4. **Task 4: Curbside Staging & Dwell Queue Manager**
   - [ ] 4.1 `feat(curbside): implement curbside staging manager with dwell tracking and overstay penalties`
   - [ ] 4.2 `test(curbside): verify dropoff/pickup allocation, staging queue, and overstay calculation`
5. **Task 5: BEV Curbside Staging & Safety HUD Visualizer**
   - [ ] 5.1 `feat(curbside): implement BEV curbside bay map, pedestrian safety circles, and queue HUD`
   - [ ] 5.2 `test(curbside): verify curbside visualizer exports valid PNG artifacts`
6. **Task 6: Package Exports, CLI Command Group & Pipeline Integration**
   - [ ] 6.1 `feat(curbside): export public curbside API from package init`
   - [ ] 6.2 `feat(cli): implement nearfield360 curbside command group (info, dropoff, summon, audit)`
   - [ ] 6.3 `feat(cli): register curbside CLI group and update release audit expected count to 21`
   - [ ] 6.4 `feat(pipeline): integrate --curbside options into surround pipeline runner`
   - [ ] 6.5 `test(cli): verify curbside CLI commands, JSON payloads, and visual outputs`
   - [ ] 6.6 `test(pipeline): verify pipeline curbside integration workflow`
7. **Task 7: End-to-End Documentation, Self-Review & Delivery**
   - [ ] 7.1 `docs: update README with Phase 29 curbside staging capabilities and CLI usage`
   - [ ] 7.2 `docs(progress): document completed Phase 29 milestone, Decision Log, and verification gates`
   - [ ] 7.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`





