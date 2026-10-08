# Autonomous Valet Parking (AVP) Development Roadmap

## Phase 23: Autonomous Valet Parking (AVP) Mission Executive, Dynamic Replanning & Multi-Stage Recovery Orchestration

### 1. Objective & Expected Outcome

Deliver an end-to-end Autonomous Valet Parking (AVP) Mission Executive and lifecycle state machine that elevates NearField360 from isolated perception, slot detection, planning, and control steps into a fully orchestrated, resilient autonomous parking system. The system autonomously manages the complete parking lifecycle: cruising and multi-frame slot discovery, optimal slot selection, approach navigation, closed-loop maneuver tracking, obstacle intrusion holding and yielding, online dynamic replanning around obstacles, and terminal precision alignment.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `MissionConfig` specifying hold timeout, maximum replan attempts, docking tolerances, approach velocity, slot tracking distance gate, and confirmation thresholds.
- **Domain Models (`src/nearfield360/mission/models.py`)**:
  - `MissionState` lifecycle enum: `STANDBY`, `SEARCHING`, `SLOT_SELECTED`, `APPROACH`, `PARKING_MANEUVER`, `OBSTACLE_HOLD`, `REPLANNING`, `FINAL_ALIGNMENT`, `COMPLETED`, `ABORTED`.
  - `MissionTrigger` event transition triggers.
  - `MissionEvent` transition event logging with timestamps and reasons.
  - `TrackedParkingSlot` multi-frame persistent spatial slot representation with Bayesian confidence tracking.
  - `RecoveryManeuver` and `MissionSummaryReport` schemas.
- **Spatial Slot Tracker (`src/nearfield360/mission/slot_tracker.py`)**:
  - Multi-frame temporal slot association using metric centroid gating and heading orientation matching.
  - Running pose filter and observation confidence scoring.
  - Pruning stale or transient false-positive slot detections.
- **Dynamic Replanner & Maneuver Recovery (`src/nearfield360/mission/replanner.py`)**:
  - Online recovery trajectory generator for blocked or deviated maneuvers: forward pull-out realignment, multi-point adjustments, and collision clearance audits.
- **AVP Mission Executive & State Machine (`src/nearfield360/mission/executive.py`)**:
  - Orchestrates sensor inputs, slot tracking, trajectory planning, closed-loop tracking control, dynamic safety monitoring, obstacle hold timers, and online replanning.
  - Handles nominal execution, transient yielding, persistent obstacle replanning, and safe abort fallbacks.
- **Visual Mission Dashboard & Telemetry Charts (`src/nearfield360/mission/viz.py`)**:
  - Multi-panel BEV execution overlay showing planned vs actual trajectory, replanned recovery paths, slot boundaries, and mission status badges.
  - Time-series mission state progression timeline chart and obstacle clearance profiles.
- **CLI Command Group & Pipeline Integration (`src/nearfield360/cli/mission.py`, `src/nearfield360/cli/pipeline.py`)**:
  - Dedicated CLI command group: `nearfield360 mission run` with scenario simulation flags (`nominal`, `transient_obstacle`, `blocked_replan`).
  - Surround pipeline integration: `nearfield360 pipeline run --mission`.
  - CLI registration and release audit update (15 expected command groups).
- **Comprehensive Verification Suite**:
  - Extensive unit and integration tests across all components.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `MissionConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for mission lifecycle states, triggers, events, and tracked slots | `uv run pytest tests/unit/mission/test_mission_models.py` |
| AC-3 | `SlotTracker` associates multi-frame slot detections, filters noise, and manages lifecycle | `uv run pytest tests/unit/mission/test_slot_tracker.py` |
| AC-4 | `ParkingReplanner` generates feasible, collision-free recovery trajectories upon blockage | `uv run pytest tests/unit/mission/test_replanner.py` |
| AC-5 | `MissionExecutive` coordinates AVP state transitions, obstacle holds, and replanning end to end | `uv run pytest tests/unit/mission/test_executive.py` |
| AC-6 | Visualization renders BEV mission dashboard overlays and state timeline charts | `uv run pytest tests/unit/mission/test_mission_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 mission run` executes simulated scenarios | `uv run pytest tests/unit/test_mission_cli.py` |
| AC-8 | Root CLI registers `mission` command group, passing release audit with 15 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline `--mission` flag executes full perception-to-mission orchestration | `uv run pytest tests/unit/test_pipeline_mission.py` |
| AC-10 | Full test suite passes with zero failures, >85% coverage, clean `mypy`, and clean `ruff` | `uv run pytest`, `uv run mypy`, `uv run ruff check .` |

### 4. Validation Plan

1. Unit tests covering config validation, domain models, slot tracking, replanning, executive state machine, and visual rendering.
2. CLI integration tests verifying `nearfield360 mission run` across nominal, transient obstacle, and blocked replan scenarios.
3. Pipeline integration tests verifying `nearfield360 pipeline run --mission` with simulated four-camera input.
4. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy --strict`.
5. Release audit: `uv run nearfield360 release audit`.
6. Full test suite with coverage: `uv run pytest --cov=nearfield360`.

### 5. Exclusions

- Cloud-based multi-vehicle fleet coordination (AVP Level 5 infrastructure-managed dispatch).
- High-definition (HD) vector map graph SLAM across multi-story parking garages.
- Hardware-in-the-loop (HIL) CAN bus transceivers (simulated discrete-time vehicle state interface used).

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Config Foundation**
   - [x] 1.1 `docs: establish Phase 23 AVP mission executive roadmap`
   - [x] 1.2 `feat(config): add MissionConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for MissionConfig validation`
2. **Task 2: Mission Domain Models**
   - [x] 2.1 `feat(mission): implement lifecycle states, triggers, events, and tracked slot models`
   - [x] 2.2 `test(mission): verify domain models serialization, hashing, and constraints`
3. **Task 3: Spatial Slot Tracker**
   - [x] 3.1 `feat(mission): implement multi-frame SlotTracker with spatial gating and confidence decay`
   - [x] 3.2 `test(mission): verify SlotTracker multi-frame association, noise rejection, and confirmation`
4. **Task 4: Dynamic Re-planner & Maneuver Recovery**
   - [x] 4.1 `feat(mission): implement ParkingReplanner for obstacle evasion and alignment recovery`
   - [x] 4.2 `test(mission): verify replanning generation, swept clearance checks, and failure handling`
5. **Task 5: AVP Mission Executive & State Machine**
   - [x] 5.1 `feat(mission): implement MissionExecutive state machine and lifecycle orchestration`
   - [x] 5.2 `feat(mission): implement simulation stepping with obstacle yielding and replanning loops`
   - [x] 5.3 `test(mission): verify executive nominal flow, transient obstacle hold, and recovery replan`
6. **Task 6: Visual Mission Dashboard & Telemetry Charts**
   - [x] 6.1 `feat(mission): implement BEV mission execution dashboard overlay`
   - [x] 6.2 `feat(mission): implement mission state timeline and clearance profile chart`
   - [x] 6.3 `test(mission): verify mission visualization rendering to valid PNG artifacts`
7. **Task 7: Package Exports & CLI Integration**
   - [x] 7.1 `feat(mission): export public mission API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 mission command group`
   - [x] 7.3 `feat(cli): register mission CLI group and update release audit expected count to 15`
   - [x] 7.4 `feat(pipeline): integrate --mission flag into four-camera pipeline runner`
   - [x] 7.5 `test(cli): verify mission CLI command options, scenarios, and artifact generation`
   - [x] 7.6 `test(pipeline): verify pipeline --mission execution flow`
8. **Task 8: End-to-End Documentation & Verification Gates**
   - [x] 8.1 `docs: update README with Phase 23 roadmap entry and mission CLI usage examples`
   - [x] 8.2 `docs(progress): document completed Phase 23 milestone and verification gates in PROGRESS.md`
   - [x] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
