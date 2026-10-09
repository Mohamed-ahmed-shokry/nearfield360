# Autonomous Valet Parking (AVP) Development Roadmap

## Phase 24: Parking Facility HD Vector Mapping, Global Topological Route Planning & Multi-Sensor Pose Graph Localization

### 1. Objective & Expected Outcome

Deliver an end-to-end Parking Facility HD Vector Mapping, Topological Route Planning, and Multi-Sensor Pose Graph Localization engine that elevates NearField360 from local, vehicle-relative maneuvering into globally aware, facility-wide autonomous valet parking. The system models structured parking garages and surface lots (driving lanes, one-way corridors, parking bays, structural boundaries, and navigation waypoints), plans kinematically admissible global routes from entrance drop-off zones to target bays, and estimates vehicle metric poses by fusing kinematic bicycle odometry with perceived 3D parking slot landmark associations.

### 2. Scope

- **Configuration (`src/nearfield360/config.py`, `configs/default.yaml`)**:
  - `MappingConfig` specifying default lane width, facility speed limits, odometry process noise, landmark association distance gates, slot observation noise, and localization uncertainty thresholds.
- **Domain Models (`src/nearfield360/mapping/models.py`)**:
  - `FacilityMap` representing complete facility metadata, spatial bounds `[x_min, x_max, y_min, y_max]`, lanes, parking bays, structural obstacles, and topological waypoints.
  - `FacilityLane` driving corridor polylines with lane width, speed limit, directionality (`ONE_WAY`, `TWO_WAY`), and connectivity.
  - `FacilitySlot` pre-mapped parking bays with metric 4-corner polygons, slot types (`PARALLEL`, `PERPENDICULAR`, `SLANTED`), access lane IDs, and reservation/vacancy status.
  - `FacilityObstacle` structural boundaries, pillars, curbs, and perimeter walls.
  - `FacilityWaypoint` topological navigation nodes (entrances, exits, intersections, slot access points).
  - `GlobalRoute` and `RouteWaypoint` schemas with curvature, speed targets, corridor bounds, and turn annotations.
  - `LocalizationState` and `PoseEstimate` with mean pose $(x, y, \theta)$ and covariance $\Sigma \in \mathbb{R}^{3 \times 3}$.
- **Facility Map Builder & Synthesizer (`src/nearfield360/mapping/builder.py`)**:
  - Programmatic synthesis of standardized multi-aisle indoor parking garages and multi-bay outdoor parking facilities.
  - Topology connectivity verification, slot accessibility verification, and boundary integrity checks.
- **Topological Graph & Global Route Planner (`src/nearfield360/mapping/router.py`)**:
  - Directed topological navigation graph built from lanes, intersections, and slot connectors.
  - A* / Dijkstra optimal route planner respecting one-way constraints, turn angle penalties, and distance weights.
  - Generates smooth discretized global routes with corridor bounds and speed profiling.
- **Multi-Sensor Pose Estimation & Localization (`src/nearfield360/mapping/localization.py`)**:
  - Dead-reckoning kinematic bicycle odometry propagation.
  - Nearest-neighbor / Mahalanobis gating associating perceived 3D parking slots with mapped facility bays.
  - Extended Kalman Filter (EKF) / Pose Graph state estimator updating vehicle pose and covariance.
  - Position uncertainty tracking ($\sqrt{\sigma_x^2 + \sigma_y^2}$), heading uncertainty ($\sigma_\theta$), and innovation residuals.
- **Visual Facility Map & Localization Dashboard (`src/nearfield360/mapping/viz.py`)**:
  - BEV facility map rendering: lanes, direction arrows, slot bays, structural boundaries, and global routes.
  - Localization overlay: ground truth vs estimated trajectory, covariance error ellipses ($2\sigma$), and landmark association vectors.
  - Multi-panel telemetry dashboard: localization error profiles and uncertainty convergence.
- **CLI Command Group & Pipeline Integration (`src/nearfield360/cli/map.py`, `src/nearfield360/cli/pipeline.py`)**:
  - Dedicated CLI command group: `nearfield360 map` with `info`, `build`, `route`, and `localize` subcommands.
  - Release readiness audit updated and verified for 16 CLI command groups.
  - Four-camera pipeline integration: `nearfield360 pipeline run --map <path> --target-slot <id>`.
- **Comprehensive Verification Suite**:
  - Unit and integration tests covering configuration, domain models, builder, router, localization, visualization, CLI, and pipeline integration.

### 3. Acceptance Criteria

| ID | Criterion | Verification Method |
|---|---|---|
| AC-1 | `MappingConfig` defined with Pydantic validation and configured in `configs/default.yaml` | `uv run pytest tests/unit/test_config.py` |
| AC-2 | Pydantic domain models for facility maps, lanes, slots, obstacles, waypoints, routes, and pose estimates | `uv run pytest tests/unit/mapping/test_mapping_models.py` |
| AC-3 | `FacilityBuilder` synthesizes valid benchmark indoor garages and outdoor lots with accessibility checks | `uv run pytest tests/unit/mapping/test_builder.py` |
| AC-4 | `GlobalRouter` constructs directed graph, enforces one-way constraints, and computes optimal A* routes | `uv run pytest tests/unit/mapping/test_router.py` |
| AC-5 | `PoseEstimator` fuses dead-reckoning odometry and slot landmark associations with covariance bounds | `uv run pytest tests/unit/mapping/test_localization.py` |
| AC-6 | Visualization renders BEV facility map overlays, route paths, and localization telemetry dashboards | `uv run pytest tests/unit/mapping/test_mapping_viz.py` |
| AC-7 | Dedicated CLI command group `nearfield360 map` executes `info`, `build`, `route`, and `localize` | `uv run pytest tests/unit/test_map_cli.py` |
| AC-8 | Root CLI registers `map` command group, passing release audit with 16 groups | `uv run nearfield360 release audit` |
| AC-9 | Surround pipeline integrates `--map` and `--target-slot` routing and localization | `uv run pytest tests/unit/test_pipeline_map.py` |
| AC-10 | Full test suite passes with zero failures, >85% coverage, clean `mypy`, and clean `ruff` | `uv run pytest`, `uv run mypy`, `uv run ruff check .` |

### 4. Validation Plan

1. Unit tests covering config validation, domain models, builder, router, localization EKF, visual rendering, and CLI commands.
2. CLI integration tests verifying `nearfield360 map info`, `map build`, `map route`, and `map localize` with valid JSON/PNG artifact generation.
3. Pipeline integration tests verifying `nearfield360 pipeline run --map <map.json> --target-slot <id>`.
4. Static analysis: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.
5. Release audit: `uv run nearfield360 release audit`.
6. Full test suite with coverage: `uv run pytest --cov=nearfield360`.

### 5. Exclusions

- Multi-floor elevator/ramp transition kinematics and 3D multi-level point cloud registration.
- V2X cellular infrastructure protocols and automated parking fee payment gateways.
- Physical CAN bus transceiver decoding (simulated discrete-time vehicle state interface used).

### 6. Task List & Micro-Commit Plan

1. **Task 1: Roadmap & Config Foundation**
   - [x] 1.1 `docs: establish Phase 24 parking facility mapping and localization roadmap`
   - [x] 1.2 `feat(config): add MappingConfig and update default yaml configuration`
   - [x] 1.3 `test(config): add unit tests for MappingConfig validation`
2. **Task 2: Facility Map Domain Models & Validation**
   - [x] 2.1 `feat(mapping): implement facility map domain models (lanes, slots, obstacles, waypoints)`
   - [x] 2.2 `feat(mapping): implement JSON serialization and spatial bounds verification`
   - [x] 2.3 `test(mapping): verify facility map domain models, serialization, and geometry checks`
3. **Task 3: Facility Map Builder & Benchmark Synthesizer**
   - [x] 3.1 `feat(mapping): implement facility map builder for multi-aisle garages and parking lots`
   - [x] 3.2 `test(mapping): verify facility map builder topology, slot accessibility, and layout generation`
4. **Task 4: Topological Graph & Global Route Planner**
   - [x] 4.1 `feat(mapping): implement topological routing graph from lanes and waypoints`
   - [x] 4.2 `feat(mapping): implement A* global route planner with turn penalties and corridor bounds`
   - [x] 4.3 `test(mapping): verify global route planner optimality, one-way enforcement, and infeasible targets`
5. **Task 5: Multi-Sensor Pose Estimation & Landmark Localization**
   - [x] 5.1 `feat(mapping): implement kinematic bicycle odometry dead-reckoning engine`
   - [x] 5.2 `feat(mapping): implement slot landmark association and EKF pose estimator`
   - [x] 5.3 `test(mapping): verify localization accuracy, covariance convergence, and noise rejection`
6. **Task 6: Facility Map & Localization Visualizer**
   - [x] 6.1 `feat(mapping): implement BEV facility map and global route rendering`
   - [x] 6.2 `feat(mapping): implement localization covariance ellipse and telemetry dashboard rendering`
   - [x] 6.3 `test(mapping): verify mapping and localization visualization rendering to valid PNG artifacts`
7. **Task 7: Package Exports & CLI Command Group**
   - [x] 7.1 `feat(mapping): export public mapping API from package init`
   - [x] 7.2 `feat(cli): implement nearfield360 map command group (info, build, route, localize)`
   - [x] 7.3 `feat(cli): register map CLI group and update release audit expected count to 16`
   - [x] 7.4 `feat(pipeline): integrate --map and --target-slot flags into pipeline runner`
   - [x] 7.5 `test(cli): verify map CLI command options, outputs, and JSON/PNG artifacts`
   - [x] 7.6 `test(pipeline): verify pipeline map integration and routing workflow`
8. **Task 8: End-to-End Documentation, Self-Review & Delivery**
   - [ ] 8.1 `docs: update README with Phase 24 mapping capabilities and CLI usage examples`
   - [ ] 8.2 `docs(progress): document completed Phase 24 milestone, Decision Log, and verification gates`
   - [ ] 8.3 `test: verify full suite, strict typing, linting, formatting, coverage, and release audit`
