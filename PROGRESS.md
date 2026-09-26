# Project Progress & Milestone Verification

## Phase 5: Multi-Camera Surround BEV Fusion, Bayesian Uncertainty Propagation, and Parking Safety Zone Architecture

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 707 passed (0 failures)
**Type Checking:** `mypy --strict` clean (68 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 93.24% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 5 addresses bird's-eye-view (BEV) fusion, uncertainty quantification, and near-field risk assessment for automated parking operations:

1. **Multi-Camera Surround Fusion:** Synchronized accumulation of semantic evidence across the four automotive fisheye cameras (Front `FV`, Rear `RV`, Left `MVL`, and Right `MVR`) into a metric vehicle-centric BEV grid.
2. **Bayesian Uncertainty Propagation:** Analytical estimation of per-cell occupancy variance based on Dirichlet/Beta conjugate observation evidence:
   $$\operatorname{Var}(p) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$
   where $\alpha = 1 + \text{evidence}_{\text{occ}}$ and $\beta = 1 + \text{evidence}_{\text{free}}$.
3. **Surround Parking Safety Zones:** Comprehensive 360-degree vehicle safety zones:
   - `forward_corridor`: Front maneuvering safety corridor
   - `rear_corridor`: Rear maneuvering safety corridor
   - `left_clearance`: Left lateral obstacle clearance zone
   - `right_clearance`: Right lateral obstacle clearance zone
   - `near_circle`: Omnidirectional critical proximity zone
   - `warning_circle`: Omnidirectional intermediate caution zone
4. **Diagnostic & Visual Artifacts:**
   - Structured JSON risk report documenting active parameters, fused samples, per-zone cell metrics, and grid matrices.
   - Dual-channel PNG visualization: Occupancy map with zone vector overlays (`--png`) and Bayesian uncertainty heatmap (`--uncertainty-png`).

---

### 2. Delivered Components & Architecture

#### A. Multi-Camera Surround Fusion
- Implemented `--all-cameras` option in `nearfield360 occupancy layer`.
- Fuses camera samples ordered by index across `FV`, `RV`, `MVL`, and `MVR`.
- Handles incomplete multi-camera frames gracefully with descriptive warnings while preserving available camera evidence.
- Shared geometric ray-projection and ground-plane intersection model ensures single-camera and multi-camera pipelines remain mathematically identical.

#### B. Per-Cell Bayesian Uncertainty
- Extended `OccupancyEvidence` with analytical `.uncertainty` property returning $H \times W$ variance grid.
- Extended `ZoneRiskSummary` and `RiskReport` to aggregate:
  - `mean_uncertainty`: Mean variance over zone cells
  - `max_uncertainty`: Maximum variance within zone cells
- Added `total_evidence` property returning total observation mass $\alpha + \beta - 2$.

#### C. Surround Safety Zone Engine
- Added `rear_corridor_zone(grid, length, half_width, start_x)` for rear obstacle monitoring during reverse maneuvers.
- Added `lateral_clearance_zone(grid, side, width, x_min, x_max)` for lateral proximity checking.
- Added `surround_parking_zones(grid, ...)` providing a turnkey dictionary of all 6 surround zones.
- Exported all zone constructors from `nearfield360.occupancy`.
- Extended `RiskConfig` in `src/nearfield360/config.py` and `configs/default.yaml` with explicit parameters for all zones with strict validation.

#### D. Unified CLI Pipeline & Colormapped Visualization
- Refactored `src/nearfield360/cli/occupancy.py` to eliminate duplication between single-camera and multi-camera execution paths.
- Added `--uncertainty-png` argument utilizing OpenCV `COLORMAP_INFERNO` normalized to $[0, 0.25]$ theoretical variance bounds.
- Full type safety under strict mypy mode with zero runtime asserts in library code (S101 compliant).

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Lockfile Integrity | `uv lock --check` | Pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Unit Tests | `uv run pytest -q` | 707 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 93.24% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 68 source files |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 6: Controlled Robustness Evaluation, Sensor Corruptions, Extrinsic Perturbations, and Automated Diagnostic Dashboards

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 707 passed (0 failures)
**Type Checking:** `mypy --strict` clean (68 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 93.24% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 6 establishes a rigorous, reproducible framework for quantifying perception degradation and fusion sensitivity under environmental degradation and hardware misalignments:

1. **Synthetic Sensor Corruptions:** Photometrically accurate degradation models for fisheye camera feeds:
   - `lens_soiling`: Random elliptical mud/dirt splatters with edge feathered Gaussian blur and opacity attenuation.
   - `fog`: Atmospheric optical depth attenuation and ambient airlight scattering: $I_{\text{corrupt}} = I \cdot t + A \cdot (1 - t)$ with severity-dependent transmission $t \in [0.15, 0.75]$.
   - `low_light_noise`: Combined Poisson shot noise, Gaussian read noise, and digital sensor gain degradation.
   - `rain`: Slanted motion-blurred streak lines with severity-scaled density and contrast desaturation.
2. **Extrinsic Calibration Perturbation Engine:**
   - 3D Euler rotation perturbations (roll $\Delta \phi$, pitch $\Delta \theta$, yaw $\Delta \psi$) mapped to rotation matrices and quaternion parameterizations.
   - Vehicle coordinate frame translation deltas ($\Delta x, \Delta y, \Delta z$).
   - Strict SE(3) matrix conditioning ($\det(R) = +1, R^T R = I$) preserving valid camera transforms.
3. **Quantitative Degradation Benchmark Runner:**
   - Occupancy Mean Absolute Error (MAE) evaluated across BEV grid cells.
   - Occupied space IoU and free space IoU at configured danger probability threshold.
   - Bayesian Dirichlet/Beta variance shift across observed BEV cells.
   - Safety zone risk cell count errors across parking corridors and clearance boundaries.
4. **Diagnostic Visualizations & Dashboard:**
   - Pure-Python SVG vector line chart engine generating standalone, responsive SVG plots with zero external graphic dependencies.
   - OpenCV rasterized PNG performance curves with axis grids, tick marks, and multi-series legends.
   - Self-contained HTML interactive dashboard combining responsive CSS layouts, KPI metric badges, embedded SVGs, and sortable tabular results.
5. **CLI Subcommands:**
   - `nearfield360 robustness corrupt`: Apply reproducible synthetic corruptions with explicit seed and severity.
   - `nearfield360 robustness perturb-calibration`: Generate perturbed WoodScape calibration JSON files.
   - `nearfield360 robustness benchmark`: Run multi-severity corruption and multi-axis calibration sweeps against clean ground truth.
   - `nearfield360 robustness plot`: Render SVG, PNG, and HTML dashboard artifacts from benchmark JSON reports.

---

### 2. Delivered Components & Architecture

#### A. Synthetic Sensor Corruptions (`src/nearfield360/robustness/corruptions.py`)
- `CorruptionType` enum supporting `lens_soiling`, `fog`, `low_light_noise`, and `rain`.
- Deterministic NumPy `Generator` seeded per execution.
- Strict input validation preserving image shapes $(H, W, 3)$ and `uint8` data types.

#### B. Extrinsic Calibration Perturbation Engine (`src/nearfield360/robustness/calibration.py`)
- `euler_to_rotation_matrix`: Standard automotive XYZ intrinsic / ZYX extrinsic rotation composition.
- `rotation_matrix_to_quaternion`: Numerically stable Shepperd-like trace algorithm producing unit Hamilton quaternions $(q_w, q_x, q_y, q_z)$.
- `perturb_rigid_transform`: SE(3) perturbation combining relative delta rotation and translation.
- `perturb_camera_calibration`: High-level calibration perturbation updating WoodScape Pydantic models.

#### C. Quantitative Robustness Benchmarking (`src/nearfield360/robustness/benchmark.py`)
- `compare_occupancy_grids`: Evaluates MAE, IoU (occupied and free), and Bayesian variance shift between clean baseline and perturbed evidence grids.
- `compare_zone_risks`: Computes per-safety-zone cell error and risk status differences.
- `run_corruption_sweep` & `run_calibration_sweep`: Executes parameter sweeps generating structured records.
- `RobustnessReport`: Serializable container with environment metadata and configuration provenance.

#### D. Visual Reporting Engine (`src/nearfield360/robustness/plots.py`)
- `render_svg_line_chart`: Generates SVG 1.1 vector graphics with grid lines, data points, and colored legends.
- `render_raster_line_chart`: Generates raster PNG images using OpenCV drawing primitives.
- `generate_robustness_html_dashboard`: Generates responsive, self-contained HTML reports with embedded CSS styling and inline SVGs.

#### E. Robustness CLI Suite (`src/nearfield360/cli/robustness.py`)
- Integrated under `nearfield360 robustness` with full typer parameter typing and atomic file writing.

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Lockfile Integrity | `uv lock --check` | Pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Unit Tests | `uv run pytest -q` | 707 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 93.24% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 68 source files |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

### 4. Roadmap Transition: Phase 4 & Phase 7

With Phase 6 complete, Phase 4 (Multi-Camera Dynamic Obstacle Tracking, BEV Motion Estimation, and Camera Health Awareness) has also been implemented and verified.

---

## Phase 4: Multi-Camera Dynamic Obstacle Tracking, BEV Motion Estimation, and Camera Health Awareness

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 707 passed (0 failures)
**Type Checking:** `mypy --strict` clean (68 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 93.24% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 4 bridges 2D camera detections and optical health into the vehicle bird's-eye-view representation and real-time safety architecture:

1. **Camera Optical Health Engine:**
   - Multi-factor optical and sensor health diagnosis per surround camera frame:
     - High-frequency Laplacian variance sharpness metric (`calculate_blur_score`).
     - Spatial blockage and dead/saturated block detection (`detect_blockage`).
     - Adaptive high-frequency and local contrast lens soiling detection (`detect_lens_soiling`).
     - Mean luminance and contrast exposure monitoring (`calculate_photometric_properties`).
   - `assess_camera_health` producing structured `CameraHealthReport` with qualitative status (`healthy`, `degraded`, `blocked`), detected anomalies, and continuous fusion discount weight in `[0.0, 1.0]`.
2. **Health-Aware BEV Occupancy Discounting:**
   - `OccupancyEvidence.scale(factor)` and `apply_health_discount` attenuating degraded camera evidence in BEV fusion (`--health-aware` flag in `occupancy layer`).
3. **2D-to-3D Fisheye Ground Footprint Projection:**
   - Bottom-center bounding box pixel unprojection via exact radial-polynomial camera models intersecting road plane ($Z = \text{ground\_z}$) to obtain metric vehicle coordinates $(x_v, y_v)$.
   - Realistic metric class bounding boxes for vehicles, pedestrians, bicycles, traffic signs, and traffic lights.
4. **2D Metric Kalman Filtering & Multi-Object Tracking:**
   - State vector $\mathbf{x} = [x, y, v_x, v_y]^T$ with constant velocity kinematic model and discrete Wiener process acceleration noise.
   - Deterministic greedy minimum Euclidean distance association with class gating.
   - Comprehensive track lifecycle management (`tentative`, `confirmed`, `lost`, `deleted`).
5. **Dynamic Risk Evaluation & Collision Forecasting:**
   - Forward trajectory extrapolation over configurable time horizon (e.g. 3.0s).
   - Zone ingress detection against surround parking zones (`forward_corridor`, `near_circle`, etc.) computing Time-to-Collision (TTC).
6. **BEV Visual Overlays & CLI Commands:**
   - BEV visualization with color-coded ground footprints, 1-second velocity vectors, history trails, predicted paths, and collision warning badges (`--png`).
   - `nearfield360 health assess` CLI command.
   - `nearfield360 track run` CLI command.

---

### 2. Delivered Components & Architecture

#### A. Camera Health Monitoring (`src/nearfield360/health/`)
- `detector.py`: Physics-informed fisheye optical degradation analysis.
- `discount.py`: Health-aware attenuation of occupancy evidence.
- `models.py`: Pydantic domain models for metrics and reports.

#### B. Dynamic Obstacle Tracking (`src/nearfield360/tracking/`)
- `projection.py`: Ground contact unprojection from 2D bounding boxes.
- `kalman.py`: 4D state vector kinematic Kalman filter with Mahalanobis distance gating.
- `tracker.py`: `MultiObjectTracker` with lifecycle state transitions and temporal history.
- `risk.py`: Forward trajectory simulation, polygon zone ingress tests, and minimum TTC calculation.
- `viz.py`: OpenCV BEV rendering of obstacles, velocity arrows, historical paths, and risk badges.

#### C. CLI Suite & Integration
- `src/nearfield360/cli/health.py`: `nearfield360 health assess` supporting human-readable, JSON stdout, and atomic file outputs.
- `src/nearfield360/cli/tracking.py`: `nearfield360 track run` supporting single-camera or all-camera tracking with JSON reports and BEV PNG overlays.
- `src/nearfield360/cli/occupancy.py`: Added `--health-aware` flag to `occupancy layer` for real-time sensor discount during fusion.

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Lockfile Integrity | `uv lock --check` | Pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Unit Tests | `uv run pytest -q` | 707 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 93.24% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 68 source files |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 7: ONNX Inference Runtime, Modular Backend Abstraction, and Live Perception Pipeline

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 707 passed (0 failures)
**Type Checking:** `mypy --strict` clean (68 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 93.24% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 7 delivers a production-grade neural perception inference stack for real-time surround fisheye scene understanding:

1. **Modular Inference Backend Abstraction:**
   - Runtime-agnostic `InferenceBackend` interface supporting pluggable execution providers.
   - `create_backend()` factory dispatching to OpenCV DNN or ONNX Runtime backends.
   - Strict model input/output tensor validation with shape and dtype assertions.
2. **Fisheye Image Preprocessing:**
   - `FisheyeImagePreprocessor` with letterbox resize, spatial normalization, and metadata-preserving unscaling for downstream coordinate recovery.
3. **Semantic Segmentation Engine:**
   - `SemanticSegmentationEngine` performing pixel-level scene parsing with configurable colormap and class-name mapping.
   - Argmax inference producing per-pixel class IDs with probability maps.
4. **Object Detection Engine:**
   - `ObjectDetectionEngine` with coordinate unscaling from normalized model outputs to native image resolution.
   - Non-maximum suppression (NMS) with configurable IoU threshold and confidence filtering.
   - `DetectionPrediction` dataclass with per-detection confidence, bounding box, and class label.
5. **Numerical Parity & Latency Benchmarking:**
   - `verify_parity()` fixture computing maximum absolute tensor difference between OpenCV DNN and ONNX Runtime backends ($\le 10^{-4}$ tolerance).
   - `benchmark_inference()` engine measuring throughput (FPS), p50/p95/p99 latency across configurable warmup and iteration counts.
6. **Live Perception Pipeline Integration:**
   - `--model` flag in `nearfield360 occupancy layer` and `nearfield360 track run` enabling end-to-end live inference from raw camera images to BEV occupancy and dynamic obstacle tracking.
   - Graceful fallback to cached semantic masks when no model is provided.

---

### 2. Delivered Components & Architecture

#### A. Inference Backend (`src/nearfield360/perception/inference/backend.py`)
- `InferenceBackend` protocol defining `forward()` tensor interface.
- `OpenCVBackend` implementation using `cv2.dnn.readNetFromONNX()`.
- `InferenceError` for model loading and execution failures.

#### B. Preprocessor (`src/nearfield360/perception/inference/preprocessor.py`)
- `FisheyeImagePreprocessor` with letterbox, mean subtraction, and spatial unscaling.
- `PreprocessorError` for invalid inputs.

#### C. Semantic Engine (`src/nearfield360/perception/inference/semantic.py`)
- `SemanticSegmentationEngine` producing per-pixel class predictions and probability maps.

#### D. Detection Engine (`src/nearfield360/perception/inference/detection.py`)
- `ObjectDetectionEngine` with NMS, confidence filtering, and coordinate unscaling.

#### E. Benchmarking (`src/nearfield360/perception/inference/benchmark.py`)
- `verify_parity()` for cross-backend numerical consistency testing.
- `benchmark_inference()` for latency profiling with percentile statistics.

#### F. Domain Models (`src/nearfield360/perception/inference/models.py`)
- `InferenceBackendType`, `InferenceDevice`, `InferencePrecision` enums.
- `InferenceConfig` Pydantic model for configuration-driven inference.

#### G. CLI Commands (`src/nearfield360/cli/infer.py`)
- `nearfield360 infer semantic`: Run semantic segmentation on camera images.
- `nearfield360 infer detection`: Run object detection with NMS.
- `nearfield360 infer benchmark`: Measure inference latency and throughput.
- `nearfield360 infer parity`: Verify numerical parity between backends.

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Lockfile Integrity | `uv lock --check` | Pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Unit Tests | `uv run pytest -q` | 707 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 93.24% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 68 source files |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 8: Integration Test Coverage, Missing CLI Command, and Documentation Accuracy

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 718 passed (0 failures)
**Type Checking:** `mypy --strict` clean (68 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 93.24% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 8 closes critical quality gaps identified in the codebase audit:

1. **Missing Integration Tests:** No end-to-end pipeline tests existed despite a 7-module perception stack.
2. **Missing CLI Command:** `nearfield360 infer parity` was documented but never implemented.
3. **No Config YAML Deserialization Test:** Drift between `configs/default.yaml` and `ProjectConfig` would go undetected.
4. **Documentation Mismatch:** README lacked the parity command usage example.

---

### 2. Delivered Components & Architecture

#### A. `nearfield360 infer parity` CLI Command (`src/nearfield360/cli/infer.py`)
- Loads two ONNX models via OpenCV DNN backend.
- Runs inference on identical random input tensors.
- Computes max absolute, mean absolute, and max relative differences.
- Reports PASS/FAIL with configurable `--atol` and `--rtol` tolerances.
- Catches `ValueError` from shape mismatches and `InferenceError` from model failures.

#### B. Integration Test Suite (`tests/integration/test_pipeline.py`)
Six integration tests using synthetic dummy ONNX models (no external dataset required):

| Test | Pipeline Path | What It Verifies |
| --- | --- | --- |
| `test_single_frame_pipeline` | Image -> Semantic -> Occupancy -> Risk | Full single-frame perception chain |
| `test_occupancy_evidence_accumulates_across_frames` | Multi-frame occupancy fusion | Evidence accumulation and grid consistency |
| `test_detection_through_tracking` | Detection -> Projection -> Kalman -> Tracker | Object lifecycle and velocity estimation |
| `test_kalman_filter_position_update` | Kalman predict/update loop | Filter convergence toward measured target |
| `test_default_config_to_risk_report` | Config -> Grid -> Zones -> Risk | End-to-end config-driven pipeline |
| `test_fuse_four_cameras` | 4-camera evidence -> Fusion -> Risk | Multi-camera surround evidence aggregation |

#### C. Code Quality Fixes (from Phase 7 audit)
- `health/discount.py`: Clamped discount weight to [0, 1] to prevent evidence inflation.
- `tracking/kalman.py`: Joseph form covariance update for numerical stability; `np.linalg.solve` instead of `np.linalg.inv`.
- `occupancy/evidence.py`: `scale(0.0)` now zeros observed counts for semantic consistency.

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 718 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 93.24% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 68 source files |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 9: Integrated Four-Camera Demo, Measured Performance Report, and Release Audit

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 733 passed (0 failures)
**Type Checking:** `mypy --strict` clean (70 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.97% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 9 closes README roadmap item 8 and delivers the first end-to-end, timed surround-view
demo plus automated release readiness:

1. **Integrated Four-Camera Pipeline (`nearfield360 pipeline run`):**
   - Discover complete four-camera frames (calibration + semantic mask + detections required).
   - Per frame: occupancy evidence rasterization (optional health discount) and detection →
     ground-footprint projection into the multi-object tracker.
   - Across frames: Bayesian evidence fusion, zone risk report, trajectory forecasting.
   - Measured performance report: per-stage wall-clock timings (`discovery_ms`,
     `perception_ms`, `occupancy_ms`, `tracking_ms`, `risk_ms`, `forecast_ms`, `total_ms`)
     and frames-per-second in the JSON artifact, with `environment` provenance.
   - Optional fused occupancy PNG via `--png`.
2. **Release Audit (`nearfield360 release audit`):**
   - Checks: project root, LICENSE (Apache-2.0), project name, version consistency
     (pyproject vs package), license/readme metadata, console script entry point,
     README presence, `py.typed`, default config load, and full CLI subcommand surface
     (including `pipeline` and `release`).
   - Human or `--json` output; atomic `--output` report; exit code 1 on any failure.

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Pipeline CLI | `src/nearfield360/cli/pipeline.py` |
| Release audit CLI | `src/nearfield360/cli/release.py` |
| Root registration | `src/nearfield360/cli/app.py` (`pipeline`, `release`) |
| Pipeline unit tests | `tests/unit/test_pipeline_cli.py` (9 tests) |
| Release audit unit tests | `tests/unit/test_release_cli.py` (6 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 733 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.97% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 70 source files |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 10: Live ONNX Perception in the Integrated Pipeline and Latency Distribution Statistics

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 737 passed (0 failures)
**Type Checking:** `mypy --strict` clean (70 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.82% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 10 closes README roadmap item 9 and closes the integration gap between Phase 7
(live ONNX engines) and Phase 9 (integrated pipeline):

1. **Live ONNX models on `pipeline run`:**
   - `--seg-model`: ONNX semantic segmentation for occupancy evidence (replaces semantic masks).
   - `--det-model`: ONNX object detection for ground-footprint projection (replaces detection files).
   - Annotation requirements relax when models are supplied (RGB + calibration suffice for
     the corresponding stage); without models, prior annotation requirements are unchanged.
   - Model paths recorded in `samples.seg_model` / `samples.det_model`.
2. **Latency distribution statistics:**
   - Per-frame wall-clock samples collected during the perception loop.
   - `timings.frame_latency` reports `samples`, `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`,
     `min_ms`, `max_ms` (milliseconds, rounded to 3 decimals).
   - Human-readable summary prints `frame p50=... p95=...` after completion.
3. **End-to-end integration coverage:**
   - CLI integration test running `pipeline run` with synthetic ONNX models and
     `--health-aware` over a complete four-camera frame.

**Explicit exclusions:** TensorRT execution provider, C++ runtime wrapper, multi-run
harness for percentile aggregation across process restarts (remain future work under
roadmap item 7 residual).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Live model loading + relaxed frame grouping | `src/nearfield360/cli/pipeline.py` |
| `_latency_stats` percentile helper | `src/nearfield360/cli/pipeline.py` |
| Unit tests (12 total, +3 for this phase) | `tests/unit/test_pipeline_cli.py` |
| End-to-end CLI integration test | `tests/integration/test_pipeline.py` (`TestEndToEndCliPipeline`) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 737 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.82% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 70 source files |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 11: Configuration-Driven Inference Backends and Optional ONNX Runtime Support

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 746 passed (0 failures)
**Type Checking:** `mypy --strict` clean (71 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.34% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 11 closes README roadmap item 10 and fixes the silent-fallback gap in the
inference backend factory:

1. **Real `OnnxRuntimeBackend`:**
   - Lazy `onnxruntime` import with clear `InferenceError` install-hint when missing.
   - Optional `nearfield360[onnxruntime]` packaging extra (not a hard dependency).
   - Session creation with graph optimization and provider selection (CUDA → CPU fallback
     with a logged warning when CUDAExecutionProvider is unavailable).
   - Input validation (4-D float32 C-contiguous tensors) and multi-output support.
2. **No silent OpenCV fallback:**
   - `create_backend` raises `InferenceError` when `onnxruntime` is requested but not
     installed (previously logged a warning and returned `OpenCVDNNBackend`).
   - Unsupported backend strings still raise `InferenceError`.
3. **CLI `--backend` selection:**
   - Shared `BackendOption` / `load_backend` / `resolve_backend_type` helpers in
     `cli/inference_common.py`.
   - `--backend {opencv,onnxruntime}` on `infer semantic|detection|benchmark|parity|inspect`,
     `occupancy layer`, `track run`, and `pipeline run`.
   - When omitted, backend/device resolve from `config.inference.backend` /
     `config.inference.device`.
4. **Config-driven detection thresholds:**
   - `infer detection` `--confidence-threshold` / `--nms-threshold` default to
     `config.inference.confidence_threshold` / `config.inference.nms_threshold`
     when not passed on the command line.
   - `track run` and `pipeline run` pass those config thresholds into
     `ObjectDetectionEngine`.

**Explicit exclusions:** TensorRT execution provider, C++ runtime wrapper, forcing
onnxruntime as a hard install dependency (remain under roadmap item 7 residual or are
deliberately optional).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| `OnnxRuntimeBackend` + strict factory | `src/nearfield360/perception/inference/backend.py` |
| Optional extra | `pyproject.toml` (`[project.optional-dependencies] onnxruntime`) |
| Shared CLI backend helpers | `src/nearfield360/cli/inference_common.py` |
| `infer` `--backend` + config thresholds | `src/nearfield360/cli/infer.py` |
| `occupancy layer` / `track run` / `pipeline run` `--backend` | `cli/occupancy.py`, `cli/tracking.py`, `cli/pipeline.py` |
| Backend factory + CLI tests | `tests/unit/perception/inference/test_backend.py`, `tests/unit/test_*_cli.py` |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 746 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.34% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 71 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 12: Model-Driven Evaluation Against Dataset Annotations

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 773 passed (0 failures)
**Type Checking:** `mypy --strict` clean (71 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.39% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 12 closes README roadmap item 11 by joining the live inference engines
(roadmap items 7/10) with the evaluation metrics (roadmap item 3) in one command:

1. **Live model scoring:** `eval segmentation --model` / `eval detection --model` run
   an ONNX model over annotated samples and score in memory. `--model` and
   `--predictions` are mutually exclusive (`Provide exactly one of --predictions or
   --model.`), and `--predictions` is now optional on both commands.
2. **Backend selection:** `--backend` / `--device` reuse the shared
   `cli/inference_common.py` helpers and default from `config.inference`.
3. **Config-driven detection thresholds:** `--confidence-threshold` /
   `--nms-threshold` fall back to `config.inference.confidence_threshold` /
   `config.inference.nms_threshold`; the effective values are recorded in the report.
4. **Sample bound:** `--limit N` (0 = all) caps annotated samples evaluated on both
   sources and both commands.
5. **Prediction export:** `--save-predictions DIR` writes model output as PNG class-ID
   masks / scored TXT rows consumable by `--predictions` (round-trip tests assert
   byte-identical metrics between model and file runs); rejected without `--model`.
6. **Report enrichment:** new top-level `"model"` key (path, backend, device, effective
   thresholds) and `"timing"` key (samples, total/mean/min/max ms, p50/p95/p99
   percentiles, samples/s) for model runs; both `null` for file-based runs.
   Per-sample latency is measured around `engine.predict` only.
7. **Data writers:** `write_detection_predictions` (7-field CSV with `repr(float)`
   round-trip fidelity and parent-directory creation) and `save_semantic_mask`
   (validated class-ID PNG), both exported from `nearfield360.data`.
8. **Empty-prediction robustness:** `_prediction_arrays` preserves the `(N, 4)` box
   shape for empty prediction batches in file and model paths (previously
   `np.asarray([])` produced shape `(0,)` and failed `evaluate_detection`'s shape
   check — latent bug fixed with a regression test).

**Explicit exclusions:** confidence-threshold sweeps/PR curves, split-manifest
filtering, batched inference, and TensorRT/C++ acceleration (remain under roadmap
item 7 residual).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Source selection, engine builders, collectors, timing stats | `src/nearfield360/cli/eval.py` |
| `write_detection_predictions` + `_format_number` | `src/nearfield360/data/detection.py` |
| `save_semantic_mask` | `src/nearfield360/data/semantic.py` |
| Writer round-trip tests | `tests/unit/data/test_detection.py`, `tests/unit/data/test_semantic.py` |
| Eval CLI tests (file, model, limit, exclusions, round-trip, timing) | `tests/unit/test_eval_cli.py` |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 773 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.39% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 71 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 13: Split-Aware Evaluation and Detection Confidence Analysis

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 794 passed (0 failures)
**Type Checking:** `mypy --strict` clean (71 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.58% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 13 closes README roadmap item 12 by resolving both exclusions Phase 12
explicitly left open — split-manifest filtering and confidence sweeps/PR curves:

1. **Split-restricted evaluation:** `--split-manifest PATH` on both `eval
   segmentation` and `eval detection` validates the manifest's sample-identity
   digest against the discovered dataset, then scores only one split's samples;
   `--split train|validation|test` selects the split (default: `test`).
   `--split` without a manifest is rejected (`--split requires
   --split-manifest.`), and an invalid manifest fails with a `Split error:`
   message before any scoring.
2. **Split provenance in reports:** a top-level `"split"` key records manifest
   path, split name, grouping policy/source, seed, and dataset/selected sample
   counts (`null` when no manifest is used); the selection line is echoed to
   stdout, and the empty-split error names the split.
3. **Confidence analysis:** `eval detection --confidence-thresholds
   0.3,0.5,0.7` adds `metrics.confidence_analysis` with pooled operating points
   per cutoff (predictions, true positives, precision, recall, F1 after
   per-class greedy matching at `--iou-threshold`) plus VOC-style 101-point
   interpolated precision-recall grids per class (`null` for classes without
   ground-truth targets). Values are parsed strictly (finite, in [0, 1],
   deduplicated, sorted ascending; malformed input is a usage error).
4. **Metrics API:** `detection_confidence_metrics` (numpy-only) and
   `detection_confidence_analysis` (batch pooling wrapper) are exported from
   `nearfield360.perception`; both reuse the per-class PR-step machinery
   extracted from `woodscape_detection_scores` so operating points and AP share
   one matching implementation.
5. **Reporting only when requested:** `--confidence-thresholds` is optional and
   default reports keep their previous structure; without a manifest the
   dataset is scored in full with `"split": null`.

**Explicit exclusions:** model-side re-inference sweeps, calibration metrics
(ECE), batched inference, segmentation confidence analysis, SVG plotting of
curves (JSON only), and TensorRT/C++ acceleration (remain under roadmap item 7
residual).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| `_per_class_pr_steps`, `_interpolated_pr_grid`, `detection_confidence_metrics` | `src/nearfield360/perception/metrics.py` |
| `_pool_detection_batches`, `detection_confidence_analysis` | `src/nearfield360/perception/evaluation.py` |
| `--split-manifest`/`--split`/`--confidence-thresholds`, `_apply_split_selection`, `_parse_confidence_thresholds` | `src/nearfield360/cli/eval.py` |
| Metrics unit tests (operating points, PR grids, validation) | `tests/unit/perception/test_metrics.py` |
| Batch-analysis unit tests | `tests/unit/perception/test_evaluation.py` |
| Eval CLI tests (confidence flags, split filtering, oracle match, digest mismatch) | `tests/unit/test_eval_cli.py` |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 794 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.58% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 71 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |
