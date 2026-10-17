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

---

## Phase 14: Segmentation Confidence Calibration and Evaluation Report Rendering

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 816 passed (0 failures)
**Type Checking:** `mypy --strict` clean (71 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.55% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 14 closes README roadmap item 13 by delivering confidence calibration quantification for semantic segmentation and vector graphic visualization for perception evaluation reports:

1. **Semantic Confidence Calibration & ECE:**
   - `eval segmentation --confidence-bins N` pools per-pixel softmax prediction confidences from live ONNX model runs into a structured reliability table.
   - Calculates Expected Calibration Error (ECE) across non-empty confidence bins:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   - Reports `metrics.confidence_analysis` with bin counts, pixel counts, per-bin metrics (accuracy, mean confidence, bounds), overall mean confidence, and ECE.
   - Validates bin count strictly ($N \ge 2$, integer); requires `--model` mode.
2. **Evaluation Report Plotting (`nearfield360 eval plot`):**
   - Renders prior eval JSON reports into publication-quality standalone SVG vector charts using the pure-Python plot engine.
   - **Detection Charts:**
     - `pr_curves.svg`: Multi-class precision-recall curves from 101-point interpolated grids.
     - `operating_points.svg`: Precision, recall, and F1 curves across confidence score cutoffs.
   - **Segmentation Reliability Diagrams:**
     - `reliability.svg`: Mean confidence vs. empirical accuracy against the diagonal perfect calibration baseline.
   - Refuses to overwrite existing SVG files unless `--overwrite` is specified.
   - Gracefully rejects malformed or non-eval reports with descriptive errors.
3. **End-to-End Pipeline Integration:**
   - Seamless workflow from evaluation execution (`eval segmentation --confidence-bins` or `eval detection --confidence-thresholds`) directly to SVG visualization (`eval plot`).

**Explicit exclusions:** Detection-side ECE, raster PNG charts, HTML dashboards, batched inference, and TensorRT/C++ acceleration (remain under roadmap item 7 residual or future work).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| `semantic_confidence_reliability` + ECE | `src/nearfield360/perception/metrics.py` |
| `semantic_confidence_analysis` pooling wrapper | `src/nearfield360/perception/evaluation.py` |
| `eval segmentation --confidence-bins` flag | `src/nearfield360/cli/eval.py` |
| `eval plot` command & `_detection_charts` / `_reliability_charts` | `src/nearfield360/cli/eval.py` |
| Metrics unit tests (calibration bins, ECE calculation) | `tests/unit/perception/test_metrics.py` |
| Evaluation pooling unit tests | `tests/unit/perception/test_evaluation.py` |
| Eval CLI unit & end-to-end tests (bins flag, plot rendering, overwrite protection, malformed data handling) | `tests/unit/test_eval_cli.py` (57 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 816 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.55% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 71 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 15: Comparative Evaluation Reporting and Regression Analysis

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 836 passed (0 failures)
**Type Checking:** `mypy --strict` clean (72 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 92.07% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 15 closes README roadmap item 14 by introducing automated comparative evaluation reporting and regression gate checking across segmentation and detection runs:

1. **Evaluation Report Comparison Engine (`src/nearfield360/perception/comparison.py`):**
   - Runtime-agnostic report diffing comparing scalar summary metrics, class-level performance, calibration ECE, and engine inference latency.
   - Enforces task identity: verifies that baseline and candidate reports evaluate the same task type (`segmentation` vs `detection`).
   - Computes absolute deltas ($c - b$), relative percentage changes ($\frac{c - b}{|b|} \times 100\%$), and latency throughput ratios ($\frac{c}{b}$).
   - Disjoint class handling: gracefully handles models evaluating non-identical class subsets with safe `None` propagation.
   - `EvaluationComparison` and `MetricDelta` frozen dataclasses with `.as_dict()` for strict JSON serializability.
2. **CLI Comparison Suite (`nearfield360 eval compare`):**
   - High-readability formatted ASCII comparison tables for summary metrics, per-class breakdowns, confidence calibration, and latency profiling.
   - `--json`: Machine-readable comparison payload streaming to stdout.
   - `--output`: Atomic output file writer for comparison artifacts, with `--overwrite` safety.
3. **Automated CI/CD Regression Gates:**
   - `--fail-under-miou-delta`: Exits with code 1 if candidate mIoU improvement falls below threshold.
   - `--fail-under-map-delta`: Exits with code 1 if candidate mAP improvement falls below threshold.
   - `--fail-over-ece-delta`: Exits with code 1 if candidate calibration error degradation exceeds threshold.
   - `--fail-over-latency-ratio`: Exits with code 1 if candidate latency degradation exceeds allowable ratio.
   - Clear colored `[PASS]` and `[FAIL]` status indicators and structured `gates` payload in report artifacts.

**Explicit exclusions:** Interactive HTML dashboards, batched inference sweeps, and native TensorRT/C++ acceleration (tracked for subsequent milestones).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Comparison engine & domain models | `src/nearfield360/perception/comparison.py` |
| Perception package exports | `src/nearfield360/perception/__init__.py` |
| `nearfield360 eval compare` CLI command | `src/nearfield360/cli/eval.py` |
| Comparison engine unit tests | `tests/unit/perception/test_comparison.py` (12 tests) |
| Eval CLI comparison tests | `tests/unit/test_eval_cli.py` (65 tests total, +8 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 836 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 92.07% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 72 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 16: Interactive Evaluation HTML Report Dashboard

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 852 passed (0 failures)
**Type Checking:** `mypy --strict` clean (73 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 91.57% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 16 closes README roadmap item 15 by delivering a self-contained, responsive HTML evaluation dashboard with zero external CDN dependencies:

1. **Dashboard Generator Engine (`src/nearfield360/perception/dashboard.py`):**
   - Pure-Python offline report generator producing zero-CDN, standalone HTML5 documents.
   - Self-contained modern styling (responsive layout, dark-mode inspired design system matching NearField360 aesthetics).
   - High-level KPI summary cards for segmentation (mIoU, pixel accuracy, pixel counts) and detection (mAP, IoU thresholds, target counts, prediction counts).
   - Class-level performance cards and tables displaying granular class metrics with support indicators.
   - Inline SVG vector chart integration:
     - Detection PR curves and operating points.
     - Segmentation confidence reliability diagrams.
     - Latency percentiles and engine execution distribution bar charts.
   - Comparative diff visualization:
     - Side-by-side metric comparison when baseline and candidate reports are provided.
     - Formatted signed deltas, percentage changes, and throughput ratios.
     - CI/CD regression gate check indicators (`[PASS]` / `[FAIL]`).
2. **CLI Integration (`nearfield360 eval dashboard`):**
   - Options for `--report`, optional `--baseline`, `--output`, `--overwrite`, and custom `--title`.
   - Integrated regression gate verification flags matching `eval compare` (`--fail-under-miou-delta`, `--fail-under-map-delta`, `--fail-over-ece-delta`, `--fail-over-latency-ratio`).
   - Atomic file output with existing file protection (`--overwrite`).
   - Clean terminal status reporting and non-zero exit codes on gate failure or malformed reports.
3. **Verification and Robustness:**
   - 100% offline self-containment assertion (no external HTTP/HTTPS script or style dependencies).
   - Validation against empty or non-eval reports, corrupted structures, and mismatched tasks.

**Explicit exclusions:** Batched inference sweeps and TensorRT/C++ acceleration (tracked for subsequent milestones).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| SVG vector bar chart generator | `src/nearfield360/robustness/plots.py` |
| Evaluation HTML dashboard generator & domain exceptions | `src/nearfield360/perception/dashboard.py` |
| Perception package exports | `src/nearfield360/perception/__init__.py` |
| `nearfield360 eval dashboard` CLI command | `src/nearfield360/cli/eval.py` |
| Bar chart unit tests | `tests/unit/robustness/test_plots.py` |
| Dashboard generator unit tests | `tests/unit/perception/test_dashboard.py` (6 tests) |
| Eval dashboard CLI integration tests | `tests/unit/test_eval_cli.py` (73 tests total, +8 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 852 passed |
| Code Coverage | `uv run pytest --cov=nearfield360` | 91.57% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 73 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 17: Batched Inference Throughput Profiling and Multi-Camera Execution

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 862 passed (0 failures)
**Type Checking:** `mypy` clean (73 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 91.45% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 17 closes README roadmap item 16 by delivering batched perception inference sweeps and multi-camera surround execution:

1. **Batched Object Detection & Segmentation:**
   - Implemented `predict_batch` and `predict_batch_annotations` on `ObjectDetectionEngine` with contiguous NCHW batched tensor forwarding, multi-head de-batching, and per-item coordinate unscaling / NMS.
   - Validated numerical parity between single-image `predict` and batched `predict_batch`.
2. **Batched Inference Benchmarking & Sweep Engine:**
   - Implemented `benchmark_batch_sweep` in `src/nearfield360/perception/inference/benchmark.py` across configurable batch sizes (e.g. 1, 2, 4, 8).
   - Profiled latency percentiles (mean, median, p90, p95, p99, min, max), effective throughput (FPS), relative speedup, and batch scaling efficiency.
   - Added structured `BatchSweepItem` and `BatchSweepSummary` data models with JSON export.
3. **CLI Integration:**
   - Updated `nearfield360 infer benchmark` with `--batch-size` / `-b` for explicit batch size profiling and `--batch-sweep` / `--batch-sizes` for automated throughput sweeps.
   - Updated `nearfield360 pipeline run` with `--batch-cameras / --no-batch-cameras` to batch all four surround cameras (`FV`, `MVL`, `MVR`, `RV`) into a single model pass.
4. **Verification & Quality Gates:**
   - Complete unit and CLI integration tests across batch engines, sweeps, and multi-camera pipeline execution.
   - Strict typing (`mypy`), linting (`ruff`), test coverage (>85%), and release readiness audit.

**Explicit exclusions:** Native TensorRT engine builds and C++ runtime bindings (remain tracked under roadmap item 7 residual / Phase 18).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Batched detection engine (`predict_batch`, `predict_batch_annotations`) | `src/nearfield360/perception/inference/detection.py` |
| Dynamic batch support in synthetic ONNX builders | `src/nearfield360/perception/inference/test_utils.py` |
| Batch sweep models (`BatchSweepItem`, `BatchSweepSummary`) | `src/nearfield360/perception/inference/models.py` |
| Batched inference throughput sweep engine (`benchmark_batch_sweep`) | `src/nearfield360/perception/inference/benchmark.py` |
| Package exports for batch sweep models and function | `src/nearfield360/perception/inference/__init__.py` |
| CLI `infer benchmark --batch-size`, `--batch-sweep`, `--batch-sizes` | `src/nearfield360/cli/infer.py` |
| CLI `pipeline run --batch-cameras` multi-camera surround execution | `src/nearfield360/cli/pipeline.py`, `src/nearfield360/cli/occupancy.py` |
| Detection engine batching unit tests | `tests/unit/perception/inference/test_detection_engine.py` |
| Batch throughput sweep engine unit tests | `tests/unit/perception/inference/test_benchmark_engine.py` |
| CLI infer benchmark batch & sweep integration tests | `tests/unit/test_infer_cli.py` |
| CLI pipeline multi-camera batching integration tests | `tests/unit/test_pipeline_cli.py` |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 862 passed in 19.86s |
| Code Coverage | `uv run pytest --cov=nearfield360` | 91.45% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 73 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 18: TensorRT Acceleration, Quantization Optimization, and Embedded C++ Runtime Deployment

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 892 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (76 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** 90.12% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 18 completes roadmap item 17 with the following major deliverables:

1. **TensorRT Inference Backend (`TensorrtBackend`):**
   - High-performance TensorRT runtime execution provider integration via ONNX Runtime and native engine execution interfaces.
   - Configurable execution provider settings: workspace memory allocation (`tensorrt_workspace_mb`), engine cache persistence (`tensorrt_cache_dir`), precision modes (`fp32`, `fp16`, `int8`), and Deep Learning Accelerator core mapping (`tensorrt_dla_core`).
   - Diagnostic hardware and provider discovery with actionable error messaging when CUDA or TensorRT dynamic libraries are unavailable.
2. **Quantization & Precision Optimization Engine (`src/nearfield360/perception/inference/optimization.py`):**
   - Half-precision FP16 model conversion with node-level weight casting and topological ONNX graph verification.
   - Dynamic INT8 quantization engine with weight scale / zero-point calibration and parameter compression profiling.
   - INT8 calibration table generation from WoodScape fisheye dataset frames for static TensorRT calibration cache export.
   - Numerical drift evaluator measuring maximum and mean absolute divergence between FP32 baseline and optimized models.
3. **Zero-Copy & Pinned Memory Buffer Management (`src/nearfield360/perception/inference/memory.py`):**
   - Pinned memory allocation interfaces (`CUDAPinnedBufferPool`) enabling page-locked host buffer reuse for DMA transfers.
   - 256-byte alignment and memory footprint planning for batched multi-camera perception workloads.
4. **Embedded Automotive C++ Deployment Architecture (`deploy/cpp/`):**
   - Production C++17/C++20 runtime wrapper with clean RAII abstractions (`TensorRTEngine`, `FisheyePreprocessor`, `PerceptionPipeline`) for automotive compute platforms (NVIDIA DRIVE AGX, Jetson Orin).
   - CMake build configuration supporting cross-compilation and verification test harnesses.
5. **CLI Integration:**
   - `nearfield360 infer providers`: Inspect available execution providers, devices, and runtime capabilities.
   - `nearfield360 infer optimize`: Convert and optimize ONNX models to FP16 or INT8 with validation and numerical drift reporting.
   - `nearfield360 infer calibrate`: Generate INT8 calibration cache tables from WoodScape sample frames.
   - `--backend tensorrt` and `--precision {fp32,fp16,int8}` integrated into `infer`, `pipeline run`, `occupancy layer`, and `track run`.
6. **Verification & Quality Gates:**
   - Strict typing (`mypy`), linting (`ruff`), release audit, and unit test suite coverage maintaining 90.12% coverage.

**Explicit exclusions:** Temporal cross-attention transformers and multi-frame 4D recurrent state networks (tracked under Phase 19 / roadmap item 18).

### 2. Delivered Components

| Component | Location |
| --- | --- |
| TensorRT inference runtime backend (`TensorrtBackend`) | `src/nearfield360/perception/inference/tensorrt_backend.py` |
| Zero-copy pinned memory management (`CUDAPinnedBufferPool`, `PinnedMemoryBuffer`, `MemoryLayoutPlan`) | `src/nearfield360/perception/inference/memory.py` |
| Precision optimization & INT8 calibration engine (`optimize_model_precision`, `generate_int8_calibration_table`, `detect_hardware_providers`) | `src/nearfield360/perception/inference/optimization.py` |
| Extended configuration models (`InferenceConfig`, `InferenceBackendType.TENSORRT`, `PrecisionType`, `OptimizationSummary`) | `src/nearfield360/config.py`, `src/nearfield360/perception/inference/models.py` |
| CLI commands (`infer providers`, `infer optimize`, `infer calibrate`, `--backend tensorrt`, `--precision`) | `src/nearfield360/cli/infer.py`, `src/nearfield360/cli/inference_common.py` |
| Embedded automotive C++ RAII engine wrapper | `deploy/cpp/include/nearfield360/tensorrt_engine.hpp` |
| Embedded automotive SIMD fisheye preprocessor | `deploy/cpp/include/nearfield360/fisheye_preprocessor.hpp` |
| Embedded automotive surround perception & BEV pipeline | `deploy/cpp/include/nearfield360/perception_pipeline.hpp` |
| Standalone embedded benchmark executable | `deploy/cpp/src/main.cpp` |
| Embedded CMake build definition & documentation | `deploy/cpp/CMakeLists.txt`, `deploy/cpp/README.md` |
| C++ deployment test suite | `tests/unit/test_cpp_deployment.py` |
| TensorRT backend unit tests | `tests/unit/perception/inference/test_tensorrt_backend.py` (8 tests) |
| Memory management unit tests | `tests/unit/perception/inference/test_memory.py` (6 tests) |
| Model optimization & calibration unit tests | `tests/unit/perception/inference/test_optimization.py` (9 tests) |
| Infer CLI integration tests | `tests/unit/test_infer_cli.py` (21 tests, +5 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 892 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | 90.12% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 76 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 19: Temporal Bird's-Eye-View (BEV) Multi-Camera Occupancy Forecasting Network

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 915 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (79 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (146 files verified)
**Test Coverage:** 90.30% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 19 completes roadmap item 18:

1. **Temporal BEV Configuration & Models (`src/nearfield360/config.py`, `src/nearfield360/occupancy/models.py`):**
   - Configurable `OccupancyForecastConfig` with horizon length, temporal discretization ($\Delta t$), memory persistence decay, dynamic flow velocity thresholds, spatial diffusion rate, and hidden recurrent channel parameters.
   - Pydantic models for temporal states (`TemporalOccupancyState`), discrete forecast steps (`OccupancyForecastStep`), 4D spatio-temporal forecast grids (`OccupancyForecastGrid`), and safety zone intrusion risks (`ZoneForecastRisk`).
2. **Spatiotemporal Cross-Attention & Recurrent Forecasting Engine (`src/nearfield360/occupancy/forecast.py`):**
   - Cross-attention multi-camera projection and camera overlap weighting.
   - Recurrent ConvGRU / gated temporal state updates fusing historical memory with new observations.
   - Dynamic velocity field estimation on the BEV plane from multi-camera sequence updates and tracked obstacle motion.
   - Autoregressive / advective forward rollout predicting per-cell occupancy probabilities, dynamic motion masks, and propagating Bayesian uncertainty diffusion over multi-second horizons ($[t+\Delta t, \dots, t+H]$).
   - Temporal safety zone risk forecasting: time-to-intrusion ($TTI$), peak occupancy envelope, and hazard trajectories across all 6 surround parking zones.
3. **ONNX Graph Exporter & Runtime (`src/nearfield360/occupancy/onnx_exporter.py`):**
   - End-to-end exportable ONNX model graph for recurrent temporal BEV forecasting with hardware acceleration fallback.
4. **CLI Integration & Multi-Horizon Visualization:**
   - `nearfield360 occupancy forecast`: Run multi-step temporal forecasting over multi-camera or single-camera sequences with JSON report export and multi-horizon PNG panel visualization ($t=0, +1s, +2s, +3s$).
   - `nearfield360 occupancy export-model`: Export the recurrent temporal BEV forecasting network as an ONNX model graph.
   - `nearfield360 pipeline run --temporal-forecast`: Unify 2D tracking velocities with the BEV grid velocity field to produce temporally consistent 4D occupancy forecasting alongside discrete track predictions.
5. **Quality Gates & Acceptance Criteria:**
   - Full test coverage (>85%), strict typing with zero mypy errors, clean ruff linter/formatter, and passing release audit.

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Temporal occupancy forecast configuration (`OccupancyForecastConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Temporal state and multi-step forecast domain models (`TemporalOccupancyState`, `OccupancyForecastStep`, `OccupancyForecastGrid`, `ZoneForecastRisk`, `TemporalForecastSummary`) | `src/nearfield360/occupancy/models.py` |
| Spatiotemporal recurrent forecasting engine (`TemporalOccupancyForecaster`, `fuse_cross_attention_occupancy`) | `src/nearfield360/occupancy/forecast.py` |
| ONNX computation graph exporter (`export_temporal_forecaster_onnx`) | `src/nearfield360/occupancy/onnx_exporter.py` |
| CLI commands (`occupancy forecast`, `occupancy export-model`, multi-step panel PNG renderer) | `src/nearfield360/cli/occupancy.py` |
| Integrated surround pipeline temporal forecasting (`pipeline run --temporal-forecast`) | `src/nearfield360/cli/pipeline.py` |
| Forecast model unit tests | `tests/unit/occupancy/test_forecast_models.py` (7 tests) |
| Forecaster engine unit tests | `tests/unit/occupancy/test_forecast.py` (7 tests) |
| ONNX exporter unit and inference tests | `tests/unit/occupancy/test_onnx_exporter.py` (2 tests) |
| Occupancy forecast CLI integration tests | `tests/unit/test_occupancy_cli.py` (+3 tests, 13 total) |
| Pipeline temporal forecast CLI integration tests | `tests/unit/test_pipeline_cli.py` (+1 test, 14 total) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 915 passed, 5 skipped (0 failures) in 38.94s |
| Code Coverage | `uv run pytest --cov=nearfield360` | 90.30% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 146 files |
| Strict Typing | `uv run mypy` | 0 errors in 79 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 20: 3D Metric Parking Slot & Free-Space Delineation Engine

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 942 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (86 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (157 files verified)
**Test Coverage:** 89.72% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 20 completes roadmap item 19:

1. **Parking Slot Domain Configuration & Data Models (`src/nearfield360/config.py`, `src/nearfield360/slots/models.py`):**
   - Configurable `ParkingSlotConfig` specifying metric slot width/length bounds, vacancy occupancy ratio thresholds, uncertainty thresholds, vehicle corridor dimensions, and safety clearance margins.
   - Pydantic models for slot geometry (`ParkingSlotType`, `SlotOccupancyStatus`, `ParkingSlotCorner`, `ParkingSlot`), approach kinematics (`SlotApproachPath`), and comprehensive detection summaries (`SlotDetectionReport`).
2. **BEV Metric Slot Detection & Geometric Fitting (`src/nearfield360/slots/detector.py`):**
   - Delineate parking slot boundary geometries from BEV road markings (`lanemarks`, `curb`) and free-space gaps between parked obstacles.
   - Extract and fit oriented 4-corner polygons with metric length, width, center, and heading estimation.
   - Slot type classification into `PARALLEL`, `PERPENDICULAR`, and `SLANTED` geometries based on vehicle frame alignment.
3. **Slot Occupancy & Vacancy Classification (`src/nearfield360/slots/classifier.py`):**
   - Rasterize slot interior footprint against metric BEV occupancy grid and Bayesian uncertainty fields.
   - Dynamic track intersection testing against active obstacles and 3D bounding boxes.
   - Classify slot status into `VACANT`, `OCCUPIED`, or `UNCERTAIN` with quantitative occupancy ratios and Bayesian variance metrics.
4. **Approach Corridor Kinematics & Feasibility Engine (`src/nearfield360/slots/corridor.py`):**
   - Calculate vehicle parking entry vector, entry waypoints, and target center parking posture.
   - Project ego vehicle swept path corridor into the slot and verify collision-free clearance against surround obstacles and occupancy grid.
   - Compute metric lateral clearance margin and determine trajectory feasibility flag (`is_feasible`).
5. **CLI Integration & Multi-Camera Pipeline Support:**
   - First-class CLI command group: `nearfield360 slots detect` with JSON reporting, `--vacant-only` filtering, and rich visual PNG rendering with slot polygons, occupancy shading, and approach vectors.
   - Full surround pipeline integration: `nearfield360 pipeline run --slots` embedding detected parking slots into multi-camera run reports.
6. **Quality Gates & Acceptance Criteria:**
   - Unit test coverage exceeding 85%, strict typing with zero mypy errors, clean ruff linter and formatter, passing release audit, and valid package builds.

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Parking slot configuration (`ParkingSlotConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Slot geometry and approach domain models (`ParkingSlot`, `SlotApproachPath`, `SlotDetectionReport`) | `src/nearfield360/slots/models.py` |
| Geometric slot detector & line fitting (`ParkingSlotDetector`) | `src/nearfield360/slots/detector.py` |
| Slot occupancy & uncertainty classifier (`SlotOccupancyClassifier`) | `src/nearfield360/slots/classifier.py` |
| Approach corridor feasibility evaluator (`ApproachCorridorEvaluator`) | `src/nearfield360/slots/corridor.py` |
| Slots package exports | `src/nearfield360/slots/__init__.py` |
| CLI command group (`nearfield360 slots detect`, `--png`, `--vacant-only`) | `src/nearfield360/cli/slots.py` |
| Surround pipeline slot integration (`pipeline run --slots`) | `src/nearfield360/cli/pipeline.py` |
| Release audit CLI consistency | `src/nearfield360/cli/release.py` |
| Slot domain model unit tests | `tests/unit/slots/test_slot_models.py` (7 tests) |
| Slot detector & classifier unit tests | `tests/unit/slots/test_slot_detector.py` (6 tests) |
| Corridor feasibility unit tests | `tests/unit/slots/test_slot_corridor.py` (3 tests) |
| Slot CLI integration tests | `tests/unit/test_slots_cli.py` (5 tests) |
| Pipeline CLI slot integration tests | `tests/unit/test_pipeline_cli.py` (+2 tests, 16 total) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 942 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | 89.72% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 157 files |
| Strict Typing | `uv run mypy` | 0 errors in 86 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 21: Autonomous Parking Trajectory Planning, Ackermann Kinematics, and Multi-Stage Maneuver Engine

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 975 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (93 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (172 files)
**Test Coverage:** 89.34% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 21 introduces an autonomous parking motion planning engine to transform detected 3D metric parking slots and BEV environment maps into kinematically feasible, collision-free vehicle trajectory maneuvers:

1. **Parking Motion Planning Configuration & Domain Models (`src/nearfield360/config.py`, `src/nearfield360/planning/models.py`):**
   - Configurable `ParkingPlannerConfig` specifying vehicle kinematics (wheelbase, front/rear overhangs, steering angle limit), dynamic limits (max speed, acceleration, jerk), spatial step size, temporal step size, and collision safety margins.
   - Pydantic models for gear states (`ManeuverGear`: `FORWARD`, `REVERSE`), maneuver phases (`ManeuverPhase`: `APPROACH`, `STEER_IN`, `ALIGN`, `DOCK`, `FINAL_ALIGN`), time-parameterized waypoints (`TrajectoryWaypoint`), multi-segment trajectories (`ManeuverSegment`, `ParkingTrajectoryPlan`), and plan reports (`ParkingPlanReport`).
2. **Ackermann Kinematics & Geometric Motion Primitives (`src/nearfield360/planning/kinematics.py`):**
   - Non-holonomic bicycle model establishing minimum turning radius $R_{\min} = L / \tan(\delta_{\max})$ and bounding curvature $|\kappa| \le 1/R_{\min}$.
   - Geometric arc and straight line segment generation.
   - Bounded vehicle footprint polygon extraction at arbitrary poses $(x, y, \theta)$ accounting for wheelbase and front/rear overhangs.
   - Reeds-Shepp curve primitives combining forward/reverse arcs and straight segments for non-holonomic pose-to-pose transitions.
3. **Continuous Swept-Footprint & Obstacle Clearance Verification (`src/nearfield360/planning/collision.py`):**
   - Continuous vehicle polygon rasterization along trajectory waypoints against metric BEV occupancy grid (`fused.occupancy()`) at configured danger threshold.
   - Uncertainty masking evaluating Dirichlet/Beta variance along swept envelope.
   - Dynamic obstacle clearance validation against active Kalman-tracked obstacles (`TrackedObstacle`) and predicted collision envelopes.
   - Quantitative trajectory minimum clearance margin computation.
4. **Multi-Stage Parking Maneuver Synthesis (`src/nearfield360/planning/planner.py`):**
   - Parallel parking planner: Multi-segment reverse S-turn (dual inflection circular arcs) followed by forward alignment dock.
   - Perpendicular parking planner: 90-degree reverse sweeping turn with tangent straight entry docking into slot center.
   - Slanted parking planner: Oriented reverse/forward docking based on slot inclination angle.
   - Automated slot selection ranking candidate vacant slots by approach feasibility, obstacle clearance, and trajectory length.
   - Smooth trapezoidal speed profiler generating time-parameterized velocity $v(t)$, acceleration $a(t)$, and arrival timestamps $t$, enforcing $v=0$ at gear shifts and target posture.
5. **Visual Trajectory Rendering & BEV Overlays (`src/nearfield360/planning/viz.py`):**
   - BEV PNG rendering overlaying planned trajectory paths colored by gear (`FORWARD` vs `REVERSE`), key vehicle footprint bounding boxes along maneuver stages, slot polygons, and occupancy background.
6. **CLI Command Group & Four-Camera Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 plan parking` with JSON reporting, slot selection, custom start poses, and `--png` visualization.
   - Surround pipeline integration: `nearfield360 pipeline run --plan-parking` generating an automated parking trajectory plan for the top vacant slot and embedding it in pipeline report and visual artifacts.
7. **Verification & Quality Gates:**
   - Unit tests covering domain models, kinematics, collision checking, parallel/perpendicular/slanted maneuver planners, speed profilers, CLI commands, and pipeline integration.
   - Strict typing under `mypy --strict`, clean `ruff` linter/formatter, passing release audit, and >85% test coverage.

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Parking motion planner configuration (`ParkingPlannerConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Trajectory planning domain models (`ParkingTrajectoryPlan`, `TrajectoryWaypoint`) | `src/nearfield360/planning/models.py` |
| Ackermann kinematics & footprint geometry (`AckermannVehicle`, Reeds-Shepp primitives) | `src/nearfield360/planning/kinematics.py` |
| Swept volume collision & clearance evaluator (`SweptFootprintEvaluator`) | `src/nearfield360/planning/collision.py` |
| Multi-stage parking maneuver planner (`ParkingTrajectoryPlanner`, speed profiler) | `src/nearfield360/planning/planner.py` |
| Trajectory BEV visualization (`render_parking_plan_bev_overlay`) | `src/nearfield360/planning/viz.py` |
| Planning package exports | `src/nearfield360/planning/__init__.py` |
| CLI command group (`nearfield360 plan parking`) | `src/nearfield360/cli/plan.py` |
| Surround pipeline integration (`pipeline run --plan-parking`) | `src/nearfield360/cli/pipeline.py` |
| Release audit CLI consistency | `src/nearfield360/cli/release.py` |
| Domain model unit tests | `tests/unit/planning/test_planning_models.py` (5 tests) |
| Kinematics & footprint unit tests | `tests/unit/planning/test_planning_kinematics.py` (6 tests) |
| Collision & swept volume unit tests | `tests/unit/planning/test_planning_collision.py` (5 tests) |
| Maneuver planner unit tests | `tests/unit/planning/test_planning_planner.py` (6 tests) |
| Trajectory visualization unit tests | `tests/unit/planning/test_planning_viz.py` (3 tests) |
| Planning package init unit tests | `tests/unit/planning/test_planning_init.py` (1 test) |
| Trajectory CLI integration tests | `tests/unit/test_plan_cli.py` (5 tests) |
| Pipeline CLI planning integration tests | `tests/unit/test_pipeline_plan.py` (2 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 975 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | 89.34% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 172 files |
| Strict Typing | `uv run mypy` | 0 errors in 93 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (13 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 22: Closed-Loop Parking Trajectory Tracking Control, Maneuver Execution Simulation, and Dynamic Safety Monitoring

**Status:** Completed
**Repository Branch:** `main`
**Test Suite:** 1012 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (100 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (186 files)
**Test Coverage:** 88.87% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 22 closes the autonomous parking loop by executing planned multi-stage trajectories through feedback control, dynamic vehicle simulation, and real-time safety monitoring:

1. **Closed-Loop Control Configuration & Domain Models (`src/nearfield360/config.py`, `src/nearfield360/control/models.py`):**
   - Configurable `ParkingControlConfig` specifying control sample interval ($\Delta t$), lookahead distance, Stanley cross-track and velocity softening gains, longitudinal PI tracking gains, steering actuator rate limits and time-constant lag, error abort thresholds, and emergency deceleration.
   - Pydantic models for control commands (`ControlCommand`), tracking error states (`TrackingErrorState`), vehicle simulation state (`VehicleSimState`), execution steps (`ManeuverExecutionStep`), control KPIs (`ControlPerformanceKPIs`), and execution reports (`ManeuverExecutionReport`).
2. **Nonlinear Path Tracking Controller (`src/nearfield360/control/controller.py`):**
   - Forward and reverse Stanley steering controller adapted for Ackermann geometry and multi-stage parking maneuvers.
   - Longitudinal velocity tracking controller combining feedforward reference acceleration and PI feedback.
   - Steering rate and angle limits enforcing physical actuator constraints.
3. **Actuator & Kinematic Vehicle Simulator (`src/nearfield360/control/simulator.py`):**
   - Discrete-time kinematic bicycle model integrating vehicle pose $(x, y, \theta)$, velocity $v$, and steering angle $\delta$.
   - First-order steering actuator lag model $\dot{\delta} = \frac{1}{\tau} (\delta_{\text{cmd}} - \delta)$ with rate clamping.
   - Sensor localization noise simulation and gear shift dwell state handling.
4. **Closed-Loop Maneuver Executor & Dynamic Safety Monitor (`src/nearfield360/control/executor.py`):**
   - Multi-stage maneuver state machine handling waypoint progress, segment switching, and gear shift transitions.
   - Real-time swept footprint safety audit checking vehicle envelope against the BEV occupancy grid and active obstacle tracks.
   - Automatic Emergency Braking (AEB) triggering an emergency stop if obstacles intrude into the vehicle clearance boundary.
   - Path tracking error watchdog aborting maneuvers if cross-track or heading errors exceed safety limits.
   - Quantitative evaluation of terminal docking accuracy ($\Delta x, \Delta y, \Delta \theta$) and control KPIs.
5. **Visual Telemetry & BEV Execution Overlays (`src/nearfield360/control/viz.py`):**
   - BEV PNG rendering overlaying planned vs. actual trajectories, footprint poses, parking slot boundaries, and collision status.
   - Time-series telemetry plots of cross-track error, heading error, speed profiles, and steering commands.
6. **CLI Command Group & Surround Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 control execute` (with `--plan-json`, `--output`, `--png`, `--telemetry-png`, `--noise-std`, `--inject-obstacle`).
   - Surround pipeline integration: `nearfield360 pipeline run --simulate-control` executing closed-loop simulation on planned trajectories.
7. **Verification & Quality Gates:**
   - Comprehensive unit and integration test suite maintaining >85% coverage.
   - Strict typing under `mypy --strict`, clean `ruff` checks, and release audit pass with 14 CLI groups.

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Parking control configuration (`ParkingControlConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Closed-loop control domain models (`ControlCommand`, `VehicleSimState`, `ManeuverExecutionReport`, etc.) | `src/nearfield360/control/models.py` |
| Forward & reverse Stanley path tracking controller (`StanleyParkingController`) | `src/nearfield360/control/controller.py` |
| Kinematic vehicle simulator with actuator lag (`VehicleKinematicSimulator`) | `src/nearfield360/control/simulator.py` |
| Closed-loop maneuver executor & dynamic safety monitor (`ManeuverExecutor`) | `src/nearfield360/control/executor.py` |
| BEV trajectory overlay and time-series telemetry renderer (`render_control_execution_bev_overlay`, `render_control_telemetry_chart`) | `src/nearfield360/control/viz.py` |
| Control package exports | `src/nearfield360/control/__init__.py` |
| CLI command group (`nearfield360 control execute`) | `src/nearfield360/cli/control.py` |
| CLI application registration & release audit (`expected_groups` updated to 14) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --simulate-control`) | `src/nearfield360/cli/pipeline.py` |
| Config & default YAML unit tests | `tests/unit/test_config.py` |
| Domain model unit tests | `tests/unit/control/test_control_models.py` (5 tests) |
| Stanley controller unit tests | `tests/unit/control/test_controller.py` (7 tests) |
| Kinematic simulator unit tests | `tests/unit/control/test_simulator.py` (6 tests) |
| Closed-loop executor & safety monitor unit tests | `tests/unit/control/test_executor.py` (8 tests) |
| Control visualization unit tests | `tests/unit/control/test_control_viz.py` (4 tests) |
| Control package init unit tests | `tests/unit/control/test_control_init.py` (1 test) |
| Control CLI integration tests | `tests/unit/test_control_cli.py` (4 tests) |
| Pipeline CLI control integration tests | `tests/unit/test_pipeline_plan.py` (2 tests) |

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1012 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | 88.87% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 186 files |
| Strict Typing | `uv run mypy` | 0 errors in 100 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (14 CLI groups registered) |
| Pre-commit Hooks | `uv run pre-commit run --all-files` | Pass |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 23: Autonomous Valet Parking (AVP) Mission Executive, Dynamic Replanning & Multi-Stage Recovery Orchestration

**Status:** Completed
**Repository Branch:** `phase/avp-mission-executive`
**Test Suite:** 1045 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (106 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 23 elevates the `nearfield360` stack from single-frame slot detection, open-loop planning, and single-trajectory execution into a fully autonomous, production-grade Autonomous Valet Parking (AVP) Mission Executive capable of multi-frame spatial memory, real-time safety yielding, dynamic evasion replanning, and terminal precision docking:

1. **AVP Mission Lifecycle Configuration & Domain Models (`src/nearfield360/config.py`, `src/nearfield360/mission/models.py`):**
   - Configurable `MissionConfig` specifying hold dwell timeout (`hold_timeout_s`), maximum replan attempts (`max_replans`), pull-out recovery distance (`replan_pull_out_dist_m`), approach speed, multi-frame slot association gate (`slot_tracking_distance_gate_m`), confirmation frame threshold (`slot_confirm_frames`), miss threshold (`slot_max_miss_frames`), and precision docking tolerances ($\Delta x, \Delta y, \Delta \theta$).
   - Strict Pydantic domain models: `MissionState` (7 operational states: `STANDBY`, `SEARCHING`, `SLOT_SELECTED`, `APPROACHING`, `PARKING_MANEUVER`, `OBSTACLE_HOLD`, `REPLANNING`, `FINAL_ALIGNMENT`, `COMPLETED`, `ABORTED`), `MissionTrigger`, `MissionEvent`, `TrackedParkingSlot`, `ReplanResult`, and `MissionSummaryReport`.
2. **Multi-Frame Spatial Slot Tracking & Memory (`src/nearfield360/mission/slot_tracker.py`):**
   - Temporal association of parking slot detections across successive frames using Euclidean distance gating and oriented polygon overlap.
   - Temporal stability scoring and confirmation filtering rejecting transient single-frame false positives while maintaining persistent slot identity.
   - Target slot selection ranking confirmed vacant candidate slots by distance and approach feasibility.
3. **Dynamic Recovery Re-Planning Engine (`src/nearfield360/mission/replanner.py`):**
   - Online recovery trajectory generator for vehicles blocked by obstacles during parking maneuvers.
   - Synthesizes reverse-to-forward pull-out realignment maneuvers followed by clean reverse docking trajectories.
   - Swept-volume footprint collision evaluation verifying clearance against BEV occupancy grids and active obstacle tracks.
4. **Autonomous Mission Executive (`src/nearfield360/mission/executive.py`):**
   - Finite state machine coordinating slot tracking, trajectory planning, closed-loop execution, dynamic safety monitoring, and recovery.
   - Real-time transient obstacle yield holding (`OBSTACLE_HOLD`): vehicle stops safely, monitors hazard clearance, and resumes execution if obstacle departs.
   - Dwell timeout fallback (`REPLANNING`): vehicle transitions to replanning or safe abort if the obstacle remains persistently stationary.
   - Terminal alignment and docking validation evaluating position errors ($\Delta x, \Delta y$) and heading error ($\Delta \theta$).
5. **BEV Mission Dashboard Overlay & Timeline Telemetry (`src/nearfield360/mission/viz.py`):**
   - Metric BEV PNG visualization displaying vehicle path, footprint envelope, tracked slot status, obstacle safety buffers, and mission telemetry overlay HUD.
   - Time-series mission state transition and velocity telemetry timeline charts.
6. **CLI Command Group & Four-Camera Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 mission run` (with `--slots-json`, `--plan-json`, `--scenario`, `--png`, `--timeline-png`, `--max-replans`, `--hold-timeout`).
   - Four-camera surround pipeline integration: `nearfield360 pipeline run --mission` orchestrating end-to-end AVP lifecycle execution on surround frames.
   - Release audit updated and verified for 15 CLI command groups.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Mission configuration (`MissionConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Mission domain models (`MissionState`, `MissionEvent`, `TrackedParkingSlot`, `MissionSummaryReport`) | `src/nearfield360/mission/models.py` |
| Multi-frame spatial slot tracker (`SlotTracker`) | `src/nearfield360/mission/slot_tracker.py` |
| Dynamic recovery replanner (`ParkingReplanner`) | `src/nearfield360/mission/replanner.py` |
| AVP Mission Executive (`MissionExecutive`) | `src/nearfield360/mission/executive.py` |
| BEV mission dashboard & timeline telemetry renderer (`viz.py`) | `src/nearfield360/mission/viz.py` |
| Mission package API exports | `src/nearfield360/mission/__init__.py` |
| CLI command group (`nearfield360 mission run`) | `src/nearfield360/cli/mission.py` |
| CLI application registration & release audit (`expected_groups` updated to 15) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --mission`) | `src/nearfield360/cli/pipeline.py` |
| Mission domain models unit tests | `tests/unit/mission/test_mission_models.py` (5 tests) |
| Slot tracker unit tests | `tests/unit/mission/test_slot_tracker.py` (6 tests) |
| Replanner unit tests | `tests/unit/mission/test_replanner.py` (4 tests) |
| Mission executive unit tests | `tests/unit/mission/test_executive.py` (5 tests) |
| Mission visualization unit tests | `tests/unit/mission/test_mission_viz.py` (3 tests) |
| Mission package exports test | `tests/unit/mission/test_mission_init.py` (1 test) |
| Mission CLI integration tests | `tests/unit/test_mission_cli.py` (7 tests) |
| Pipeline CLI mission integration tests | `tests/unit/test_pipeline_mission.py` (3 tests) |

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1045 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 106 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (15 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 24: Parking Facility HD Vector Mapping, Global Topological Route Planning & Multi-Sensor Pose Graph Localization

**Status:** Completed
**Repository Branch:** `phase/parking-facility-mapping-slam` (merged into `main` via PR #2)
**Test Suite:** 1089 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (114 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)

---

### 1. Milestone Objectives & Scope

Phase 24 establishes an end-to-end Parking Facility HD Vector Mapping, Topological Route Planning, and Multi-Sensor Pose Graph Localization engine elevating NearField360 from local vehicle-relative maneuvering into globally aware, facility-wide autonomous valet parking:

1. **Configuration (`MappingConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Default lane width, facility speed limits, odometry process noise, landmark association distance gates, slot observation noise, and localization uncertainty thresholds.
2. **HD Vector Map Domain Models (`src/nearfield360/mapping/models.py`):**
   - Strict Pydantic v2 models: `FacilityBounds`, `FacilityLane`, `FacilitySlot`, `FacilityObstacle`, `FacilityWaypoint`, `GlobalRoute`, `RouteWaypoint`, `PoseEstimate`, `LandmarkObservation`, and `LocalizationReport`.
3. **Facility Map Builder & Synthesizer (`src/nearfield360/mapping/builder.py`):**
   - Programmatic synthesis of benchmark multi-aisle garages and surface parking lots with topological connectivity, boundary integrity, and slot accessibility checks.
4. **Topological Graph & A\* Router (`src/nearfield360/mapping/router.py`):**
   - Directed topological graph enforcing one-way rules, geometric turn penalties, corridor boundary generation, and Bezier waypoint densification.
5. **Multi-Sensor Pose Estimation (EKF) (`src/nearfield360/mapping/localization.py`):**
   - Kinematic bicycle dead-reckoning odometry fused with range/bearing observations of slot corner landmarks, $3\sigma$ covariance tracking, and innovation gating.
6. **BEV Facility Map & Telemetry Dashboard (`src/nearfield360/mapping/viz.py`):**
   - Bird's-eye-view facility rendering and dual-panel tracking error and covariance convergence plots.
7. **CLI Command Group & Four-Camera Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 map` (`info`, `build`, `route`, `localize`).
   - Release audit updated and verified for 16 CLI command groups.
   - Four-camera surround pipeline integration: `nearfield360 pipeline run --map <path> --target-slot <id>`.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Mapping configuration (`MappingConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Mapping domain models (`FacilityMap`, `FacilityLane`, `FacilitySlot`, etc.) | `src/nearfield360/mapping/models.py` |
| Facility map synthesizer (`FacilityBuilder`, `build_benchmark_garage`) | `src/nearfield360/mapping/builder.py` |
| Topological router (`GlobalRouter`, `RoutingGraph`) | `src/nearfield360/mapping/router.py` |
| EKF Pose estimator & simulator (`PoseEstimator`, `LocalizationSimulator`) | `src/nearfield360/mapping/localization.py` |
| BEV facility renderer & telemetry dashboard (`viz.py`) | `src/nearfield360/mapping/viz.py` |
| Mapping package API exports | `src/nearfield360/mapping/__init__.py` |
| CLI command group (`nearfield360 map`) | `src/nearfield360/cli/map.py` |
| CLI app registration & release audit (`expected_groups` updated to 16) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --map ... --target-slot ...`) | `src/nearfield360/cli/pipeline.py` |
| Mapping domain models unit tests | `tests/unit/mapping/test_mapping_models.py` (7 tests) |
| Facility builder unit tests | `tests/unit/mapping/test_builder.py` (4 tests) |
| Topological router unit tests | `tests/unit/mapping/test_router.py` (7 tests) |
| Localization unit tests | `tests/unit/mapping/test_localization.py` (7 tests) |
| Visualization unit tests | `tests/unit/mapping/test_mapping_viz.py` (4 tests) |
| Config unit tests | `tests/unit/test_config.py` (4 tests for mapping) |
| Map CLI integration tests | `tests/unit/test_map_cli.py` (6 tests) |
| Pipeline map integration tests | `tests/unit/test_pipeline_map.py` (2 tests) |

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1089 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 216 files |
| Strict Typing | `uv run mypy` | 0 errors in 114 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (16 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |
| Metadata Validation | `uv run twine check dist/*` | Pass |

---

## Phase 25: Cloud Fleet Coordination, V2X Facility Dispatch & Multi-Agent Conflict Deconfliction

**Status:** Completed
**Repository Branch:** `phase/v2x-cloud-fleet-coordination`
**Test Suite:** 1118 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (121 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (230 files)
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (17 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 25 establishes a comprehensive cloud-edge fleet coordination, V2X facility dispatch, and multi-agent spatiotemporal conflict deconfliction subsystem, taking NearField360 from single-vehicle autonomous valet parking to cooperative multi-agent smart parking facility operations:

1. **Configuration (`FleetConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - V2X protocol version, heartbeat timeout, reservation lease duration, arrival buffer window, swept-volume collision radius, headway time, velocity dampening factor, max iterations, packet drop rate, and communication latency.
2. **V2X & Fleet Domain Models (`src/nearfield360/fleet/models.py`):**
   - Pydantic v2 schemas: `V2XMessage`, `VehicleHeartbeat`, `DispatchAssignment`, `SpatiotemporalWaypoint`, `TrajectoryProfile`, `ConflictPoint`, `FacilityOccupancySummary`, and `FleetDispatchReport`.
3. **Dynamic Slot Dispatcher (`SlotDispatcher` in `src/nearfield360/fleet/dispatcher.py`):**
   - Distance- and aisle congestion-weighted bay selection, reservation locks, lease renewals and expirations, and double-booking avoidance.
4. **4D Spatiotemporal Deconfliction Engine (`DeconflictionEngine` in `src/nearfield360/fleet/deconfliction.py`):**
   - Swept-volume boundary intersection detection, headway time violation checks, virtual token priority scheduling, and yield velocity profiling.
5. **Discrete-Time Centralized Fleet Coordinator (`FleetCoordinator` in `src/nearfield360/fleet/coordinator.py`):**
   - Multi-vehicle concurrent simulation, stochastic V2I wireless packet drop and latency simulation, collision-free waypoint progression, and fleet throughput KPIs.
6. **BEV Fleet Coordination Dashboard (`viz.py` in `src/nearfield360/fleet/viz.py`):**
   - Facility layout rendering with vehicle footprints, planned trajectory paths, conflict hotspot badges, and facility occupancy gauges.
7. **CLI Command Group & Surround Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 fleet` (`info`, `dispatch`, `simulate`, `deconflict`).
   - Release audit updated and verified for 17 CLI command groups.
   - Four-camera surround pipeline runner integration via `--fleet` and `--fleet-vehicles`.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Fleet configuration (`FleetConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Fleet domain models (`V2XMessage`, `VehicleHeartbeat`, etc.) | `src/nearfield360/fleet/models.py` |
| Dynamic slot dispatcher (`SlotDispatcher`) | `src/nearfield360/fleet/dispatcher.py` |
| 4D deconfliction engine (`DeconflictionEngine`) | `src/nearfield360/fleet/deconfliction.py` |
| Fleet coordinator & simulator (`FleetCoordinator`) | `src/nearfield360/fleet/coordinator.py` |
| BEV fleet visualizer & dashboard (`viz.py`) | `src/nearfield360/fleet/viz.py` |
| Fleet package API exports | `src/nearfield360/fleet/__init__.py` |
| CLI command group (`nearfield360 fleet`) | `src/nearfield360/cli/fleet.py` |
| CLI app registration & release audit (`expected_groups` updated to 17) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --fleet ...`) | `src/nearfield360/cli/pipeline.py` |
| Fleet config unit tests | `tests/unit/test_config.py` (2 tests for fleet) |
| Fleet models unit tests | `tests/unit/fleet/test_fleet_models.py` (6 tests) |
| Slot dispatcher unit tests | `tests/unit/fleet/test_dispatcher.py` (5 tests) |
| Deconfliction engine unit tests | `tests/unit/fleet/test_deconfliction.py` (5 tests) |
| Fleet coordinator unit tests | `tests/unit/fleet/test_coordinator.py` (5 tests) |
| Fleet visualizer unit tests | `tests/unit/fleet/test_fleet_viz.py` (3 tests) |
| Fleet CLI integration tests | `tests/unit/test_fleet_cli.py` (5 tests) |
| Pipeline fleet integration tests | `tests/unit/test_pipeline_fleet.py` (1 test) |

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1118 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 230 files |
| Strict Typing | `uv run mypy` | 0 errors in 121 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (17 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 26: Autonomous EV Charging Dispatch, Power Grid Management & Robot/Automated Bay Scheduling

**Status:** Completed
**Repository Branch:** `phase/ev-charging-dispatch-scheduling`
**Test Suite:** 1141 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (128 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (244 files)
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (18 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 26 delivers an end-to-end Autonomous EV Charging Dispatch, Power Grid Management, and Automated Bay Scheduling subsystem for NearField360. Building upon Phase 24's HD Vector Mapping and Phase 25's Cloud Fleet Coordination, Phase 26 enables smart AVP facilities to coordinate electric vehicle charging sessions, enforce transformer peak-shaving limits via active load balancing, model non-linear Constant Current / Constant Voltage (CC/CV) lithium-ion charging curves, and orchestrate automated relocation dispatches to free up charging bays once target State-of-Charge (SoC) is reached:

1. **Configuration (`ChargingConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Facility transformer grid capacity (`facility_grid_power_kw`), peak-shaving threshold ratio (`peak_shaving_threshold_ratio`), default target SoC (`default_target_soc`), critical low SoC threshold (`critical_soc_threshold`), idle penalty tariff (`idle_penalty_per_min`), robotic plug docking dwell time (`plug_dock_dwell_s`), default charger power (`default_charger_power_kw`), and minimum curtailment allocation (`min_power_allocation_kw`).
2. **EV Charging & Grid Domain Models (`src/nearfield360/charging/models.py`):**
   - Strictly validated Pydantic v2 schemas: `ChargerType`, `ChargerPortStatus`, `ChargingSessionStatus`, `ChargerBay`, `EVBatteryState`, `ChargingSession`, `RelocationOrder`, `FacilityPowerSummary`, and `ChargingDispatchReport`.
3. **Dynamic Power Balancer & Peak Shaving Engine (`GridPowerManager` in `src/nearfield360/charging/grid.py`):**
   - Real-time allocation of electrical power across active charging sessions guaranteeing aggregate load stays within `facility_grid_power_kw`, with priority weighting boosting critically depleted batteries during curtailment.
4. **EV Charging Queue Scheduler & Relocation Manager (`ChargingScheduler` in `src/nearfield360/charging/scheduler.py`):**
   - Priority scoring based on critical battery depletion, SoC deficit, and waiting time; bay assignment; completion monitoring; and automated generation of `RelocationOrder` dispatches to staging bays.
5. **Charging Hardware & Non-Linear CC/CV Simulator (`ChargingHardwareSimulator` in `src/nearfield360/charging/hardware.py`):**
   - Models robotic connector docking dwell delays, charger efficiency losses, Constant Current (CC) full power acceptance below 80% SoC, and Constant Voltage (CV) exponential tapering above 80% SoC.
6. **BEV Charging Dashboard (`render_charging_dashboard` in `src/nearfield360/charging/viz.py`):**
   - Dual-panel BEV facility visualization highlighting color-coded charging bays, vehicle SoC badges, grid utilization progress bars, and active session telemetry.
7. **CLI Command Group & Surround Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 charging` (`info`, `grid`, `schedule`, `simulate`).
   - Release audit updated and verified for 18 CLI command groups.
   - Four-camera surround pipeline runner integration via `nearfield360 pipeline run --charging`.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Charging configuration (`ChargingConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Charging domain models (`ChargerBay`, `EVBatteryState`, `ChargingSession`, etc.) | `src/nearfield360/charging/models.py` |
| Grid power manager (`GridPowerManager`) | `src/nearfield360/charging/grid.py` |
| EV charging scheduler (`ChargingScheduler`, `ChargingRequest`) | `src/nearfield360/charging/scheduler.py` |
| Hardware & CC/CV simulator (`ChargingHardwareSimulator`) | `src/nearfield360/charging/hardware.py` |
| BEV charging visualizer (`render_charging_dashboard`) | `src/nearfield360/charging/viz.py` |
| Charging package API exports | `src/nearfield360/charging/__init__.py` |
| CLI command group (`nearfield360 charging`) | `src/nearfield360/cli/charging.py` |
| CLI app registration & release audit (`expected_groups` updated to 18) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --charging`) | `src/nearfield360/cli/pipeline.py` |
| Charging config unit tests | `tests/unit/test_config.py` (4 tests for charging) |
| Charging models unit tests | `tests/unit/charging/test_charging_models.py` (5 tests) |
| Grid power manager unit tests | `tests/unit/charging/test_grid.py` (3 tests) |
| Charging scheduler unit tests | `tests/unit/charging/test_scheduler.py` (2 tests) |
| Hardware & CC/CV unit tests | `tests/unit/charging/test_hardware.py` (3 tests) |
| Charging visualizer unit tests | `tests/unit/charging/test_charging_viz.py` (1 test) |
| Charging CLI integration tests | `tests/unit/test_charging_cli.py` (4 tests) |
| Pipeline charging integration tests | `tests/unit/test_pipeline_charging.py` (1 test) |

---

### 3. Decision Log

1. **Decision**: Mutable `ChargingSession` (`frozen=False`) vs. Frozen `ChargerBay` & `EVBatteryState`
   - *Alternatives Considered*: Making `ChargingSession` immutable and returning new session instances at every simulation step.
   - *Rationale & Evidence*: In line with `ConflictPoint` in `nearfield360.fleet.models`, discrete-time simulation steps update `current_soc`, `energy_delivered_kwh`, `allocated_power_kw`, and `status` in place across active sessions while keeping static facility specifications immutable.
2. **Decision**: Non-Linear Exponential CC/CV Tapering Above 80% SoC
   - *Alternatives Considered*: Linear constant-power charging from 0% to 100% SoC.
   - *Rationale & Evidence*: Real-world lithium-ion EV batteries transition from Constant Current (CC) to Constant Voltage (CV) around 80% SoC to prevent cell plating and over-voltage. Modeling exponential decay $P(\text{SoC}) = P_{\max}\exp(-k(\text{SoC}-0.80))$ accurately reflects topping-off behavior and grid load release.
3. **Decision**: Priority-Weighted Power Curtailment in `GridPowerManager`
   - *Alternatives Considered*: Equal flat division of grid power across all active chargers during peak demand.
   - *Rationale & Evidence*: Vehicles below `critical_soc_threshold` (20%) require urgent energy replenishment. Weighting power shares by $(1 - \text{SoC})$ plus a critical boost ensures depleted vehicles charge faster while still strictly respecting `facility_grid_power_kw`.
4. **Decision**: Increment Expected CLI Group Count from 17 to 18 in Release Audit
   - *Alternatives Considered*: Exempting new command groups from `release.py`.
   - *Rationale & Evidence*: `nearfield360 release audit` enforces strict surface verification across all registered Typer subcommands.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1141 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors in 244 files |
| Strict Typing | `uv run mypy` | 0 errors in 128 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (18 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 27: Autonomous Valet Parking Smart Billing, Dynamic Tariffs & Cryptographic Session Receipts

**Status:** Completed
**Repository Branch:** `phase/automated-valet-parking-payment-billing`
**Test Suite:** 1157 passed, 5 skipped (0 failures)
**Type Checking:** `mypy --strict` clean (134 source files)
**Linting & Style:** `ruff check` and `ruff format` clean (254 files)
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (19 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 27 delivers an end-to-end Autonomous Valet Parking Smart Billing, Dynamic Tariffs, and Cryptographically Signed Session Receipts subsystem for NearField360. Building upon Phase 24's HD Vector Mapping, Phase 25's Cloud Fleet Coordination, and Phase 26's EV Charging Dispatch, Phase 27 equips facilities with dynamic occupancy-based congestion pricing, itemized multi-category charges (parking dwell, EV charging kilowatt-hours, charging bay overstay fines, and valet concierge fees), facility grace periods, and cryptographic SHA-256 HMAC receipt signing and verification:

1. **Configuration (`BillingConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable base parking rate (`base_hourly_rate`), peak congestion multiplier (`peak_multiplier`), EV electricity tariff (`kwh_tariff`), charging overstay penalty rate (`idle_overstay_per_min`), valet service fee (`valet_service_fee`), free grace period (`grace_period_minutes`), sales tax rate (`tax_rate`), and facility HMAC secret key (`receipt_hmac_secret`).
2. **Domain Models & Verifiable Receipts (`src/nearfield360/billing/models.py`):**
   - Pydantic v2 schemas: `TariffTier`, `PaymentStatus`, `BillingCategory`, `BillingItem`, `ParkingReceipt`, and `FinancialSettlementReport`.
   - Canonical deterministic JSON serialization (`to_canonical_json`) excluding the signature digest for tamper-proof hashing.
3. **Dynamic Tariff Engine & Congestion Pricing (`TariffEngine` in `src/nearfield360/billing/tariffs.py`):**
   - Computes effective rates based on occupancy load factors (elevated multipliers at >=75% and >=90% occupancy), overnight discounts, and grace period deductions.
4. **Billing Calculator & Cryptographic Signer (`BillingCalculator` in `src/nearfield360/billing/calculator.py`):**
   - Computes line items, subtotal, tax, and total charged. Computes and validates SHA-256 HMAC digital signatures, reliably detecting forged or altered receipts.
5. **Financial Settlement Aggregator (`SettlementAggregator` in `src/nearfield360/billing/settlement.py`):**
   - Aggregates multi-session receipts into an audited financial report with revenue breakdown across categories, tax collected, and average dwell duration.
6. **BEV Revenue & Financial Visualizer (`render_billing_dashboard` in `src/nearfield360/billing/viz.py`):**
   - Dual-panel BEV layout rendering facility bays with payment badges and a side panel with revenue breakdown progress bars and financial KPIs.
7. **CLI Command Group & Surround Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 billing` (`info`, `calculate`, `receipt`, `settle`).
   - Release audit updated and verified for 19 CLI command groups.
   - Four-camera surround pipeline runner integration via `nearfield360 pipeline run --billing`.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Billing configuration (`BillingConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Billing domain models (`BillingItem`, `ParkingReceipt`, etc.) | `src/nearfield360/billing/models.py` |
| Dynamic tariff engine (`TariffEngine`) | `src/nearfield360/billing/tariffs.py` |
| Billing calculator & HMAC signer (`BillingCalculator`) | `src/nearfield360/billing/calculator.py` |
| Financial settlement aggregator (`SettlementAggregator`) | `src/nearfield360/billing/settlement.py` |
| BEV revenue visualizer (`render_billing_dashboard`) | `src/nearfield360/billing/viz.py` |
| Billing package API exports | `src/nearfield360/billing/__init__.py` |
| CLI command group (`nearfield360 billing`) | `src/nearfield360/cli/billing.py` |
| CLI app registration & release audit (`expected_groups` updated to 19) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --billing`) | `src/nearfield360/cli/pipeline.py` |
| Billing config unit tests | `tests/unit/test_config.py` (4 tests) |
| Billing models unit tests | `tests/unit/billing/test_billing_models.py` (4 tests) |
| Tariff engine unit tests | `tests/unit/billing/test_tariffs.py` (2 tests) |
| Billing calculator unit tests | `tests/unit/billing/test_calculator.py` (3 tests) |
| Financial settlement unit tests | `tests/unit/billing/test_settlement.py` (2 tests) |
| Billing visualizer unit tests | `tests/unit/billing/test_billing_viz.py` (1 test) |
| Billing CLI integration tests | `tests/unit/test_billing_cli.py` (4 tests) |
| Pipeline billing integration tests | `tests/unit/test_pipeline_billing.py` (1 test) |

---

### 3. Decision Log

1. **Decision**: SHA-256 HMAC with Deterministic Canonical JSON over Asymmetric Public Key
   - *Alternatives Considered*: RSA or Ed25519 asymmetric signatures.
   - *Rationale & Evidence*: Within an AVP facility server ecosystem, HMAC provides symmetric authentication and tampering protection without introducing heavy cryptography dependencies (`cryptography` library), staying lean with Python standard library `hashlib` and `hmac`.
2. **Decision**: Strict Grace Period Zero-Billing Deduction
   - *Alternatives Considered*: Prorating grace period by the minute.
   - *Rationale & Evidence*: Standard commercial parking and valet practices provide a 15-minute free drop-off/pick-up window. If duration <= `grace_period_minutes`, zero parking fee is assessed.
3. **Decision**: Increment Expected CLI Group Count from 18 to 19 in Release Audit
   - *Alternatives Considered*: Not testing `billing` in `release.py`.
   - *Rationale & Evidence*: `nearfield360 release audit` verifies full CLI subcommand discovery; adding `billing` maintains release integrity.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1157 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 134 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (19 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 28: Multi-Tenant Facility Partitioning, Fleet Operator SLA Management & Reserved VIP Corridor Access

**Status:** Completed
**Repository Branch:** `phase/multi-tenant-facility-sla-corridors`
**Test Suite:** 1187 passed, 5 skipped (0 failures)
**Type Checking:** `mypy` clean (140 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (20 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 28 delivers an end-to-end Multi-Tenant Facility Partitioning, Fleet Operator SLA Management, and Reserved VIP Express Corridor Access subsystem for NearField360. Building on Phase 24's HD Vector Mapping, Phase 25's Cloud Fleet Coordination, Phase 26's EV Charging Dispatch, and Phase 27's Smart Billing, Phase 28 enables autonomous valet parking garages to partition facilities into dedicated tenant zones, enforce operator quota allocations, monitor Service Level Agreements (SLAs: wait times, turnaround guarantees, breach penalties), and schedule exclusive VIP express corridors to grant prioritized vehicles right-of-way through congested facility bottlenecks:

1. **Configuration (`TenantConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable default quota ratio (`default_quota_ratio`), guaranteed dispatch latency (`sla_max_dispatch_wait_s`), breach penalty tariff (`sla_breach_penalty_per_min`), VIP priority multiplier (`vip_priority_multiplier`), preemption temporal buffer (`express_preemption_buffer_s`), and partition count limits (`max_tenant_partitions`).
2. **Domain Models & Schemas (`src/nearfield360/tenant/models.py`):**
   - Pydantic v2 schemas: `TenantTier` (`PUBLIC_TRANSIENT`, `COMMERCIAL_FLEET`, `CAR_SHARING`, `VIP_EXECUTIVE`), `AccessPermission` (`STANDARD_ACCESS`, `PRIORITY_ACCESS`, `RESTRICTED_ACCESS`, `EXCLUSIVE_VIP`), `TenantPartition`, `TenantSLAContract`, `VIPExpressCorridor`, and `TenantPartitionReport`.
3. **Partition & Quota Enforcement Engine (`TenantManager` in `src/nearfield360/tenant/manager.py`):**
   - Partition creation, dedicated bay quota enforcement, vehicle-to-tenant registration, unauthorized access prevention, and real-time slot occupancy tracking.
4. **Fleet Operator SLA Monitor & Breach Auditor (`SLAMonitor` in `src/nearfield360/tenant/sla.py`):**
   - Turnaround dispatch latency measurement, SLA compliance rate evaluation, and automated breach penalty calculations.
5. **VIP Express Corridor Reservation & Preemption Engine (`VIPCorridorManager` in `src/nearfield360/tenant/corridor.py`):**
   - Dedicated express corridor reservation across facility lanes, spatiotemporal conflict detection, and right-of-way preemption for high-priority emergency and VIP vehicles.
6. **BEV Multi-Tenant Partition Visualizer (`render_tenant_dashboard` in `src/nearfield360/tenant/viz.py`):**
   - Dual-panel BEV layout rendering facility bays color-coded by tenant operator, highlighted VIP express corridor lanes, quota gauge bars, and a live SLA telemetry HUD.
7. **CLI Command Group & Surround Pipeline Integration:**
   - Dedicated CLI command group: `nearfield360 tenant` (`info`, `partition`, `corridor`, `audit`).
   - Release audit updated and verified for 20 CLI command groups.
   - Four-camera surround pipeline runner integration via `nearfield360 pipeline run --tenant`.

---

### 2. Delivered Components

| Component | Location |
| --- | --- |
| Tenant configuration (`TenantConfig`) | `src/nearfield360/config.py`, `configs/default.yaml` |
| Tenant domain models (`TenantPartition`, `TenantSLAContract`, `VIPExpressCorridor`, etc.) | `src/nearfield360/tenant/models.py` |
| Partition & quota manager (`TenantManager`) | `src/nearfield360/tenant/manager.py` |
| SLA monitor & penalty auditor (`SLAMonitor`) | `src/nearfield360/tenant/sla.py` |
| VIP express corridor manager (`VIPCorridorManager`) | `src/nearfield360/tenant/corridor.py` |
| BEV tenant partition visualizer (`render_tenant_dashboard`) | `src/nearfield360/tenant/viz.py` |
| Tenant package API exports | `src/nearfield360/tenant/__init__.py` |
| CLI command group (`nearfield360 tenant`) | `src/nearfield360/cli/tenant.py` |
| CLI app registration & release audit (`expected_groups` updated to 20) | `src/nearfield360/cli/app.py`, `src/nearfield360/cli/release.py` |
| Surround pipeline integration (`pipeline run --tenant`) | `src/nearfield360/cli/pipeline.py` |
| Tenant config unit tests | `tests/unit/test_config.py` |
| Tenant models unit tests | `tests/unit/tenant/test_tenant_models.py` |
| Tenant manager unit tests | `tests/unit/tenant/test_manager.py` |
| SLA monitor unit tests | `tests/unit/tenant/test_sla.py` |
| VIP corridor unit tests | `tests/unit/tenant/test_corridor.py` |
| Tenant visualizer unit tests | `tests/unit/tenant/test_tenant_viz.py` |
| Tenant CLI integration tests | `tests/unit/test_tenant_cli.py` |
| Pipeline tenant integration tests | `tests/unit/test_pipeline_tenant.py` |

---

### 3. Decision Log

1. **Decision**: Explicit Spatiotemporal Preemption over Deadlock Waiting for VIP Corridors
   - *Alternatives Considered*: Queueing lower-priority vehicles until they clear before granting the VIP reservation.
   - *Rationale & Evidence*: Commercial and hospital VIP/emergency operations require guaranteed lane clearance. When a higher-priority corridor reservation is activated, conflicting reservations are preempted immediately with an explicit preemption hold token.
2. **Decision**: Strict Quota and Tier Access Enforcement with Dedicated Exception Types
   - *Alternatives Considered*: Generic runtime exceptions or soft warnings.
   - *Rationale & Evidence*: Defining `QuotaExceededError` and `UnauthorizedAccessError` ensures explicit, recoverable error handling across the dispatch and billing layers.
3. **Decision**: Increment Expected CLI Group Count from 19 to 20 in Release Audit
   - *Alternatives Considered*: Not including `tenant` in the release audit group list.
   - *Rationale & Evidence*: Every functional subsystem CLI command group must be verified by `nearfield360 release audit` to ensure zero command regressions.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1187 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 140 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (20 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 29: Curbside Drop-off/Pick-up Staging, Dwell Management & Pedestrian Safety Interlocks

**Status:** Completed
**Repository Branch:** `phase/curbside-dropoff-staging-pedestrian-safety`
**Test Suite:** 1219 passed, 5 skipped (0 failures)
**Type Checking:** `mypy` clean (148 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (21 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 29 delivers an end-to-end Curbside Drop-off & Pick-up (DOPU) Staging, Dwell Management, and Pedestrian Safety Interlock subsystem for NearField360. Building on Phase 24's HD Vector Mapping, Phase 25's Fleet Coordination, Phase 26's Charging Dispatch, Phase 27's Smart Billing, and Phase 28's Multi-Tenant Partitioning, Phase 29 implements the critical Human-Machine Interface Handover Zone (ISO 23374 AVP Type 2). The system manages dedicated curbside drop-off and pick-up staging bays, tracks driver/passenger egress and ingress, enforces safety interlocks that inhibit autonomous motion while doors/trunks are open or pedestrians are detected in the active clearance envelope, optimizes curbside dwell times, queues vehicles awaiting available pick-up bays, and visualizes curbside telemetry and safety interlock statuses:

1. **Configuration (`CurbsideConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable max drop-off dwell time (`max_dropoff_dwell_s`), max pick-up dwell time (`max_pickup_dwell_s`), pedestrian critical safety radius (`pedestrian_safety_radius_m`), interlock clearance hold debounce duration (`interlock_clearance_hold_s`), curbside staging bays capacity (`staging_capacity`), and overstay penalty tariff (`overstay_penalty_per_min`).
2. **Domain Models & Handover Schemas (`src/nearfield360/curbside/models.py`):**
   - Pydantic v2 frozen models: `HandoverType` (`DROPOFF`, `PICKUP`), `BayStatus` (`VACANT`, `OCCUPIED_UNLOADING`, `OCCUPIED_BOARDING`, `HANDOVER_READY`, `CLEARING`), `SafetyInterlockStatus` (`ENGAGED_DOOR_OPEN`, `ENGAGED_PEDESTRIAN_PROXIMITY`, `ENGAGED_TRUNK_OPEN`, `DISENGAGED_CLEAR`), `CurbsideBay`, `CurbsideMission`, `PedestrianDetection`, and `CurbsideReport`.
3. **Pedestrian Safety Interlock Engine (`SafetyInterlockEngine` in `src/nearfield360/curbside/safety.py`):**
   - Latch and sensor monitoring (door, trunk, pedestrian proximity).
   - Enforces motion inhibition (`is_motion_permitted == False`) when doors or trunk are open, or when pedestrians are detected within `pedestrian_safety_radius_m`.
   - Debounce clearance hold logic: requires uninterrupted clearance for `interlock_clearance_hold_s` before transitioning to `DISENGAGED_CLEAR`.
4. **Curbside Staging & Dwell Queue Manager (`CurbsideManager` in `src/nearfield360/curbside/manager.py`):**
   - Staging bay allocation, drop-off completion, passenger summon request queueing, dwell duration tracking, and overstay tariff penalty calculations.
5. **BEV Curbside & Safety HUD Visualizer (`render_curbside_dashboard` in `src/nearfield360/curbside/viz.py`):**
   - Dual-panel BEV layout rendering facility layout, dedicated curbside staging bays, vehicle footprints, safety envelope circles around vehicles (green=clear, red=interlocked), pedestrian positions, and right-panel queue status and safety KPIs.
6. **CLI Command Group & Four-Camera Pipeline Integration (`src/nearfield360/cli/curbside.py`, `src/nearfield360/cli/pipeline.py`):**
   - CLI command group `nearfield360 curbside` with `info`, `dropoff`, `summon`, and `audit` subcommands.
   - Updated release audit expected group count from 20 to 21.
   - Integrated `--curbside` flag into `nearfield360 pipeline run`.

---

### 2. Delivered Components & Architecture

- `src/nearfield360/config.py`: Added `CurbsideConfig` with validation rules.
- `configs/default.yaml`: Added `curbside` default parameters.
- `src/nearfield360/curbside/models.py`: Pydantic models for bays, missions, pedestrian detections, interlocks, and audit reports.
- `src/nearfield360/curbside/safety.py`: Pedestrian safety interlock engine with temporal debounce clearance.
- `src/nearfield360/curbside/manager.py`: Curbside staging bay and dwell queue manager.
- `src/nearfield360/curbside/viz.py`: Dual-panel BEV visualization of curbside handover zones and safety envelope HUD.
- `src/nearfield360/curbside/__init__.py`: Package exports for all public curbside components.
- `src/nearfield360/cli/curbside.py`: Full Typer CLI command group `nearfield360 curbside`.
- `src/nearfield360/cli/app.py`: Registered `curbside_app` in root CLI.
- `src/nearfield360/cli/release.py`: Added `curbside` to audited CLI command groups (now 21).
- `src/nearfield360/cli/pipeline.py`: Added `--curbside` option, stage execution, samples payload, and output report fields.
- `tests/unit/test_config.py`: Tests for `CurbsideConfig`.
- `tests/unit/curbside/test_curbside_models.py`: Serialization and validation unit tests.
- `tests/unit/curbside/test_safety.py`: Pedestrian safety interlock unit tests.
- `tests/unit/curbside/test_manager.py`: Curbside staging manager and dwell penalty tests.
- `tests/unit/curbside/test_viz.py`: Visualizer PNG generation tests.
- `tests/unit/test_curbside_cli.py`: Curbside CLI unit tests with JSON and PNG assertions.
- `tests/unit/test_pipeline_curbside.py`: Surround pipeline `--curbside` integration tests.

---

### 3. Decision Log

1. **Decision**: Explicit Temporal Debounce Hold Before Disengaging Safety Interlocks
   - *Alternatives Considered*: Immediate disengagement as soon as a pedestrian steps outside the critical radius.
   - *Rationale & Evidence*: In real-world curbside environments, pedestrians often hesitate, turn back for luggage, or oscillate near vehicle boundaries. ISO 23374 requires a mandatory debounce hold duration (default 3.0 s) after clearance before autonomous motion can resume.
2. **Decision**: Dual Egress/Ingress Dwell Timers and Minute-Rate Overstay Penalties
   - *Alternatives Considered*: Single static dwell timer for all handover events.
   - *Rationale & Evidence*: Passenger drop-off egress is significantly faster (default 120 s) than summon boarding (default 180 s), where passengers locate the vehicle and load luggage. Differentiated dwell limits optimize facility curbside throughput while fairly penalizing excessive overstays.
3. **Decision**: Increment Expected CLI Group Count from 20 to 21 in Release Audit
   - *Alternatives Considered*: Excluding `curbside` from release readiness checks.
   - *Rationale & Evidence*: Every functional subsystem CLI command group must be verified by `nearfield360 release audit` to maintain 100% command availability.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1219 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 148 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (21 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 30: Remote Teleoperation Assistance, Video Fallback Streaming & Emergency Minimum Risk Maneuver (MRM) Execution

**Status:** Completed
**Repository Branch:** `phase/remote-teleoperation-assistance-mrm`
**Test Suite:** 1243 passed, 5 skipped (0 failures)
**Type Checking:** `mypy` clean (153 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (22 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 30 delivers an end-to-end Remote Teleoperation Assistance, Latency-Adaptive Video Fallback Streaming, and Emergency Minimum Risk Maneuver (MRM) Execution subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3) and SAE J3016 operational safety concepts, Phase 30 provides the critical remote human fallback layer when autonomous vehicles encounter edge-case blockages, perception ambiguities, or communication degradation:

1. **Configuration (`TeleopConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable heartbeat watchdog timeout (`heartbeat_timeout_s`), direct driving maximum latency (`max_rtt_ms`), critical latency threshold (`critical_rtt_ms`), target streaming frame rate (`video_target_fps`), fallback streaming frame rate (`video_fallback_fps`), MRM deceleration profile (`mrm_deceleration_mps2`), safe stop standstill speed threshold (`safe_stop_standstill_speed_mps`), and automatic failsafe trigger flag (`auto_mrm_on_heartbeat_loss`).
2. **Domain Models & Teleoperation Protocol (`src/nearfield360/teleop/models.py`):**
   - Pydantic v2 frozen models: `TeleopMode` (`IDLE_AUTONOMOUS`, `TELE_ASSISTANCE`, `TELE_GUIDANCE`, `DIRECT_DRIVING`, `MRM_SAFE_STOP`), `MRMStage` (`INACTIVE`, `DECELERATING`, `PULLING_TO_SHOULDER`, `STANDSTILL_LOCKED`, `EMERGENCY_BEACON`), `NetworkQuality` (`OPTIMAL`, `DEGRADED`, `CRITICAL`, `SEVERED`), `TeleopWaypoint`, `TeleopCommand`, `TeleopHeartbeat`, `TeleopSession`, and `TeleopReport`.
3. **Bandwidth-Adaptive Video Fallback & Network Monitor (`NetworkMonitor` in `src/nearfield360/teleop/network.py`):**
   - RTT exponential moving average (EMA) smoothing, packet loss sequence tracking, jitter estimation, network health classification, and dynamic video streaming throttling recommendations (framerate downscaling and compression scaling).
4. **Minimum Risk Maneuver (MRM) Safe Stop Engine (`MRMEngine` in `src/nearfield360/teleop/mrm.py`):**
   - Deterministic deceleration physics: $v(t) = \max(0, v_0 - a \cdot t)$, stopping distance calculation, safe stop waypoint trajectory synthesis, mechanical parking brake locking, and hazard beacon actuation.
5. **Remote Assistance Executive & Session Coordinator (`TeleopManager` in `src/nearfield360/teleop/manager.py`):**
   - Multi-vehicle session management, failsafe heartbeat watchdog timer, operator command routing, and strict latency safety enforcement (direct driving rejected if link is not `OPTIMAL`).
6. **BEV Teleoperation Cockpit & Telemetry HUD Visualizer (`render_teleop_dashboard` in `src/nearfield360/teleop/viz.py`):**
   - Dual-panel BEV layout rendering facility layout, vehicle footprint, suggested remote path overlays, MRM emergency halos, latency meter bar, and operator cockpit HUD.
7. **CLI Command Group & Four-Camera Pipeline Integration (`src/nearfield360/cli/teleop.py`, `src/nearfield360/cli/pipeline.py`):**
   - Dedicated CLI command group `nearfield360 teleop` with `info`, `session`, `fallback`, and `audit` subcommands.
   - Release audit verified across 22 command groups.
   - Surround pipeline runner integrated with `--teleop` option.

---

### 2. Delivered Components & Architecture

- `src/nearfield360/config.py`: Added `TeleopConfig` with validation rules.
- `configs/default.yaml`: Added `teleop` default parameters.
- `src/nearfield360/teleop/models.py`: Pydantic models for modes, stages, network quality, commands, heartbeats, sessions, and reports.
- `src/nearfield360/teleop/network.py`: Network quality monitoring, RTT tracking, and adaptive video throttling.
- `src/nearfield360/teleop/mrm.py`: Deterministic MRM state machine, deceleration physics, and parking brake lock.
- `src/nearfield360/teleop/manager.py`: Teleoperation session manager and safety command dispatcher.
- `src/nearfield360/teleop/viz.py`: Dual-panel BEV cockpit visualizer and telemetry HUD.
- `src/nearfield360/teleop/__init__.py`: Package exports for all public teleop components.
- `src/nearfield360/cli/teleop.py`: Full Typer CLI command group `nearfield360 teleop`.
- `src/nearfield360/cli/app.py`: Registered `teleop_app` in root CLI.
- `src/nearfield360/cli/release.py`: Added `teleop` to audited CLI command groups (now 22).
- `src/nearfield360/cli/pipeline.py`: Added `--teleop` option, stage execution, samples payload, and output report fields.
- `tests/unit/test_config.py`: Tests for `TeleopConfig`.
- `tests/unit/teleop/test_teleop_models.py`: Serialization and validation unit tests.
- `tests/unit/teleop/test_network.py`: Network quality classification and video throttling tests.
- `tests/unit/teleop/test_mrm.py`: MRM safe stop execution and deceleration physics tests.
- `tests/unit/teleop/test_teleop_manager.py`: TeleopManager session lifecycle, latency rejection, and watchdog tests.
- `tests/unit/teleop/test_teleop_viz.py`: Visualizer PNG generation tests.
- `tests/unit/test_teleop_cli.py`: Teleoperation CLI unit tests with JSON and PNG assertions.
- `tests/unit/test_pipeline_teleop.py`: Surround pipeline `--teleop` integration tests.

---

### 3. Decision Log

1. **Decision**: Combined Instantaneous and Exponential Moving Average (EMA) for Network Quality Classification
   - *Alternatives Considered*: Pure EMA smoothing only.
   - *Rationale & Evidence*: In teleoperation safety, waiting 5+ packets for an EMA to catch up to an instantaneous latency spike (e.g. cellular handoff or network freeze) creates an unacceptable collision hazard. Evaluating `effective_rtt = max(ema_rtt_ms, last_rtt_ms)` immediately inhibits direct steering upon the first anomalous packet.
2. **Decision**: Automatic Failsafe Standstill Lockdown on Watchdog Timeout
   - *Alternatives Considered*: Allowing vehicle to maintain current velocity while waiting for reconnect.
   - *Rationale & Evidence*: ISO 23374 and SAE J3016 mandate that loss of teleoperation heartbeat must automatically transition the automated vehicle to a Minimum Risk State within bounded time (`heartbeat_timeout_s = 1.5s`).
3. **Decision**: Increment Expected CLI Group Count from 21 to 22 in Release Audit
   - *Alternatives Considered*: Excluding `teleop` from release readiness checks.
   - *Rationale & Evidence*: All functional subsystem CLI command groups must be validated by `nearfield360 release audit` to ensure complete CLI surface availability.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1243 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 153 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (22 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 31: Multi-Floor Facility Topology, Spiral Ramp Kinematics, Steep Incline Pitch Dynamics & Automated Vehicle Elevator Dispatch (ISO 23374 Vertical Transit)

**Status:** Completed
**Repository Branch:** `phase/multi-floor-ramp-elevator-transit`
**Test Suite:** 1272 passed, 5 skipped (0 failures)
**Type Checking:** `mypy` clean (160 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (23 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 31 delivers an end-to-end Multi-Floor Facility Topology, Spiral & Straight Ramp Kinematics, Steep Incline Pitch Dynamics, and Automated Vehicle Elevator Dispatch subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3) multi-level facility navigation standards, Phase 31 equips automated vehicles with the capability to transit vertically between multiple garage floors (subterranean basements, ground floors, multi-tier elevated decks, and rooftops):

1. **Configuration (`MultiFloorConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable default floor height (`default_floor_height_m`), maximum safe ramp grade (`max_ramp_grade_pct`), chassis ground breakover clearance threshold (`min_breakover_clearance_m`), ramp speed limit (`ramp_speed_limit_mps`), vehicle elevator hoist speed (`elevator_hoist_speed_mps`), door open/close dwell cycle time (`elevator_door_dwell_s`), platform vertical leveling tolerance (`elevator_leveling_tolerance_m`), single-lane ramp token lease timeout (`ramp_token_timeout_s`), and maximum supported floors (`max_floors_supported`).
2. **Domain Models & Vertical Transit Protocol (`src/nearfield360/multilevel/models.py`):**
   - Pydantic v2 frozen models: `PortalType` (`STRAIGHT_RAMP`, `SPIRAL_RAMP`, `VEHICLE_ELEVATOR`), `ElevatorState` (`IDLE`, `CALLED`, `TRANSITING`, `DOORS_OPENING`, `LOADING`, `LEVELING`, `HOISTING`, `UNLOADING`, `HOLD`), `RampDirection` (`ASCENDING`, `DESCENDING`, `BIDIRECTIONAL`), `RampTokenPriority` (`ASCENDING_FIRST`, `DESCENDING_FIRST`, `FIFO`), `FloorLevel`, `VerticalPortal`, `VehiclePitchState`, `Waypoint3D`, `ElevatorCabin`, `RampToken`, `MultiFloorRoute`, and `MultiFloorFacilityReport`.
3. **Multi-Floor Topological Router & Graph Builder (`MultiFloorGraph` in `src/nearfield360/multilevel/graph.py`):**
   - 3D multi-floor directed graph representing intra-floor aisles, parking bays, and vertical portals. Dijkstra/A* multi-level routing synthesizing 3D waypoints with altitude $z$, horizontal distances, transit times, and incline pitch gradients.
4. **Ramp Kinematics, Pitch Dynamics & Chassis Breakover Checker (`RampDynamicsEngine` in `src/nearfield360/multilevel/ramp.py`):**
   - Slope angle $\theta = \arctan(G / 100)$, longitudinal gravity load $F_g = mg \sin\theta$, normal force $F_N = mg \cos\theta$, required tractive friction $\mu_{\text{req}} = \tan\theta$, traction margin $\mu - \mu_{\text{req}}$, spiral continuous curvature yaw rate $\omega = v_{\text{horiz}} / R$, lateral centripetal acceleration, vertical climb rate $\dot{z}$, and underbody chassis breakover clearance checking against crest angle $\Delta \theta$.
5. **Single-Lane Bidirectional Ramp Token Arbiter (`RampTokenArbiter` in `src/nearfield360/multilevel/arbiter.py`):**
   - Mutual exclusion token arbitration for single-lane ramps. Directional priority resolution (`ASCENDING_FIRST` to prevent heavy vehicles stalling or rollback on steep inclines) and automatic lease timeout eviction to prevent deadlocks.
6. **Automated Vehicle Elevator Dispatcher & Leveling Interlocks (`VehicleElevatorDispatcher` in `src/nearfield360/multilevel/elevator.py`):**
   - Scheduling queue, closest idle cabin selection, door dwell cycles, capacity payload enforcement ($\le 3500$ kg), vertical hoist velocity profiling, and safety interlock validation (`is_drive_in_permitted` and `is_drive_out_permitted` verifying platform leveling $\le 15$ mm).
7. **Multi-Floor BEV & Elevation HUD Visualizer (`render_multilevel_dashboard` in `src/nearfield360/multilevel/viz.py`):**
   - Dual-panel BEV layout rendering stacked multi-floor floorplans, aisles, bays, straight and spiral ramps, elevator shafts, active vehicle footprints, 3D route trajectory, and right-panel vertical transit HUD.
8. **CLI Command Group & Four-Camera Pipeline Integration (`src/nearfield360/cli/floor.py`, `src/nearfield360/cli/pipeline.py`):**
   - Dedicated CLI command group `nearfield360 floor` with `info`, `route`, `ramp`, `elevator`, and `audit` subcommands.
   - Release audit verified across 23 registered command groups.
   - Four-camera pipeline runner integrated with `--floor` option.

---

### 2. Delivered Components & Architecture

- `src/nearfield360/config.py`: Added `MultiFloorConfig` with validation rules.
- `configs/default.yaml`: Added `floor` default parameters.
- `src/nearfield360/multilevel/models.py`: Pydantic models for floors, portals, pitch states, elevator cabins, tokens, routes, and facility reports.
- `src/nearfield360/multilevel/graph.py`: 3D multi-floor topological graph and Dijkstra vertical routing engine.
- `src/nearfield360/multilevel/ramp.py`: Ramp kinematics, incline gravity physics, spiral curvature, and breakover clearance engine.
- `src/nearfield360/multilevel/arbiter.py`: Single-lane ramp token arbiter with ascending priority and timeout release.
- `src/nearfield360/multilevel/elevator.py`: Automated car elevator dispatcher with door dwell and leveling safety interlocks.
- `src/nearfield360/multilevel/viz.py`: Dual-panel stacked multi-floor BEV visualizer and telemetry HUD.
- `src/nearfield360/multilevel/__init__.py`: Package exports for all public multi-floor components.
- `src/nearfield360/cli/floor.py`: Full Typer CLI command group `nearfield360 floor`.
- `src/nearfield360/cli/app.py`: Registered `floor_app` in root CLI.
- `src/nearfield360/cli/release.py`: Added `floor` to audited CLI command groups (now 23).
- `src/nearfield360/cli/pipeline.py`: Added `--floor` option, stage execution, samples payload, and output report fields.
- `tests/unit/test_config.py`: Tests for `MultiFloorConfig`.
- `tests/unit/multilevel/test_multilevel_models.py`: Serialization and validation unit tests.
- `tests/unit/multilevel/test_graph.py`: Multi-floor topological graph and 3D routing tests.
- `tests/unit/multilevel/test_ramp.py`: Ramp pitch dynamics, gravity force, and breakover clearance tests.
- `tests/unit/multilevel/test_arbiter.py`: Single-lane ramp mutual exclusion and ascending priority tests.
- `tests/unit/multilevel/test_elevator.py`: Vehicle elevator dispatch, hoisting, and leveling interlock tests.
- `tests/unit/multilevel/test_multilevel_viz.py`: Visualizer PNG generation tests.
- `tests/unit/test_floor_cli.py`: Floor CLI unit tests with JSON and PNG assertions.
- `tests/unit/test_pipeline_floor.py`: Surround pipeline `--floor` integration tests.

---

### 3. Decision Log

1. **Decision**: Ascending-First Directional Priority on Single-Lane Ramps
   - *Alternatives Considered*: Standard first-come first-served (FIFO) queue.
   - *Rationale & Evidence*: In multi-story automotive parking garages with steep ramp grades (up to 15%), stopping an ascending vehicle mid-ramp to yield to a descending vehicle creates severe rollback hazards, high transmission strain, and traction loss on damp concrete. Prioritizing ascending vehicles clears uphill momentum safely while descending vehicles can easily hold on flat deck waiting bays.
2. **Decision**: Platform Leveling Sensor Interlock Gate ($\le 15$ mm) for Car Hoists
   - *Alternatives Considered*: Allowing vehicle drive-in as soon as the hoist reaches the target floor elevation without measuring leveling offset.
   - *Rationale & Evidence*: Vehicle underbody ground clearance and low-profile wheel rims can suffer severe structural damage if driven across an uneven vertical step greater than 15 mm between garage deck and elevator platform. Gating drive-in/drive-out behind `abs(leveling_offset_m) <= leveling_tolerance_m` guarantees safe mechanical transitions.
3. **Decision**: Increment Expected CLI Group Count from 22 to 23 in Release Audit
   - *Alternatives Considered*: Subsuming multi-floor operations under the existing 2D `map` command.
   - *Rationale & Evidence*: Multi-floor vertical transit (ramps, elevators, pitch dynamics, tokens) represents a distinct operational layer separate from 2D vector lane SLAM, and warrants a first-class CLI command group `nearfield360 floor` verified by `nearfield360 release audit`.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1272 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 160 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (23 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 32: Emergency Vehicle Preemption (EVP), Facility Fire Evacuation, Dynamic Hazard Containment & Life Safety Dispatch (ISO 23374 / NFPA 88A)

**Status:** Completed
**Repository Branch:** `phase/emergency-preemption-evacuation-life-safety`
**Test Suite:** 1304 passed, 5 skipped (0 failures)
**Type Checking:** `mypy` clean (164 source files)
**Linting & Style:** `ruff check` and `ruff format` clean
**Test Coverage:** >88.5% (exceeds required 85.0% threshold)
**Release Audit:** Passed (24 registered CLI command groups)

---

### 1. Milestone Objectives & Scope

Phase 32 delivers an end-to-end Emergency Vehicle Preemption (EVP), Facility Fire Evacuation, Dynamic Hazard Containment, and Life Safety Dispatch subsystem for NearField360. Conforming to ISO 23374 (AVP Type 2 / Type 3), NFPA 88A (Standard for Parking Structures), and UL 3300 operational safety standards, Phase 32 equips autonomous parking facilities with life-safety response capabilities during facility emergencies:

1. **Configuration (`EmergencyConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable preemption radius (`preemption_radius_m`), dynamic hazard buffer distance (`hazard_buffer_radius_m`), emergency egress speed limit (`evacuation_speed_limit_mps`), fire door closure clearance window (`fire_door_clearance_timeout_s`), wave departure stagger interval (`stagger_interval_s`), emergency watchdog heartbeat timeout (`emergency_heartbeat_timeout_s`), and EV battery pack thermal runaway threshold (`thermal_runaway_temp_c`).
2. **Domain Models & Life Safety Protocols (`src/nearfield360/emergency/models.py`):**
   - Pydantic v2 frozen models: `EmergencyType` (`FIRE_ALARM`, `EV_THERMAL_RUNAWAY`, `GAS_LEAK`, `STRUCTURAL_HAZARD`, `EMERGENCY_VEHICLE_APPROACH`), `EmergencySeverity` (`ADVISORY`, `EVP_PREEMPTION`, `PARTIAL_EVACUATION`, `FULL_FACILITY_EVACUATION`), `VehicleEvacuationStatus` (`STATIONARY_SAFE`, `HOLD_YIELDING`, `EVACUATING`, `EVACUATED`, `TRAPPED_IN_HAZARD`), `FireDoorStatus` (`OPEN`, `CLOSING`, `LOCKED_SHUT`), `ResponderType` (`FIRE_ENGINE`, `AMBULANCE`, `POLICE`), `HazardZone`, `FireDoorInterlock`, `EmergencyVehicle`, `EvacuationWaypoint`, `EvacuationRoute`, `EvacuationWave`, and `LifeSafetyReport`.
3. **Emergency Vehicle Preemption (EVP) Arbiter (`EmergencyPreemptionArbiter` in `src/nearfield360/emergency/preemption.py`):**
   - Tracks incoming emergency vehicles (fire engines, ambulances, police cruisers) equipped with sirens and emergency strobes. Evaluates Euclidean proximity and corridor centerline segment projections toward destination incident scenes. Detects right-of-way conflicts and issues mandatory yield holds (`HOLD_YIELDING`) to clear emergency transit paths.
4. **Dynamic Hazard Containment & Fire Door Interlocks (`HazardContainmentEngine` in `src/nearfield360/emergency/containment.py`):**
   - Dynamic 2D/3D hazard exclusion zones with configurable perimeter safety buffers. Real-time battery temperature telemetry monitoring detecting EV thermal runaway ($\ge 70.0^\circ\text{C}$) and establishing automated localized containment perimeters. Fire barrier rolling shutter clearance interlocks ensuring vehicles clear closing compartments before descending shutters seal the sector.
5. **Multi-Wave Facility Evacuation Planner (`FacilityEvacuationPlanner` in `src/nearfield360/emergency/evacuation.py`):**
   - Synthesizes emergency egress route trajectories from parking bays to the nearest uncompromised facility exit portals. Staggers egress departures in prioritized waves (`EvacuationWave`) based on hazard proximity urgency to prevent gate bottlenecks and traffic jams.
6. **Life Safety BEV Tactical Map & Situational Visualizer (`render_emergency_dashboard` in `src/nearfield360/emergency/viz.py`):**
   - Dual-panel BEV layout rendering facility layout, danger halo rings around hazard perimeters, fire door barriers, emergency responder vectors and siren waves, vehicle markers color-coded by evacuation status, and tactical Life Safety Command HUD.
7. **CLI Command Group & Four-Camera Pipeline Integration (`src/nearfield360/cli/emergency.py`, `src/nearfield360/cli/pipeline.py`):**
   - Dedicated CLI command group `nearfield360 emergency` with `info`, `preempt`, `hazard`, `evacuate`, and `audit` subcommands.
   - Release audit verified across 24 registered command groups.
   - Four-camera pipeline runner integrated with `--emergency` option.

---

### 2. Delivered Components & Architecture

- `src/nearfield360/config.py`: Added `EmergencyConfig` with validation rules.
- `configs/default.yaml`: Added `emergency` default parameters.
- `src/nearfield360/emergency/models.py`: Pydantic models for incident types, severity, vehicle evacuation statuses, hazard zones, fire doors, emergency responders, routes, waves, and reports.
- `src/nearfield360/emergency/preemption.py`: Emergency vehicle preemption arbiter and corridor projection distance algorithms.
- `src/nearfield360/emergency/containment.py`: Dynamic hazard exclusion engine, EV thermal runaway detector, and fire door clearance interlocks.
- `src/nearfield360/emergency/evacuation.py`: Multi-wave egress planner and prioritized route synthesizer.
- `src/nearfield360/emergency/viz.py`: Dual-panel tactical BEV incident map and life safety command HUD visualizer.
- `src/nearfield360/emergency/__init__.py`: Package exports for all public emergency components.
- `src/nearfield360/cli/emergency.py`: Full Typer CLI command group `nearfield360 emergency`.
- `src/nearfield360/cli/app.py`: Registered `emergency_app` in root CLI.
- `src/nearfield360/cli/release.py`: Added `emergency` to audited CLI command groups (now 24).
- `src/nearfield360/cli/pipeline.py`: Added `--emergency` option, stage execution, samples payload, and output report fields.
- `tests/unit/test_config.py`: Unit tests for `EmergencyConfig` validation and environment overrides.
- `tests/unit/emergency/test_emergency_models.py`: Serialization and constraint validation tests for emergency domain models.
- `tests/unit/emergency/test_preemption.py`: Unit tests for preemption corridor detection, yield holds, and right-of-way clearance.
- `tests/unit/emergency/test_containment.py`: Unit tests for hazard buffers, EV thermal runaway handling, and fire door countdown timeouts.
- `tests/unit/emergency/test_evacuation.py`: Unit tests for route clearance, wave staggering, and exit selection.
- `tests/unit/emergency/test_emergency_viz.py`: Unit tests verifying tactical dashboard PNG rendering and dimensions.
- `tests/unit/test_emergency_cli.py`: CLI unit tests with JSON and PNG assertions across all subcommands.
- `tests/unit/test_pipeline_emergency.py`: Surround pipeline `--emergency` integration tests.

---

### 3. Decision Log

1. **Decision**: Corridor Centerline Segment Projection for Emergency Vehicle Preemption
   - *Alternatives Considered*: Pure radial proximity circle around responder.
   - *Rationale & Evidence*: Emergency responders travel at higher velocities ($v \ge 6$ m/s) along predictable ingress corridors towards incident points. A purely static radial bubble fails to warn or yield vehicles positioned 40 meters ahead along the responder's path until the responder is right next to them. Evaluating perpendicular distance to the responder's trajectory segment `(start_x, start_y) -> (target_x, target_y)` clears the entire corridor before the responder arrives.
2. **Decision**: Automatic EV Thermal Runaway Hazard Creation at 70°C Battery Pack Threshold
   - *Alternatives Considered*: Manual operator confirmation required before marking battery hazard.
   - *Rationale & Evidence*: Lithium-ion battery thermal runaway exhibits exponential self-heating once cell temperature reaches ~70°C. In dense automated parking facilities, waiting for human dispatcher verification risks catastrophic fire spread to adjacent bays. Automatic immediate boundary fencing safely quarantines the suspect vehicle and initiates perimeter evacuation without delay.
3. **Decision**: Increment Expected CLI Group Count from 23 to 24 in Release Audit
   - *Alternatives Considered*: Nesting emergency commands under the existing `control` or `mission` CLI groups.
   - *Rationale & Evidence*: Life safety and emergency preemption constitutes a critical standalone operational domain under ISO 23374 and NFPA 88A, warranting a first-class `nearfield360 emergency` CLI command group verified by `nearfield360 release audit`.

---

### 4. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Unit Tests | `uv run pytest -q` | 1304 passed, 5 skipped (0 failures) |
| Code Coverage | `uv run pytest --cov=nearfield360` | >88.5% (exceeds 85% requirement) |
| Static Analysis | `uv run ruff check .` | 0 errors |
| Code Formatting | `uv run ruff format --check .` | 0 errors |
| Strict Typing | `uv run mypy` | 0 errors in 164 source files |
| Lockfile | `uv lock --check` | Pass |
| Release Audit | `uv run nearfield360 release audit` | All checks pass (24 CLI groups registered) |
| Package Build | `uv build` | Success (sdist + wheel) |

---

## Phase 33: Facility Access Control, Barrier Gate Interlocks, License Plate Recognition (ALPR) & Vehicle Clearance Enveloping (ISO 23374 Gateway Operations)

**Status:** In Progress (Task 1 Complete)
**Repository Branch:** `phase/facility-access-control-barrier-gates`
**Pull Request:** Draft PR #13

### 1. Milestone Objectives & Scope

Phase 33 delivers an end-to-end Automated Facility Access Control, Barrier Gate Interlock, Automated License Plate Recognition (ALPR), and Vehicle Clearance Enveloping subsystem for NearField360 conforming to ISO 23374. The subsystem governs vehicle ingress and egress at facility perimeter portals, authenticating credentials, checking 3D dimensional clearance envelopes against portal structures, managing barrier arm kinematics and four-loop inductive sensor transitions with anti-tailgating interlocks, rendering tactical BEV gateway dashboards, exposing a dedicated `nearfield360 gate` CLI command group, and integrating gateway operations into the surround camera pipeline:

1. **Configuration (`GateConfig` in `src/nearfield360/config.py`, `configs/default.yaml`):**
   - Configurable portal clearance height (`portal_clearance_height_m`: 2.10 m), portal clearance width (`portal_clearance_width_m`: 2.60 m), minimum ground clearance (`ground_clearance_min_m`: 0.12 m), barrier arm length (`barrier_arm_length_m`: 3.0 m), opening time (`barrier_opening_time_s`: 1.8 s), hold time (`barrier_holding_time_s`: 5.0 s), closing time (`barrier_closing_time_s`: 2.2 s), ALPR confidence threshold (`alpr_confidence_threshold`: 0.85), anti-tailgating timeout (`anti_tailgating_timeout_s`: 2.0 s), safety loop debounce (`safety_loop_debounce_ms`: 50.0 ms), and UWB RSSI threshold (`uwb_rssi_threshold_dbm`: -75.0 dBm).
2. **Domain Models & Access Protocols (`src/nearfield360/gate/models.py`):**
   - Enums: `GateDirection` (`ENTRY`, `EXIT`, `BIDIRECTIONAL`), `BarrierArmState` (`LOWERED_CLOSED`, `OPENING`, `RAISED_OPEN`, `CLOSING`, `SAFETY_REVERSING`, `FAULT_LOCKED`), `CredentialType` (`ALPR_PLATE`, `UWB_DIGITAL_KEY`, `RFID_BADGE`, `QR_BARCODE`, `V2X_DIRECT_AUTH`), `AccessDecisionStatus` (`GRANTED`, `REJECTED_UNKNOWN_CREDENTIAL`, `REJECTED_OVER_HEIGHT`, `REJECTED_OVER_WIDTH`, `REJECTED_LOW_GROUND_CLEARANCE`, `REJECTED_EXPIRED_RESERVATION`, `TAILGATING_LOCKOUT`), `InductiveLoopId` (`LOOP_A_APPROACH`, `LOOP_B_PRESENCE`, `LOOP_C_SAFETY`, `LOOP_D_DEPARTURE`).
   - Domain models: `VehicleDimensionEnvelope`, `VehicleCredential`, `InductiveLoopState`, `GateAccessRequest`, `GateAccessDecision`, `BarrierKinematicsState`, `GateTransitEvent`, `GateAuditReport`.
3. **Access Credential Validator & ALPR Optical Matcher (`src/nearfield360/gate/validator.py`):**
   - String normalization (`normalize_plate_string`), Levenshtein similarity metric (`compute_plate_similarity`), registry CRUD methods, optical confidence cutoff checking (`alpr_confidence_threshold`), reservation validity window verification, and pass ID generation.
4. **Dimensional Clearance & Envelope Profiler (`src/nearfield360/gate/envelope.py`):**
   - In progress (Task 4).
5. **Inductive Loop & Anti-Tailgating Barrier Interlock Controller (`src/nearfield360/gate/controller.py`):**
   - Planned (Task 5).
6. **Gateway BEV Tactical Map & Visualizer (`src/nearfield360/gate/viz.py`):**
   - Planned (Task 6).
7. **Package Exports, CLI Command Group & Surround Pipeline Integration:**
   - Planned (Task 7).

---

### 2. Decision Log

1. **Decision**: Explicit 3D Dimensional Envelope Validation Before Physical Barrier Gate Actuation
   - *Alternatives Considered*: Allowing barrier arm opening solely upon credential match and delegating collision avoidance to vehicle ultrasonic sensors.
   - *Rationale & Evidence*: In automated valet parking structures, low ceiling pipes, ventilation ducts, and overhead signage frequently impose hard headroom limits (typically 2.1m). If an over-height vehicle (e.g. SUV with roof box) or low-clearance vehicle is admitted, structural damage or ramp bottoming occurs within the facility. Pre-barrier gateway clearance profiling ensures structural compliance before physical admission.

---

### 3. Verification & Quality Gates

| Check | Command | Result |
| --- | --- | --- |
| Gate Config Defaults & Overrides | `uv run pytest tests/unit/test_config.py` | 87 passed |
| Gate Domain Models & Constraints | `uv run pytest tests/unit/gate/test_gate_models.py` | 6 passed |
| Access Credential Validator & ALPR | `uv run pytest tests/unit/gate/test_validator.py` | 7 passed |
