# NearField360

**Real-Time Multi-Camera Surround-View Perception for Automated Parking**

[![CI](https://github.com/Mohamed-ahmed-shokry/nearfield360/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-ahmed-shokry/nearfield360/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

NearField360 is an engineering-focused computer vision system for calibrated automotive
fisheye cameras. Its target pipeline combines semantic perception, dynamic-object tracking,
camera-health awareness, and transparent geometric fusion into a local bird's-eye-view
representation around a vehicle.

The repository provides a reproducible, tested foundation: a WoodScape data layer, explicit
fisheye/vehicle geometry with ground-plane and BEV scaffolding, label-space perception
metrics, and inspection CLIs. Accuracy, latency, and FPS are deliberately not reported until
the corresponding experiments have run.

## Capability status

| Area | Status | Evidence |
| --- | --- | --- |
| Reproducible Python package | Implemented | Locked uv environment; wheel and sdist isolated-install smokes |
| Typed configuration and logging | Implemented | Strict validation, environment overrides, human/JSON logs; geometry and BEV sections |
| CLI and environment diagnostics | Implemented | `config validate`, `config show`, `doctor`, `geometry`, `data`, `occupancy`, `robustness`, `slots`, `plan`, and `control` commands |
| Integrated four-camera pipeline | Implemented | `nearfield360 pipeline run` with per-stage timings, FPS, health-aware fusion, tracking, risk, and occupancy PNG |
| Release readiness audit | Implemented | `nearfield360 release audit` verifying license, version consistency, default config, and CLI surface |
| Automated quality gates | Implemented | Ruff, strict mypy, pytest coverage, pre-commit, cross-platform CI |
| WoodScape data layer | Implemented core | Discovery, bounded readers, semantic/calibration/detection contracts, integrity, statistics, explicit-group splits, semantic overlays; synthetic tests |
| Fisheye calibration and geometry | Implemented core | Fourth-order radial projection/inverse, rigid transforms, vehicle cameras, surround rig, ground intersection, BEV grid; synthetic tests |
| Segmentation and detection metrics | Implemented | Numpy-only confusion/IoU and XYXY box IoU, file-based and live model-driven evaluation (`eval segmentation|detection --model`), split-aware scoring (`--split-manifest`/`--split`), detection confidence operating points & PR curves (`--confidence-thresholds`), semantic confidence calibration & ECE (`--confidence-bins`), SVG chart rendering (`eval plot`), comparative regression evaluation (`eval compare`), interactive HTML report dashboards (`eval dashboard`), prediction export, sample limits, and latency statistics |
| BEV occupancy and risk | Implemented | Multi-camera surround fusion (`--all-cameras`), Bayesian uncertainty propagation, 360-degree parking zones (corridors/clearance/circles), occupancy & uncertainty rendering |
| Controlled robustness evaluation | Implemented | Synthetic sensor corruptions (soiling, fog, noise, rain), extrinsic calibration perturbation engine, quantitative metrics (occupancy MAE, IoU, risk error), SVG/PNG charts, interactive HTML report dashboard |
| Dynamic obstacle tracking and forecasting | Implemented core | 2D-to-3D ground footprint projection, 2D metric Kalman filter, state lifecycle, forward trajectory forecasting, collision TTC ingress, and BEV visual overlays |
| Camera health monitoring and discounting | Implemented core | Laplacian blur variance, adaptive high-frequency lens soiling detection, blockage/underexposure detection, and health-aware Bayesian evidence discounting (`--health-aware`) |
| Modular ONNX perception runtime & inference engines | Implemented | OpenCV DNN backend, letterboxing/normalization, semantic segmentation & object detection engines, numerical parity verification, latency benchmarking, and live pipeline integration |
| 3D metric parking slot and free-space delineation | Implemented | Oriented 4-corner polygon fitting from road markings and obstacle gaps, occupancy/vacancy classification, approach corridor kinematics and collision feasibility, BEV visual overlays, `nearfield360 slots detect`, and surround pipeline integration (`pipeline run --slots`) |
| Autonomous parking trajectory planning and maneuvers | Implemented | Non-holonomic Ackermann vehicle kinematics, multi-stage maneuvers (parallel reverse S-turn, perpendicular/slanted reverse dock), continuous swept footprint collision validation against BEV occupancy and dynamic obstacles, time-parameterized speed profiling, `nearfield360 plan parking`, and surround pipeline integration (`pipeline run --plan-parking`) |
| Closed-loop trajectory tracking control & execution simulation | Implemented | Forward/reverse Stanley path follower, curvature feedforward, longitudinal PI control, kinematic bicycle simulator with actuator lag, real-time swept footprint safety monitoring and AEB, docking accuracy KPIs, `nearfield360 control execute`, and surround pipeline integration (`pipeline run --simulate-control`) |
| AVP mission executive & dynamic recovery orchestration | Implemented | Finite state machine lifecycle manager (7 states), multi-frame spatial slot tracking/memory (`SlotTracker`), dynamic obstacle yield holding, recovery re-planning around obstacles (`ParkingReplanner`), terminal docking evaluation, BEV mission dashboard & timeline telemetry charts (`viz.py`), `nearfield360 mission run`, and surround pipeline integration (`pipeline run --mission`) |
| PyTorch training & native TensorRT/C++ accelerators | Planned | No proprietary training checkpoints or CUDA toolchains assumed; pure ONNX and OpenCV DNN execution |

## System design

```mermaid
flowchart LR
    CAM[Four calibrated fisheye cameras] --> DATA[Validated synchronized samples]
    DATA --> GEO[Fisheye rays and vehicle transforms]
    DATA --> PER[Semantic and object perception]
    PER --> TRACK[Temporal tracking]
    DATA --> HEALTH[Camera health]
    GEO --> BEV[Local BEV / occupancy fusion]
    TRACK --> BEV
    HEALTH --> BEV
    BEV --> RISK[Explainable near-field risk zones]
    BEV --> SLOTS[3D metric parking slots]
    SLOTS --> PLAN[Autonomous trajectory planning & maneuvers]
    PER --> EXPORT[ONNX / TensorRT]
    EXPORT --> CPP[C++ runtime and profiling]
```

The design keeps raw fisheye imagery whenever possible. It does not assume that rectilinear
undistortion can preserve a field of view greater than 180 degrees, and it does not infer metric
3D structure without a documented geometric or learned assumption.

Vehicle axes are X forward, Y left, and Z up. Camera axes are X right, Y down, and Z forward.
The ground plane is `Z == ground_z` (default zero). Pixels alone never determine depth: the
geometry layer returns unit rays and, with an explicit ground assumption, footprints with
Euclidean distances. Rays without a forward intersection stay unknown. The BEV grid covers
`[x_min, x_max) x [y_min, y_max)` with square cells and half-open bounds, matching the
fisheye image-bounds convention.

## Quick start

[uv](https://docs.astral.sh/uv/) is the supported environment manager. The project pins Python
3.12 for development and CI also verifies Python 3.11 and 3.13.

```powershell
git clone git@github.com:Mohamed-ahmed-shokry/nearfield360.git
Set-Location nearfield360
uv sync --locked --group dev
uv run nearfield360 --version
uv run nearfield360 doctor
```

The same commands work in POSIX shells after replacing `Set-Location` with `cd`.

### Configuration

Validate the repository defaults and inspect the effective settings:

```powershell
uv run nearfield360 --config configs/default.yaml config validate
uv run nearfield360 config show
```

Environment variables use the `NEARFIELD360_` prefix and `__` for nested fields. Environment
values override YAML values:

```powershell
$env:NEARFIELD360_PATHS__DATASET_ROOT = "D:\datasets\woodscape"
$env:NEARFIELD360_GEOMETRY__THETA_MAX = "2.2"
$env:NEARFIELD360_BEV__RESOLUTION = "0.05"
uv run nearfield360 config show
```

`configs/default.yaml` documents the geometry angular limit (`geometry.theta_max`), the ground
plane (`geometry.ground_z`), the footprint range gate (`geometry.max_distance`), the local
BEV extent (`bev.x_min/x_max/y_min/y_max/resolution`), the occupancy policy
(`occupancy.free_classes`/`occupied_classes`/`confidence_slope`/`min_evidence`), and the risk
filter (`risk.front_length`/`half_width`/`start_x`/`rear_length`/`rear_start_x`/`lateral_width`/
`vehicle_x_min`/`vehicle_x_max`/`near_radius`/`warning_radius`/`danger_occupancy`), and the robustness
benchmark settings (`robustness.severities`/`seed`/`rotation_perturbations_deg`/
`translation_perturbations_m`). Older configuration files without these sections continue to load with
these defaults.

Relative paths are interpreted from the process working directory. Dataset presence is not
required for `--help`, `--version`, configuration validation, or the test suite.

### Geometry inspection

Validate a per-image calibration and map pixels to ground footprints without inferring depth:

```powershell
uv run nearfield360 geometry info --calibration D:\datasets\woodscape\calibration_data\00001_FV.json --json
uv run nearfield360 geometry ground --calibration D:\datasets\woodscape\calibration_data\00001_FV.json --pixel 640,480 --pixel 800,480 --json
uv run nearfield360 geometry bev --json
```

`--theta-max` overrides the configured angular limit; `--check-bounds/--ignore-bounds` toggles
the half-open image rectangle. Invalid rays and intersections behind the ray origin report as
unknown rather than extrapolated footprints.

### Occupancy fusion and risk

Fuse semantic-grounded frames into a local bird's-eye-view occupancy layer and evaluate
configured safety zones. Supports single-camera inspection or synchronized surround fusion
(`--all-cameras` combining front, rear, left, and right cameras). Only pixels whose semantic
class votes free or occupied are rasterized; each measurement is weighted by
`1 / (1 + confidence_slope * distance)` so distant observations contribute less. Per-cell
Bayesian uncertainty `Var(p) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))` is
tracked alongside occupancy probability, with mean and maximum uncertainty reported per zone:

```powershell
uv run nearfield360 occupancy layer --root D:\datasets\woodscape --all-cameras --output outputs/occupancy/surround.json --png outputs/occupancy/surround.png --uncertainty-png outputs/occupancy/uncertainty.png
uv run nearfield360 occupancy zones --json
```

`--samples N` limits how many frames per camera are fused (0 fuses all). The command outputs a
structured JSON risk report documenting active parameters, fused samples, per-zone cell metrics
(total, confident, danger count, danger ratio, mean and max uncertainty), and grid matrices.
Optional PNG outputs render:
- `--png`: BEV occupancy map shading cells from free (green) to occupied (red) with vector zone
  overlays (`forward_corridor`, `rear_corridor`, `left_clearance`, `right_clearance`,
  `near_circle`, and `warning_circle`).
- `--uncertainty-png`: Bayesian variance heatmap visualized using an inferno colormap.
- `--health-aware`: Assess camera optical health and discount degraded/soiled evidence in BEV fusion.

The scene-level policy is configurable: `occupancy.free_classes` vote for free space,
`occupancy.occupied_classes` vote for occupancy, `min_evidence` gates cell confidence, and
`risk.danger_occupancy` sets the occupancy threshold above which confident cells are marked
dangerous. Single-frame CLIs (`geometry ground`) and multi-camera fusion (`occupancy layer`)
share the identical ray, calibration, and grid models.

### Temporal BEV occupancy forecasting

Forecast future bird's-eye-view space occupancy across a multi-second parking horizon ($[t+\Delta t, \dots, t+H]$)
using spatiotemporal recurrence and dynamic velocity flow fields:

```powershell
# Forecast 3.0 seconds into the future over 0.5s increments with multi-step panel visualization:
uv run nearfield360 occupancy forecast --root D:\datasets\woodscape --all-cameras --horizon 3.0 --step 0.5 --output outputs/occupancy/forecast.json --png outputs/occupancy/forecast_panels.png

# Export the recurrent temporal forecasting network as an ONNX model graph:
uv run nearfield360 occupancy export-model --output models/temporal_forecaster.onnx --hidden-channels 16 --horizon-steps 6
```

### Camera health assessment

Evaluate camera optical health, lens soiling, blur, and exposure anomalies for any surround camera frame:

```powershell
# Assess a single camera image and print formatted diagnostics
uv run nearfield360 health assess --image D:\datasets\woodscape\rgb_images\00001_FV.png --camera FV

# Emit machine-readable health metrics JSON or write atomically to an artifact
uv run nearfield360 health assess --image D:\datasets\woodscape\rgb_images\00001_FV.png --camera FV --json
uv run nearfield360 health assess --image D:\datasets\woodscape\rgb_images\00001_FV.png --output outputs/health/00001_FV.json
```

The report provides:
- **Status & Anomalies:** `healthy`, `degraded`, or `blocked` classification with specific anomaly tags (`soiling`, `blur`, `blockage`, `underexposure`, `overexposure`).
- **Quantitative Metrics:** Localized high-frequency Laplacian energy variance (sharpness), spatial blockage ratio, mean brightness, and contrast.
- **Discount Weight:** Evidence multiplier in `[0.0, 1.0]` recommended for BEV fusion attenuation.

### Dynamic obstacle tracking and collision forecasting

Track dynamic obstacles across consecutive temporal frames, project 2D detections into vehicle-centric ground footprints via fisheye ray unprojection, filter motion with a 2D metric Kalman filter, forecast future trajectories, and evaluate collision Time-to-Collision (TTC) with surround parking zones:

```powershell
# Track obstacles from a single camera view
uv run nearfield360 track run --root D:\datasets\woodscape --camera FV --output outputs/tracking/report.json --png outputs/tracking/overlay.png

# Multi-camera temporal tracking across all 4 synchronized surround cameras
uv run nearfield360 track run --root D:\datasets\woodscape --all-cameras --max-frames 20 --output outputs/tracking/multi_cam.json --png outputs/tracking/multi_cam.png
```

The tracking engine delivers:
- **Fisheye Ground Projection:** Unprojects bottom-center contact points through radial-polynomial camera models onto the road plane (`Z == ground_z`) with class-specific 3D metric bounding footprints.
- **2D Metric Kalman Filter:** State vector `[x, y, vx, vy]` under a constant velocity kinematic model with track lifecycle management (`tentative`, `confirmed`, `lost`, `deleted`).
- **Trajectory Forecasting & Safety Zone Ingress:** Forward trajectory extrapolation over configurable horizon evaluating spatial intersection and Time-to-Collision against all surround safety zones (`forward_corridor`, `near_circle`, etc.).
- **BEV Visual Overlay Rendering (`--png`):** Color-coded ground footprints, 1-second velocity heading arrows, past position trails, predicted forward trajectories, and collision warning badges.

### Robustness evaluation and diagnostic plots

Evaluate BEV occupancy resilience under controlled sensor corruptions and extrinsic calibration
misalignment, and export publication-quality diagnostic plots or interactive HTML dashboards:

```powershell
# 1. Apply synthetic sensor corruptions (lens_soiling, fog, low_light_noise, rain)
uv run nearfield360 robustness corrupt --image sample.png --type lens_soiling --severity 3 --output outputs/soiled.png

# 2. Simulate extrinsic calibration misalignment (Euler rotation angles and translation deltas)
uv run nearfield360 robustness perturb-calibration --calibration D:\datasets\woodscape\calibration_data\00001_FV.json --roll 2.0 --dx 0.1 --output outputs/perturbed_calib.json

# 3. Run full automated robustness benchmark sweeps against clean ground truth
uv run nearfield360 robustness benchmark --root D:\datasets\woodscape --camera FV --samples 5 --output outputs/robustness/report.json

# 4. Generate SVG vector charts, raster PNGs, and a self-contained interactive HTML dashboard
uv run nearfield360 robustness plot --report outputs/robustness/report.json --output-dir outputs/robustness/plots
```

The benchmark computes quantitative degradation metrics across corruptions and calibration axes:
- **Occupancy Mean Absolute Error (MAE):** Difference in occupancy probability against clean ground truth.
- **Occupied & Free Space IoU:** Cell-level binary segmentation preservation at the configured risk threshold.
- **Bayesian Uncertainty Shift:** Mean shift in Dirichlet/Beta variance across observed BEV cells.
- **Parking Safety Zone Risk Error:** Absolute error in zone-level occupied risk cell count.

The `plot` subcommand generates:
- Pure-Python vector SVG performance degradation curves (`corruption_degradation.svg`, `calibration_sensitivity.svg`).
- OpenCV rasterized PNG comparison charts (`corruption_degradation.png`, `calibration_sensitivity.png`).
- A self-contained, responsive HTML diagnostic dashboard (`index.html`) with embedded vector charts, metric tables, and environmental metadata.

### Neural perception inference and benchmarking

Run live semantic segmentation or 2D object detection directly on fisheye camera feeds, benchmark ONNX model latency, or inspect neural network tensor shapes:

```powershell
# 1. Run semantic segmentation on a fisheye image with WoodScape palette visualization:
uv run nearfield360 infer semantic --model models/segmentation.onnx --image D:\datasets\woodscape\rgb_images\00001_FV.png --output outputs/mask.png

# 2. Run 2D object detection with box unscaling and non-maximum suppression (NMS):
uv run nearfield360 infer detection --model models/detection.onnx --image D:\datasets\woodscape\rgb_images\00001_FV.png --output outputs/detections.json --json

# 3. Benchmark inference latency distribution and throughput (single batch or scaling sweep):
uv run nearfield360 infer benchmark --model models/segmentation.onnx --batch-size 4 --iterations 50 --warmup 10
uv run nearfield360 infer benchmark --model models/segmentation.onnx --batch-sweep --batch-sizes 1,2,4,8 --output outputs/sweep.json

# 4. Inspect ONNX model input/output shapes and metadata:
uv run nearfield360 infer inspect --model models/segmentation.onnx

# 5. Verify numerical parity between two ONNX models (e.g. after quantization):
uv run nearfield360 infer parity --model-a models/original.onnx --model-b models/quantized.onnx

# 6. Run live perception directly within multi-camera BEV occupancy mapping and tracking:
uv run nearfield360 occupancy layer --root D:\datasets\woodscape --model models/segmentation.onnx --output outputs/occupancy.json
uv run nearfield360 track run --root D:\datasets\woodscape --model models/detection.onnx --output outputs/tracking.json
```

### Reproducible evaluation against dataset annotations

Score semantic masks or detected boxes against annotated WoodScape samples. Source selection is
mutually exclusive: use `--predictions` for exported per-sample files (PNG class-ID masks /
scored TXT rows) or `--model` for a live ONNX run in the same command. Reports are reproducible
JSON artifacts with environment/config metadata, sample accounting, metrics, and — for model
runs — engine identity, effective detection thresholds, and latency statistics:

```powershell
# 1. Score exported prediction files against ground-truth annotations:
uv run nearfield360 eval segmentation --root D:\datasets\woodscape --predictions outputs/predictions --output outputs/segmentation_report.json
uv run nearfield360 eval detection --root D:\datasets\woodscape --predictions outputs/predictions --output outputs/detection_report.json

# 2. Run a live ONNX model over the annotated dataset (exclusive with --predictions):
uv run nearfield360 eval segmentation --root D:\datasets\woodscape --model models/segmentation.onnx --output outputs/segmentation_report.json
uv run nearfield360 eval detection --root D:\datasets\woodscape --model models/detection.onnx --iou-threshold 0.5 --output outputs/detection_report.json

# 3. Bound the run, override config-driven thresholds, and export model predictions for reuse:
uv run nearfield360 eval detection --root D:\datasets\woodscape --model models/detection.onnx --limit 50 --confidence-threshold 0.4 --nms-threshold 0.5 --save-predictions outputs/predictions --output outputs/detection_subset.json

# 4. Score only one split from `nearfield360 data split` (--split defaults to test):
uv run nearfield360 eval segmentation --root D:\datasets\woodscape --predictions outputs/predictions --split-manifest outputs/splits.json --split test --output outputs/segmentation_test.json

# 5. Add pooled confidence operating points and per-class PR grids to a detection report:
uv run nearfield360 eval detection --root D:\datasets\woodscape --predictions outputs/predictions --confidence-thresholds 0.3,0.5,0.7 --output outputs/detection_confidence.json

# 6. Pool per-pixel softmax confidence into a calibration reliability table with ECE:
uv run nearfield360 eval segmentation --root D:\datasets\woodscape --model models/segmentation.onnx --confidence-bins 10 --output outputs/segmentation_calibration.json

# 7. Render eval report confidence analysis to vector SVG charts:
uv run nearfield360 eval plot --report outputs/detection_confidence.json --output-dir outputs/charts
uv run nearfield360 eval plot --report outputs/segmentation_calibration.json --output-dir outputs/charts

# 8. Compare two evaluation reports with delta metrics, tabular output, and regression gates:
uv run nearfield360 eval compare --baseline outputs/baseline_report.json --candidate outputs/model_report.json
uv run nearfield360 eval compare --baseline outputs/baseline.json --candidate outputs/candidate.json --fail-under-miou-delta -0.01 --fail-over-latency-ratio 1.15 --output outputs/comparison.json

# 9. Render an interactive, standalone HTML evaluation dashboard (zero external CDN dependencies):
uv run nearfield360 eval dashboard --report outputs/segmentation_report.json --output outputs/dashboard.html
uv run nearfield360 eval dashboard --report outputs/candidate.json --baseline outputs/baseline.json --fail-under-miou-delta 0.0 --output outputs/comparison_dashboard.html
```

With `--split-manifest`, the manifest's sample-identity digest is validated against the
discovered dataset before scoring, and the report gains a top-level `"split"` key (manifest
path, split name, grouping provenance, seed, and dataset/selected sample counts). With
`--confidence-thresholds`, `metrics.confidence_analysis` records pooled operating points
(predictions, true positives, precision, recall, F1 per cutoff) and VOC-style 101-point
interpolated precision-recall curves per class. With `--confidence-bins`, segmentation reports
record an expected calibration error (ECE) and a confidence reliability table with mean confidence
and observed accuracy per bin. `nearfield360 eval plot` renders these analyses to standalone SVG
vector charts (`pr_curves.svg`, `operating_points.svg`, `reliability.svg`).

`nearfield360 eval compare` contrasts two evaluation JSON reports of the same task (segmentation or
detection), displaying summary deltas, class-level breakdowns, timing differences, and calibration ECE
shifts. Regression gates (`--fail-under-miou-delta`, `--fail-under-map-delta`, `--fail-over-ece-delta`,
`--fail-over-latency-ratio`) enforce automated CI thresholds, exiting with code 1 if any gate fails.
Machine-readable delta payloads can be exported via `--output` or streamed via `--json`.

`nearfield360 eval dashboard` generates a self-contained, responsive HTML evaluation dashboard
combining KPI summary cards, per-class performance meters, embedded SVG PR curves, reliability diagrams,
latency distribution bar charts, and comparative diff views with zero external CDN dependencies. When
provided with `--baseline`, CI/CD regression gates are verified and embedded directly into the report.

## Dataset policy

[WoodScape](https://github.com/valeoai/WoodScape) is the primary target dataset because it
provides four automotive surround-view cameras and annotations for complementary perception
tasks. Its official repository labels the **data license as proprietary** even though its tools
have a separate open-source license. Download and accept the dataset terms through Valeo's
official channel; do not add WoodScape images, annotations, or calibration bundles to this
repository.

Inspect an already acquired dataset without changing its files:

```powershell
uv run nearfield360 data verify --root D:\datasets\woodscape --require-semantic --require-calibration
uv run nearfield360 data stats --root D:\datasets\woodscape --semantic --json
```

`--root` overrides the configured dataset path. Verification exits with status 1 for integrity
errors and 2 for a missing/invalid root. Statistics decode RGB files, report actual file sizes and
resolutions, and optionally count pixels in available semantic masks. They are not model metrics.
Unit tests use small synthetic fixtures, so contributors and CI do not need restricted data.

Render inspection overlays for samples that have semantic masks (samples without masks are
skipped, not fabricated):

```powershell
uv run nearfield360 data viz --root D:\datasets\woodscape --output outputs/viz --max-images 20
```

Split generation can keep explicitly supplied recording/sequence groups together. Its default
groups equal filename identifiers only: those identifiers do **not** establish camera
synchronization or recording-level separation. Evaluation must document verified grouping
provenance; do not treat the fallback as a leakage-free benchmark split.

## Development checks

Run the same core gates used in CI:

```powershell
uv lock --check
uv sync --locked --group dev
uv run pre-commit run --all-files
uv run pytest -q --cov=nearfield360 --cov-report=term-missing
uv build --clear
uv run twine check dist/*
```

Tests that eventually require external datasets, a GPU, or long runtimes are marked separately;
the default suite remains CPU-only and synthetic. Current CI runs on Linux with Python 3.11,
3.12, and 3.13, and on Windows with Python 3.12.

## Reproducibility and results policy

- Configuration, seeds, commands, code revisions, and environment metadata accompany each
  meaningful experiment.
- Dataset files, credentials, large checkpoints, generated engines, and machine-local output are
  ignored by Git.
- Measured results must identify hardware, software versions, input resolution, precision,
  dataset split, warmup, and timing boundaries.
- Missing CUDA, TensorRT, native compiler, or proprietary-data evidence is reported as
  unavailable—not silently replaced with estimates.

## Roadmap

1. ~~WoodScape discovery, parsing, integrity checks, statistics, and visualization.~~ Done (core):
   discovery, bounded RGB/label readers, semantic/calibration/detection contracts, integrity,
   statistics, explicit-group splits with manifests, and semantic overlay rendering.
2. ~~Fourth-order fisheye projection/inverse projection and camera-to-vehicle transforms.~~ Done
   (core): radial-polynomial projection/inverse, rigid transforms, calibrated cameras, surround
   rig, ground-plane intersection, and BEV grid scaffolding with geometry CLIs.
3. ~~Deployment-oriented semantic segmentation, metrics, and reproducible experiments.~~ Done
   (core): numpy-only confusion/IoU, box-IoU metrics, prediction parsing, evaluation CLI,
   modular OpenCV DNN inference engine, letterbox preprocessing, continuous confidence heatmaps,
   and `nearfield360 infer semantic` command.
4. ~~Object detection, temporal tracking, and camera-soiling awareness.~~ Done (core):
   WoodScape 2D-to-3D ground footprint projection, 2D metric Kalman filtering with constant velocity kinematics, multi-object tracker with lifecycle management (tentative, confirmed, lost, deleted), forward trajectory forecasting over time horizon, collision TTC zone ingress detection, BEV visual overlays with velocity vectors and predicted paths, camera optical health assessment (soiling, blur, underexposure, overexposure, blockage), and health-aware occupancy evidence discounting (`--health-aware`).
5. ~~Multi-camera BEV fusion, uncertainty propagation, and transparent safety zones.~~ Done
   (core): 360-degree multi-camera surround fusion (`--all-cameras`), distance-decayed evidence
   rasterization, per-cell Bayesian uncertainty estimation, transparent parking safety zones
   (forward/rear corridors, lateral clearance, near/warning circles), and dual
   occupancy/uncertainty map rendering.
6. ~~Controlled robustness evaluation and automated plots.~~ Done (core): synthetic sensor corruptions
   (lens soiling, fog, low-light noise, rain), extrinsic calibration perturbation engine (Euler
   rotations and 3D translation shifts), quantitative occupancy/risk degradation benchmark runner,
   pure-Python vector SVG and OpenCV raster PNG plot engines, self-contained interactive HTML
   dashboard, and Typer `robustness` CLI suite.
7. ~~ONNX perception runtime, numerical parity, and latency benchmarking.~~ Done (core):
   Modular `InferenceBackend` protocol, `OpenCVDNNBackend`, `FisheyeImagePreprocessor`,
   `ObjectDetectionEngine` with NMS and spatial unscaling, numerical parity verification
   (`verify_numerical_parity`), statistical latency benchmark engine (`BenchmarkSummary`, FPS,
   p50/p95/p99), `nearfield360 infer` CLI suite, and live `--model` integration into `occupancy layer`
   and `track run`. Next: native TensorRT execution provider and C++ runtime wrapper.
8. ~~Integrated four-camera demo, measured performance report, and release audit.~~ Done:
   `nearfield360 pipeline run` orchestrates health, multi-camera occupancy fusion, dynamic
   tracking, and risk evaluation over complete four-camera frames with per-stage timing and
   FPS metrics in a JSON report; `nearfield360 release audit` verifies packaging metadata,
   license, default config, and CLI surface consistency for release readiness.
9. ~~Live ONNX perception in the integrated pipeline and latency distribution statistics.~~ Done:
   `--seg-model`/`--det-model` on `pipeline run` for annotation-free live inference,
   per-frame latency samples with p50/p95/p99 percentiles in the performance report, and
   end-to-end integration coverage with synthetic ONNX models. Excludes TensorRT/C++
   acceleration (remains under roadmap item 7 residual work).
10. ~~Configuration-driven inference backends and optional ONNX Runtime support.~~ Done:
    real `OnnxRuntimeBackend` behind an installable extra, no silent OpenCV fallback when
    ONNX Runtime is requested, `--backend` selection on live perception CLIs defaulting from
    `config.inference.backend`, and detection confidence/NMS thresholds wired from config
    into tracking and pipeline engines. Excludes TensorRT/C++ (roadmap item 7 residual).
11. ~~Model-driven evaluation against dataset annotations.~~ Done:
    `eval segmentation --model` / `eval detection --model` run a live ONNX model over
    annotated WoodScape samples and score predictions in the same command (joining the
    inference engines from roadmap items 7/10 with the metrics from item 3), with
    `--backend`/`--device` selection, config-driven detection thresholds, `--limit`,
    `--save-predictions` export for the file-based workflow, and per-sample latency
    statistics in the report. Excludes confidence-threshold sweeps/PR curves, split-manifest
    filtering, batched inference, and TensorRT/C++ (roadmap item 7 residual).
12. ~~Split-aware evaluation and detection confidence analysis.~~ Done:
    `--split-manifest`/`--split` on both `eval` commands restrict scoring to a held-out
    split created by `nearfield360 data split` (manifest digest-validated against the
    dataset and recorded in the report), and `eval detection --confidence-thresholds`
    adds pooled operating points (precision/recall/F1 per score threshold) plus per-class
    101-point interpolated precision-recall curves to the metrics report. Excludes
    model-side re-inference sweeps, calibration metrics (ECE), batched inference, and
    TensorRT/C++ (roadmap item 7 residual).
13. ~~Segmentation confidence calibration and evaluation report rendering.~~ Done:
    `eval segmentation --confidence-bins` pools per-pixel softmax confidences from
    `--model` runs into a reliability table with expected calibration error (ECE) under
    `metrics.confidence_analysis`, and `eval plot` renders prior eval reports to
    vector SVG charts (detection PR curves and operating points, semantic reliability
    diagram) through the shared plot engine. Excludes detection-side ECE, raster PNG
    charts, HTML dashboards, batched inference, and TensorRT/C++ (roadmap item 7
    residual).
14. ~~Comparative evaluation reporting and regression analysis.~~ Done:
    `eval compare` contrasts two evaluation JSON reports (e.g., baseline vs. model,
    split-to-split, or backend comparison) with delta metrics (mIoU/AP differences, ECE
    shifts, latency ratios), comparative tabular output, regression gate thresholds
    (`--fail-under-miou-delta`, `--fail-under-map-delta`, `--fail-over-ece-delta`,
    `--fail-over-latency-ratio`), and machine-readable delta JSON artifacts. Excludes HTML
    dashboards, batched inference, and TensorRT/C++ (roadmap item 7 residual).
15. ~~Interactive evaluation HTML report dashboard.~~ Done:
    `eval dashboard` renders self-contained, responsive HTML evaluation reports combining
    summary metrics, class-level performance cards, embedded SVG PR curves, reliability diagrams,
    latency distribution bar charts, and comparative diff views into a standalone report with
    zero external CDN dependencies. Excludes batched inference and TensorRT/C++ (roadmap item 7 residual).
16. ~~Batched inference throughput profiling and execution.~~ Done:
    Batched 2D object detection and semantic segmentation inference engines (`predict_batch`
    and `predict_batch_annotations`), automated throughput sweep engine (`benchmark_batch_sweep`)
    evaluating latency percentiles, throughput (FPS), relative speedup, and scaling efficiency;
    CLI integration with `infer benchmark --batch-size` and `--batch-sweep / --batch-sizes`;
    and multi-camera surround batched perception execution (`pipeline run --batch-cameras`).
    Excludes native TensorRT/C++ (roadmap item 7 residual).
17. ~~Native TensorRT acceleration and C++ runtime wrapper.~~ Done:
    High-performance `TensorrtBackend` integration via ONNX Runtime `TensorrtExecutionProvider`
    and native engine execution interfaces, configuration-driven workspace and cache parameters,
    FP16 half-precision conversion and dynamic INT8 quantization engines (`infer optimize`),
    TensorRT INT8 calibration cache table generator (`infer calibrate`), host acceleration
    provider discovery (`infer providers`), zero-copy CUDA pinned memory buffer pools
    (`CUDAPinnedBufferPool`), and standalone embedded automotive C++ deployment architecture
    with multi-camera surround batching and BEV ground-plane unprojection (`deploy/cpp/`).
18. ~~Temporal bird's-eye-view multi-camera occupancy forecasting network.~~ Done:
    Spatiotemporal recurrent BEV perception engine (`TemporalOccupancyForecaster`), cross-attention
    multi-camera boundary fusion (`fuse_cross_attention_occupancy`), ConvGRU temporal recurrence
    modeling occlusion memory and belief persistence, 2D continuous metric velocity field estimation
    anchored by Kalman-tracked obstacles (`TrackedObstacle`), multi-step forward forecasting rollout
    (`OccupancyForecastGrid`) with flow advection, static decay, and Bayesian uncertainty diffusion;
    temporal safety zone risk forecasting (time-to-intrusion, peak hazard envelopes); exportable
    ONNX computation graph generator (`export_temporal_forecaster_onnx`); CLI commands
    `occupancy forecast` and `occupancy export-model` with multi-horizon panel visualization;
    and surround pipeline integration (`pipeline run --temporal-forecast`).
19. ~~3D metric parking slot and free-space delineation engine.~~ Done:
    Metric slot boundary extraction from BEV road markings (`lanemarks`, `curb`) and obstacle
    free-space gaps, oriented 4-corner polygon fitting, slot type categorization (`parallel`,
    `perpendicular`, `slanted`), interior occupancy and Bayesian uncertainty classification
    (`vacant`, `occupied`, `uncertain`), dynamic obstacle clearance validation, approach corridor
    kinematics and collision feasibility evaluation, CLI command group `nearfield360 slots detect`
    (with `--vacant-only` and `--png` vector overlay), and surround pipeline integration
    (`pipeline run --slots`).
20. ~~Autonomous parking trajectory planning, Ackermann kinematics, and multi-stage maneuver engine.~~ Done:
    Non-holonomic Ackermann vehicle kinematics model ($R_{\min} = L / \tan(\delta_{\max})$), vehicle footprint
    geometry with overhangs, multi-stage maneuver generators (parallel reverse S-turn, perpendicular/slanted reverse
    docking), continuous swept footprint rasterization and collision clearance evaluation against BEV occupancy grids
    and dynamic obstacles, smooth time-parameterized trapezoidal speed profiler, CLI command group `nearfield360 plan parking`
    (with `--slots-json`, custom start pose, and `--png` visualization), and surround pipeline integration
    (`pipeline run --plan-parking`).
21. ~~Closed-loop parking trajectory tracking control, maneuver execution simulation, and dynamic safety monitoring.~~ Done:
    Nonlinear forward and reverse Stanley path tracking controller, feedforward curvature steering and longitudinal PI velocity profiling, kinematic bicycle simulator with first-order actuator lag ($\tau$) and Gaussian localization noise, multi-stage maneuver state machine with gear shift dwell, real-time swept footprint safety monitoring and automated emergency braking (AEB), tracking error watchdog abort protection, terminal docking accuracy KPIs ($\Delta x, \Delta y, \Delta \theta$), CLI command group `nearfield360 control execute` (with `--plan-json`, `--output`, `--png`, `--telemetry-png`, and `--inject-obstacle`), and surround pipeline integration (`pipeline run --simulate-control`).
22. ~~Autonomous Valet Parking (AVP) mission executive, dynamic replanning, and multi-stage recovery orchestration.~~ Done:
    Finite state machine lifecycle manager (`MissionExecutive`) governing 7 operational states (`STANDBY`, `SEARCHING`, `SLOT_SELECTED`, `APPROACHING`, `PARKING_MANEUVER`, `OBSTACLE_HOLD`, `REPLANNING`, `FINAL_ALIGNMENT`, `COMPLETED`, `ABORTED`), multi-frame spatial slot tracking with Hungarian/Euclidean association and temporal confidence decay (`SlotTracker`), online dynamic obstacle yield holding and dwell timeout management, multi-stage recovery re-planning around persistent obstructions (`ParkingReplanner`), terminal precision docking evaluation, rich BEV mission dashboard overlay and timeline telemetry charts (`viz.py`), dedicated CLI command group `nearfield360 mission run` (with `--slots-json`, `--plan-json`, `--scenario`, `--png`, `--timeline-png`, `--max-replans`, `--hold-timeout`), release audit verification (15 CLI command groups), and surround pipeline integration (`pipeline run --mission`).

## Usage: integrated four-camera pipeline and release audit

Run the full four-camera surround pipeline (discovery → health/occupancy/tracking → risk)
with a measured performance report:

```powershell
# Process all complete four-camera frames and write a timed JSON report
uv run nearfield360 pipeline run --root D:\datasets\woodscape --output outputs/pipeline/report.json

# Limit frames, render fused occupancy, and discount degraded camera evidence
uv run nearfield360 pipeline run --root D:\datasets\woodscape --samples 5 --health-aware --output outputs/pipeline/report.json --png outputs/pipeline/occupancy.png

# Live surround perception with temporal BEV occupancy forecasting:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --temporal-forecast --output outputs/pipeline/report.json

# Live surround perception with 3D metric parking slot and approach corridor delineation:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --slots --output outputs/pipeline/slots.json --png outputs/pipeline/slots.png

# Live surround perception with autonomous parking trajectory planning into the best vacant slot:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --slots --plan-parking --output outputs/pipeline/plan_report.json --png outputs/pipeline/plan_overlay.png

# Live surround perception with closed-loop parking execution simulation & docking verification:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --simulate-control --output outputs/pipeline/control_report.json --png outputs/pipeline/control_bev.png

# Execute a precomputed trajectory plan with BEV overlay and time-series telemetry charts:
uv run nearfield360 control execute --plan-json outputs/pipeline/plan_report.json --output outputs/control/exec_report.json --png outputs/control/bev_exec.png --telemetry-png outputs/control/telemetry.png

# Live surround perception with multi-camera batched neural network execution:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --seg-model models/seg.onnx --det-model models/det.onnx --batch-cameras --output outputs/pipeline/report.json
```

The report includes `environment` provenance, effective `config`, per-stage timings
(`discovery_ms`, `perception_ms`, `occupancy_ms`, `tracking_ms`, `risk_ms`, `forecast_ms`,
`total_ms`), `frames_per_second`, per-frame latency distribution (`frame_latency` with
mean/p50/p95/p99/min/max in milliseconds), fused `evidence`, per-zone `risk`, active
`tracks`, and `forecasts`. Frames missing any of the four cameras, calibration, semantic
masks (unless `--seg-model` is provided), or detections (unless `--det-model` is provided)
are skipped with a warning.

For live neural perception without precomputed annotations, pass ONNX models:

```powershell
# Annotation-free live pipeline: RGB + calibration only, models supply masks and boxes
uv run nearfield360 pipeline run --root D:\datasets\woodscape --seg-model models/segmentation.onnx --det-model models/detection.onnx --output outputs/pipeline/live.json

# Select the inference backend (defaults to config.inference.backend; opencv | onnxruntime | tensorrt)
uv run nearfield360 pipeline run --root D:\datasets\woodscape --seg-model models/segmentation.onnx --det-model models/detection.onnx --backend opencv --output outputs/pipeline/live.json
```

`--seg-model` replaces semantic masks for occupancy evidence; `--det-model` replaces
detection files for ground-footprint projection. Model paths are recorded under
`samples.seg_model` / `samples.det_model` in the report.

### 3D metric parking slot detection and feasibility

Delineate 3D metric parking slots from road markings (`lanemarks`, `curb`) and obstacle free-space gaps, classify occupancy, and evaluate approach corridor feasibility:

```powershell
# Detect parking slots across surround cameras with PNG BEV overlay:
uv run nearfield360 slots detect --root D:\datasets\woodscape --all-cameras --output outputs/slots/report.json --png outputs/slots/bev.png

# Single-camera front view slot detection filtering to vacant slots only:
uv run nearfield360 slots detect --root D:\datasets\woodscape --camera FV --vacant-only --output outputs/slots/vacant.json
```

The output JSON report documents detected slots (`slot_id`, `slot_type`, `corners`, `center`, `heading_rad`, `width_m`, `length_m`, `status`, `confidence`, and `approach_path` feasibility), along with a summary of total, vacant, occupied, uncertain, and feasible slot counts.

### Autonomous parking trajectory planning and maneuvers

Plan kinematically feasible, multi-stage parking maneuvers into detected vacant parking slots using non-holonomic Ackermann bicycle kinematics ($R_{\min} = L / \tan(\delta_{\max})$), continuous swept footprint rasterization, and trapezoidal speed profiling:

```powershell
# Plan parking trajectory from an existing slots detection JSON report:
uv run nearfield360 plan parking --slots-json outputs/slots/report.json --output outputs/plan/trajectory.json --png outputs/plan/bev.png

# Detect slots and plan parking end-to-end from a dataset root:
uv run nearfield360 plan parking --root D:\datasets\woodscape --output outputs/plan/trajectory.json --png outputs/plan/bev.png

# Specify custom vehicle start pose (x_m, y_m, heading_rad):
uv run nearfield360 plan parking --root D:\datasets\woodscape --start-pose "3.0,1.5,0.0" --output outputs/plan/trajectory.json

# Integrated surround pipeline perception with live parking trajectory generation:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --slots --plan-parking --output outputs/pipeline/plan_report.json --png outputs/pipeline/plan_bev.png
```

The output JSON report details the target slot, vehicle kinematic configuration, trajectory feasibility status (`feasible`, `collision_violated`, `kinematically_infeasible`), individual maneuver segments (`REVERSE_ARC`, `FORWARD_ALIGNMENT`, etc.) with gear selections, waypoints `(x, y, heading, curvature, speed, acceleration, timestamp)`, total travel length, and duration.

### Autonomous Valet Parking (AVP) mission executive and recovery

Orchestrate the complete end-to-end Autonomous Valet Parking mission lifecycle, coordinating multi-frame spatial slot tracking/memory (`SlotTracker`), closed-loop trajectory tracking, online dynamic obstacle yield holding (`OBSTACLE_HOLD`), multi-stage recovery re-planning around persistent obstructions (`ParkingReplanner`), and terminal precision docking:

```powershell
# Run nominal AVP mission simulation with BEV dashboard overlay and timeline telemetry charts:
uv run nearfield360 mission run --output outputs/mission/nominal.json --scenario nominal --png outputs/mission/dashboard.png --timeline-png outputs/mission/timeline.png

# Simulate dynamic obstacle yield and automatic resumption:
uv run nearfield360 mission run --output outputs/mission/transient.json --scenario transient_obstacle --hold-timeout 10.0

# Simulate recovery re-planning evasion around a persistently blocked slot entrance:
uv run nearfield360 mission run --output outputs/mission/replan.json --scenario blocked_replan --max-replans 3

# Run AVP mission lifecycle on a custom precomputed trajectory plan or slots report:
uv run nearfield360 mission run --plan-json outputs/plan/trajectory.json --output outputs/mission/plan_exec.json --png outputs/mission/bev.png

# Integrated surround pipeline perception with end-to-end AVP mission lifecycle execution:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --mission --output outputs/pipeline/mission_report.json --png outputs/pipeline/mission_bev.png
```

The output JSON report details the mission identifier, terminal state (`COMPLETED`, `ABORTED`), target slot ID, total elapsed duration, total execution steps, replan count, discrete state transition events with timestamps and triggers, and final terminal docking KPIs ($\Delta x, \Delta y, \Delta \theta$).

### Parking facility HD vector mapping, routing, and localization

Build structured high-definition vector facility maps for multi-aisle garages and surface parking lots, compute kinematically-feasible global topological routes with turn penalties and corridor boundaries, and perform multi-sensor EKF pose graph localization fusing kinematic odometry with landmark parking slot observations:

```powershell
# 1. Inspect facility map geometry, lane topology, and slot capacities:
uv run nearfield360 map info --map maps/garage.json --json

# 2. Synthesize benchmark parking facility maps (garage or surface lot) with BEV rendering:
uv run nearfield360 map build --type garage --aisles 3 --slots-per-aisle 8 --output outputs/map/garage.json --png outputs/map/garage_bev.png
uv run nearfield360 map build --type lot --aisles 4 --slots-per-aisle 10 --output outputs/map/lot.json --png outputs/map/lot_bev.png

# 3. Plan optimal global topological routes from facility entrance to a target slot:
uv run nearfield360 map route --map outputs/map/garage.json --target-slot SLOT_A1_02 --output outputs/map/route.json --png outputs/map/route_bev.png

# 4. Run closed-loop multi-sensor EKF localization fusing odometry and slot landmark updates:
uv run nearfield360 map localize --map outputs/map/garage.json --route outputs/map/route.json --steps 40 --output outputs/map/slam.json --png outputs/map/telemetry.png

# 5. Integrate facility vector mapping directly into the surround camera pipeline:
uv run nearfield360 pipeline run --root D:\datasets\woodscape --map outputs/map/garage.json --target-slot SLOT_A1_02 --output outputs/pipeline/map_pipeline.json
```

Key capabilities:
- **HD Vector Map Specification:** Pydantic v2 domain models for `FacilityMap`, `FacilityLane`, `FacilitySlot`, `FacilityObstacle`, `FacilityWaypoint`, and `GlobalRoute` with complete JSON schema validation and topological integrity verification.
- **Topological A\* Router:** Graph search enforcing lane one-way traversal, geometry turn penalties, smooth Bezier corridor waypoint densification, and slot access point snapping.
- **Multi-Sensor Pose Estimator (EKF):** Non-linear kinematic bicycle prediction fused with range/bearing observations of marked slot corners and entrance landmarks, complete with $3\sigma$ covariance uncertainty estimation and noise rejection.
- **Dual-Panel Telemetry Dashboard:** Bird's-eye-view vector facility rendering paired with real-time localization tracking error ($\Delta x, \Delta y, \Delta \theta$) and covariance determinant convergence plots.


### Inference backends and model optimization

Live perception commands (`infer`, `occupancy layer`, `track run`, `pipeline run`) accept
`--backend {opencv,onnxruntime,tensorrt}` and `--precision {fp32,fp16,int8}`. When omitted,
the backend, device, and precision come from `config.inference.backend`, `config.inference.device`,
and `config.inference.precision`. Detection confidence and NMS thresholds default from
`config.inference.confidence_threshold` and `config.inference.nms_threshold`.

Inspect hardware acceleration execution providers on the current host:

```powershell
uv run nearfield360 infer providers
uv run nearfield360 infer providers --json
```

Optimize an ONNX model with half-precision FP16 or dynamic INT8 quantization:

```powershell
# Convert to FP16 with numerical drift verification against FP32 baseline:
uv run nearfield360 infer optimize --model models/detection.onnx --output models/detection_fp16.onnx --precision fp16 --check-drift

# Quantize to dynamic INT8:
uv run nearfield360 infer optimize --model models/segmentation.onnx --output models/segmentation_int8.onnx --precision int8
```

Generate a TensorRT-compatible INT8 calibration cache table from fisheye samples:

```powershell
uv run nearfield360 infer calibrate --model models/detection.onnx --output outputs/calibration.cache --samples 50
```

ONNX Runtime is an optional extra — install with:

```powershell
uv sync --extra onnxruntime
# or: pip install "nearfield360[onnxruntime]"
```

If `onnxruntime` is not installed and `--backend onnxruntime` is requested, the CLI exits
with a clear install hint rather than silently falling back to OpenCV.

Verify release readiness (packaging metadata, Apache-2.0 license, version consistency,
default config, and CLI subcommand surface):

```powershell
uv run nearfield360 release audit
uv run nearfield360 release audit --json --output outputs/release/audit.json
```

The audit exits with status 1 if any check fails.

## License

NearField360 source code is licensed under the [Apache License 2.0](LICENSE). External datasets,
pretrained weights, and third-party components remain subject to their own licenses.
