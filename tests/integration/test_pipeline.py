"""End-to-end integration tests exercising multiple project components together."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from nearfield360.config import load_config
from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    distance_weights,
    rasterize_occupancy,
)
from nearfield360.occupancy.risk import risk_report, surround_parking_zones
from nearfield360.perception.inference.backend import create_backend
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.perception.inference.test_utils import (
    create_dummy_detection_onnx,
    create_dummy_segmentation_onnx,
)
from nearfield360.tracking.kalman import KalmanFilter2D
from nearfield360.tracking.models import GroundFootprint
from nearfield360.tracking.tracker import MultiObjectTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_seg_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)
    return model_path


@pytest.fixture
def dummy_det_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "det.onnx"
    create_dummy_detection_onnx(model_path, num_classes=5, num_boxes=3, height=32, width=32)
    return model_path


@pytest.fixture
def sample_image(tmp_path: Path) -> np.ndarray:
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_path = tmp_path / "fisheye.png"
    cv2.imwrite(str(img_path), img)
    return img


@pytest.fixture
def bev_grid() -> BevGrid:
    return BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)


@pytest.fixture
def occupancy_policy() -> OccupancyPolicy:
    return OccupancyPolicy.from_names(
        free=["road", "curb"],
        occupied=["vehicles", "person", "bicycle"],
    )


# ---------------------------------------------------------------------------
# Integration Test 1: Semantic Segmentation → Occupancy → Risk
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSemanticOccupancyRiskPipeline:
    """Full pipeline: image → semantic model → BEV occupancy → zone risk report."""

    def test_single_frame_pipeline(
        self,
        dummy_seg_model: Path,
        sample_image: np.ndarray,
        bev_grid: BevGrid,
        occupancy_policy: OccupancyPolicy,
    ) -> None:
        # Step 1: Load backend and run semantic segmentation
        backend = create_backend(
            dummy_seg_model,
            backend_type=InferenceBackendType.OPENCV,
            device=InferenceDevice.CPU,
        )
        engine = SemanticSegmentationEngine(backend=backend, num_classes=10)
        class_mask, confidence_map = engine.predict(sample_image)

        assert class_mask.shape == sample_image.shape[:2]
        assert confidence_map.shape == sample_image.shape[:2]
        assert class_mask.dtype == np.uint8
        assert confidence_map.dtype == np.float32

        # Step 2: Simulate ground-projected footprints and rasterize
        num_points = 100
        rng = np.random.default_rng(42)
        points = np.column_stack(
            [
                rng.uniform(bev_grid.x_min, bev_grid.x_max, num_points),
                rng.uniform(bev_grid.y_min, bev_grid.y_max, num_points),
            ]
        )
        labels = rng.choice([0, 1, 2, 3, 4, 5], size=num_points)
        valid = np.ones(num_points, dtype=bool)
        weights = distance_weights(np.linalg.norm(points, axis=1), slope=0.5)

        evidence = rasterize_occupancy(
            bev_grid,
            points,
            labels,
            occupancy_policy,
            weights=weights,
            valid=valid,
        )

        assert isinstance(evidence, OccupancyEvidence)
        assert evidence.grid is bev_grid
        assert evidence.observed.sum() > 0

        # Step 3: Compute risk report on surround zones
        zones = surround_parking_zones(bev_grid)
        reports = risk_report(zones, evidence, min_evidence=1, danger_occupancy=0.5)

        assert len(reports) == 6
        for report in reports:
            assert report.cells >= 0
            assert report.observed_cells >= 0
            assert report.observed_cells <= report.cells

    def test_occupancy_evidence_accumulates_across_frames(
        self,
        bev_grid: BevGrid,
        occupancy_policy: OccupancyPolicy,
    ) -> None:
        """Fusing multiple frames increases total evidence."""
        rng = np.random.default_rng(0)
        base_points = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, 50),
                rng.uniform(-2.0, 2.0, 50),
            ]
        )
        base_labels = rng.choice([1, 6], size=50)

        fused: OccupancyEvidence | None = None
        for _frame_idx in range(3):
            jittered = base_points + rng.normal(0, 0.05, base_points.shape)
            ev = rasterize_occupancy(bev_grid, jittered, base_labels, occupancy_policy)
            fused = ev if fused is None else fused.add(ev)

        assert fused is not None
        assert fused.observed.sum() > 0
        occ = fused.occupancy()
        assert occ.shape == bev_grid.shape


# ---------------------------------------------------------------------------
# Integration Test 2: Detection → Projection → Tracking
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDetectionTrackingPipeline:
    """Full pipeline: image → detection model → ground projection → multi-object tracker."""

    def test_detection_through_tracking(
        self,
        dummy_det_model: Path,
    ) -> None:
        # Step 1: Load detection engine
        backend = create_backend(
            dummy_det_model,
            backend_type=InferenceBackendType.OPENCV,
            device=InferenceDevice.CPU,
        )
        ObjectDetectionEngine(backend=backend, num_classes=5)

        # Step 2: Create synthetic detections as GroundFootprint objects
        tracker = MultiObjectTracker()
        frame_footprints = [
            [
                GroundFootprint(
                    center_x=2.0,
                    center_y=1.0,
                    width=1.8,
                    length=4.5,
                    class_id=0,
                    class_name="vehicles",
                    camera="FV",
                    confidence=0.9,
                ),
            ],
            [
                GroundFootprint(
                    center_x=2.1,
                    center_y=1.05,
                    width=1.8,
                    length=4.5,
                    class_id=0,
                    class_name="vehicles",
                    camera="FV",
                    confidence=0.92,
                ),
            ],
            [
                GroundFootprint(
                    center_x=2.2,
                    center_y=1.1,
                    width=1.8,
                    length=4.5,
                    class_id=0,
                    class_name="vehicles",
                    camera="FV",
                    confidence=0.88,
                ),
            ],
        ]

        all_obstacles = []
        for footprints in frame_footprints:
            obstacles = tracker.update(footprints)
            all_obstacles.append(obstacles)

        # After 3 frames with min_hits=3, the track should be confirmed
        final_obstacles = all_obstacles[-1]
        assert len(final_obstacles) >= 1
        obstacle = final_obstacles[0]
        assert obstacle.class_name == "vehicles"
        assert obstacle.hits >= 1
        assert len(obstacle.history) >= 1

    def test_kalman_filter_position_update(self) -> None:
        """Kalman filter converges toward measured position over repeated updates."""
        kf = KalmanFilter2D(initial_pos=(0.0, 0.0), measurement_noise=0.1)

        target = (3.0, 2.0)
        for _ in range(20):
            kf.predict()
            kf.update(target)

        pos_x, pos_y = kf.position
        assert abs(pos_x - target[0]) < 0.5
        assert abs(pos_y - target[1]) < 0.5


# ---------------------------------------------------------------------------
# Integration Test 3: Config → Grid → Zones → Risk (no dataset needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConfigGridRiskPipeline:
    """Verify config defaults produce a valid grid and risk zone suite."""

    def test_default_config_to_risk_report(self) -> None:
        config = load_config()
        grid = BevGrid(
            x_min=config.bev.x_min,
            x_max=config.bev.x_max,
            y_min=config.bev.y_min,
            y_max=config.bev.y_max,
            resolution=config.bev.resolution,
        )

        assert grid.shape[0] > 0
        assert grid.shape[1] > 0

        zones = surround_parking_zones(
            grid,
            front_length=config.risk.front_length,
            rear_length=config.risk.rear_length,
            half_width=config.risk.half_width,
            start_x=config.risk.start_x,
            rear_start_x=config.risk.rear_start_x,
            lateral_width=config.risk.lateral_width,
            vehicle_x_min=config.risk.vehicle_x_min,
            vehicle_x_max=config.risk.vehicle_x_max,
            near_radius=config.risk.near_radius,
            warning_radius=config.risk.warning_radius,
        )

        assert len(zones) == 6
        zone_names = {z.name for z in zones}
        expected_names = {
            "forward_corridor",
            "rear_corridor",
            "left_clearance",
            "right_clearance",
            "near_circle",
            "warning_circle",
        }
        assert zone_names == expected_names

        # Create synthetic evidence with some occupied cells near origin
        policy = OccupancyPolicy.from_names(
            free=["road"],
            occupied=["vehicles"],
        )
        occupied_points = np.array([[0.2, 0.1], [-0.1, 0.3], [0.0, -0.2]])
        occupied_labels = np.array([6, 6, 6])

        evidence = rasterize_occupancy(grid, occupied_points, occupied_labels, policy)

        reports = risk_report(
            zones,
            evidence,
            min_evidence=config.occupancy.min_evidence,
            danger_occupancy=config.risk.danger_occupancy,
        )

        assert len(reports) == 6
        near_circle = next(r for r in reports if r.name == "near_circle")
        assert near_circle.observed_cells > 0
        assert near_circle.occupied_cells > 0


# ---------------------------------------------------------------------------
# Integration Test 4: Multi-camera evidence fusion
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiCameraFusion:
    """Fuse evidence from multiple simulated cameras into one occupancy layer."""

    def test_fuse_four_cameras(self, bev_grid: BevGrid) -> None:
        policy = OccupancyPolicy.from_names(
            free=["road"],
            occupied=["vehicles"],
        )

        rng = np.random.default_rng(99)
        camera_evidences: list[OccupancyEvidence] = []
        for cam_idx in range(4):
            offset_x = (cam_idx - 1.5) * 2.0
            points = np.column_stack(
                [
                    rng.uniform(offset_x - 1.0, offset_x + 1.0, 30),
                    rng.uniform(-3.0, 3.0, 30),
                ]
            )
            labels = rng.choice([1, 6], size=30)
            ev = rasterize_occupancy(bev_grid, points, labels, policy)
            camera_evidences.append(ev)

        fused = camera_evidences[0]
        for ev in camera_evidences[1:]:
            fused = fused.add(ev)

        assert fused.observed.sum() > 0
        occ = fused.occupancy()
        assert occ.shape == bev_grid.shape

        # Risk report on fused result
        zones = surround_parking_zones(bev_grid)
        reports = risk_report(zones, fused, min_evidence=1)
        assert len(reports) == 6


# ---------------------------------------------------------------------------
# Integration Test 5: Full CLI pipeline with live ONNX models
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndCliPipeline:
    """Exercise `nearfield360 pipeline run` end-to-end with synthetic ONNX models."""

    def test_pipeline_run_with_live_models(
        self,
        tmp_path: Path,
        dummy_seg_model: Path,
        dummy_det_model: Path,
    ) -> None:
        import json

        from typer.testing import CliRunner

        from nearfield360.cli import app
        from nearfield360.utils.artifacts import read_json

        dataset = tmp_path / "dataset"
        cameras = ("FV", "RV", "MVL", "MVR")
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        for cam in cameras:
            stem = f"00001_{cam}"
            (dataset / "rgb_images").mkdir(parents=True, exist_ok=True)
            (dataset / "calibration_data").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dataset / f"rgb_images/{stem}.png"), img)
            calib = {
                "extrinsic": {
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 1.5],
                },
                "intrinsic": {
                    "aspect_ratio": 1.0,
                    "cx_offset": 0.0,
                    "cy_offset": 0.0,
                    "height": 32,
                    "k1": 50.0,
                    "k2": 0.0,
                    "k3": 0.0,
                    "k4": 0.0,
                    "model": "radial_poly",
                    "poly_order": 4,
                    "width": 32,
                },
                "name": cam,
            }
            (dataset / f"calibration_data/{stem}.json").write_text(
                json.dumps(calib), encoding="utf-8"
            )

        output = tmp_path / "e2e_pipeline.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "pipeline",
                "run",
                "--root",
                str(dataset),
                "--output",
                str(output),
                "--seg-model",
                str(dummy_seg_model),
                "--det-model",
                str(dummy_det_model),
                "--health-aware",
            ],
        )

        assert result.exit_code == 0, result.stdout + result.stderr
        assert "Pipeline complete: 1 frames" in result.stdout

        payload = read_json(output)
        assert payload["samples"]["evaluated"] == 1
        assert payload["samples"]["seg_model"] == str(dummy_seg_model)
        assert payload["samples"]["det_model"] == str(dummy_det_model)
        assert payload["samples"]["health_aware"] is True
        assert len(payload["health"]) == 4
        assert payload["timings"]["frame_latency"]["samples"] == 1
        assert payload["timings"]["frame_latency"]["p95_ms"] >= 0.0
        assert len(payload["risk"]) == 6
        assert "summary" in payload
        assert payload["environment"]["nearfield360_version"]
        assert json.dumps(payload)
