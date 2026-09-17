from __future__ import annotations

import numpy as np
import pytest

from nearfield360.config import ProjectConfig
from nearfield360.data.calibration import IntrinsicParameters
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.bev import BevGrid
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.fisheye import RadialPolynomialFisheye
from nearfield360.geometry.transforms import RigidTransform
from nearfield360.occupancy.evidence import OccupancyEvidence
from nearfield360.occupancy.risk import RiskZone, ZoneRisk
from nearfield360.perception.evaluation import environment_metadata
from nearfield360.robustness.benchmark import (
    RobustnessReport,
    compare_occupancy_grids,
    compare_zone_risks,
    run_calibration_sweep,
    run_corruption_sweep,
)
from nearfield360.robustness.corruptions import CorruptionType


@pytest.fixture
def test_grid() -> BevGrid:
    return BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=0.5)


@pytest.fixture
def sample_evidence(test_grid: BevGrid) -> OccupancyEvidence:
    occupied = np.zeros(test_grid.shape, dtype=np.float64)
    free = np.zeros(test_grid.shape, dtype=np.float64)
    observed = np.zeros(test_grid.shape, dtype=np.int64)
    occupied[2:6, 2:6] = 5.0
    free[0:2, 0:2] = 5.0
    observed[2:6, 2:6] = 5
    observed[0:2, 0:2] = 5
    return OccupancyEvidence(grid=test_grid, occupied=occupied, free=free, observed=observed)


@pytest.fixture
def sample_camera() -> CalibratedCamera:
    intrinsic = IntrinsicParameters(
        aspect_ratio=1.0,
        cx_offset=0.0,
        cy_offset=0.0,
        height=480,
        k1=200.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        model="radial_poly",
        poly_order=4,
        width=640,
    )
    rot = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    transform = RigidTransform(rot, (2.0, 0.0, 0.7))
    return CalibratedCamera(
        name=CameraId.FRONT,
        model=RadialPolynomialFisheye(intrinsic, theta_max=2.2),
        camera_to_vehicle=transform,
    )


def test_compare_occupancy_grids_identical(sample_evidence: OccupancyEvidence) -> None:
    comp = compare_occupancy_grids(sample_evidence, sample_evidence)
    assert comp.mae == 0.0
    assert comp.occupied_iou == 1.0
    assert comp.free_iou == 1.0
    assert comp.uncertainty_shift == 0.0
    assert comp.confident_cells_clean == comp.confident_cells_perturbed

    d = comp.as_dict()
    assert d["mae"] == 0.0
    assert d["occupied_iou"] == 1.0


def test_compare_occupancy_grids_difference(
    test_grid: BevGrid, sample_evidence: OccupancyEvidence
) -> None:
    occupied = np.zeros(test_grid.shape, dtype=np.float64)
    free = np.zeros(test_grid.shape, dtype=np.float64)
    observed = np.zeros(test_grid.shape, dtype=np.int64)
    occupied[2:6, 2:6] = 1.0
    free[2:6, 2:6] = 4.0
    observed[2:6, 2:6] = 5
    perturbed = OccupancyEvidence(grid=test_grid, occupied=occupied, free=free, observed=observed)

    comp = compare_occupancy_grids(sample_evidence, perturbed)
    assert comp.mae > 0.0
    assert comp.occupied_iou < 1.0


def test_compare_zone_risks() -> None:
    clean_zr = [
        ZoneRisk(
            name="corridor",
            cells=10,
            observed_cells=8,
            occupied_cells=3,
            area_m2=2.5,
            observed_area_m2=2.0,
            occupied_area_m2=0.75,
            mean_occupancy=0.6,
            max_occupancy=0.9,
            mean_uncertainty=0.04,
            max_uncertainty=0.08,
        )
    ]
    pert_zr = [
        ZoneRisk(
            name="corridor",
            cells=10,
            observed_cells=8,
            occupied_cells=5,
            area_m2=2.5,
            observed_area_m2=2.0,
            occupied_area_m2=1.25,
            mean_occupancy=0.7,
            max_occupancy=0.95,
            mean_uncertainty=0.06,
            max_uncertainty=0.10,
        )
    ]

    res = compare_zone_risks(clean_zr, pert_zr)
    assert "corridor" in res
    comp = res["corridor"]
    assert comp.clean_occupied == 3
    assert comp.perturbed_occupied == 5
    assert comp.occupied_diff == 2
    assert comp.clean_uncertainty == 0.04
    assert comp.perturbed_uncertainty == 0.06

    d = comp.as_dict()
    assert d["occupied_diff"] == 2


def test_run_corruption_sweep(test_grid: BevGrid, sample_evidence: OccupancyEvidence) -> None:
    zone = RiskZone(name="test_zone", mask=np.ones(test_grid.shape, dtype=bool))
    config = ProjectConfig()

    records = run_corruption_sweep(
        sample_evidence,
        grid=test_grid,
        zones=[zone],
        config=config,
        severities=[1, 2, 3],
        corruptions=[CorruptionType.LENS_SOILING, CorruptionType.FOG],
    )

    assert len(records) == 2 * 3
    for rec in records:
        assert "corruption" in rec
        assert "severity" in rec
        assert "occupancy_metrics" in rec
        assert "zone_metrics" in rec
        assert "test_zone" in rec["zone_metrics"]


def test_run_calibration_sweep(
    test_grid: BevGrid, sample_evidence: OccupancyEvidence, sample_camera: CalibratedCamera
) -> None:
    zone = RiskZone(name="test_zone", mask=np.ones(test_grid.shape, dtype=bool))
    config = ProjectConfig()
    pixels = np.array([[320.0, 240.0], [300.0, 200.0]])
    classes = np.array([10, 1], dtype=np.int64)  # 10: vehicles, 1: road

    records = run_calibration_sweep(
        sample_evidence,
        sample_camera,
        pixels,
        classes,
        grid=test_grid,
        zones=[zone],
        config=config,
        rotation_perturbations_deg=[1.0, 2.0],
        translation_perturbations_m=[0.05],
    )

    # 3 rotation axes * 2 angles + 2 translation axes * 1 dist = 6 + 2 = 8 records
    assert len(records) == 8
    for rec in records:
        assert "axis" in rec
        assert "unit" in rec
        assert "magnitude" in rec
        assert "occupancy_metrics" in rec
        assert "zone_metrics" in rec


def test_robustness_report_serialization() -> None:
    rep = RobustnessReport(
        environment=environment_metadata(),
        config={"runtime": {"seed": 42}},
        corruption_sweeps=[{"corruption": "fog", "severity": 1}],
        calibration_sweeps=[{"axis": "pitch", "magnitude": 1.0}],
    )
    d = rep.as_dict()
    assert d["environment"]["python_version"] is not None
    assert len(d["corruption_sweeps"]) == 1
    assert len(d["calibration_sweeps"]) == 1
