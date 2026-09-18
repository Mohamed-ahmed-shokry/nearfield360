from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from nearfield360.robustness.plots import (
    generate_robustness_html_dashboard,
    render_raster_line_chart,
    render_svg_line_chart,
)


@pytest.fixture
def sample_series() -> dict[str, list[tuple[float, float]]]:
    return {
        "soiling": [(1.0, 0.05), (2.0, 0.12), (3.0, 0.22), (4.0, 0.35), (5.0, 0.52)],
        "fog": [(1.0, 0.03), (2.0, 0.08), (3.0, 0.15), (4.0, 0.26), (5.0, 0.41)],
    }


def test_render_svg_line_chart(sample_series: dict[str, list[tuple[float, float]]]) -> None:
    svg = render_svg_line_chart(
        title="Test Metric vs. Severity",
        x_label="Severity",
        y_label="MAE",
        series_dict=sample_series,
        width=800,
        height=500,
    )

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "Test Metric vs. Severity" in svg
    assert "Severity" in svg
    assert "MAE" in svg
    assert "soiling" in svg
    assert "fog" in svg
    assert "<path" in svg
    assert "<circle" in svg


def test_render_svg_line_chart_handles_edge_cases() -> None:
    # Empty series
    svg_empty = render_svg_line_chart("Empty", "X", "Y", {})
    assert "<svg" in svg_empty

    # Single point
    svg_single = render_svg_line_chart("Single", "X", "Y", {"series": [(1.0, 1.0)]})
    assert "<circle" in svg_single

    # Non-finite values
    svg_nan = render_svg_line_chart(
        "NaNs", "X", "Y", {"series": [(1.0, float("nan")), (2.0, 0.5), (3.0, float("inf"))]}
    )
    assert "<circle" in svg_nan


def test_render_raster_line_chart(
    sample_series: dict[str, list[tuple[float, float]]], tmp_path: Path
) -> None:
    output_png = tmp_path / "chart.png"
    render_raster_line_chart(
        title="Raster Chart",
        x_label="Severity",
        y_label="Error",
        series_dict=sample_series,
        output_path=output_png,
        width=640,
        height=400,
    )

    assert output_png.is_file()
    img = cv2.imread(str(output_png))
    assert img is not None
    assert img.shape == (400, 640, 3)


def test_generate_robustness_html_dashboard() -> None:
    report = {
        "environment": {
            "nearfield360_version": "0.1.0",
            "python_version": "3.12.1",
            "platform": "TestPlatform",
        },
        "corruption_sweeps": [
            {
                "corruption": "lens_soiling",
                "severity": 1,
                "occupancy_metrics": {
                    "mae": 0.05,
                    "occupied_iou": 0.92,
                    "free_iou": 0.95,
                    "uncertainty_shift": 0.01,
                },
            },
            {
                "corruption": "lens_soiling",
                "severity": 5,
                "occupancy_metrics": {
                    "mae": 0.35,
                    "occupied_iou": 0.55,
                    "free_iou": 0.68,
                    "uncertainty_shift": 0.08,
                },
            },
        ],
        "calibration_sweeps": [
            {
                "axis": "pitch",
                "unit": "degrees",
                "magnitude": 1.0,
                "occupancy_metrics": {
                    "mae": 0.04,
                    "occupied_iou": 0.88,
                    "free_iou": 0.91,
                    "uncertainty_shift": 0.005,
                },
            }
        ],
    }

    html = generate_robustness_html_dashboard(report)

    assert "<!DOCTYPE html>" in html
    assert "NearField360 Robustness &amp; Sensitivity Report" in html
    assert "TestPlatform" in html
    assert "<svg" in html
    assert "lens_soiling" in html
    assert "pitch" in html
    assert "0.350 MAE" in html  # Max corruption error KPI
