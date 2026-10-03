"""Unit tests for temporal BEV occupancy forecasting ONNX exporter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from nearfield360.occupancy import export_temporal_forecaster_onnx


def test_export_temporal_forecaster_onnx(tmp_path: Path) -> None:
    output_path = tmp_path / "forecaster.onnx"
    result_path = export_temporal_forecaster_onnx(
        output_path,
        grid_shape=(32, 32),
        in_channels=4,
        hidden_channels=8,
        horizon_steps=4,
        opset_version=13,
    )

    assert result_path == output_path.resolve()
    assert result_path.is_file()

    model = onnx.load(str(result_path))
    onnx.checker.check_model(model)

    # Validate inputs
    inputs = {inp.name: inp for inp in model.graph.input}
    assert "bev_features" in inputs
    assert "hidden_state" in inputs

    # Validate outputs
    outputs = {out.name: out for out in model.graph.output}
    assert "updated_hidden_state" in outputs
    assert "forecast_occupancy" in outputs
    assert "velocity_field" in outputs


def test_exported_onnx_inference(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")

    model_path = tmp_path / "forecaster_eval.onnx"
    export_temporal_forecaster_onnx(
        model_path,
        grid_shape=(16, 16),
        in_channels=4,
        hidden_channels=8,
        horizon_steps=3,
        opset_version=14,
    )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    bev_input = np.random.uniform(0.0, 1.0, size=(1, 4, 16, 16)).astype(np.float32)
    hid_input = np.zeros((1, 8, 16, 16), dtype=np.float32)

    outputs = session.run(
        None,
        {
            "bev_features": bev_input,
            "hidden_state": hid_input,
        },
    )

    assert len(outputs) == 3
    hid_out, occ_out, vel_out = outputs

    assert hid_out.shape == (1, 8, 16, 16)
    assert occ_out.shape == (1, 3, 16, 16)
    assert vel_out.shape == (1, 2, 16, 16)

    # Occupancy predictions should be valid probabilities in [0, 1]
    assert np.all(occ_out >= 0.0)
    assert np.all(occ_out <= 1.0)
