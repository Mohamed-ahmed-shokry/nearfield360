from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from nearfield360.perception.inference.models import PrecisionType, QuantizationType
from nearfield360.perception.inference.optimization import (
    OptimizationError,
    convert_model_to_fp16,
    detect_hardware_providers,
    generate_int8_calibration_table,
    optimize_model_precision,
    quantize_model_dynamic_int8,
)
from nearfield360.perception.inference.test_utils import create_dummy_segmentation_onnx


@pytest.fixture
def dummy_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.onnx"
    return create_dummy_segmentation_onnx(model_path, num_classes=5, height=32, width=32)


def test_convert_model_to_fp16_keep_io(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "model_fp16.onnx"
    total_nodes, converted_nodes = convert_model_to_fp16(dummy_model, out_path, keep_io_types=True)

    assert out_path.is_file()
    assert total_nodes > 0
    assert converted_nodes > 0

    # Load and verify ONNX graph
    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)

    # Initializers should be FLOAT16 (10)
    for init in model.graph.initializer:
        assert init.data_type == onnx.TensorProto.FLOAT16

    # Model boundary IO should still be FLOAT (1) with internal Cast nodes
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_convert_model_to_fp16_raw_io(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "model_fp16_raw.onnx"
    total_nodes, converted_nodes = convert_model_to_fp16(dummy_model, out_path, keep_io_types=False)

    assert out_path.is_file()
    assert total_nodes > 0
    assert converted_nodes > 0

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_quantize_model_dynamic_int8(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "model_int8.onnx"
    total_nodes, quantized_nodes = quantize_model_dynamic_int8(dummy_model, out_path)

    assert out_path.is_file()
    assert total_nodes > 0
    assert quantized_nodes > 0

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)


def test_optimize_model_precision_fp16_with_drift(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "optimized_fp16.onnx"
    sample = np.random.default_rng(42).standard_normal((1, 3, 32, 32)).astype(np.float32)

    summary = optimize_model_precision(
        dummy_model,
        out_path,
        target_precision=PrecisionType.FP16,
        test_sample=sample,
    )

    assert summary.source_path == str(dummy_model)
    assert summary.optimized_path == str(out_path)
    assert summary.source_precision == "fp32"
    assert summary.target_precision == "fp16"
    assert summary.quantization_type == QuantizationType.FP16.value
    assert summary.source_size_bytes > 0
    assert summary.optimized_size_bytes > 0
    assert summary.compression_ratio > 0.0
    assert summary.node_count > 0
    assert summary.quantized_node_count > 0
    assert summary.max_absolute_drift is not None
    assert summary.max_absolute_drift < 0.05
    assert summary.mean_absolute_drift is not None


def test_optimize_model_precision_int8(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "optimized_int8.onnx"
    summary = optimize_model_precision(
        dummy_model,
        out_path,
        target_precision="int8",
    )
    assert summary.target_precision == "int8"
    assert summary.quantization_type == QuantizationType.DYNAMIC_INT8.value
    assert out_path.is_file()


def test_optimize_model_precision_errors(dummy_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.onnx"

    # Missing source file
    with pytest.raises(OptimizationError, match="Source model file not found"):
        optimize_model_precision(tmp_path / "nonexistent.onnx", out_path)

    # Destination exists and overwrite=False
    out_path.write_bytes(b"content")
    with pytest.raises(OptimizationError, match="already exists and overwrite is disabled"):
        optimize_model_precision(dummy_model, out_path, overwrite=False)

    # Unsupported precision
    with pytest.raises(OptimizationError, match="Unsupported target precision"):
        optimize_model_precision(dummy_model, out_path, target_precision="bfloat16")


def test_generate_int8_calibration_table(dummy_model: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "calibration.cache"
    samples = [
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        np.ones((1, 3, 32, 32), dtype=np.float32) * 0.5,
    ]

    summary = generate_int8_calibration_table(
        dummy_model,
        samples,
        cache_path,
        method="entropy",
    )

    assert cache_path.is_file()
    assert summary.model_path == str(dummy_model)
    assert summary.cache_path == str(cache_path)
    assert summary.num_samples == 2
    assert summary.calibration_method == "entropy"
    assert "output" in summary.tensor_ranges

    content = cache_path.read_text(encoding="utf-8")
    assert "TRT-8400-EntropyCalibration2" in content
    assert "output:" in content


def test_generate_int8_calibration_table_errors(dummy_model: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "calib.cache"
    with pytest.raises(OptimizationError, match="Model file not found"):
        generate_int8_calibration_table(tmp_path / "absent.onnx", [], cache_path)

    with pytest.raises(OptimizationError, match="At least one calibration sample"):
        generate_int8_calibration_table(dummy_model, [], cache_path)


def test_detect_hardware_providers() -> None:
    report = detect_hardware_providers()
    assert len(report.providers) >= 2
    # CPU should always be present and available
    cpu_p = next((p for p in report.providers if p.name == "CPUExecutionProvider"), None)
    assert cpu_p is not None
    assert cpu_p.available is True
    assert report.recommended_backend in ("tensorrt", "onnxruntime", "opencv")
    assert report.recommended_device in ("cuda", "cpu")
