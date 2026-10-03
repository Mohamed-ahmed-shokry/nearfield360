from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.perception.inference.models import (
    BenchmarkSummary,
    CalibrationSummary,
    InferenceBackendType,
    InferenceDevice,
    InferenceResult,
    ModelMetadata,
    OptimizationSummary,
    PrecisionType,
    ProviderInfo,
    ProvidersReport,
    QuantizationType,
)


def test_inference_enums() -> None:
    assert InferenceDevice.CPU.value == "cpu"
    assert InferenceDevice.CUDA.value == "cuda"
    assert InferenceDevice.DIRECTML.value == "directml"

    assert InferenceBackendType.OPENCV.value == "opencv"
    assert InferenceBackendType.ONNXRUNTIME.value == "onnxruntime"
    assert InferenceBackendType.TENSORRT.value == "tensorrt"

    assert PrecisionType.FP32.value == "fp32"
    assert PrecisionType.FP16.value == "fp16"
    assert PrecisionType.INT8.value == "int8"

    assert QuantizationType.FP16.value == "fp16"
    assert QuantizationType.DYNAMIC_INT8.value == "dynamic_int8"
    assert QuantizationType.STATIC_INT8.value == "static_int8"


def test_model_metadata_instantiation() -> None:
    meta = ModelMetadata(
        model_path="models/seg.onnx",
        backend="opencv",
        device="cpu",
        input_names=("input",),
        input_shapes=((1, 3, 480, 640),),
        output_names=("output",),
        output_shapes=((1, 10, 480, 640),),
    )
    assert meta.model_path == "models/seg.onnx"
    assert meta.input_names == ("input",)
    assert meta.output_shapes == ((1, 10, 480, 640),)

    dumped = meta.model_dump()
    assert dumped["backend"] == "opencv"


def test_benchmark_summary_validation() -> None:
    summary = BenchmarkSummary(
        iterations=50,
        warmup=10,
        backend="opencv",
        device="cpu",
        input_shape=(1, 3, 480, 640),
        mean_latency_ms=12.5,
        median_latency_ms=12.3,
        p90_latency_ms=13.0,
        p95_latency_ms=13.5,
        p99_latency_ms=14.2,
        min_latency_ms=11.8,
        max_latency_ms=15.0,
        fps=80.0,
    )
    assert summary.iterations == 50
    assert summary.fps == 80.0

    with pytest.raises(ValidationError):
        BenchmarkSummary(
            iterations=0,  # invalid
            warmup=10,
            backend="opencv",
            device="cpu",
            input_shape=(1, 3, 480, 640),
            mean_latency_ms=12.5,
            median_latency_ms=12.3,
            p90_latency_ms=13.0,
            p95_latency_ms=13.5,
            p99_latency_ms=14.2,
            min_latency_ms=11.8,
            max_latency_ms=15.0,
            fps=80.0,
        )


def test_inference_result() -> None:
    res = InferenceResult(latency_ms=15.2, input_shape=(1, 3, 480, 640))
    assert res.latency_ms == 15.2
    assert res.input_shape == (1, 3, 480, 640)


def test_optimization_summary() -> None:
    opt = OptimizationSummary(
        source_path="models/model.onnx",
        optimized_path="models/model_fp16.onnx",
        source_precision="fp32",
        target_precision="fp16",
        quantization_type="fp16",
        source_size_bytes=2048000,
        optimized_size_bytes=1024000,
        compression_ratio=2.0,
        node_count=45,
        quantized_node_count=40,
        max_absolute_drift=0.0012,
        mean_absolute_drift=0.00015,
    )
    assert opt.compression_ratio == 2.0
    assert opt.target_precision == "fp16"
    assert opt.max_absolute_drift == 0.0012


def test_calibration_summary() -> None:
    calib = CalibrationSummary(
        model_path="models/model.onnx",
        cache_path="models/calibration.cache",
        num_samples=50,
        tensor_ranges={"input": (-2.1, 2.6), "output": (0.0, 1.0)},
        calibration_method="entropy",
    )
    assert calib.num_samples == 50
    assert "input" in calib.tensor_ranges


def test_providers_report() -> None:
    p1 = ProviderInfo(
        name="TensorrtExecutionProvider",
        available=False,
        priority=1,
        device="cuda",
        details="Requires TensorRT library",
    )
    p2 = ProviderInfo(
        name="CPUExecutionProvider",
        available=True,
        priority=3,
        device="cpu",
        details="Default CPU provider",
    )
    report = ProvidersReport(
        providers=(p1, p2),
        recommended_backend="onnxruntime",
        recommended_device="cpu",
        cuda_available=False,
        tensorrt_available=False,
    )
    assert len(report.providers) == 2
    assert report.recommended_backend == "onnxruntime"
    assert not report.tensorrt_available
