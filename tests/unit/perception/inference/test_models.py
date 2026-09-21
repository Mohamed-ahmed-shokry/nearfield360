from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.perception.inference.models import (
    BenchmarkSummary,
    InferenceBackendType,
    InferenceDevice,
    InferenceResult,
    ModelMetadata,
)


def test_inference_enums() -> None:
    assert InferenceDevice.CPU.value == "cpu"
    assert InferenceDevice.CUDA.value == "cuda"
    assert InferenceDevice.DIRECTML.value == "directml"

    assert InferenceBackendType.OPENCV.value == "opencv"
    assert InferenceBackendType.ONNXRUNTIME.value == "onnxruntime"


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
