"""Inference runtime, preprocessing, and model execution engines."""

from __future__ import annotations

from nearfield360.perception.inference.backend import (
    InferenceBackend,
    InferenceError,
    OnnxRuntimeBackend,
    OpenCVDNNBackend,
    create_backend,
)
from nearfield360.perception.inference.benchmark import (
    ParityComparison,
    benchmark_inference,
    compare_numerical_parity,
    verify_numerical_parity,
)
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.models import (
    BenchmarkSummary,
    InferenceBackendType,
    InferenceDevice,
    InferenceResult,
    ModelMetadata,
)
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
    PreprocessorError,
    PreprocessTransform,
)
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine

__all__ = [
    "BenchmarkSummary",
    "FisheyeImagePreprocessor",
    "InferenceBackend",
    "InferenceBackendType",
    "InferenceDevice",
    "InferenceError",
    "InferenceResult",
    "ModelMetadata",
    "ObjectDetectionEngine",
    "OnnxRuntimeBackend",
    "OpenCVDNNBackend",
    "ParityComparison",
    "PreprocessTransform",
    "PreprocessorError",
    "SemanticSegmentationEngine",
    "benchmark_inference",
    "compare_numerical_parity",
    "create_backend",
    "verify_numerical_parity",
]
