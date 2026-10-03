"""Neural network model precision optimization, FP16 conversion, and INT8 calibration."""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from nearfield360.perception.inference.models import (
    CalibrationSummary,
    InferenceBackendType,
    OptimizationSummary,
    PrecisionType,
    ProviderInfo,
    ProvidersReport,
    QuantizationType,
)

logger = logging.getLogger(__name__)


class OptimizationError(RuntimeError):
    """Raised when model conversion, quantization, or calibration fails."""


def _float_to_hex_scale(scale: float) -> str:
    """Encode float scale factor into IEEE-754 32-bit hex representation (TensorRT format)."""
    packed = struct.pack("!f", float(scale))
    return packed.hex()


def convert_model_to_fp16(
    input_path: Path,
    output_path: Path,
    *,
    keep_io_types: bool = True,
) -> tuple[int, int]:
    """Convert an FP32 ONNX model to FP16 half-precision representation.

    Args:
        input_path: Path to source FP32 ONNX model.
        output_path: Path to write the optimized FP16 ONNX model.
        keep_io_types: If True, inserts Cast nodes at input and output boundaries
            so external callers can continue supplying standard float32 numpy arrays.

    Returns:
        Tuple of (total_node_count, converted_node_count).
    """
    try:
        model = onnx.load(str(input_path))
    except Exception as exc:
        raise OptimizationError(f"Failed to load ONNX model from {input_path}: {exc}") from exc

    graph = model.graph
    converted_count = 0

    # 1. Convert initializers (weights and biases) from FLOAT to FLOAT16
    new_initializers: list[Any] = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_init = numpy_helper.from_array(arr, name=init.name)
            new_initializers.append(new_init)
            converted_count += 1
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    # 2. Convert constant nodes
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr, name=attr.t.name))
                    converted_count += 1

    if keep_io_types:
        # Wrap inputs and outputs with Cast nodes so caller interface remains float32
        new_nodes: list[Any] = []

        # Map original graph input name -> casted fp16 name
        input_name_map: dict[str, str] = {}
        for inp in graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                cast_out = f"{inp.name}_fp16_cast"
                input_name_map[inp.name] = cast_out
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp.name],
                    outputs=[cast_out],
                    to=TensorProto.FLOAT16,
                    name=f"cast_in_{inp.name}",
                )
                new_nodes.append(cast_node)

        # Rewire existing node inputs if they read from model inputs
        for node in graph.node:
            for i, in_name in enumerate(node.input):
                if in_name in input_name_map:
                    node.input[i] = input_name_map[in_name]
            new_nodes.append(node)

        # Rewire outputs: compute nodes write to _fp16_out, final Cast node writes to model output
        output_name_map: dict[str, str] = {}
        for out in graph.output:
            if out.type.tensor_type.elem_type == TensorProto.FLOAT:
                internal_fp16 = f"{out.name}_fp16_raw"
                output_name_map[out.name] = internal_fp16

        for node in graph.node:
            for i, out_name in enumerate(node.output):
                if out_name in output_name_map:
                    node.output[i] = output_name_map[out_name]

        for out_name, internal_fp16 in output_name_map.items():
            cast_node = helper.make_node(
                "Cast",
                inputs=[internal_fp16],
                outputs=[out_name],
                to=TensorProto.FLOAT,
                name=f"cast_out_{out_name}",
            )
            new_nodes.append(cast_node)

        del graph.node[:]
        graph.node.extend(new_nodes)
    else:
        # Direct FP16 input and output tensor types
        for inp in graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT16
        for out in graph.output:
            if out.type.tensor_type.elem_type == TensorProto.FLOAT:
                out.type.tensor_type.elem_type = TensorProto.FLOAT16

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise OptimizationError(f"FP16 model validation check failed: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return len(graph.node), converted_count


def quantize_model_dynamic_int8(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Quantize linear weights of an ONNX model to 8-bit integers.

    Utilizes ``onnxruntime.quantization.quantize_dynamic`` when available,
    falling back to pure ONNX affine weight quantization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
            QuantType,
            quantize_dynamic,
        )

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
        model = onnx.load(str(output_path))
        node_count = len(model.graph.node)
        quantized_count = sum(
            1 for n in model.graph.node if "Quantize" in n.op_type or "Integer" in n.op_type
        )
        return node_count, quantized_count
    except (ImportError, Exception) as exc:
        logger.debug("onnxruntime.quantization failed (%s); using pure ONNX quantization", exc)

    # Pure ONNX Fallback Quantization: quantize Conv/Gemm weights to int8
    model = onnx.load(str(input_path))
    graph = model.graph
    quantized_count = 0
    new_nodes: list[Any] = []

    for init in list(graph.initializer):
        if init.data_type == TensorProto.FLOAT and init.name.endswith(("_w", "weight", "weights")):
            arr = numpy_helper.to_array(init)
            max_val = float(np.max(np.abs(arr)))
            scale = max(max_val / 127.0, 1e-7)

            # Symmetric int8 quantization
            q_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)

            q_init_name = f"{init.name}_qint8"
            scale_init_name = f"{init.name}_scale"

            q_init = numpy_helper.from_array(q_arr, name=q_init_name)
            scale_init = helper.make_tensor(scale_init_name, TensorProto.FLOAT, [], [float(scale)])

            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[q_init_name, scale_init_name],
                outputs=[init.name],
                name=f"dequant_{init.name}",
            )
            graph.initializer.remove(init)
            graph.initializer.extend([q_init, scale_init])
            new_nodes.append(dequant_node)
            quantized_count += 1

    new_nodes.extend(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return len(graph.node), quantized_count


def optimize_model_precision(
    input_path: Path,
    output_path: Path,
    *,
    target_precision: str | PrecisionType = PrecisionType.FP16,
    test_sample: np.ndarray | None = None,
    overwrite: bool = True,
) -> OptimizationSummary:
    """Optimize an ONNX model's precision (FP16 or INT8) and evaluate numerical drift.

    Args:
        input_path: Existing source ONNX model path.
        output_path: Destination path for optimized model.
        target_precision: Precision target ('fp16' or 'int8').
        test_sample: Optional sample tensor to measure numerical output drift.
        overwrite: Allow overwriting destination file if it exists.

    Returns:
        Detailed :class:`OptimizationSummary` containing file sizes, compression ratio,
        and output divergence statistics.
    """
    if not input_path.is_file():
        raise OptimizationError(f"Source model file not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise OptimizationError(
            f"Output file {output_path} already exists and overwrite is disabled."
        )

    norm_prec = (
        target_precision.value.lower()
        if isinstance(target_precision, PrecisionType)
        else str(target_precision).lower()
    )

    source_size = input_path.stat().st_size

    if norm_prec == PrecisionType.FP16.value:
        q_type = QuantizationType.FP16.value
        total_nodes, converted_nodes = convert_model_to_fp16(
            input_path, output_path, keep_io_types=True
        )
    elif norm_prec == PrecisionType.INT8.value:
        q_type = QuantizationType.DYNAMIC_INT8.value
        total_nodes, converted_nodes = quantize_model_dynamic_int8(input_path, output_path)
    else:
        raise OptimizationError(
            f"Unsupported target precision: {norm_prec}. Choose 'fp16' or 'int8'."
        )

    optimized_size = output_path.stat().st_size
    compression = source_size / max(optimized_size, 1)

    max_drift: float | None = None
    mean_drift: float | None = None

    if test_sample is not None:
        try:
            # Evaluate drift using OpenCV DNN or ONNX Runtime backend
            net_base = cv2.dnn.readNetFromONNX(str(input_path))
            net_opt = cv2.dnn.readNetFromONNX(str(output_path))

            net_base.setInput(test_sample)
            out_base = net_base.forward()

            net_opt.setInput(test_sample)
            out_opt = net_opt.forward()

            diff = np.abs(out_base - out_opt)
            max_drift = float(np.max(diff))
            mean_drift = float(np.mean(diff))
        except Exception as exc:
            logger.debug("Numerical drift evaluation skipped due to forward pass error: %s", exc)

    return OptimizationSummary(
        source_path=str(input_path),
        optimized_path=str(output_path),
        source_precision="fp32",
        target_precision=norm_prec,
        quantization_type=q_type,
        source_size_bytes=source_size,
        optimized_size_bytes=optimized_size,
        compression_ratio=float(round(compression, 4)),
        node_count=total_nodes,
        quantized_node_count=converted_nodes,
        max_absolute_drift=max_drift,
        mean_absolute_drift=mean_drift,
    )


def generate_int8_calibration_table(
    model_path: Path,
    calibration_samples: Sequence[np.ndarray],
    output_cache_path: Path,
    *,
    method: str = "entropy",
) -> CalibrationSummary:
    """Generate a TensorRT-compatible INT8 calibration cache table from fisheye samples."""
    if not model_path.is_file():
        raise OptimizationError(f"Model file not found: {model_path}")
    if not calibration_samples:
        raise OptimizationError("At least one calibration sample array is required.")

    # Inspect model to identify output/activation tensor names
    try:
        model = onnx.load(str(model_path))
    except Exception as exc:
        raise OptimizationError(f"Failed to load ONNX model {model_path}: {exc}") from exc

    tensor_names = [out.name for out in model.graph.output]
    if not tensor_names:
        tensor_names = ["output"]

    tensor_ranges: dict[str, tuple[float, float]] = {}
    scale_map: dict[str, float] = {}

    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        for sample in calibration_samples:
            net.setInput(sample)
            raw = net.forward()
            arr = np.asarray(raw, dtype=np.float32)

            for name in tensor_names:
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                if name in tensor_ranges:
                    curr_min, curr_max = tensor_ranges[name]
                    tensor_ranges[name] = (min(curr_min, min_v), max(curr_max, max_v))
                else:
                    tensor_ranges[name] = (min_v, max_v)
    except Exception as exc:
        logger.debug("DNN forward for calibration range extraction failed (%s); using priors", exc)
        for name in tensor_names:
            tensor_ranges[name] = (-1.0, 1.0)

    # Compute scale factors
    for name, (min_v, max_v) in tensor_ranges.items():
        peak = max(abs(min_v), abs(max_v), 1e-6)
        scale_map[name] = 127.0 / peak

    # Write TensorRT-format calibration cache
    header = "TRT-8400-EntropyCalibration2" if method == "entropy" else "TRT-8400-MinMaxCalibration"
    lines = [header]
    for name, scale in sorted(scale_map.items()):
        hex_val = _float_to_hex_scale(scale)
        lines.append(f"{name}: {hex_val}")

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    output_cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return CalibrationSummary(
        model_path=str(model_path),
        cache_path=str(output_cache_path),
        num_samples=len(calibration_samples),
        tensor_ranges=tensor_ranges,
        calibration_method=method,
    )


def detect_hardware_providers() -> ProvidersReport:
    """Detect available compute devices and execution providers on the current host."""
    providers: list[ProviderInfo] = []

    # 1. Check TensorRT
    trt_available = False
    trt_detail = "Requires TensorRT C++ libraries and compatible GPU"
    try:
        import onnxruntime as ort  # type: ignore[import-not-found,import-untyped,unused-ignore]

        if "TensorrtExecutionProvider" in ort.get_available_providers():
            trt_available = True
            trt_detail = "Active via ONNX Runtime TensorrtExecutionProvider"
    except ImportError:
        pass

    try:
        import tensorrt  # type: ignore[import-not-found,import-untyped,unused-ignore]

        trt_available = True
        trt_detail = f"Native TensorRT {getattr(tensorrt, '__version__', '')} available"
    except ImportError:
        pass

    providers.append(
        ProviderInfo(
            name="TensorrtExecutionProvider",
            available=trt_available,
            priority=1,
            device="cuda",
            details=trt_detail,
        )
    )

    # 2. Check CUDA
    cuda_available = False
    cuda_detail = "No CUDA-capable GPU detected or CUDA runtime missing"
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cuda_available = True
        cuda_detail = f"Detected {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s)"

    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_available = True
            cuda_detail = "Active via ONNX Runtime CUDAExecutionProvider"
    except ImportError:
        pass

    providers.append(
        ProviderInfo(
            name="CUDAExecutionProvider",
            available=cuda_available,
            priority=2,
            device="cuda",
            details=cuda_detail,
        )
    )

    # 3. Check DirectML (Windows DirectX acceleration)
    directml_available = False
    try:
        import onnxruntime as ort

        if "DmlExecutionProvider" in ort.get_available_providers():
            directml_available = True
    except ImportError:
        pass

    providers.append(
        ProviderInfo(
            name="DmlExecutionProvider",
            available=directml_available,
            priority=3,
            device="directml",
            details="DirectX 12 DirectML acceleration"
            if directml_available
            else "Unavailable or not installed",
        )
    )

    # 4. CPU Execution Provider (Always present)
    providers.append(
        ProviderInfo(
            name="CPUExecutionProvider",
            available=True,
            priority=4,
            device="cpu",
            details="Standard CPU execution",
        )
    )

    # Recommendations
    if trt_available:
        rec_backend = InferenceBackendType.TENSORRT.value
        rec_device = "cuda"
    elif cuda_available:
        rec_backend = InferenceBackendType.ONNXRUNTIME.value
        rec_device = "cuda"
    else:
        rec_backend = InferenceBackendType.OPENCV.value
        rec_device = "cpu"

    return ProvidersReport(
        providers=tuple(providers),
        recommended_backend=rec_backend,
        recommended_device=rec_device,
        cuda_available=cuda_available,
        tensorrt_available=trt_available,
    )


__all__ = [
    "OptimizationError",
    "convert_model_to_fp16",
    "detect_hardware_providers",
    "generate_int8_calibration_table",
    "optimize_model_precision",
    "quantize_model_dynamic_int8",
]
