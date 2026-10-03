"""CLI commands for neural perception inference, benchmarking, and inspection."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer

from nearfield360.cli.inference_common import (
    BackendOption,
    PrecisionOption,
    resolve_backend_type,
    resolve_precision,
)
from nearfield360.cli.state import get_state
from nearfield360.data.detection import DetectionPrediction
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import WOODSCAPE_SEMANTIC_CLASSES
from nearfield360.perception.inference.backend import (
    InferenceError,
    create_backend,
)
from nearfield360.perception.inference.benchmark import (
    benchmark_batch_sweep,
    benchmark_inference,
    compare_numerical_parity,
)
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.models import (
    InferenceDevice,
    PrecisionType,
)
from nearfield360.perception.inference.optimization import (
    OptimizationError,
    detect_hardware_providers,
    generate_int8_calibration_table,
    optimize_model_precision,
)
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
    PreprocessorError,
)
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.utils.artifacts import write_json

logger = logging.getLogger(__name__)

infer_app = typer.Typer(
    help="Neural network perception inference, benchmarking, and model inspection.",
    no_args_is_help=True,
)


@infer_app.command("semantic")
def infer_semantic(
    context: typer.Context,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to semantic segmentation ONNX model file.",
        ),
    ],
    image: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to input fisheye image.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Path to save output mask image (.png) or inference summary (.json).",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            "-d",
            help="Execution device target (cpu, cuda, directml).",
        ),
    ] = "cpu",
    backend: BackendOption = None,
    palette: Annotated[
        bool,
        typer.Option(
            "--palette/--raw-mask",
            help="Colorize output mask using official WoodScape palette.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--no-json",
            help="Output structured JSON metrics to standard output.",
        ),
    ] = False,
    preserve_aspect_ratio: Annotated[
        bool,
        typer.Option(
            "--preserve-aspect-ratio/--direct-resize",
            help="Preserve aspect ratio via letterboxing during preprocessing.",
        ),
    ] = True,
) -> None:
    """Run semantic segmentation inference on an automotive fisheye image."""
    try:
        rgb_img = load_rgb_image(image)
    except ImageReadError as exc:
        typer.secho(f"Failed to read image {image}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        dev_enum = InferenceDevice(device.lower())
    except ValueError:
        valid_devs = ", ".join(d.value for d in InferenceDevice)
        typer.secho(
            f"Invalid device {device!r}. Must be one of: {valid_devs}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        b_type = resolve_backend_type(context, backend)
        backend_obj = create_backend(
            model,
            backend_type=b_type,
            device=dev_enum,
        )
        preprocessor = FisheyeImagePreprocessor(
            target_size=(480, 640),
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        engine = SemanticSegmentationEngine(
            backend=backend_obj,
            preprocessor=preprocessor,
            num_classes=len(WOODSCAPE_SEMANTIC_CLASSES),
        )
    except (InferenceError, PreprocessorError, ValueError) as exc:
        typer.secho(f"Failed to initialize engine: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    t0 = time.perf_counter()
    try:
        mask, conf = engine.predict(rgb_img)
    except (InferenceError, PreprocessorError) as exc:
        typer.secho(f"Inference execution failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate class distribution
    unique_classes, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    class_dist: dict[str, Any] = {}
    for c_id, cnt in zip(unique_classes, counts, strict=True):
        c_name = (
            WOODSCAPE_SEMANTIC_CLASSES[c_id].name
            if c_id < len(WOODSCAPE_SEMANTIC_CLASSES)
            else f"class_{c_id}"
        )
        class_dist[c_name] = {
            "class_id": int(c_id),
            "pixels": int(cnt),
            "percentage": round(float(cnt / total_pixels) * 100.0, 2),
        }

    summary: dict[str, Any] = {
        "model": str(model),
        "image": str(image),
        "device": backend_obj.device.value,
        "input_shape": list(rgb_img.shape),
        "mask_shape": list(mask.shape),
        "latency_ms": round(latency_ms, 2),
        "mean_confidence": round(float(np.mean(conf)), 4),
        "classes_present": class_dist,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            write_json(output, summary, overwrite=True)
            if not json_output:
                typer.echo(f"Wrote inference summary to {output}")
        else:
            if palette:
                palette_bgr = np.array(
                    [c.color_bgr for c in WOODSCAPE_SEMANTIC_CLASSES], dtype=np.uint8
                )
                safe_mask = np.clip(mask, 0, len(palette_bgr) - 1)
                colorized_bgr = palette_bgr[safe_mask]
                cv2.imwrite(str(output), colorized_bgr)
            else:
                cv2.imwrite(str(output), mask)
            if not json_output:
                typer.echo(f"Saved predicted segmentation mask to {output}")

    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    else:
        typer.secho(
            f"Segmentation complete in {latency_ms:.2f} ms ({backend_obj.device.value.upper()})",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Mask shape: {mask.shape} | Mean confidence: {np.mean(conf):.3f}")
        typer.echo("Classes detected:")
        for name, info in class_dist.items():
            typer.echo(f"  - {name}: {info['percentage']}% ({info['pixels']} px)")


@infer_app.command("detection")
def infer_detection(
    context: typer.Context,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to object detection ONNX model file.",
        ),
    ],
    image: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to input fisheye image.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Path to save annotated image (.png/.jpg) or detections (.json).",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            "-d",
            help="Execution device target (cpu, cuda, directml).",
        ),
    ] = "cpu",
    confidence_threshold: Annotated[
        float | None,
        typer.Option(
            "--confidence-threshold",
            "-c",
            min=0.0,
            max=1.0,
            help="Minimum confidence threshold (defaults to config value).",
        ),
    ] = None,
    nms_threshold: Annotated[
        float | None,
        typer.Option(
            "--nms-threshold",
            min=0.0,
            max=1.0,
            help="NMS IoU threshold (defaults to config value).",
        ),
    ] = None,
    backend: BackendOption = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--no-json",
            help="Output structured JSON detections to standard output.",
        ),
    ] = False,
    preserve_aspect_ratio: Annotated[
        bool,
        typer.Option(
            "--preserve-aspect-ratio/--direct-resize",
            help="Preserve aspect ratio via letterboxing during preprocessing.",
        ),
    ] = True,
) -> None:
    """Run 2D object detection inference on an automotive fisheye image."""
    try:
        rgb_img = load_rgb_image(image)
    except ImageReadError as exc:
        typer.secho(f"Failed to read image {image}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        dev_enum = InferenceDevice(device.lower())
    except ValueError:
        valid_devs = ", ".join(d.value for d in InferenceDevice)
        typer.secho(
            f"Invalid device {device!r}. Must be one of: {valid_devs}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        b_type = resolve_backend_type(context, backend)
        conf_thr = (
            confidence_threshold
            if confidence_threshold is not None
            else get_state(context).config.inference.confidence_threshold
        )
        nms_thr = (
            nms_threshold
            if nms_threshold is not None
            else get_state(context).config.inference.nms_threshold
        )
        backend_obj = create_backend(
            model,
            backend_type=b_type,
            device=dev_enum,
        )
        preprocessor = FisheyeImagePreprocessor(
            target_size=(480, 640),
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        engine = ObjectDetectionEngine(
            backend=backend_obj,
            preprocessor=preprocessor,
            confidence_threshold=conf_thr,
            nms_threshold=nms_thr,
            num_classes=5,
        )
    except (InferenceError, PreprocessorError, ValueError) as exc:
        typer.secho(f"Failed to initialize engine: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    t0 = time.perf_counter()
    try:
        predictions: tuple[DetectionPrediction, ...] = engine.predict(rgb_img)
    except (InferenceError, PreprocessorError) as exc:
        typer.secho(f"Inference execution failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    dets_data: list[dict[str, Any]] = [
        {
            "class_id": p.class_id,
            "class_name": p.class_name,
            "box_xyxy": [
                round(p.x_min, 1),
                round(p.y_min, 1),
                round(p.x_max, 1),
                round(p.y_max, 1),
            ],
            "score": round(p.score, 4),
        }
        for p in predictions
    ]

    summary: dict[str, Any] = {
        "model": str(model),
        "image": str(image),
        "device": backend_obj.device.value,
        "input_shape": list(rgb_img.shape),
        "latency_ms": round(latency_ms, 2),
        "num_detections": len(predictions),
        "detections": dets_data,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            write_json(output, summary, overwrite=True)
            if not json_output:
                typer.echo(f"Wrote detections to {output}")
        else:
            # Draw boxes on BGR canvas
            canvas = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            for p in predictions:
                pt1 = (round(p.x_min), round(p.y_min))
                pt2 = (round(p.x_max), round(p.y_max))
                cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), 2)
                label = f"{p.class_name}: {p.score:.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (pt1[0], max(15, pt1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(output), canvas)
            if not json_output:
                typer.echo(f"Saved annotated image to {output}")

    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    else:
        typer.secho(
            f"Detection complete in {latency_ms:.2f} ms ({backend_obj.device.value.upper()})",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Found {len(predictions)} objects:")
        for d in dets_data:
            typer.echo(f"  - [{d['class_name']}] score={d['score']:.2f}, box={d['box_xyxy']}")


@infer_app.command("benchmark")
def infer_benchmark(
    context: typer.Context,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to ONNX model file.",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            "--device",
            "-d",
            help="Execution device target (cpu, cuda, directml).",
        ),
    ] = "cpu",
    backend: BackendOption = None,
    precision: PrecisionOption = None,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", min=1, help="Number of timed iterations."),
    ] = 50,
    warmup: Annotated[
        int,
        typer.Option("--warmup", min=0, help="Number of warmup iterations."),
    ] = 10,
    height: Annotated[
        int,
        typer.Option("--height", min=1, help="Input tensor height in pixels."),
    ] = 480,
    width: Annotated[
        int,
        typer.Option("--width", min=1, help="Input tensor width in pixels."),
    ] = 640,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", min=1, help="Batch size for single-run benchmark."),
    ] = 1,
    batch_sweep: Annotated[
        bool,
        typer.Option(
            "--batch-sweep",
            help="Run a throughput scaling sweep across multiple batch sizes.",
        ),
    ] = False,
    batch_sizes: Annotated[
        str,
        typer.Option(
            "--batch-sizes",
            help="Comma-separated batch sizes to evaluate during sweep (e.g. '1,2,4,8').",
        ),
    ] = "1,2,4,8",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", resolve_path=True, help="Save summary JSON to file."),
    ] = None,
) -> None:
    """Benchmark inference latency and throughput of an ONNX model."""
    try:
        dev_enum = InferenceDevice(device.lower())
    except ValueError:
        valid_devs = ", ".join(d.value for d in InferenceDevice)
        typer.secho(
            f"Invalid device {device!r}. Must be one of: {valid_devs}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        b_type = resolve_backend_type(context, backend)
        prec = resolve_precision(context, precision)
        backend_obj = create_backend(
            model,
            backend_type=b_type,
            device=dev_enum,
            precision=prec,
        )
    except (InferenceError, ValueError) as exc:
        typer.secho(f"Failed to initialize backend: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    target_info = f"{backend_obj.backend_type.value.upper()} on {backend_obj.device.value.upper()}"

    if batch_sweep:
        try:
            parsed_batch_sizes = [
                int(item.strip()) for item in batch_sizes.split(",") if item.strip()
            ]
            if not parsed_batch_sizes:
                raise ValueError("No batch sizes provided.")
            if any(b <= 0 for b in parsed_batch_sizes):
                raise ValueError("All batch sizes must be positive integers.")
        except ValueError as exc:
            typer.secho(f"Invalid --batch-sizes: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None

        try:
            sweep_summary = benchmark_batch_sweep(
                backend_obj,
                batch_sizes=parsed_batch_sizes,
                iterations=iterations,
                warmup=warmup,
                base_shape=(3, height, width),
            )
        except (InferenceError, ValueError) as exc:
            typer.secho(f"Batch sweep failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None

        sweep_dict = sweep_summary.model_dump()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, sweep_dict, overwrite=True)
            typer.echo(f"Wrote batch sweep report to {output}")

        typer.secho(
            f"Batch Sweep Complete ({target_info})",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo(f"  Base Shape:        {sweep_summary.base_input_shape}")
        typer.echo(
            f"  Iterations:        {sweep_summary.iterations} (warmup: {sweep_summary.warmup})"
        )
        typer.echo(
            f"  Optimal Batch:     {sweep_summary.optimal_batch_size} "
            f"({sweep_summary.max_fps:.1f} fps)"
        )
        typer.echo("")
        header = (
            f"  {'Batch':<7} {'Latency (ms)':<14} {'p95 (ms)':<10} "
            f"{'Throughput (fps)':<18} {'Speedup':<9} {'Efficiency':<10}"
        )
        typer.echo(header)
        typer.echo("  " + "-" * (len(header) - 2))
        for item in sweep_summary.items:
            eff_pct = item.scaling_efficiency * 100.0
            typer.echo(
                f"  {item.batch_size:<7} "
                f"{item.mean_latency_ms:<14.2f} "
                f"{item.p95_latency_ms:<10.2f} "
                f"{item.fps:<18.1f} "
                f"{item.speedup:<9.2f}x "
                f"{eff_pct:<10.1f}%"
            )
        return

    try:
        summary = benchmark_inference(
            backend_obj,
            input_shape=(batch_size, 3, height, width),
            iterations=iterations,
            warmup=warmup,
        )
    except (InferenceError, ValueError) as exc:
        typer.secho(f"Benchmark failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    summary_dict = summary.model_dump()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, summary_dict, overwrite=True)
        typer.echo(f"Wrote benchmark report to {output}")

    typer.secho(
        f"Benchmark Complete ({target_info})",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  Input Shape:       {summary.input_shape}")
    typer.echo(f"  Batch Size:        {batch_size}")
    typer.echo(f"  Iterations:        {summary.iterations} (warmup: {summary.warmup})")
    typer.echo(f"  Mean Latency:      {summary.mean_latency_ms:.2f} ms")
    typer.echo(f"  Median Latency:    {summary.median_latency_ms:.2f} ms")
    typer.echo(f"  90th Percentile:   {summary.p90_latency_ms:.2f} ms")
    typer.echo(f"  95th Percentile:   {summary.p95_latency_ms:.2f} ms")
    typer.echo(f"  99th Percentile:   {summary.p99_latency_ms:.2f} ms")
    typer.echo(f"  Throughput (FPS):  {summary.fps:.1f} fps")


@infer_app.command("parity")
def infer_parity(
    context: typer.Context,
    model_a: Annotated[
        Path,
        typer.Option(
            "--model-a",
            "-a",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the first ONNX model file (actual).",
        ),
    ],
    model_b: Annotated[
        Path,
        typer.Option(
            "--model-b",
            "-b",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the second ONNX model file (reference).",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            "--device",
            "-d",
            help="Execution device target (cpu, cuda, directml).",
        ),
    ] = "cpu",
    height: Annotated[
        int,
        typer.Option("--height", min=1, help="Input tensor height in pixels."),
    ] = 480,
    width: Annotated[
        int,
        typer.Option("--width", min=1, help="Input tensor width in pixels."),
    ] = 640,
    atol: Annotated[
        float,
        typer.Option("--atol", help="Absolute tolerance for parity check."),
    ] = 1e-4,
    rtol: Annotated[
        float,
        typer.Option("--rtol", help="Relative tolerance for parity check."),
    ] = 1e-4,
    backend: BackendOption = None,
) -> None:
    """Verify numerical parity between two ONNX models on the same input."""
    try:
        dev_enum = InferenceDevice(device.lower())
    except ValueError:
        valid_devs = ", ".join(d.value for d in InferenceDevice)
        typer.secho(
            f"Invalid device {device!r}. Must be one of: {valid_devs}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        b_type = resolve_backend_type(context, backend)
        backend_a = create_backend(model_a, backend_type=b_type, device=dev_enum)
        backend_b = create_backend(model_b, backend_type=b_type, device=dev_enum)
    except InferenceError as exc:
        typer.secho(f"Failed to load model: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    dummy_input = np.random.randn(1, 3, height, width).astype(np.float32)
    try:
        output_a = backend_a.forward(dummy_input)
        output_b = backend_b.forward(dummy_input)
    except InferenceError as exc:
        typer.secho(f"Inference failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        result = compare_numerical_parity(output_a, output_b, atol=atol, rtol=rtol)
    except (ValueError, TypeError) as exc:
        typer.secho(f"Parity comparison failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if result.is_match:
        typer.secho(
            "Parity check PASSED",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        typer.secho(
            "Parity check FAILED",
            fg=typer.colors.RED,
            bold=True,
        )
    typer.echo(f"  Max Absolute Diff:  {result.max_abs_diff:.2e}")
    typer.echo(f"  Mean Absolute Diff: {result.mean_abs_diff:.2e}")
    typer.echo(f"  Max Relative Diff:  {result.max_rel_diff:.2e}")
    typer.echo(f"  ATol:               {result.atol}")
    typer.echo(f"  RTol:               {result.rtol}")

    if not result.is_match:
        raise typer.Exit(code=1)


@infer_app.command("inspect")
def infer_inspect(
    context: typer.Context,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to ONNX model file.",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            "--device",
            "-d",
            help="Execution device target (cpu, cuda, directml).",
        ),
    ] = "cpu",
    backend: BackendOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", resolve_path=True, help="Save metadata JSON to file."),
    ] = None,
) -> None:
    """Inspect input/output tensor shapes and structural metadata of an ONNX model."""
    try:
        dev_enum = InferenceDevice(device.lower())
        b_type = resolve_backend_type(context, backend)
        backend_obj = create_backend(
            model,
            backend_type=b_type,
            device=dev_enum,
        )
    except Exception as exc:
        typer.secho(f"Failed to inspect model: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    meta = backend_obj.metadata.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, meta, overwrite=True)

    typer.echo(json.dumps(meta, indent=2))


@infer_app.command("providers")
def infer_providers(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON providers report."),
    ] = False,
) -> None:
    """Inspect detected execution providers (TensorRT, CUDA, DirectML, CPU)."""
    report = detect_hardware_providers()
    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    typer.secho("Execution Providers & Acceleration Discovery", bold=True)
    typer.echo(f"  Recommended Backend: {report.recommended_backend}")
    typer.echo(f"  Recommended Device:  {report.recommended_device}")
    typer.echo(f"  NVIDIA CUDA:         {'Detected' if report.cuda_available else 'Not detected'}")
    typer.echo(
        f"  TensorRT Support:    {'Detected' if report.tensorrt_available else 'Not detected'}"
    )
    typer.echo("")
    typer.echo(f"  {'Provider':<28} {'Available':<11} {'Device':<10} {'Priority':<10} Details")
    typer.echo("  " + "-" * 78)
    for p in report.providers:
        avail_str = "[YES]" if p.available else "[NO]"
        color = typer.colors.GREEN if p.available else typer.colors.YELLOW
        typer.echo(
            f"  {p.name:<28} "
            + typer.style(f"{avail_str:<11}", fg=color)
            + f"{p.device:<10} {p.priority:<10} {p.details}"
        )


@infer_app.command("optimize")
def infer_optimize(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to source FP32 ONNX model file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Path to save optimized ONNX model.",
        ),
    ],
    precision: Annotated[
        PrecisionType,
        typer.Option(
            "--precision",
            "-p",
            case_sensitive=False,
            help="Target precision ('fp16' or 'int8').",
        ),
    ] = PrecisionType.FP16,
    check_drift: Annotated[
        bool,
        typer.Option(
            "--check-drift/--no-check-drift",
            help="Evaluate numerical output drift against FP32 baseline on sample input.",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            help="Allow overwriting target file if it already exists.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print machine-readable JSON optimization summary.",
        ),
    ] = False,
) -> None:
    """Optimize an ONNX model with FP16 half-precision conversion or dynamic INT8 quantization."""
    sample: np.ndarray | None = None
    if check_drift:
        sample = np.random.default_rng(42).standard_normal((1, 3, 480, 640)).astype(np.float32)

    try:
        summary = optimize_model_precision(
            model,
            output,
            target_precision=precision,
            test_sample=sample,
            overwrite=overwrite,
        )
    except OptimizationError as exc:
        typer.secho(f"Model optimization failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(summary.model_dump(), indent=2))
        return

    typer.secho(
        f"Model Precision Optimization ({precision.value.upper()}) Complete",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  Source Model:        {summary.source_path}")
    typer.echo(f"  Optimized Model:     {summary.optimized_path}")
    typer.echo(f"  Quantization Method: {summary.quantization_type}")
    typer.echo(f"  Source Size:         {summary.source_size_bytes / (1024 * 1024):.2f} MB")
    typer.echo(f"  Optimized Size:      {summary.optimized_size_bytes / (1024 * 1024):.2f} MB")
    typer.echo(f"  Compression Ratio:   {summary.compression_ratio:.2f}x")
    typer.echo(
        f"  Node Count:          {summary.node_count} ({summary.quantized_node_count} converted)"
    )
    if summary.max_absolute_drift is not None:
        typer.echo(f"  Max Absolute Drift:  {summary.max_absolute_drift:.2e}")
        typer.echo(f"  Mean Absolute Drift: {summary.mean_absolute_drift:.2e}")


@infer_app.command("calibrate")
def infer_calibrate(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to ONNX model to profile for INT8 calibration.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Path to save TensorRT calibration cache table.",
        ),
    ],
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            "-s",
            min=1,
            help="Number of calibration samples to evaluate.",
        ),
    ] = 20,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="Calibration method ('entropy' or 'minmax').",
        ),
    ] = "entropy",
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            "-r",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Path to WoodScape dataset root or image directory.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print machine-readable JSON calibration summary.",
        ),
    ] = False,
) -> None:
    """Generate a TensorRT-compatible INT8 calibration cache table from fisheye samples."""
    sample_tensors: list[np.ndarray] = []
    if root is not None:
        img_paths = list(root.glob("**/*.png")) + list(root.glob("**/*.jpg"))
        for p in img_paths[:samples]:
            try:
                img = load_rgb_image(p)
                preproc = FisheyeImagePreprocessor()
                blob, _ = preproc.preprocess(img)
                sample_tensors.append(blob)
            except Exception as exc:
                logger.debug("Failed reading calibration sample %s: %s", p, exc)
                continue

    if not sample_tensors:
        rng = np.random.default_rng(42)
        sample_tensors.extend(
            rng.standard_normal((1, 3, 480, 640)).astype(np.float32) for _ in range(samples)
        )

    try:
        calib = generate_int8_calibration_table(
            model,
            sample_tensors,
            output,
            method=method,
        )
    except OptimizationError as exc:
        typer.secho(f"Calibration failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(calib.model_dump(), indent=2))
        return

    typer.secho(
        "INT8 Calibration Table Generated",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  Model:               {calib.model_path}")
    typer.echo(f"  Cache Table:         {calib.cache_path}")
    typer.echo(f"  Samples Evaluated:   {calib.num_samples}")
    typer.echo(f"  Calibration Method:  {calib.calibration_method}")
    typer.echo(f"  Tensors Profiled:    {len(calib.tensor_ranges)}")
    for name, (min_v, max_v) in calib.tensor_ranges.items():
        typer.echo(f"    - {name:<20} [{min_v:.3f}, {max_v:.3f}]")


__all__ = [
    "infer_app",
]
