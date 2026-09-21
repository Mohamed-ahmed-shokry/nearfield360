"""CLI commands for neural perception inference, benchmarking, and inspection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer

from nearfield360.data.detection import DetectionPrediction
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import WOODSCAPE_SEMANTIC_CLASSES
from nearfield360.perception.inference.backend import (
    InferenceError,
    create_backend,
)
from nearfield360.perception.inference.models import (
    InferenceBackendType,
    InferenceDevice,
)
from nearfield360.perception.inference.benchmark import benchmark_inference
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
    PreprocessorError,
)
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.utils.artifacts import write_json

infer_app = typer.Typer(
    help="Neural network perception inference, benchmarking, and model inspection.",
    no_args_is_help=True,
)


@infer_app.command("semantic")
def infer_semantic(
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
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=dev_enum,
        )
        preprocessor = FisheyeImagePreprocessor(
            target_size=(480, 640),
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        engine = SemanticSegmentationEngine(
            backend=backend,
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
        "device": backend.device.value,
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
            f"Segmentation complete in {latency_ms:.2f} ms ({backend.device.value.upper()})",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Mask shape: {mask.shape} | Mean confidence: {np.mean(conf):.3f}")
        typer.echo("Classes detected:")
        for name, info in class_dist.items():
            typer.echo(f"  - {name}: {info['percentage']}% ({info['pixels']} px)")


@infer_app.command("detection")
def infer_detection(
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
        float,
        typer.Option(
            "--confidence-threshold",
            "-c",
            min=0.0,
            max=1.0,
            help="Minimum confidence threshold.",
        ),
    ] = 0.25,
    nms_threshold: Annotated[
        float,
        typer.Option(
            "--nms-threshold",
            min=0.0,
            max=1.0,
            help="Non-maximum suppression IoU threshold.",
        ),
    ] = 0.45,
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
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=dev_enum,
        )
        preprocessor = FisheyeImagePreprocessor(
            target_size=(480, 640),
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        engine = ObjectDetectionEngine(
            backend=backend,
            preprocessor=preprocessor,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
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
        "device": backend.device.value,
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
            f"Detection complete in {latency_ms:.2f} ms ({backend.device.value.upper()})",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"Found {len(predictions)} objects:")
        for d in dets_data:
            typer.echo(f"  - [{d['class_name']}] score={d['score']:.2f}, box={d['box_xyxy']}")


@infer_app.command("benchmark")
def infer_benchmark(
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
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=dev_enum,
        )
        summary = benchmark_inference(
            backend,
            input_shape=(1, 3, height, width),
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

    target_info = f"{backend.backend_type.value.upper()} on {backend.device.value.upper()}"
    typer.secho(
        f"Benchmark Complete ({target_info})",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  Input Shape:       {summary.input_shape}")
    typer.echo(f"  Iterations:        {summary.iterations} (warmup: {summary.warmup})")
    typer.echo(f"  Mean Latency:      {summary.mean_latency_ms:.2f} ms")
    typer.echo(f"  Median Latency:    {summary.median_latency_ms:.2f} ms")
    typer.echo(f"  90th Percentile:   {summary.p90_latency_ms:.2f} ms")
    typer.echo(f"  95th Percentile:   {summary.p95_latency_ms:.2f} ms")
    typer.echo(f"  99th Percentile:   {summary.p99_latency_ms:.2f} ms")
    typer.echo(f"  Throughput (FPS):  {summary.fps:.1f} fps")


@infer_app.command("inspect")
def infer_inspect(
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
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", resolve_path=True, help="Save metadata JSON to file."),
    ] = None,
) -> None:
    """Inspect input/output tensor shapes and structural metadata of an ONNX model."""
    try:
        dev_enum = InferenceDevice(device.lower())
        backend = create_backend(
            model,
            backend_type=InferenceBackendType.OPENCV,
            device=dev_enum,
        )
    except Exception as exc:
        typer.secho(f"Failed to inspect model: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    meta = backend.metadata.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, meta, overwrite=True)

    typer.echo(json.dumps(meta, indent=2))


__all__ = [
    "infer_app",
]
