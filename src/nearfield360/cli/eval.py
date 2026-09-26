"""Reproducible evaluation commands that publish JSON metric artifacts."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Never

import numpy as np
import typer

from nearfield360.cli.data_common import DatasetRootOption, discover_dataset
from nearfield360.cli.inference_common import (
    BackendOption,
    DeviceOption,
    load_backend,
    resolve_backend_type,
    resolve_device,
)
from nearfield360.cli.state import get_state
from nearfield360.data.detection import (
    WOODSCAPE_DETECTION_CLASSES,
    DetectionAnnotationError,
    DetectionPrediction,
    load_detection_annotations,
    load_detection_predictions,
    write_detection_predictions,
)
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticMaskError,
    load_semantic_mask,
    save_semantic_mask,
)
from nearfield360.data.woodscape import IMAGE_SUFFIXES, WoodScapeDataset
from nearfield360.perception.evaluation import (
    detection_confidence_analysis,
    environment_metadata,
    evaluate_detection,
    evaluate_semantic,
)
from nearfield360.perception.inference.backend import InferenceError
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice
from nearfield360.perception.inference.preprocessor import PreprocessorError
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.utils.artifacts import ArtifactError, write_json

eval_app = typer.Typer(
    help="Reproducible perception evaluation against labelled WoodScape data.",
    no_args_is_help=True,
)

PredictionsDirOption = Annotated[
    Path | None,
    typer.Option(
        "--predictions",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory of per-sample prediction files (exclusive with --model).",
    ),
]

ModelOption = Annotated[
    Path | None,
    typer.Option(
        "--model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="ONNX model to score live against dataset annotations (exclusive with --predictions).",
    ),
]

ThresholdOption = Annotated[
    float,
    typer.Option(
        "--iou-threshold",
        min=0.0,
        max=1.0,
        help="Match threshold for detection IoU (inclusive).",
    ),
]

LimitOption = Annotated[
    int,
    typer.Option(
        "--limit",
        min=0,
        help="Maximum annotated samples to evaluate (0 evaluates all).",
    ),
]

ConfidenceOption = Annotated[
    float | None,
    typer.Option(
        "--confidence-threshold",
        min=0.0,
        max=1.0,
        help="Minimum detection confidence (defaults to config.inference.confidence_threshold).",
    ),
]

NmsOption = Annotated[
    float | None,
    typer.Option(
        "--nms-threshold",
        min=0.0,
        max=1.0,
        help="Detection NMS IoU threshold (defaults to config.inference.nms_threshold).",
    ),
]

SavePredictionsOption = Annotated[
    Path | None,
    typer.Option(
        "--save-predictions",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Write per-sample predictions (PNG masks / TXT rows) for reuse with --predictions.",
    ),
]

ConfidenceThresholdsOption = Annotated[
    str | None,
    typer.Option(
        "--confidence-thresholds",
        help=(
            "Comma-separated confidence cutoffs in [0, 1] (e.g. 0.3,0.5,0.7) that add a "
            "confidence_analysis section with operating points and PR grids to the report."
        ),
    ),
]


def _find_prediction_file(directory: Path, stem: str, suffixes: frozenset[str]) -> Path | None:
    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _writing_output(output: Path, overwrite: bool, payload: dict[str, Any]) -> None:
    try:
        write_json(output, payload, overwrite=overwrite)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Wrote {output}")


def _report_payload(
    context: typer.Context,
    *,
    expected: int,
    evaluated: int,
    missing: list[str],
    metrics: dict[str, Any],
    model: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = get_state(context)
    return {
        "environment": environment_metadata(),
        "config": state.config.model_dump(mode="json"),
        "seed": state.config.runtime.seed,
        "model": model,
        "timing": timing,
        "samples": {
            "expected": expected,
            "evaluated": evaluated,
            "missing_predictions": missing,
        },
        "metrics": metrics,
    }


def _reject_incomplete(missing: list[str]) -> None:
    if not missing:
        return
    examples = ", ".join(missing[:5])
    remainder = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
    typer.secho(
        f"Missing predictions for {len(missing)} sample(s): {examples}{remainder}",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1) from None


def _reject_source() -> Never:
    typer.secho(
        "Provide exactly one of --predictions or --model.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1) from None


def _reject_save_without_model(save_predictions: Path | None) -> None:
    if save_predictions is None:
        return
    typer.secho(
        "--save-predictions requires --model.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1) from None


def _parse_confidence_thresholds(raw: str | None) -> list[float] | None:
    """Parse ``--confidence-thresholds`` into a deduplicated ascending list."""
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if not any(parts):
        raise typer.BadParameter("Provide a comma-separated list of cutoffs, e.g. '0.3,0.5,0.7'")
    values: set[float] = set()
    for part in parts:
        if not part:
            raise typer.BadParameter(f"Confidence thresholds must be non-empty values, got {raw!r}")
        try:
            value = float(part)
        except ValueError as exc:
            raise typer.BadParameter(
                f"Confidence thresholds must be numbers, got {part!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise typer.BadParameter(
                f"Confidence thresholds must be finite values in [0, 1], got {part!r}"
            )
        values.add(value)
    return sorted(values)


def _timing_stats(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Summarize per-sample engine latency samples in milliseconds."""
    if not samples_ms:
        return {
            "samples": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "samples_per_second": 0.0,
        }
    arr = np.asarray(samples_ms, dtype=np.float64)
    total = float(np.sum(arr))
    mean = total / len(samples_ms)
    return {
        "samples": len(samples_ms),
        "total_ms": round(total, 3),
        "mean_ms": round(mean, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "samples_per_second": round(1000.0 / mean, 3) if mean > 0.0 else 0.0,
    }


def _echo_timing(timing: dict[str, Any]) -> None:
    typer.echo(
        f"Timing: {timing['samples']} samples, mean={timing['mean_ms']:.3f} ms, "
        f"p95={timing['p95_ms']:.3f} ms, {timing['samples_per_second']:.1f} samples/s"
    )


def _build_segmentation_engine(
    context: typer.Context,
    model: Path,
    *,
    backend: InferenceBackendType | None,
    device: InferenceDevice | None,
) -> SemanticSegmentationEngine:
    loaded = load_backend(context, model, backend=backend, device=device)
    return SemanticSegmentationEngine(
        backend=loaded,
        num_classes=len(WOODSCAPE_SEMANTIC_CLASSES),
    )


def _segmentation_model_pairs(
    engine: SemanticSegmentationEngine,
    dataset: WoodScapeDataset,
    limit: int,
    save_predictions: Path | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, list[float]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    latencies_ms: list[float] = []
    expected = 0
    for sample in dataset:
        if sample.semantic_mask_path is None:
            continue
        if limit > 0 and expected >= limit:
            break
        expected += 1
        try:
            target = load_semantic_mask(sample.semantic_mask_path)
            image = load_rgb_image(sample.image_path)
            start = time.perf_counter()
            predicted, _confidence = engine.predict(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if save_predictions is not None:
                save_semantic_mask(save_predictions / f"{sample.key.stem}.png", predicted)
        except (ImageReadError, SemanticMaskError, InferenceError, PreprocessorError) as exc:
            _mask_error(sample.key.stem, exc)
        pairs.append((predicted, target))
    return pairs, expected, latencies_ms


def _prediction_arrays(
    predictions: Sequence[DetectionPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert parsed predictions to aligned arrays; empty files keep (N, 4) shape."""
    if not predictions:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    boxes = np.asarray([item.xyxy for item in predictions], dtype=np.float64)
    scores = np.asarray([item.score for item in predictions], dtype=np.float64)
    classes = np.asarray([item.class_id for item in predictions], dtype=np.int64)
    return boxes, scores, classes


def _build_detection_engine(
    context: typer.Context,
    model: Path,
    *,
    backend: InferenceBackendType | None,
    device: InferenceDevice | None,
    confidence_threshold: float | None,
    nms_threshold: float | None,
) -> ObjectDetectionEngine:
    inference = get_state(context).config.inference
    loaded = load_backend(context, model, backend=backend, device=device)
    return ObjectDetectionEngine(
        backend=loaded,
        confidence_threshold=(
            inference.confidence_threshold if confidence_threshold is None else confidence_threshold
        ),
        nms_threshold=inference.nms_threshold if nms_threshold is None else nms_threshold,
        num_classes=len(WOODSCAPE_DETECTION_CLASSES),
    )


def _detection_model_batches(
    engine: ObjectDetectionEngine,
    dataset: WoodScapeDataset,
    limit: int,
    save_predictions: Path | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[list[Any]], int, list[float]]:
    prediction_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    target_batches: list[list[Any]] = []
    latencies_ms: list[float] = []
    expected = 0
    for sample in dataset:
        if sample.detection_path is None:
            continue
        if limit > 0 and expected >= limit:
            break
        expected += 1
        try:
            image = load_rgb_image(sample.image_path)
            image_size = (int(image.shape[0]), int(image.shape[1]))
            targets = load_detection_annotations(sample.detection_path, image_size=image_size)
            start = time.perf_counter()
            raw_predictions = engine.predict(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if save_predictions is not None:
                write_detection_predictions(
                    save_predictions / f"{sample.key.stem}.txt", raw_predictions
                )
        except (ImageReadError, DetectionAnnotationError, InferenceError, PreprocessorError) as exc:
            typer.secho(
                f"Detection error for {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1) from None
        prediction_batches.append(_prediction_arrays(raw_predictions))
        target_batches.append(list(targets))
    return prediction_batches, target_batches, expected, latencies_ms


def _mask_error(stem: str, exc: Exception) -> None:
    typer.secho(f"Mask error for {stem}: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from None


@eval_app.command("segmentation")
def evaluate_segmentation(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="JSON report artifact (created atomically; refuses to overwrite).",
        ),
    ],
    predictions: PredictionsDirOption = None,
    model: ModelOption = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing report artifact.")
    ] = False,
    root: DatasetRootOption = None,
    limit: LimitOption = 0,
    backend: BackendOption = None,
    device: DeviceOption = None,
    save_predictions: SavePredictionsOption = None,
) -> None:
    """Score predicted masks against WoodScape semantic ground truth."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    missing: list[str] = []
    expected = 0
    model_info: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    if model is not None:
        if predictions is not None:
            _reject_source()
        dataset = discover_dataset(context, root)
        engine = _build_segmentation_engine(context, model, backend=backend, device=device)
        pairs, expected, latencies_ms = _segmentation_model_pairs(
            engine, dataset, limit, save_predictions
        )
        timing = _timing_stats(latencies_ms)
        model_info = {
            "path": str(model),
            "backend": resolve_backend_type(context, backend).value,
            "device": resolve_device(context, device).value,
        }
    else:
        if predictions is None:
            _reject_source()
        _reject_save_without_model(save_predictions)
        dataset = discover_dataset(context, root)
        for sample in dataset:
            if sample.semantic_mask_path is None:
                continue
            if limit > 0 and expected >= limit:
                break
            expected += 1
            prediction_file = _find_prediction_file(predictions, sample.key.stem, IMAGE_SUFFIXES)
            if prediction_file is None:
                missing.append(sample.key.stem)
                continue
            try:
                target = load_semantic_mask(sample.semantic_mask_path)
                predicted = load_semantic_mask(prediction_file)
            except (ImageReadError, SemanticMaskError) as exc:
                _mask_error(sample.key.stem, exc)
            pairs.append((predicted, target))

    _reject_incomplete(missing)
    if expected == 0:
        typer.secho(
            "Dataset contains no semantic masks to evaluate against.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        evaluation = evaluate_semantic(pairs)
    except ValueError as exc:
        _mask_error("segmentation", exc)
    payload = _report_payload(
        context,
        expected=expected,
        evaluated=len(pairs),
        missing=missing,
        metrics=evaluation.as_dict(),
        model=model_info,
        timing=timing,
    )
    _writing_output(output, overwrite, payload)
    typer.echo(f"Evaluated {len(pairs)}/{expected} samples; mIoU={evaluation.mean_iou:.3f}")
    if timing is not None:
        _echo_timing(timing)


@eval_app.command("detection")
def evaluate_detection_command(
    context: typer.Context,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="JSON report artifact (created atomically; refuses to overwrite).",
        ),
    ],
    predictions: PredictionsDirOption = None,
    model: ModelOption = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing report artifact.")
    ] = False,
    iou_threshold: ThresholdOption = 0.5,
    root: DatasetRootOption = None,
    limit: LimitOption = 0,
    backend: BackendOption = None,
    device: DeviceOption = None,
    confidence_threshold: ConfidenceOption = None,
    nms_threshold: NmsOption = None,
    save_predictions: SavePredictionsOption = None,
    confidence_thresholds: ConfidenceThresholdsOption = None,
) -> None:
    """Score detected boxes (``*.txt``) against WoodScape detection annotations."""
    thresholds = _parse_confidence_thresholds(confidence_thresholds)
    prediction_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    target_batches: list[list[Any]] = []
    missing: list[str] = []
    expected = 0
    model_info: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    confidence_analysis: dict[str, Any] | None = None
    if model is not None:
        if predictions is not None:
            _reject_source()
        dataset = discover_dataset(context, root)
        engine = _build_detection_engine(
            context,
            model,
            backend=backend,
            device=device,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        prediction_batches, target_batches, expected, latencies_ms = _detection_model_batches(
            engine, dataset, limit, save_predictions
        )
        timing = _timing_stats(latencies_ms)
        model_info = {
            "path": str(model),
            "backend": resolve_backend_type(context, backend).value,
            "device": resolve_device(context, device).value,
            "confidence_threshold": engine.confidence_threshold,
            "nms_threshold": engine.nms_threshold,
        }
    else:
        if predictions is None:
            _reject_source()
        _reject_save_without_model(save_predictions)
        dataset = discover_dataset(context, root)
        for sample in dataset:
            if sample.detection_path is None:
                continue
            if limit > 0 and expected >= limit:
                break
            expected += 1
            prediction_file = _find_prediction_file(
                predictions, sample.key.stem, frozenset({".txt"})
            )
            if prediction_file is None:
                missing.append(sample.key.stem)
                continue
            try:
                image = load_rgb_image(sample.image_path)
                image_size = (int(image.shape[0]), int(image.shape[1]))
                targets = load_detection_annotations(sample.detection_path, image_size=image_size)
                predictions_list = load_detection_predictions(
                    prediction_file, image_size=image_size
                )
            except (ImageReadError, DetectionAnnotationError) as exc:
                typer.secho(
                    f"Detection error for {sample.key.stem}: {exc}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1) from None
            prediction_batches.append(_prediction_arrays(predictions_list))
            target_batches.append(list(targets))

    _reject_incomplete(missing)
    if expected == 0:
        typer.secho(
            "Dataset contains no detection annotations to evaluate against.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        evaluation = evaluate_detection(
            prediction_batches, target_batches, iou_threshold=iou_threshold
        )
        if thresholds is not None:
            confidence_analysis = detection_confidence_analysis(
                prediction_batches,
                target_batches,
                iou_threshold=iou_threshold,
                thresholds=thresholds,
            )
    except ValueError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    metrics = evaluation.as_dict()
    if confidence_analysis is not None:
        metrics["confidence_analysis"] = confidence_analysis
    payload = _report_payload(
        context,
        expected=expected,
        evaluated=len(prediction_batches),
        missing=missing,
        metrics=metrics,
        model=model_info,
        timing=timing,
    )
    _writing_output(output, overwrite, payload)
    typer.echo(
        f"Evaluated {len(prediction_batches)}/{expected} samples; "
        f"mAP={evaluation.mean_average_precision:.3f}"
    )
    if timing is not None:
        _echo_timing(timing)


__all__ = ["eval_app"]
