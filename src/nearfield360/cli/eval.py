"""Reproducible evaluation commands that publish JSON metric artifacts."""

from __future__ import annotations

import json
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
from nearfield360.data.manifests import load_split_manifest
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticMaskError,
    load_semantic_mask,
    save_semantic_mask,
)
from nearfield360.data.splits import DatasetSplit, SplitError
from nearfield360.data.woodscape import IMAGE_SUFFIXES, WoodScapeDataset
from nearfield360.perception.comparison import (
    ComparisonError,
    EvaluationComparison,
    compare_evaluations,
)
from nearfield360.perception.evaluation import (
    detection_confidence_analysis,
    environment_metadata,
    evaluate_detection,
    evaluate_semantic,
    semantic_confidence_analysis,
)
from nearfield360.perception.inference.backend import InferenceError
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice
from nearfield360.perception.inference.preprocessor import PreprocessorError
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.robustness.plots import render_svg_line_chart
from nearfield360.utils.artifacts import ArtifactError, read_json, write_json

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

ConfidenceBinsOption = Annotated[
    int | None,
    typer.Option(
        "--confidence-bins",
        min=1,
        max=1000,
        help=(
            "Equal-width reliability bins for per-pixel confidence analysis "
            "(adds metrics.confidence_analysis; requires --model)."
        ),
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

SplitManifestOption = Annotated[
    Path | None,
    typer.Option(
        "--split-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Restrict evaluation to one split's samples from this manifest.",
    ),
]

SplitOption = Annotated[
    DatasetSplit | None,
    typer.Option(
        "--split",
        help="Split to evaluate (defaults to test when --split-manifest is given).",
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
    split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = get_state(context)
    return {
        "environment": environment_metadata(),
        "config": state.config.model_dump(mode="json"),
        "seed": state.config.runtime.seed,
        "split": split,
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


def _reject_confidence_bins_without_model(confidence_bins: int | None) -> None:
    if confidence_bins is None:
        return
    typer.secho(
        "--confidence-bins requires --model.",
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


def _apply_split_selection(
    dataset: WoodScapeDataset,
    split_manifest: Path | None,
    split: DatasetSplit | None,
) -> tuple[WoodScapeDataset, dict[str, Any] | None]:
    """Restrict a dataset to the manifest's split; an unchanged selection returns None info."""
    if split_manifest is None:
        if split is not None:
            typer.secho(
                "--split requires --split-manifest.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        return dataset, None
    effective = DatasetSplit.TEST if split is None else split
    try:
        splits = load_split_manifest(split_manifest, dataset)
        selected = splits.samples(dataset, effective)
    except SplitError as exc:
        typer.secho(f"Split error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    info = {
        "manifest": str(split_manifest),
        "split": effective.value,
        "grouping": splits.grouping,
        "group_source": splits.group_source,
        "seed": splits.seed,
        "dataset_samples": len(dataset),
        "selected_samples": len(selected),
    }
    typer.echo(f"Split {effective.value}: selected {len(selected)}/{len(dataset)} samples.")
    return WoodScapeDataset(dataset.root, selected), info


def _split_label(split_info: dict[str, Any] | None) -> str:
    if split_info is None:
        return ""
    return f" in split '{split_info['split']}'"


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
    *,
    collect_confidence: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], int, list[float]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    confidences: list[np.ndarray] = []
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
            predicted, confidence = engine.predict(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if save_predictions is not None:
                save_semantic_mask(save_predictions / f"{sample.key.stem}.png", predicted)
        except (ImageReadError, SemanticMaskError, InferenceError, PreprocessorError) as exc:
            _mask_error(sample.key.stem, exc)
        pairs.append((predicted, target))
        if collect_confidence:
            confidences.append(confidence)
    return pairs, confidences, expected, latencies_ms


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
    split_manifest: SplitManifestOption = None,
    split: SplitOption = None,
    limit: LimitOption = 0,
    backend: BackendOption = None,
    device: DeviceOption = None,
    save_predictions: SavePredictionsOption = None,
    confidence_bins: ConfidenceBinsOption = None,
) -> None:
    """Score predicted masks against WoodScape semantic ground truth."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    confidences: list[np.ndarray] = []
    missing: list[str] = []
    expected = 0
    model_info: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    split_info: dict[str, Any] | None = None
    confidence_analysis: dict[str, Any] | None = None
    if model is not None:
        if predictions is not None:
            _reject_source()
        dataset = discover_dataset(context, root)
        dataset, split_info = _apply_split_selection(dataset, split_manifest, split)
        engine = _build_segmentation_engine(context, model, backend=backend, device=device)
        pairs, confidences, expected, latencies_ms = _segmentation_model_pairs(
            engine,
            dataset,
            limit,
            save_predictions,
            collect_confidence=confidence_bins is not None,
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
        _reject_confidence_bins_without_model(confidence_bins)
        dataset = discover_dataset(context, root)
        dataset, split_info = _apply_split_selection(dataset, split_manifest, split)
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
            f"Dataset contains no semantic masks to evaluate against{_split_label(split_info)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        evaluation = evaluate_semantic(pairs)
    except ValueError as exc:
        _mask_error("segmentation", exc)
    if confidence_bins is not None:
        samples = [
            (confidence, predicted, target)
            for confidence, (predicted, target) in zip(confidences, pairs, strict=True)
        ]
        try:
            confidence_analysis = semantic_confidence_analysis(samples, num_bins=confidence_bins)
        except ValueError as exc:
            typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
    metrics = evaluation.as_dict()
    if confidence_analysis is not None:
        metrics["confidence_analysis"] = confidence_analysis
    payload = _report_payload(
        context,
        expected=expected,
        evaluated=len(pairs),
        missing=missing,
        metrics=metrics,
        model=model_info,
        timing=timing,
        split=split_info,
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
    split_manifest: SplitManifestOption = None,
    split: SplitOption = None,
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
    split_info: dict[str, Any] | None = None
    if model is not None:
        if predictions is not None:
            _reject_source()
        dataset = discover_dataset(context, root)
        dataset, split_info = _apply_split_selection(dataset, split_manifest, split)
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
        dataset, split_info = _apply_split_selection(dataset, split_manifest, split)
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
            "Dataset contains no detection annotations to evaluate against"
            f"{_split_label(split_info)}.",
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
        split=split_info,
    )
    _writing_output(output, overwrite, payload)
    typer.echo(
        f"Evaluated {len(prediction_batches)}/{expected} samples; "
        f"mAP={evaluation.mean_average_precision:.3f}"
    )
    if timing is not None:
        _echo_timing(timing)


def _detection_charts(analysis: dict[str, Any]) -> list[tuple[str, str]]:
    """Build PR-curve and operating-point SVG charts from a detection analysis."""
    charts: list[tuple[str, str]] = []
    curves = analysis.get("pr_curves")
    if isinstance(curves, dict):
        series: dict[str, list[tuple[float, float]]] = {}
        for name, points in curves.items():
            if points is None:
                continue
            series[str(name)] = [
                (float(point["recall"]), float(point["precision"])) for point in points
            ]
        if series:
            charts.append(
                (
                    "pr_curves.svg",
                    render_svg_line_chart(
                        "Detection Precision-Recall Curves",
                        "Recall",
                        "Precision",
                        series,
                        y_min=0.0,
                        y_max=1.0,
                    ),
                )
            )
    thresholds = analysis.get("thresholds")
    if isinstance(thresholds, list) and thresholds:
        operating: dict[str, list[tuple[float, float]]] = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for point in thresholds:
            confidence = float(point["confidence"])
            operating["precision"].append((confidence, float(point["precision"])))
            operating["recall"].append((confidence, float(point["recall"])))
            operating["f1"].append((confidence, float(point["f1"])))
        charts.append(
            (
                "operating_points.svg",
                render_svg_line_chart(
                    "Detection Operating Points",
                    "Confidence threshold",
                    "Score",
                    operating,
                    y_min=0.0,
                    y_max=1.0,
                ),
            )
        )
    return charts


def _reliability_charts(analysis: dict[str, Any]) -> list[tuple[str, str]]:
    """Build a reliability-diagram SVG chart from a semantic confidence analysis."""
    bins = analysis.get("bins")
    if not isinstance(bins, list) or not bins:
        return []
    model: list[tuple[float, float]] = []
    for item in bins:
        if not isinstance(item, dict):
            raise TypeError(f"Bin entry is not a mapping: {item!r}")
        if item.get("pixels"):
            model.append((float(item["mean_confidence"]), float(item["accuracy"])))
    if not model:
        return []
    return [
        (
            "reliability.svg",
            render_svg_line_chart(
                "Segmentation Reliability Diagram",
                "Mean confidence",
                "Accuracy",
                {
                    "Model": model,
                    "Perfect calibration": [(0.0, 0.0), (1.0, 1.0)],
                },
                y_min=0.0,
                y_max=1.0,
            ),
        )
    ]


def _build_report_charts(report: Any) -> list[tuple[str, str]]:
    """Return ``(filename, svg)`` charts for an eval report's confidence analysis."""
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError("Not an eval report: missing 'metrics' section")
    analysis = report["metrics"].get("confidence_analysis")
    if not isinstance(analysis, dict):
        raise ValueError(
            "Report has no confidence analysis; rerun eval with --confidence-thresholds "
            "(detection) or --confidence-bins (segmentation)"
        )
    try:
        if "bins" in analysis:
            charts = _reliability_charts(analysis)
        elif "pr_curves" in analysis or "thresholds" in analysis:
            charts = _detection_charts(analysis)
        else:
            charts = []
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Report confidence analysis is malformed: {exc}") from exc
    if not charts:
        raise ValueError("Report confidence analysis contains no plottable data")
    return charts


@eval_app.command("plot")
def plot_evaluation_report(
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Eval JSON report whose confidence analysis will be rendered.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            resolve_path=True,
            help="Directory for rendered SVG charts (created if missing).",
        ),
    ],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing SVG charts.")
    ] = False,
) -> None:
    """Render an eval report's confidence analysis to vector SVG charts."""
    try:
        payload = read_json(report)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    try:
        charts = _build_report_charts(payload)
    except ValueError as exc:
        typer.secho(f"Plot error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    if output_dir.exists() and not output_dir.is_dir():
        typer.secho(
            f"Plot error: {output_dir} exists and is not a directory",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    existing = [output_dir / name for name, _ in charts if (output_dir / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        typer.secho(
            f"Artifact error: {rendered} already exist; pass --overwrite to replace.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, svg in charts:
            destination = output_dir / name
            destination.write_text(f"{svg}\n", encoding="utf-8")
            typer.echo(f"Wrote {destination}")
    except OSError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None


def _format_delta_val(val: float | None, precision: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}"


def _format_signed_delta(delta: float | None, precision: int = 4) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.{precision}f}"


def _format_percent(pct: float | None) -> str:
    if pct is None:
        return "-"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _format_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "-"
    return f"{ratio:.3f}x"


def _print_comparison_summary(
    cmp: EvaluationComparison, baseline_path: Path, candidate_path: Path
) -> None:
    typer.echo(f"Evaluation Comparison (Task: {cmp.task.upper()})")
    typer.echo(f"  Baseline:  {baseline_path}")
    typer.echo(f"  Candidate: {candidate_path}")
    typer.echo("")
    typer.echo(f"{'Metric':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12} {'% Change':>10}")
    typer.echo("-" * 72)
    for name, delta in cmp.summary_deltas.items():
        b_str = _format_delta_val(delta.baseline)
        c_str = _format_delta_val(delta.candidate)
        d_str = _format_signed_delta(delta.delta)
        p_str = _format_percent(delta.percent_change)
        typer.echo(f"{name:<24} {b_str:>12} {c_str:>12} {d_str:>12} {p_str:>10}")

    if cmp.class_deltas:
        metric_name = "IoU" if cmp.task == "segmentation" else "AP"
        typer.echo(f"\nClass Breakdown ({metric_name})")
        typer.echo("-" * 72)
        typer.echo(
            f"{'Class':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12} {'% Change':>10}"
        )
        typer.echo("-" * 72)
        for name, delta in cmp.class_deltas.items():
            b_str = _format_delta_val(delta.baseline)
            c_str = _format_delta_val(delta.candidate)
            d_str = _format_signed_delta(delta.delta)
            p_str = _format_percent(delta.percent_change)
            typer.echo(f"{name:<24} {b_str:>12} {c_str:>12} {d_str:>12} {p_str:>10}")

    if cmp.calibration_deltas:
        typer.echo("\nCalibration & Confidence")
        typer.echo("-" * 72)
        typer.echo(
            f"{'Metric':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12} {'% Change':>10}"
        )
        typer.echo("-" * 72)
        for name, delta in cmp.calibration_deltas.items():
            b_str = _format_delta_val(delta.baseline)
            c_str = _format_delta_val(delta.candidate)
            d_str = _format_signed_delta(delta.delta)
            p_str = _format_percent(delta.percent_change)
            typer.echo(f"{name:<24} {b_str:>12} {c_str:>12} {d_str:>12} {p_str:>10}")

    if cmp.timing_deltas:
        typer.echo("\nTiming & Latency")
        typer.echo("-" * 72)
        typer.echo(f"{'Metric':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12} {'Ratio':>10}")
        typer.echo("-" * 72)
        for name, delta in cmp.timing_deltas.items():
            b_str = _format_delta_val(delta.baseline, precision=3)
            c_str = _format_delta_val(delta.candidate, precision=3)
            d_str = _format_signed_delta(delta.delta, precision=3)
            r_str = _format_ratio(delta.ratio)
            typer.echo(f"{name:<24} {b_str:>12} {c_str:>12} {d_str:>12} {r_str:>10}")


def _evaluate_gates(
    cmp: EvaluationComparison,
    *,
    fail_under_miou_delta: float | None,
    fail_under_map_delta: float | None,
    fail_over_ece_delta: float | None,
    fail_over_latency_ratio: float | None,
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    overall_ok = True

    if fail_under_miou_delta is not None:
        if cmp.task != "segmentation":
            raise ComparisonError(
                "--fail-under-miou-delta is only applicable to segmentation reports"
            )
        delta = cmp.summary_deltas["mean_iou"].delta
        passed = delta is not None and delta >= fail_under_miou_delta
        if not passed:
            overall_ok = False
        actual_str = f"{delta:+.4f}" if delta is not None else "missing"
        checks.append(
            {
                "gate": "fail_under_miou_delta",
                "status": "pass" if passed else "fail",
                "threshold": fail_under_miou_delta,
                "actual": delta,
                "detail": f"mIoU delta {actual_str} (min required: {fail_under_miou_delta:+.4f})",
            }
        )

    if fail_under_map_delta is not None:
        if cmp.task != "detection":
            raise ComparisonError("--fail-under-map-delta is only applicable to detection reports")
        delta = cmp.summary_deltas["mean_average_precision"].delta
        passed = delta is not None and delta >= fail_under_map_delta
        if not passed:
            overall_ok = False
        actual_str = f"{delta:+.4f}" if delta is not None else "missing"
        checks.append(
            {
                "gate": "fail_under_map_delta",
                "status": "pass" if passed else "fail",
                "threshold": fail_under_map_delta,
                "actual": delta,
                "detail": f"mAP delta {actual_str} (min required: {fail_under_map_delta:+.4f})",
            }
        )

    if fail_over_ece_delta is not None:
        if cmp.calibration_deltas is None or cmp.calibration_deltas.get("ece") is None:
            raise ComparisonError(
                "--fail-over-ece-delta requires confidence calibration data (ECE) in both reports"
            )
        delta = cmp.calibration_deltas["ece"].delta
        passed = delta is not None and delta <= fail_over_ece_delta
        if not passed:
            overall_ok = False
        actual_str = f"{delta:+.4f}" if delta is not None else "missing"
        checks.append(
            {
                "gate": "fail_over_ece_delta",
                "status": "pass" if passed else "fail",
                "threshold": fail_over_ece_delta,
                "actual": delta,
                "detail": f"ECE delta {actual_str} (max allowed: {fail_over_ece_delta:+.4f})",
            }
        )

    if fail_over_latency_ratio is not None:
        if cmp.timing_deltas is None or cmp.timing_deltas.get("mean_ms") is None:
            raise ComparisonError(
                "--fail-over-latency-ratio requires timing information in both reports"
            )
        ratio = cmp.timing_deltas["mean_ms"].ratio
        passed = ratio is not None and ratio <= fail_over_latency_ratio
        if not passed:
            overall_ok = False
        actual_str = f"{ratio:.3f}x" if ratio is not None else "missing"
        checks.append(
            {
                "gate": "fail_over_latency_ratio",
                "status": "pass" if passed else "fail",
                "threshold": fail_over_latency_ratio,
                "actual": ratio,
                "detail": f"Latency ratio {actual_str} (max: {fail_over_latency_ratio:.3f}x)",
            }
        )

    return overall_ok, checks


@eval_app.command("compare")
def compare_evaluation_reports(
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            "-b",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Baseline evaluation JSON report.",
        ),
    ],
    candidate: Annotated[
        Path,
        typer.Option(
            "--candidate",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Candidate evaluation JSON report to compare against baseline.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional path to write comparison JSON artifact.",
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing output artifact.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable comparison JSON to stdout.")
    ] = False,
    fail_under_miou_delta: Annotated[
        float | None,
        typer.Option(
            "--fail-under-miou-delta",
            help="Exit 1 if candidate mIoU - baseline mIoU is less than this value.",
        ),
    ] = None,
    fail_under_map_delta: Annotated[
        float | None,
        typer.Option(
            "--fail-under-map-delta",
            help="Exit 1 if candidate mAP - baseline mAP is less than this value.",
        ),
    ] = None,
    fail_over_ece_delta: Annotated[
        float | None,
        typer.Option(
            "--fail-over-ece-delta",
            help="Exit 1 if candidate ECE - baseline ECE exceeds this value.",
        ),
    ] = None,
    fail_over_latency_ratio: Annotated[
        float | None,
        typer.Option(
            "--fail-over-latency-ratio",
            help="Exit 1 if candidate/baseline mean latency ratio exceeds this value.",
        ),
    ] = None,
) -> None:
    """Compare two evaluation JSON reports and evaluate regression gates."""
    try:
        baseline_data = read_json(baseline)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        candidate_data = read_json(candidate)
    except ArtifactError as exc:
        typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    try:
        cmp = compare_evaluations(baseline_data, candidate_data)
        overall_ok, checks = _evaluate_gates(
            cmp,
            fail_under_miou_delta=fail_under_miou_delta,
            fail_under_map_delta=fail_under_map_delta,
            fail_over_ece_delta=fail_over_ece_delta,
            fail_over_latency_ratio=fail_over_latency_ratio,
        )
    except ComparisonError as exc:
        typer.secho(f"Compare error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    payload = cmp.as_dict()
    if checks:
        payload["gates"] = {
            "passed": overall_ok,
            "checks": checks,
        }

    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_comparison_summary(cmp, baseline, candidate)
        if checks:
            typer.echo("\nRegression Gate Checks")
            typer.echo("-" * 76)
            for c in checks:
                tag = "PASS" if c["status"] == "pass" else "FAIL"
                color = typer.colors.GREEN if c["status"] == "pass" else typer.colors.RED
                typer.secho(f"  [{tag}] {c['gate']}: {c['detail']}", fg=color)
            if not overall_ok:
                typer.secho(
                    "\nOne or more regression gate checks failed.",
                    fg=typer.colors.RED,
                    err=True,
                )
            else:
                typer.echo("\nAll regression gate checks passed.")

    if output is not None:
        try:
            write_json(output, payload, overwrite=overwrite)
        except ArtifactError as exc:
            typer.secho(f"Artifact error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        if not as_json:
            typer.echo(f"Wrote {output}")

    if not overall_ok:
        raise typer.Exit(code=1)


__all__ = ["eval_app"]
