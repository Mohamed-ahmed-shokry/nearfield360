"""Reproducible evaluation commands that publish JSON metric artifacts."""

from __future__ import annotations

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
    DetectionAnnotationError,
    load_detection_annotations,
    load_detection_predictions,
)
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticMaskError,
    load_semantic_mask,
)
from nearfield360.data.woodscape import IMAGE_SUFFIXES, WoodScapeDataset
from nearfield360.perception.evaluation import (
    environment_metadata,
    evaluate_detection,
    evaluate_semantic,
)
from nearfield360.perception.inference.backend import InferenceError
from nearfield360.perception.inference.models import InferenceBackendType, InferenceDevice
from nearfield360.perception.inference.preprocessor import PreprocessorError
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.utils.artifacts import ArtifactError, write_json

eval_app = typer.Typer(
    help="Reproducible perception evaluation against labelled WoodScape data.",
    no_args_is_help=True,
)

PredictionsOption = Annotated[
    Path,
    typer.Option(
        "--predictions",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory of per-sample prediction files.",
    ),
]

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
) -> dict[str, Any]:
    state = get_state(context)
    return {
        "environment": environment_metadata(),
        "config": state.config.model_dump(mode="json"),
        "seed": state.config.runtime.seed,
        "model": model,
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
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
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
            predicted, _confidence = engine.predict(image)
        except (ImageReadError, SemanticMaskError, InferenceError, PreprocessorError) as exc:
            _mask_error(sample.key.stem, exc)
        pairs.append((predicted, target))
    return pairs, expected


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
) -> None:
    """Score predicted masks against WoodScape semantic ground truth."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    missing: list[str] = []
    expected = 0
    model_info: dict[str, Any] | None = None
    if model is not None:
        if predictions is not None:
            _reject_source()
        dataset = discover_dataset(context, root)
        engine = _build_segmentation_engine(context, model, backend=backend, device=device)
        pairs, expected = _segmentation_model_pairs(engine, dataset, limit)
        model_info = {
            "path": str(model),
            "backend": resolve_backend_type(context, backend).value,
            "device": resolve_device(context, device).value,
        }
    else:
        if predictions is None:
            _reject_source()
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
    )
    _writing_output(output, overwrite, payload)
    typer.echo(f"Evaluated {len(pairs)}/{expected} samples; mIoU={evaluation.mean_iou:.3f}")


@eval_app.command("detection")
def evaluate_detection_command(
    context: typer.Context,
    predictions: PredictionsOption,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="JSON report artifact (created atomically; refuses to overwrite).",
        ),
    ],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing report artifact.")
    ] = False,
    iou_threshold: ThresholdOption = 0.5,
    root: DatasetRootOption = None,
    limit: LimitOption = 0,
) -> None:
    """Score detected boxes (``*.txt``) against WoodScape detection annotations."""
    dataset = discover_dataset(context, root)
    prediction_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    target_batches: list[list[Any]] = []
    missing: list[str] = []
    expected = 0
    for sample in dataset:
        if sample.detection_path is None:
            continue
        if limit > 0 and expected >= limit:
            break
        expected += 1
        prediction_file = _find_prediction_file(predictions, sample.key.stem, frozenset({".txt"}))
        if prediction_file is None:
            missing.append(sample.key.stem)
            continue
        try:
            image = load_rgb_image(sample.image_path)
            image_size = (int(image.shape[0]), int(image.shape[1]))
            targets = load_detection_annotations(sample.detection_path, image_size=image_size)
            predictions_list = load_detection_predictions(prediction_file, image_size=image_size)
        except (ImageReadError, DetectionAnnotationError) as exc:
            typer.secho(
                f"Detection error for {sample.key.stem}: {exc}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1) from None
        boxes = np.asarray([item.xyxy for item in predictions_list], dtype=np.float64)
        scores = np.asarray([item.score for item in predictions_list], dtype=np.float64)
        classes = np.asarray([item.class_id for item in predictions_list], dtype=np.int64)
        prediction_batches.append((boxes, scores, classes))
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
    except ValueError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    payload = _report_payload(
        context,
        expected=expected,
        evaluated=len(prediction_batches),
        missing=missing,
        metrics=evaluation.as_dict(),
    )
    _writing_output(output, overwrite, payload)
    typer.echo(
        f"Evaluated {len(prediction_batches)}/{expected} samples; "
        f"mAP={evaluation.mean_average_precision:.3f}"
    )


__all__ = ["eval_app"]
