"""Automated interactive evaluation HTML report dashboard generation.

Generates self-contained, responsive HTML evaluation reports combining summary metrics,
class-level performance cards, embedded SVG PR curves, reliability diagrams,
latency distribution charts, and comparative diff views with zero external CDN dependencies.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from nearfield360.perception.comparison import (
    ComparisonError,
    EvaluationComparison,
    compare_evaluations,
)
from nearfield360.robustness.plots import render_svg_bar_chart, render_svg_line_chart


class DashboardError(ValueError):
    """Raised when evaluation report artifacts cannot be parsed or rendered into a dashboard."""


def _format_val(val: float | int | None, precision: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, int):
        return f"{val:,}"
    if abs(val) >= 100:
        return f"{val:.1f}"
    return f"{val:.{precision}f}"


def _format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100.0:.2f}%"


def _format_signed_delta(delta: float | None, precision: int = 4) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.{precision}f}"


def _format_signed_pct(pct: float | None) -> str:
    if pct is None:
        return "-"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _format_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "-"
    return f"{ratio:.3f}x"


def _detect_task(report: Mapping[str, Any]) -> str:
    """Infer the task type (segmentation or detection) from report structure."""
    if "task" in report and isinstance(report["task"], str):
        return report["task"].lower()
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DashboardError("Report missing 'metrics' section")
    if "mean_iou" in metrics:
        return "segmentation"
    if "mean_average_precision" in metrics:
        return "detection"
    raise DashboardError(
        "Unrecognized evaluation report format: expected 'mean_iou' or "
        "'mean_average_precision' in metrics"
    )


def _render_detection_charts(confidence_analysis: Mapping[str, Any]) -> list[str]:
    """Generate inline SVG charts for detection confidence analysis."""
    charts: list[str] = []
    pr_curves = confidence_analysis.get("pr_curves")
    if isinstance(pr_curves, Mapping) and pr_curves:
        series: dict[str, list[tuple[float, float]]] = {}
        for cls_name, points in pr_curves.items():
            if not isinstance(points, Sequence):
                continue
            series[str(cls_name)] = [
                (float(p["recall"]), float(p["precision"]))
                for p in points
                if isinstance(p, Mapping) and "recall" in p and "precision" in p
            ]
        if series:
            charts.append(
                render_svg_line_chart(
                    "Detection Precision-Recall Curves",
                    "Recall",
                    "Precision",
                    series,
                    y_min=0.0,
                    y_max=1.0,
                )
            )

    thresholds = confidence_analysis.get("thresholds")
    if isinstance(thresholds, Sequence) and thresholds:
        operating: dict[str, list[tuple[float, float]]] = {
            "Precision": [],
            "Recall": [],
            "F1": [],
        }
        for point in thresholds:
            if isinstance(point, Mapping) and "confidence" in point:
                conf = float(point["confidence"])
                operating["Precision"].append((conf, float(point.get("precision", 0.0))))
                operating["Recall"].append((conf, float(point.get("recall", 0.0))))
                operating["F1"].append((conf, float(point.get("f1", 0.0))))
        charts.append(
            render_svg_line_chart(
                "Detection Operating Points",
                "Confidence Threshold",
                "Score",
                operating,
                y_min=0.0,
                y_max=1.0,
            )
        )
    return charts


def _render_segmentation_charts(confidence_analysis: Mapping[str, Any]) -> list[str]:
    """Generate inline SVG reliability diagram for segmentation confidence calibration."""
    charts: list[str] = []
    bins = confidence_analysis.get("bins")
    if isinstance(bins, Sequence) and bins:
        model_pts = [
            (float(b["mean_confidence"]), float(b["accuracy"]))
            for b in bins
            if isinstance(b, Mapping) and b.get("pixels", 0)
        ]
        if model_pts:
            charts.append(
                render_svg_line_chart(
                    "Segmentation Reliability Diagram",
                    "Mean Confidence",
                    "Accuracy",
                    {
                        "Model": model_pts,
                        "Perfect calibration": [(0.0, 0.0), (1.0, 1.0)],
                    },
                    y_min=0.0,
                    y_max=1.0,
                )
            )
    return charts


def _render_latency_chart(timing: Mapping[str, Any]) -> str | None:
    """Generate inline SVG latency distribution bar chart if timing metrics are present."""
    if not timing or int(timing.get("samples", 0)) <= 0:
        return None
    keys = ["min_ms", "p50_ms", "mean_ms", "p95_ms", "p99_ms", "max_ms"]
    labels = ["Min", "P50", "Mean", "P95", "P99", "Max"]
    values = [float(timing.get(k, 0.0)) for k in keys]
    fps = float(timing.get("samples_per_second", 0.0))
    samples = int(timing.get("samples", 0))
    return render_svg_bar_chart(
        f"Inference Latency Profile ({fps:.1f} FPS, {samples} samples)",
        "Latency Metric",
        "Latency (ms)",
        labels,
        values,
        bar_color="#38bdf8",
    )


def generate_evaluation_html_dashboard(
    report: Mapping[str, Any],
    *,
    baseline_report: Mapping[str, Any] | None = None,
    comparison: EvaluationComparison | Mapping[str, Any] | None = None,
    gate_results: Sequence[Mapping[str, Any]] | None = None,
    title: str | None = None,
) -> str:
    """Render a self-contained, responsive HTML evaluation dashboard with zero CDN dependencies.

    Parameters
    ----------
    report:
        Primary evaluation JSON report (candidate report).
    baseline_report:
        Optional baseline evaluation JSON report. When provided, automatically performs
        comparative analysis and embeds delta diff tables.
    comparison:
        Optional precomputed EvaluationComparison domain object or dictionary payload.
    gate_results:
        Optional list of regression gate evaluations from CLI gate checks.
    title:
        Optional custom title for the dashboard header.

    Returns
    -------
    str
        Self-contained, valid HTML5 report string.
    """
    if not isinstance(report, Mapping):
        raise DashboardError("Report artifact must be a mapping dictionary")

    # Check if report is directly a comparison artifact
    is_direct_comparison = "summary_deltas" in report and "task" in report
    if is_direct_comparison:
        task = str(report["task"]).lower()
        cmp_dict: Mapping[str, Any] | None = report
    else:
        task = _detect_task(report)
        cmp_dict = None

    # Resolve comparison if baseline report is provided
    if baseline_report is not None:
        if not isinstance(baseline_report, Mapping):
            raise DashboardError("Baseline report artifact must be a mapping dictionary")
        try:
            comparison_obj = compare_evaluations(dict(baseline_report), dict(report))
            cmp_dict = comparison_obj.as_dict()
        except ComparisonError as exc:
            raise DashboardError(f"Comparative dashboard error: {exc}") from exc
    elif comparison is not None:
        if isinstance(comparison, EvaluationComparison):
            cmp_dict = comparison.as_dict()
        elif isinstance(comparison, Mapping):
            cmp_dict = comparison
        else:
            raise DashboardError("Invalid comparison object type")

    # Resolve default title
    if title is None:
        task_label = "Semantic Segmentation" if task == "segmentation" else "Object Detection"
        title = (
            f"Comparative {task_label} Evaluation Report"
            if cmp_dict is not None
            else f"{task_label} Evaluation Report"
        )

    # Extract metadata & environment
    env = report.get("environment", {}) if not is_direct_comparison else {}
    samples_info = report.get("samples", {}) if not is_direct_comparison else {}
    model_info = report.get("model") if not is_direct_comparison else None
    split_info = report.get("split") if not is_direct_comparison else None
    timing_info = report.get("timing") if not is_direct_comparison else None
    metrics_info = report.get("metrics", {}) if not is_direct_comparison else {}

    # Extract charts
    charts: list[str] = []
    if not is_direct_comparison:
        conf_analysis = metrics_info.get("confidence_analysis")
        if isinstance(conf_analysis, Mapping):
            if task == "detection":
                charts.extend(_render_detection_charts(conf_analysis))
            elif task == "segmentation":
                charts.extend(_render_segmentation_charts(conf_analysis))

        if isinstance(timing_info, Mapping):
            lat_chart = _render_latency_chart(timing_info)
            if lat_chart:
                charts.append(lat_chart)

    # Build KPI summary cards
    kpi_cards: list[tuple[str, str, str | None]] = []
    if not is_direct_comparison:
        if task == "segmentation":
            miou_val = metrics_info.get("mean_iou")
            acc_val = metrics_info.get("pixel_accuracy")
            kpi_cards.append(
                ("Mean IoU (mIoU)", _format_pct(miou_val) if miou_val is not None else "N/A", None)
            )
            kpi_cards.append(
                (
                    "Pixel Accuracy",
                    _format_pct(acc_val) if acc_val is not None else "N/A",
                    None,
                )
            )
        else:
            map_val = metrics_info.get("mean_average_precision")
            iou_thresh = metrics_info.get("iou_threshold", 0.5)
            kpi_cards.append(
                (
                    f"mAP @ IoU {iou_thresh}",
                    _format_pct(map_val) if map_val is not None else "N/A",
                    None,
                )
            )

        # Confidence calibration KPI
        conf_analysis = metrics_info.get("confidence_analysis")
        if isinstance(conf_analysis, Mapping):
            if "ece" in conf_analysis:
                kpi_cards.append(
                    ("Expected Calibration Error", f"{float(conf_analysis['ece']):.4f}", None)
                )
            if "mean_confidence" in conf_analysis:
                kpi_cards.append(
                    (
                        "Mean Confidence",
                        _format_pct(float(conf_analysis["mean_confidence"])),
                        None,
                    )
                )

        # Timing KPI
        if isinstance(timing_info, Mapping) and timing_info.get("samples"):
            mean_ms = float(timing_info.get("mean_ms", 0.0))
            fps = float(timing_info.get("samples_per_second", 0.0))
            kpi_cards.append(("Mean Latency", f"{mean_ms:.2f} ms", f"{fps:.1f} FPS"))

        # Sample coverage KPI
        evaluated_cnt = samples_info.get("evaluated", 0)
        expected_cnt = samples_info.get("expected", evaluated_cnt)
        kpi_cards.append(("Samples Evaluated", f"{evaluated_cnt} / {expected_cnt}", None))

    # Build Class performance table
    class_rows: list[dict[str, Any]] = []
    if not is_direct_comparison:
        classes_dict = metrics_info.get("classes")
        if isinstance(classes_dict, Mapping):
            for cls_name, cls_data in classes_dict.items():
                if not isinstance(cls_data, Mapping):
                    continue
                score = (
                    cls_data.get("iou")
                    if task == "segmentation"
                    else cls_data.get("average_precision")
                )
                class_rows.append(
                    {
                        "name": cls_name,
                        "score": score,
                        "data": cls_data,
                    }
                )

    # HTML Construction
    header_meta = (
        f"Task: <strong>{html.escape(task.upper())}</strong> &bull; "
        f"NearField360 v{html.escape(str(env.get('nearfield360_version', '0.1.0')))} &bull; "
        f"Python {html.escape(str(env.get('python_version', '3.12')))} &bull; "
        f"Platform {html.escape(str(env.get('platform', 'unknown')))}"
    )

    doc: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"  <title>{html.escape(title)}</title>",
        "  <style>",
        "    :root {",
        "      --bg: #0f172a;",
        "      --card-bg: #1e293b;",
        "      --border: #334155;",
        "      --text: #f8fafc;",
        "      --text-muted: #94a3b8;",
        "      --primary: #38bdf8;",
        "      --danger: #f87171;",
        "      --success: #4ade80;",
        "      --warning: #fbbf24;",
        "      --bar-bg: #334155;",
        "    }",
        "    * { box-sizing: border-box; }",
        "    body {",
        '      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        "Helvetica, Arial, sans-serif;",
        "      background: var(--bg);",
        "      color: var(--text);",
        "      margin: 0;",
        "      padding: 24px;",
        "      line-height: 1.5;",
        "    }",
        "    .container { max-width: 1200px; margin: 0 auto; }",
        "    header { margin-bottom: 24px; }",
        "    h1 { font-size: 26px; font-weight: 700; margin: 0 0 8px 0; }",
        "    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }",
        "    .badge {",
        "      display: inline-block;",
        "      padding: 4px 10px;",
        "      font-size: 12px;",
        "      font-weight: 600;",
        "      border-radius: 9999px;",
        "      background: #334155;",
        "      color: var(--text);",
        "    }",
        "    .badge-primary { background: #0369a1; color: #e0f2fe; }",
        "    .badge-success { background: #15803d; color: #dcfce7; }",
        "    .badge-danger { background: #b91c1c; color: #fee2e2; }",
        "    .meta { color: var(--text-muted); font-size: 13px; margin-bottom: 20px; }",
        "    .kpi-grid {",
        "      display: grid;",
        "      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));",
        "      gap: 16px;",
        "      margin-bottom: 32px;",
        "    }",
        "    .kpi-card {",
        "      background: var(--card-bg);",
        "      border: 1px solid var(--border);",
        "      border-radius: 8px;",
        "      padding: 16px;",
        "    }",
        "    .kpi-title {",
        "      font-size: 11px;",
        "      color: var(--text-muted);",
        "      text-transform: uppercase;",
        "      font-weight: 600;",
        "      letter-spacing: 0.5px;",
        "    }",
        "    .kpi-value {",
        "      font-size: 24px;",
        "      font-weight: 700;",
        "      color: var(--primary);",
        "      margin-top: 4px;",
        "    }",
        "    .kpi-sub {",
        "      font-size: 12px;",
        "      color: var(--text-muted);",
        "      margin-top: 2px;",
        "    }",
        "    .card {",
        "      background: var(--card-bg);",
        "      border: 1px solid var(--border);",
        "      border-radius: 8px;",
        "      padding: 20px;",
        "      margin-bottom: 32px;",
        "      overflow-x: auto;",
        "    }",
        "    .card h2 {",
        "      font-size: 18px;",
        "      font-weight: 600;",
        "      margin-top: 0;",
        "      margin-bottom: 16px;",
        "      color: var(--text);",
        "    }",
        "    .chart-card {",
        "      background: #ffffff;",
        "      border-radius: 8px;",
        "      padding: 16px;",
        "      margin-bottom: 32px;",
        "      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);",
        "      overflow-x: auto;",
        "      display: flex;",
        "      justify-content: center;",
        "    }",
        "    table {",
        "      width: 100%;",
        "      border-collapse: collapse;",
        "      font-size: 13px;",
        "    }",
        "    th, td {",
        "      padding: 10px 12px;",
        "      text-align: left;",
        "      border-bottom: 1px solid var(--border);",
        "    }",
        "    th {",
        "      background: #0f172a;",
        "      color: var(--text-muted);",
        "      font-weight: 600;",
        "      text-transform: uppercase;",
        "      font-size: 11px;",
        "      letter-spacing: 0.5px;",
        "    }",
        "    tr:hover td { background: rgba(255, 255, 255, 0.02); }",
        "    .progress-bar-bg {",
        "      background: var(--bar-bg);",
        "      border-radius: 4px;",
        "      height: 8px;",
        "      width: 100%;",
        "      overflow: hidden;",
        "      margin-top: 4px;",
        "    }",
        "    .progress-bar-fill {",
        "      height: 100%;",
        "      border-radius: 4px;",
        "      background: var(--primary);",
        "    }",
        "    .delta-pos { color: var(--success); font-weight: 600; }",
        "    .delta-neg { color: var(--danger); font-weight: 600; }",
        "    .delta-neutral { color: var(--text-muted); }",
        "    .gate-pass { color: var(--success); font-weight: 600; }",
        "    .gate-fail { color: var(--danger); font-weight: 600; }",
        "  </style>",
        "</head>",
        "<body>",
        '  <div class="container">',
        "    <header>",
        f"      <h1>{html.escape(title)}</h1>",
        f'      <div class="meta">{header_meta}</div>',
        '      <div class="badges">',
        f'        <span class="badge badge-primary">{html.escape(task.upper())}</span>',
    ]

    if model_info and isinstance(model_info, Mapping):
        backend_str = str(model_info.get("backend", "opencv"))
        device_str = str(model_info.get("device", "cpu"))
        doc.append(f'        <span class="badge">Backend: {html.escape(backend_str)}</span>')
        doc.append(f'        <span class="badge">Device: {html.escape(device_str)}</span>')

    if split_info and isinstance(split_info, Mapping):
        split_name = str(split_info.get("split", "unknown"))
        doc.append(f'        <span class="badge">Split: {html.escape(split_name)}</span>')

    doc.extend(["      </div>", "    </header>"])

    # Regression Gate Checks & Comparative Diff Views
    if cmp_dict is not None:
        # Check for gates inside comparison or parameter
        effective_gates: Sequence[Mapping[str, Any]] = ()
        if gate_results:
            effective_gates = gate_results
        elif isinstance(cmp_dict.get("gates"), Mapping):
            gates_obj = cmp_dict["gates"]
            if isinstance(gates_obj.get("checks"), Sequence):
                effective_gates = gates_obj["checks"]

        if effective_gates:
            all_gates_pass = all(c.get("status") == "pass" for c in effective_gates)
            banner_class = "badge-success" if all_gates_pass else "badge-danger"
            status_text = "ALL GATES PASSED" if all_gates_pass else "GATE REGRESSION DETECTED"
            gate_header = (
                f"<h2>Regression Gate Verification: "
                f'<span class="badge {banner_class}">{status_text}</span></h2>'
            )
            doc.extend(
                [
                    '    <div class="card">',
                    f"      {gate_header}",
                    "      <table>",
                    "        <thead>",
                    "          <tr>",
                    "            <th>Gate</th>",
                    "            <th>Threshold</th>",
                    "            <th>Actual</th>",
                    "            <th>Status</th>",
                    "            <th>Detail</th>",
                    "          </tr>",
                    "        </thead>",
                    "        <tbody>",
                ]
            )
            for gate in effective_gates:
                g_status = gate.get("status", "pass")
                status_class = "gate-pass" if g_status == "pass" else "gate-fail"
                status_label = "[PASS]" if g_status == "pass" else "[FAIL]"
                gate_name_html = html.escape(str(gate.get("gate", "")))
                doc.append("          <tr>")
                doc.append(f"            <td><strong>{gate_name_html}</strong></td>")
                doc.append(f"            <td>{_format_val(gate.get('threshold'))}</td>")
                doc.append(f"            <td>{_format_val(gate.get('actual'))}</td>")
                doc.append(f'            <td class="{status_class}">{status_label}</td>')
                doc.append(f"            <td>{html.escape(str(gate.get('detail', '')))}</td>")
                doc.append("          </tr>")
            doc.extend(["        </tbody>", "      </table>", "    </div>"])

        # Comparative Summary Deltas
        sum_deltas = cmp_dict.get("summary_deltas")
        if isinstance(sum_deltas, Mapping) and sum_deltas:
            doc.extend(
                [
                    '    <div class="card">',
                    "      <h2>Summary Metric Deltas (Baseline vs Candidate)</h2>",
                    "      <table>",
                    "        <thead>",
                    "          <tr>",
                    "            <th>Metric</th>",
                    "            <th>Baseline</th>",
                    "            <th>Candidate</th>",
                    "            <th>Delta</th>",
                    "            <th>% Change</th>",
                    "          </tr>",
                    "        </thead>",
                    "        <tbody>",
                ]
            )
            for m_name, d_obj in sum_deltas.items():
                if isinstance(d_obj, Mapping):
                    delta_num = d_obj.get("delta")
                    delta_cls = (
                        "delta-pos"
                        if delta_num and delta_num > 0
                        else ("delta-neg" if delta_num and delta_num < 0 else "delta-neutral")
                    )
                    delta_str = _format_signed_delta(delta_num)
                    pct_str = _format_signed_pct(d_obj.get("percent_change"))
                    doc.append("          <tr>")
                    doc.append(f"            <td><strong>{html.escape(m_name)}</strong></td>")
                    doc.append(f"            <td>{_format_val(d_obj.get('baseline'))}</td>")
                    doc.append(f"            <td>{_format_val(d_obj.get('candidate'))}</td>")
                    doc.append(f'            <td class="{delta_cls}">{delta_str}</td>')
                    doc.append(f'            <td class="{delta_cls}">{pct_str}</td>')
                    doc.append("          </tr>")
            doc.extend(["        </tbody>", "      </table>", "    </div>"])

        # Class Breakdown Deltas
        cls_deltas = cmp_dict.get("class_deltas")
        if isinstance(cls_deltas, Mapping) and cls_deltas:
            metric_label = "IoU" if task == "segmentation" else "AP"
            doc.extend(
                [
                    '    <div class="card">',
                    f"      <h2>Class Performance Deltas ({metric_label})</h2>",
                    "      <table>",
                    "        <thead>",
                    "          <tr>",
                    "            <th>Class</th>",
                    "            <th>Baseline</th>",
                    "            <th>Candidate</th>",
                    "            <th>Delta</th>",
                    "            <th>% Change</th>",
                    "          </tr>",
                    "        </thead>",
                    "        <tbody>",
                ]
            )
            for c_name, d_obj in cls_deltas.items():
                if isinstance(d_obj, Mapping):
                    delta_num = d_obj.get("delta")
                    delta_cls = (
                        "delta-pos"
                        if delta_num and delta_num > 0
                        else ("delta-neg" if delta_num and delta_num < 0 else "delta-neutral")
                    )
                    delta_str = _format_signed_delta(delta_num)
                    pct_str = _format_signed_pct(d_obj.get("percent_change"))
                    doc.append("          <tr>")
                    doc.append(f"            <td><strong>{html.escape(c_name)}</strong></td>")
                    doc.append(f"            <td>{_format_val(d_obj.get('baseline'))}</td>")
                    doc.append(f"            <td>{_format_val(d_obj.get('candidate'))}</td>")
                    doc.append(f'            <td class="{delta_cls}">{delta_str}</td>')
                    doc.append(f'            <td class="{delta_cls}">{pct_str}</td>')
                    doc.append("          </tr>")
            doc.extend(["        </tbody>", "      </table>", "    </div>"])

        # Latency / Timing Deltas
        tim_deltas = cmp_dict.get("timing_deltas")
        if isinstance(tim_deltas, Mapping) and tim_deltas:
            doc.extend(
                [
                    '    <div class="card">',
                    "      <h2>Timing & Latency Comparison</h2>",
                    "      <table>",
                    "        <thead>",
                    "          <tr>",
                    "            <th>Timing Metric</th>",
                    "            <th>Baseline</th>",
                    "            <th>Candidate</th>",
                    "            <th>Delta (ms)</th>",
                    "            <th>Latency Ratio</th>",
                    "          </tr>",
                    "        </thead>",
                    "        <tbody>",
                ]
            )
            for t_name, d_obj in tim_deltas.items():
                if isinstance(d_obj, Mapping):
                    delta_num = d_obj.get("delta")
                    ratio_num = d_obj.get("ratio")
                    # For latency, lower is better: delta < 0 is good, delta > 0 is bad
                    lat_cls = (
                        "delta-pos"
                        if delta_num and delta_num < 0
                        else ("delta-neg" if delta_num and delta_num > 0 else "delta-neutral")
                    )
                    b_val = _format_val(d_obj.get("baseline"), precision=2)
                    c_val = _format_val(d_obj.get("candidate"), precision=2)
                    d_val = _format_signed_delta(delta_num, precision=2)
                    r_val = _format_ratio(ratio_num)
                    doc.append("          <tr>")
                    doc.append(f"            <td><strong>{html.escape(t_name)}</strong></td>")
                    doc.append(f"            <td>{b_val}</td>")
                    doc.append(f"            <td>{c_val}</td>")
                    doc.append(f'            <td class="{lat_cls}">{d_val}</td>')
                    doc.append(f'            <td class="{lat_cls}">{r_val}</td>')
                    doc.append("          </tr>")
            doc.extend(["        </tbody>", "      </table>", "    </div>"])

    # KPI Summary Cards (for primary/candidate run)
    if kpi_cards:
        doc.append('    <div class="kpi-grid">')
        for card_title, card_val, card_sub in kpi_cards:
            doc.append('      <div class="kpi-card">')
            doc.append(f'        <div class="kpi-title">{html.escape(card_title)}</div>')
            doc.append(f'        <div class="kpi-value">{html.escape(card_val)}</div>')
            if card_sub:
                doc.append(f'        <div class="kpi-sub">{html.escape(card_sub)}</div>')
            doc.append("      </div>")
        doc.append("    </div>")

    # Embedded SVG Charts
    doc.extend(
        f'    <div class="chart-card">\n      {chart_svg}\n    </div>' for chart_svg in charts
    )

    # Per-Class Performance Table (Candidate)
    if class_rows:
        score_metric = "IoU" if task == "segmentation" else "AP"
        doc.extend(
            [
                '    <div class="card">',
                f"      <h2>Per-Class Performance Overview ({score_metric})</h2>",
                "      <table>",
                "        <thead>",
                "          <tr>",
                "            <th style='width: 25%;'>Class</th>",
                f"            <th style='width: 15%;'>{score_metric}</th>",
                "            <th style='width: 35%;'>Performance Meter</th>",
                "            <th style='width: 25%;'>Details / Support</th>",
                "          </tr>",
                "        </thead>",
                "        <tbody>",
            ]
        )
        for row in sorted(
            class_rows,
            key=lambda r: float(r["score"]) if r["score"] is not None else -1.0,
            reverse=True,
        ):
            c_name = row["name"]
            score = row["score"]
            c_data = row["data"]
            score_str = _format_pct(score) if score is not None else "N/A"
            score_pct_float = (
                min(100.0, max(0.0, float(score) * 100.0)) if score is not None else 0.0
            )

            # Determine progress bar fill color
            fill_color = "var(--primary)"
            if score is not None:
                if score >= 0.70:
                    fill_color = "var(--success)"
                elif score < 0.40:
                    fill_color = "var(--danger)"

            # Support detail string
            if task == "segmentation":
                px_count = c_data.get("pixels", c_data.get("pixel_count"))
                detail_str = f"{px_count:,} pixels" if px_count is not None else "-"
            else:
                preds = c_data.get("predictions", "-")
                targs = c_data.get("targets", "-")
                detail_str = f"{preds} preds / {targs} targets"

            bar_html = (
                f'              <div class="progress-bar-bg">\n'
                f'                <div class="progress-bar-fill" style="width: '
                f'{score_pct_float:.1f}%; background: {fill_color};"></div>\n'
                f"              </div>"
            )

            doc.append("          <tr>")
            doc.append(f"            <td><strong>{html.escape(c_name)}</strong></td>")
            doc.append(f"            <td><strong>{score_str}</strong></td>")
            doc.append(f"            <td>\n{bar_html}\n            </td>")
            doc.append(f"            <td>{html.escape(str(detail_str))}</td>")
            doc.append("          </tr>")
        doc.extend(["        </tbody>", "      </table>", "    </div>"])

    # Closing document
    doc.extend(
        [
            "  </div>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(doc)


__all__ = [
    "DashboardError",
    "generate_evaluation_html_dashboard",
]
