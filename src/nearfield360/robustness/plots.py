"""Automated diagnostic plotting and visual HTML report generation.

Generates publication-quality SVG vector plots, standalone raster PNG charts,
and a self-contained offline HTML dashboard summarizing sensor corruption and
calibration sensitivity benchmarks without external heavy plotting dependencies.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Modern aesthetic palette for multi-series plots
PALETTE = (
    "#2563eb",  # Blue
    "#dc2626",  # Red
    "#16a34a",  # Green
    "#d97706",  # Amber
    "#9333ea",  # Purple
    "#0891b2",  # Cyan
)


def _format_num(val: float) -> str:
    if abs(val) < 1e-4 and val != 0.0:
        return f"{val:.2e}"
    if abs(val) >= 100:
        return f"{val:.1f}"
    return f"{val:.3f}".rstrip("0").rstrip(".") if "." in f"{val:.3f}" else f"{val:.1f}"


def render_svg_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series_dict: Mapping[str, Sequence[tuple[float, float]]],
    *,
    width: int = 760,
    height: int = 440,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """Render a responsive multi-series vector line plot as an SVG string.

    Parameters
    ----------
    title:
        Main plot title.
    x_label:
        Label along horizontal axis.
    y_label:
        Label along vertical axis.
    series_dict:
        Mapping of series names to sequences of ``(x, y)`` data points.
    width:
        SVG canvas width in pixels.
    height:
        SVG canvas height in pixels.
    y_min:
        Optional fixed lower bound for Y axis.
    y_max:
        Optional fixed upper bound for Y axis.

    Returns
    -------
    str
        Valid SVG document string.
    """
    margin_l = 80
    margin_r = 160
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x: list[float] = []
    all_y: list[float] = []
    for pts in series_dict.values():
        for x, y in pts:
            if np.isfinite(x) and np.isfinite(y):
                all_x.append(float(x))
                all_y.append(float(y))

    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 1.0
    if min_x == max_x:
        max_x += 1.0

    calc_y_min = min(all_y) if all_y else 0.0
    calc_y_max = max(all_y) if all_y else 1.0
    if calc_y_min == calc_y_max:
        calc_y_max += 1.0

    eff_y_min = y_min if y_min is not None else calc_y_min
    eff_y_max = y_max if y_max is not None else calc_y_max
    if eff_y_min == eff_y_max:
        eff_y_max += 1.0

    def to_svg_x(x: float) -> float:
        return margin_l + ((x - min_x) / (max_x - min_x)) * plot_w

    def to_svg_y(y: float) -> float:
        return margin_t + plot_h - ((y - eff_y_min) / (eff_y_max - eff_y_min)) * plot_h

    # Build SVG elements
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="font-family: -apple-system, BlinkMacSystemFont, '
        f'Segoe UI, Roboto, Helvetica, Arial, sans-serif; background: #ffffff;">',
        # Background and grid
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="36" font-size="18" font-weight="600" fill="#1e293b">'
        f"{html.escape(title)}</text>",
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>',
    ]

    # Y-axis ticks and horizontal gridlines (5 ticks)
    for i in range(6):
        val = eff_y_min + (i / 5.0) * (eff_y_max - eff_y_min)
        y_pos = to_svg_y(val)
        svg.append(
            f'<line x1="{margin_l}" y1="{y_pos:.1f}" x2="{margin_l + plot_w}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'<text x="{margin_l - 12}" y="{y_pos + 4:.1f}" font-size="11" fill="#64748b" '
            f'text-anchor="end">{_format_num(val)}</text>'
        )

    # X-axis ticks (5 ticks)
    for i in range(6):
        val = min_x + (i / 5.0) * (max_x - min_x)
        x_pos = to_svg_x(val)
        svg.append(
            f'<line x1="{x_pos:.1f}" y1="{margin_t}" x2="{x_pos:.1f}" y2="{margin_t + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'<text x="{x_pos:.1f}" y="{margin_t + plot_h + 20}" font-size="11" fill="#64748b" '
            f'text-anchor="middle">{_format_num(val)}</text>'
        )

    # Axis titles
    svg.append(
        f'<text x="{margin_l + plot_w / 2}" y="{height - 18}" font-size="13" font-weight="500" '
        f'fill="#334155" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="24" y="{margin_t + plot_h / 2}" font-size="13" font-weight="500" '
        f'fill="#334155" text-anchor="middle" transform="rotate(-90 24 {margin_t + plot_h / 2})">'
        f"{html.escape(y_label)}</text>"
    )

    # Series lines, points, and legend
    legend_x = margin_l + plot_w + 24
    legend_y = margin_t + 16

    for s_idx, (name, points) in enumerate(series_dict.items()):
        color = PALETTE[s_idx % len(PALETTE)]
        valid_pts = [
            (to_svg_x(x), to_svg_y(y))
            for x, y in points
            if np.isfinite(x) and np.isfinite(y)
        ]
        if not valid_pts:
            continue

        # Line path
        d_str = " ".join(
            f"{'M' if idx == 0 else 'L'} {px:.1f} {py:.1f}"
            for idx, (px, py) in enumerate(valid_pts)
        )
        svg.append(
            f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

        # Marker dots
        for px, py in valid_pts:
            svg.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.5"/>'
            )

        # Legend entry
        entry_y = legend_y + s_idx * 24
        svg.append(
            f'<line x1="{legend_x}" y1="{entry_y}" x2="{legend_x + 18}" y2="{entry_y}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        svg.append(
            f'<circle cx="{legend_x + 9}" cy="{entry_y}" r="3.5" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 26}" y="{entry_y + 4}" font-size="12" fill="#334155">'
            f"{html.escape(name)}</text>"
        )

    svg.append("</svg>")
    return "\n".join(svg)


def render_raster_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series_dict: Mapping[str, Sequence[tuple[float, float]]],
    output_path: Path,
    *,
    width: int = 800,
    height: int = 500,
) -> None:
    """Render a raster PNG line chart using OpenCV canvas operations."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    margin_l = 80
    margin_r = 160
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x: list[float] = []
    all_y: list[float] = []
    for pts in series_dict.values():
        for x, y in pts:
            if np.isfinite(x) and np.isfinite(y):
                all_x.append(float(x))
                all_y.append(float(y))

    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 1.0
    if min_x == max_x:
        max_x += 1.0

    min_y = min(all_y) if all_y else 0.0
    max_y = max(all_y) if all_y else 1.0
    if min_y == max_y:
        max_y += 1.0

    def to_cv_x(x: float) -> int:
        return int(margin_l + ((x - min_x) / (max_x - min_x)) * plot_w)

    def to_cv_y(y: float) -> int:
        return int(margin_t + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h)

    # Plot background and bounding box
    cv2.rectangle(
        canvas,
        (margin_l, margin_t),
        (margin_l + plot_w, margin_t + plot_h),
        (245, 245, 245),
        thickness=-1,
    )
    cv2.rectangle(
        canvas,
        (margin_l, margin_t),
        (margin_l + plot_w, margin_t + plot_h),
        (210, 210, 210),
        thickness=1,
    )

    # Title
    cv2.putText(
        canvas,
        title,
        (margin_l, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # Ticks along Y
    for i in range(6):
        val = min_y + (i / 5.0) * (max_y - min_y)
        y_pos = to_cv_y(val)
        cv2.line(canvas, (margin_l, y_pos), (margin_l + plot_w, y_pos), (230, 230, 230), 1)
        cv2.putText(
            canvas,
            _format_num(val),
            (margin_l - 60, y_pos + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
            lineType=cv2.LINE_AA,
        )

    # Ticks along X
    for i in range(6):
        val = min_x + (i / 5.0) * (max_x - min_x)
        x_pos = to_cv_x(val)
        cv2.line(canvas, (x_pos, margin_t), (x_pos, margin_t + plot_h), (230, 230, 230), 1)
        cv2.putText(
            canvas,
            _format_num(val),
            (x_pos - 15, margin_t + plot_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
            lineType=cv2.LINE_AA,
        )

    # Axis labels
    cv2.putText(
        canvas,
        x_label,
        (margin_l + plot_w // 2 - 40, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 50, 50),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        y_label,
        (15, margin_t + plot_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 50, 50),
        1,
        lineType=cv2.LINE_AA,
    )

    # Palette in BGR for OpenCV
    bgr_palette = [
        (235, 99, 37),   # Blue
        (38, 38, 220),   # Red
        (74, 163, 22),   # Green
        (6, 119, 217),   # Amber
        (234, 51, 147),  # Purple
    ]

    legend_x = margin_l + plot_w + 15
    legend_y = margin_t + 20

    for s_idx, (name, points) in enumerate(series_dict.items()):
        color = bgr_palette[s_idx % len(bgr_palette)]
        valid_pts = [
            (to_cv_x(x), to_cv_y(y))
            for x, y in points
            if np.isfinite(x) and np.isfinite(y)
        ]
        for idx in range(len(valid_pts) - 1):
            cv2.line(canvas, valid_pts[idx], valid_pts[idx + 1], color, 2, lineType=cv2.LINE_AA)
        for pt in valid_pts:
            cv2.circle(canvas, pt, 4, color, -1, lineType=cv2.LINE_AA)

        # Legend
        entry_y = legend_y + s_idx * 24
        cv2.line(
            canvas,
            (legend_x, entry_y),
            (legend_x + 18, entry_y),
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(canvas, (legend_x + 9, entry_y), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            name[:15],
            (legend_x + 24, entry_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def generate_robustness_html_dashboard(
    report_dict: dict[str, Any],
    *,
    title: str = "NearField360 Robustness & Sensitivity Report",
) -> str:
    """Generate a responsive, offline-ready HTML report embedding interactive SVG charts."""
    env = report_dict.get("environment", {})
    corr_sweeps = report_dict.get("corruption_sweeps", [])
    calib_sweeps = report_dict.get("calibration_sweeps", [])

    # Group corruption series: MAE vs Severity
    corr_series: dict[str, list[tuple[float, float]]] = {}
    for rec in corr_sweeps:
        c_type = rec["corruption"]
        sev = float(rec["severity"])
        mae = float(rec["occupancy_metrics"]["mae"])
        corr_series.setdefault(c_type, []).append((sev, mae))

    # Group calibration series: MAE vs Perturbation Magnitude
    calib_series: dict[str, list[tuple[float, float]]] = {}
    for rec in calib_sweeps:
        axis = f"{rec['axis']} ({rec['unit']})"
        mag = float(rec["magnitude"])
        mae = float(rec["occupancy_metrics"]["mae"])
        calib_series.setdefault(axis, []).append((mag, mae))

    svg_corr = render_svg_line_chart(
        title="BEV Occupancy Degradation (MAE) vs. Sensor Corruption Severity",
        x_label="Corruption Severity (1 to 5)",
        y_label="Occupancy Mean Absolute Error (MAE)",
        series_dict=corr_series,
    )

    svg_calib = render_svg_line_chart(
        title="BEV Occupancy Degradation (MAE) vs. Calibration Misalignment",
        x_label="Perturbation Magnitude",
        y_label="Occupancy Mean Absolute Error (MAE)",
        series_dict=calib_series,
    )

    # Compute key performance indicators
    max_corr_mae = max(
        (rec["occupancy_metrics"]["mae"] for rec in corr_sweeps), default=0.0
    )
    max_calib_mae = max(
        (rec["occupancy_metrics"]["mae"] for rec in calib_sweeps), default=0.0
    )
    worst_corr = max(
        corr_sweeps, key=lambda r: r["occupancy_metrics"]["mae"], default=None
    )
    worst_corr_name = (
        f"{worst_corr['corruption']} (sev {worst_corr['severity']})"
        if worst_corr
        else "N/A"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --danger: #f87171;
      --success: #4ade80;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .meta {{
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 24px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .kpi-title {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .kpi-value {{
      font-size: 24px;
      font-weight: 700;
      color: var(--primary);
      margin-top: 4px;
    }}
    .chart-card {{
      background: #ffffff;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 32px;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #1e293b;
      color: var(--text-muted);
      font-weight: 600;
    }}
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{html.escape(title)}</h1>
    <div class="meta">
      NearField360 v{html.escape(str(env.get('nearfield360_version', '0.1.0')))} &bull;
      Python {html.escape(str(env.get('python_version', '3.12')))} &bull;
      Platform {html.escape(str(env.get('platform', 'unknown')))}
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Max Corruption Error</div>
        <div class="kpi-value">{max_corr_mae:.3f} MAE</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Max Calibration Error</div>
        <div class="kpi-value">{max_calib_mae:.3f} MAE</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Worst Corruption Impact</div>
        <div class="kpi-value" style="font-size: 18px; color: var(--danger);">
          {html.escape(worst_corr_name)}
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Calibration Sweeps Evaluated</div>
        <div class="kpi-value">{len(calib_sweeps)}</div>
      </div>
    </div>

    <div class="chart-card">
      {svg_corr}
    </div>

    <div class="chart-card">
      {svg_calib}
    </div>

    <div class="table-container">
      <h3 style="margin-top: 0;">Sensor Corruption Summary</h3>
      <table>
        <thead>
          <tr>
            <th>Corruption</th>
            <th>Severity</th>
            <th>Occupancy MAE</th>
            <th>Occupied IoU</th>
            <th>Free IoU</th>
            <th>Uncertainty Shift</th>
          </tr>
        </thead>
        <tbody>
"""
    for rec in corr_sweeps:
        occ = rec["occupancy_metrics"]
        html_content += f"""          <tr>
            <td>{html.escape(str(rec['corruption']))}</td>
            <td>{rec['severity']}</td>
            <td>{occ['mae']:.4f}</td>
            <td>{occ['occupied_iou']:.4f}</td>
            <td>{occ['free_iou']:.4f}</td>
            <td>{occ['uncertainty_shift']:+.4f}</td>
          </tr>
"""

    html_content += """        </tbody>
      </table>
    </div>

    <div class="table-container">
      <h3 style="margin-top: 0;">Extrinsic Calibration Sensitivity Summary</h3>
      <table>
        <thead>
          <tr>
            <th>Axis</th>
            <th>Unit</th>
            <th>Magnitude</th>
            <th>Occupancy MAE</th>
            <th>Occupied IoU</th>
            <th>Free IoU</th>
            <th>Uncertainty Shift</th>
          </tr>
        </thead>
        <tbody>
"""
    for rec in calib_sweeps:
        occ = rec["occupancy_metrics"]
        html_content += f"""          <tr>
            <td>{html.escape(str(rec['axis']))}</td>
            <td>{html.escape(str(rec['unit']))}</td>
            <td>{rec['magnitude']}</td>
            <td>{occ['mae']:.4f}</td>
            <td>{occ['occupied_iou']:.4f}</td>
            <td>{occ['free_iou']:.4f}</td>
            <td>{occ['uncertainty_shift']:+.4f}</td>
          </tr>
"""

    html_content += """        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    return html_content


__all__ = [
    "generate_robustness_html_dashboard",
    "render_raster_line_chart",
    "render_svg_line_chart",
]
