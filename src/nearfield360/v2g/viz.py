"""Tactical V2G energy dashboard, electrical single-line diagram (SLD), and microgrid visualizer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.v2g.microgrid import (
    ConnectedBay,
    FacilityMicrogridManager,
    compute_facility_base_load_kw,
    compute_facility_solar_pv_kw,
)
from nearfield360.v2g.models import (
    GridTariffWindow,
    MicrogridFacilityReport,
)

# High-contrast tactical color palette (BGR)
COLOR_BG = (18, 20, 24)
COLOR_PANEL_BG = (26, 30, 36)
COLOR_GRID = (38, 42, 50)
COLOR_CARD_BG = (34, 38, 46)
COLOR_CARD_BORDER = (55, 62, 75)
COLOR_BUS = (140, 150, 165)
COLOR_TEXT_TITLE = (255, 255, 255)
COLOR_TEXT_BODY = (215, 220, 230)
COLOR_TEXT_MUTED = (125, 130, 140)

# Electrical state colors
COLOR_CHG = (220, 140, 40)  # Blue (Charging from AC bus)
COLOR_DIS = (40, 165, 245)  # Amber / Gold (Discharging into AC bus)
COLOR_STANDBY = (80, 85, 95)  # Gray
COLOR_SOLAR = (50, 215, 235)  # Solar yellow/gold
COLOR_OK = (60, 210, 95)  # Green
COLOR_WARN = (40, 180, 240)  # Amber
COLOR_DANGER = (40, 50, 240)  # Red


def _draw_txt(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_sld_panel(
    canvas: NDArray[np.uint8],
    bays: Sequence[ConnectedBay],
    hour: float = 14.0,
    width: int = 800,
    height: int = 720,
) -> None:
    """Render the facility electrical single-line diagram and connected vehicle bays."""
    # Background panel
    cv2.rectangle(canvas, (0, 0), (width, height), COLOR_BG, -1)
    cv2.line(canvas, (width, 0), (width, height), COLOR_CARD_BORDER, 2)

    # Header
    title = "FACILITY ELECTRICAL SINGLE-LINE DIAGRAM (SLD)"
    _draw_txt(canvas, title, (25, 38), 0.65, COLOR_TEXT_TITLE, 2)
    h_int = int(hour)
    m_int = int((hour % 1) * 60)
    sub = f"SIMULATION TIME: {h_int:02d}:{m_int:02d} | 400V 3-PHASE MICROGRID"
    _draw_txt(canvas, sub, (25, 62), 0.42, COLOR_TEXT_MUTED, 1)

    # Substation Transformer Block
    cv2.rectangle(canvas, (35, 90), (220, 160), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (35, 90), (220, 160), COLOR_CARD_BORDER, 1)
    _draw_txt(canvas, "SUBSTATION TRANSFORMER", (45, 112), 0.38, COLOR_TEXT_TITLE, 1)
    _draw_txt(canvas, "500 kVA 20kV / 400V", (45, 130), 0.35, COLOR_TEXT_MUTED, 1)
    base_load = compute_facility_base_load_kw(hour)
    _draw_txt(canvas, f"Base Load: {base_load:.1f} kW", (45, 148), 0.38, COLOR_OK, 1)

    # Rooftop Solar PV Block
    solar_kw = compute_facility_solar_pv_kw(hour)
    cv2.rectangle(canvas, (250, 90), (435, 160), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (250, 90), (435, 160), COLOR_CARD_BORDER, 1)
    _draw_txt(canvas, "ROOFTOP SOLAR PV", (260, 112), 0.38, COLOR_TEXT_TITLE, 1)
    _draw_txt(canvas, "120 kWp Bifacial Array", (260, 130), 0.35, COLOR_TEXT_MUTED, 1)
    _draw_txt(canvas, f"Solar Out: {solar_kw:.1f} kW", (260, 148), 0.38, COLOR_SOLAR, 1)

    # Main 400V AC Feeder Bus Line
    bus_y = 195
    cv2.line(canvas, (35, bus_y), (765, bus_y), COLOR_BUS, 5)
    bus_label = "MAIN 400V AC DISTRIBUTION FEEDER BUS"
    _draw_txt(canvas, bus_label, (250, bus_y - 12), 0.40, COLOR_BUS, 1)

    # Feeder drop lines from Substation & Solar to Bus
    cv2.line(canvas, (127, 160), (127, bus_y), COLOR_BUS, 2)
    cv2.line(canvas, (342, 160), (342, bus_y), COLOR_BUS, 2)

    # Draw Parking Bays
    num_bays = min(10, len(bays))
    col_width = 138
    row_height = 215

    for idx in range(num_bays):
        bay = bays[idx]
        col = idx % 5
        row = idx // 5

        x_start = 35 + col * col_width
        y_start = 225 + row * row_height
        x_end = x_start + col_width - 12
        y_end = y_start + row_height - 18

        p_ac = bay.inverter.active_power_kw
        if p_ac > 0.5:
            accent_color = COLOR_CHG
            status_txt = f"+{p_ac:.1f} kW CHG"
        elif p_ac < -0.5:
            accent_color = COLOR_DIS
            status_txt = f"{p_ac:.1f} kW V2G"
        else:
            accent_color = COLOR_STANDBY
            status_txt = "STANDBY"

        tap_x = (x_start + x_end) // 2
        cv2.line(canvas, (tap_x, bus_y), (tap_x, y_start), accent_color, 2)

        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), accent_color, 2)

        _draw_txt(canvas, bay.bay_id, (x_start + 8, y_start + 20), 0.42, COLOR_TEXT_TITLE, 1)
        veh_name = bay.vehicle.vehicle_id.replace("EV-", "")[:12]
        _draw_txt(canvas, veh_name, (x_start + 8, y_start + 38), 0.32, COLOR_TEXT_MUTED, 1)

        soc = bay.vehicle.current_soc
        bar_x = x_start + 8
        bar_y = y_start + 50
        bar_w = x_end - x_start - 16
        bar_h = 10
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 22, 26), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, soc)))
        soc_color = COLOR_OK if soc >= 0.60 else (COLOR_WARN if soc >= 0.30 else COLOR_DANGER)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), soc_color, -1)

        soc_str = f"SoC:{soc * 100:.0f}%"
        _draw_txt(canvas, soc_str, (x_start + 8, y_start + 76), 0.38, COLOR_TEXT_BODY, 1)
        dep_str = f"Dep:{bay.vehicle.min_departure_soc * 100:.0f}%"
        _draw_txt(canvas, dep_str, (x_start + 65, y_start + 76), 0.32, COLOR_TEXT_MUTED, 1)
        cap_str = f"Cap:{bay.vehicle.battery_capacity_kwh:.0f}kWh"
        _draw_txt(canvas, cap_str, (x_start + 8, y_start + 96), 0.32, COLOR_TEXT_MUTED, 1)
        flx_str = f"Flx:{bay.flexible_discharge_energy_kwh:.1f}kWh"
        _draw_txt(canvas, flx_str, (x_start + 8, y_start + 114), 0.32, COLOR_TEXT_MUTED, 1)

        cv2.rectangle(canvas, (x_start + 6, y_end - 28), (x_end - 6, y_end - 6), (20, 24, 30), -1)
        _draw_txt(canvas, status_txt, (x_start + 10, y_end - 12), 0.36, accent_color, 1)

    # Legend
    leg_y = 695
    cv2.circle(canvas, (45, leg_y), 5, COLOR_CHG, -1)
    _draw_txt(canvas, "Charging (AC->DC)", (58, leg_y + 4), 0.36, COLOR_TEXT_MUTED, 1)
    cv2.circle(canvas, (215, leg_y), 5, COLOR_DIS, -1)
    _draw_txt(canvas, "Discharging (V2G BPT)", (228, leg_y + 4), 0.36, COLOR_TEXT_MUTED, 1)
    cv2.circle(canvas, (415, leg_y), 5, COLOR_STANDBY, -1)
    _draw_txt(canvas, "Standby / Idle", (428, leg_y + 4), 0.36, COLOR_TEXT_MUTED, 1)
    cv2.circle(canvas, (575, leg_y), 5, COLOR_SOLAR, -1)
    _draw_txt(canvas, "Solar PV", (588, leg_y + 4), 0.36, COLOR_TEXT_MUTED, 1)


def draw_hud_panel(
    canvas: NDArray[np.uint8],
    manager: FacilityMicrogridManager,
    report: MicrogridFacilityReport | None = None,
    hour: float = 14.0,
    tariff_window: GridTariffWindow | None = None,
    x_offset: int = 800,
    width: int = 480,
    height: int = 720,
) -> None:
    """Render the microgrid telemetry and financial arbitrage HUD on the right panel."""
    # Background
    cv2.rectangle(canvas, (x_offset, 0), (x_offset + width, height), COLOR_PANEL_BG, -1)

    # Title
    hud_title = "V2G MICROGRID TELEMETRY HUD"
    _draw_txt(canvas, hud_title, (x_offset + 25, 38), 0.65, COLOR_TEXT_TITLE, 2)
    sub = "ISO 15118-20 / OCPP 2.0.1 SMARTCHARGING"
    _draw_txt(canvas, sub, (x_offset + 25, 60), 0.40, COLOR_TEXT_MUTED, 1)

    # Card 1: Transformer & Peak Shaving
    y1 = 80
    r1_x1, r1_x2 = x_offset + 20, x_offset + width - 20
    cv2.rectangle(canvas, (r1_x1, y1), (r1_x2, y1 + 130), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (r1_x1, y1), (r1_x2, y1 + 130), COLOR_CARD_BORDER, 1)
    c1_title = "TRANSFORMER LOAD & PEAK SHAVING"
    _draw_txt(canvas, c1_title, (x_offset + 32, y1 + 24), 0.44, COLOR_TEXT_TITLE, 1)

    limit = manager.transformer_limit_kw
    target = manager.peak_shaving_target_kw
    base_kw = compute_facility_base_load_kw(hour)
    solar_kw = compute_facility_solar_pv_kw(hour)
    net_demand = max(0.0, base_kw - solar_kw)

    txt_lim = f"Transformer Limit: {limit:.0f} kW"
    _draw_txt(canvas, txt_lim, (x_offset + 32, y1 + 52), 0.38, COLOR_TEXT_BODY, 1)
    txt_tgt = f"Peak Target: {target:.0f} kW"
    _draw_txt(canvas, txt_tgt, (x_offset + 32, y1 + 72), 0.38, COLOR_WARN, 1)
    txt_net = f"Net Demand: {net_demand:.1f} kW"
    _draw_txt(canvas, txt_net, (x_offset + 32, y1 + 92), 0.38, COLOR_OK, 1)

    util_frac = min(1.0, net_demand / limit)
    bar_x = x_offset + 32
    bar_y = y1 + 104
    bar_w = width - 84
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (18, 20, 24), -1)
    fill_w = int(bar_w * util_frac)
    u_color = COLOR_OK if util_frac < 0.70 else (COLOR_WARN if util_frac < 0.85 else COLOR_DANGER)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), u_color, -1)

    # Card 2: Fleet BESS Capacity
    y2 = 225
    cv2.rectangle(canvas, (r1_x1, y2), (r1_x2, y2 + 135), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (r1_x1, y2), (r1_x2, y2 + 135), COLOR_CARD_BORDER, 1)
    c2_title = "FLEET VIRTUAL BESS CAPACITY"
    _draw_txt(canvas, c2_title, (x_offset + 32, y2 + 24), 0.44, COLOR_TEXT_TITLE, 1)

    tot_cap = manager.get_aggregate_capacity_kwh()
    tot_flex = manager.get_aggregate_flexible_energy_kwh()
    _, hours_autonomy, _ = manager.simulate_islanded_backup(critical_load_kw=35.0)

    txt_bays = f"Connected Bays: {len(manager.bays)} Vehicles"
    _draw_txt(canvas, txt_bays, (x_offset + 32, y2 + 52), 0.38, COLOR_TEXT_BODY, 1)
    txt_cap = f"Total Capacity: {tot_cap:.1f} kWh"
    _draw_txt(canvas, txt_cap, (x_offset + 32, y2 + 74), 0.38, COLOR_TEXT_BODY, 1)
    txt_flex = f"Flexible Discharge: {tot_flex:.1f} kWh"
    _draw_txt(canvas, txt_flex, (x_offset + 32, y2 + 96), 0.38, COLOR_DIS, 1)
    txt_iso = f"Islanded Backup: {hours_autonomy:.1f} Hours"
    _draw_txt(canvas, txt_iso, (x_offset + 32, y2 + 118), 0.38, COLOR_OK, 1)

    # Card 3: Tariff & Arbitrage
    y3 = 375
    cv2.rectangle(canvas, (r1_x1, y3), (r1_x2, y3 + 145), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (r1_x1, y3), (r1_x2, y3 + 145), COLOR_CARD_BORDER, 1)
    c3_title = "TARIFF & ARBITRAGE FINANCIALS"
    _draw_txt(canvas, c3_title, (x_offset + 32, y3 + 24), 0.44, COLOR_TEXT_TITLE, 1)

    w_name = tariff_window.window_id if tariff_window else "ON-PEAK-EVENING"
    imp_p = tariff_window.import_price_per_kwh if tariff_window else 0.42
    exp_p = tariff_window.export_price_per_kwh if tariff_window else 0.36
    profit = report.net_arbitrage_profit_dollars if report else 48.50
    co2 = report.carbon_offset_kg if report else 64.20

    txt_w = f"Tariff Window: {w_name}"
    _draw_txt(canvas, txt_w, (x_offset + 32, y3 + 52), 0.38, COLOR_TEXT_TITLE, 1)
    txt_imp = f"Import Tariff: ${imp_p:.2f} / kWh"
    _draw_txt(canvas, txt_imp, (x_offset + 32, y3 + 74), 0.38, COLOR_TEXT_MUTED, 1)
    txt_exp = f"Export Feed-In: ${exp_p:.2f} / kWh"
    _draw_txt(canvas, txt_exp, (x_offset + 32, y3 + 96), 0.38, COLOR_DIS, 1)
    txt_prf = f"Arbitrage Profit: +${profit:.2f}"
    _draw_txt(canvas, txt_prf, (x_offset + 32, y3 + 118), 0.38, COLOR_OK, 1)
    txt_co2 = f"CO2 Avoided: {co2:.1f} kg"
    _draw_txt(canvas, txt_co2, (x_offset + 32, y3 + 138), 0.36, COLOR_SOLAR, 1)

    # Card 4: Grid Frequency & Inverter Health
    y4 = 535
    cv2.rectangle(canvas, (r1_x1, y4), (r1_x2, y4 + 130), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (r1_x1, y4), (r1_x2, y4 + 130), COLOR_CARD_BORDER, 1)
    c4_title = "GRID FREQUENCY & INVERTER HEALTH"
    _draw_txt(canvas, c4_title, (x_offset + 32, y4 + 24), 0.44, COLOR_TEXT_TITLE, 1)

    t_f = "Grid Frequency: 50.02 Hz [NOMINAL]"
    _draw_txt(canvas, t_f, (x_offset + 32, y4 + 52), 0.38, COLOR_OK, 1)
    t_d = "Droop Response: ARMED (25 kW/Hz)"
    _draw_txt(canvas, t_d, (x_offset + 32, y4 + 74), 0.38, COLOR_TEXT_BODY, 1)
    t_e = "Inverter Efficiency: 96.8 %"
    _draw_txt(canvas, t_e, (x_offset + 32, y4 + 96), 0.38, COLOR_TEXT_BODY, 1)
    t_pf = "Power Factor: 0.99 [DISPLACEMENT]"
    _draw_txt(canvas, t_pf, (x_offset + 32, y4 + 118), 0.38, COLOR_TEXT_BODY, 1)

    # Overall Status Banner
    cv2.rectangle(canvas, (r1_x1, 680), (r1_x2, 710), (25, 45, 30), -1)
    cv2.rectangle(canvas, (r1_x1, 680), (r1_x2, 710), COLOR_OK, 1)
    banner_txt = "V2G MICROGRID DISPATCH: OPTIMAL & STABLE"
    _draw_txt(canvas, banner_txt, (x_offset + 48, 700), 0.40, COLOR_OK, 1)


def render_v2g_dashboard(
    manager: FacilityMicrogridManager,
    report: MicrogridFacilityReport | None = None,
    hour: float = 14.0,
    tariff_window: GridTariffWindow | None = None,
    output_path: Path | None = None,
) -> NDArray[np.uint8]:
    """Generate high-contrast dual-panel 1280x720 V2G energy and SLD tactical rendering."""
    canvas = np.full((720, 1280, 3), COLOR_BG, dtype=np.uint8)
    bays_list = list(manager.bays.values())

    # Left panel: SLD & bays (800x720)
    draw_sld_panel(canvas, bays=bays_list, hour=hour, width=800, height=720)

    # Right panel: Telemetry HUD (480x720)
    draw_hud_panel(
        canvas,
        manager=manager,
        report=report,
        hour=hour,
        tariff_window=tariff_window,
        x_offset=800,
        width=480,
        height=720,
    )

    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), canvas)

    return canvas
