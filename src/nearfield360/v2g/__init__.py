"""Vehicle-to-Grid (V2G) Bi-Directional Power Flow, Peak Shaving & Inverter Telemetry."""

from __future__ import annotations

from nearfield360.v2g.arbitrage import (
    ArbitrageScheduler,
    find_tariff_window,
    get_default_tariff_schedule,
    is_discharge_arbitrage_profitable,
)
from nearfield360.v2g.inverter import SmartInverter
from nearfield360.v2g.microgrid import (
    ConnectedBay,
    FacilityMicrogridManager,
    compute_facility_base_load_kw,
    compute_facility_solar_pv_kw,
    create_default_demonstrator_fleet,
)
from nearfield360.v2g.models import (
    GridTariffWindow,
    InverterPhase,
    InverterTelemetry,
    MicrogridFacilityReport,
    V2GArbitrageSchedule,
    V2GPowerMode,
    V2GTimeSlotDispatch,
    VehicleBPTProfile,
)
from nearfield360.v2g.viz import draw_hud_panel, draw_sld_panel, render_v2g_dashboard

__all__ = [
    "ArbitrageScheduler",
    "ConnectedBay",
    "FacilityMicrogridManager",
    "GridTariffWindow",
    "InverterPhase",
    "InverterTelemetry",
    "MicrogridFacilityReport",
    "SmartInverter",
    "V2GArbitrageSchedule",
    "V2GPowerMode",
    "V2GTimeSlotDispatch",
    "VehicleBPTProfile",
    "compute_facility_base_load_kw",
    "compute_facility_solar_pv_kw",
    "create_default_demonstrator_fleet",
    "draw_hud_panel",
    "draw_sld_panel",
    "find_tariff_window",
    "get_default_tariff_schedule",
    "is_discharge_arbitrage_profitable",
    "render_v2g_dashboard",
]
