"""UDS ISO 14229 diagnostic service exporter and regulatory report generator."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from nearfield360.dssad.models import DSSADReport, EDRIncidentRecord, TelemetrySnapshot


class DSSADExporter:
    """Exporter for UN R157/R160 regulatory reports, CSV telemetry, and UDS diagnostic frames."""

    @staticmethod
    def export_json(report: DSSADReport, indent: int = 2) -> str:
        """Export comprehensive forensic package as validated formatted JSON."""
        return json.dumps(report.model_dump(), indent=indent)

    @staticmethod
    def export_telemetry_csv(telemetry: list[TelemetrySnapshot]) -> str:
        """Export high-frequency telemetry samples as comma-separated values (CSV)."""
        output = io.StringIO()
        fieldnames = [
            "timestamp_ns",
            "timestamp_iso",
            "speed_mps",
            "longitudinal_accel_mps2",
            "lateral_accel_mps2",
            "yaw_rate_rads",
            "steering_angle_rad",
            "brake_pressure_bar",
            "throttle_pct",
            "gear",
            "x_m",
            "y_m",
            "yaw_rad",
            "odd_state",
            "aeb_active",
            "active_dtcs",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for s in telemetry:
            writer.writerow(
                {
                    "timestamp_ns": s.timestamp_ns,
                    "timestamp_iso": s.timestamp_iso,
                    "speed_mps": f"{s.speed_mps:.3f}",
                    "longitudinal_accel_mps2": f"{s.longitudinal_accel_mps2:.3f}",
                    "lateral_accel_mps2": f"{s.lateral_accel_mps2:.3f}",
                    "yaw_rate_rads": f"{s.yaw_rate_rads:.4f}",
                    "steering_angle_rad": f"{s.steering_angle_rad:.4f}",
                    "brake_pressure_bar": f"{s.brake_pressure_bar:.2f}",
                    "throttle_pct": f"{s.throttle_pct:.1f}",
                    "gear": s.gear,
                    "x_m": f"{s.x_m:.2f}",
                    "y_m": f"{s.y_m:.2f}",
                    "yaw_rad": f"{s.yaw_rad:.3f}",
                    "odd_state": s.odd_state,
                    "aeb_active": "1" if s.aeb_active else "0",
                    "active_dtcs": ";".join(s.active_dtcs),
                }
            )

        return output.getvalue()

    @staticmethod
    def export_uds_did_frame(incident: EDRIncidentRecord, did: int = 0x19ED) -> dict[str, Any]:
        """Encode incident metadata into a standardized ISO 14229 UDS diagnostic DID readout.

        Simulates Service 0x22 (ReadDataByIdentifier) response:
        [Positive Response 0x62] + [DID MSB] + [DID LSB] + [Record Data Payload]
        """
        # Pack high-level forensic metrics:
        # 2 bytes: Delta-V in units of 0.1 m/s (0-655.3 m/s)
        delta_v_raw = min(round(incident.delta_v_mps * 10), 0xFFFF)
        # 2 bytes: Peak Accel in units of 0.01 g (0-655.35 g)
        peak_g_raw = min(round(incident.peak_accel_g * 100), 0xFFFF)
        # 1 byte: Event severity ranking
        severity_code = {"INFO": 1, "WARNING": 2, "CRITICAL": 3, "FATAL": 4}.get(
            incident.trigger_event.severity.value, 0
        )
        # 1 byte: Locked flag
        locked_byte = 1 if incident.is_locked else 0
        # 16 bytes: truncated root hash digest
        hash_bytes = bytes.fromhex(incident.tamper_chain_digest[:32].ljust(32, "0"))

        payload_bytes = (
            delta_v_raw.to_bytes(2, byteorder="big")
            + peak_g_raw.to_bytes(2, byteorder="big")
            + severity_code.to_bytes(1, byteorder="big")
            + locked_byte.to_bytes(1, byteorder="big")
            + hash_bytes
        )

        response_bytes = bytes([0x62, (did >> 8) & 0xFF, did & 0xFF]) + payload_bytes

        return {
            "service_id": "0x62",
            "did": f"0x{did:04X}",
            "incident_id": incident.incident_id,
            "raw_hex": response_bytes.hex().upper(),
            "delta_v_mps": incident.delta_v_mps,
            "peak_accel_g": incident.peak_accel_g,
            "severity": incident.trigger_event.severity.value,
            "is_locked": incident.is_locked,
            "chain_digest_prefix": incident.tamper_chain_digest[:16],
        }
