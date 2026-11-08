"""Automotive nearfield acoustic sensing, siren detection, DoA, and yield interlocks."""

from __future__ import annotations

from nearfield360.acoustic.detector import (
    SirenDetector,
    compute_stft,
    generate_synthetic_siren,
)
from nearfield360.acoustic.doa import (
    AcousticDoAEstimator,
    build_default_microphone_array,
    gcc_phat,
    simulate_multichannel_siren,
)
from nearfield360.acoustic.executive import AcousticExecutive
from nearfield360.acoustic.models import (
    AcousticDoAResult,
    AcousticReport,
    AudioVisualSectorMatch,
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    MicrophoneSpec,
    SirenDetectionResult,
    SirenType,
    azimuth_to_camera_sector,
)
from nearfield360.acoustic.viz import render_acoustic_dashboard

__all__ = [
    "AcousticDoAEstimator",
    "AcousticDoAResult",
    "AcousticExecutive",
    "AcousticReport",
    "AudioVisualSectorMatch",
    "CameraSector",
    "DopplerState",
    "EmergencyYieldState",
    "MicrophoneSpec",
    "SirenDetectionResult",
    "SirenDetector",
    "SirenType",
    "azimuth_to_camera_sector",
    "build_default_microphone_array",
    "compute_stft",
    "gcc_phat",
    "generate_synthetic_siren",
    "render_acoustic_dashboard",
    "simulate_multichannel_siren",
]
