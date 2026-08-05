"""Hardware-facing adapters for physical mobile-robot deployments.

This package is intentionally independent of MuJoCo and ROS.  Hardware
adapters produce the same dependency-light contracts used by the planner so
the prediction, risk and MPPI implementations remain shared with simulation.
"""

from .livox_scan_adapter import (
    LivoxPointCloudFrame,
    LivoxScanAdapter,
    LivoxScanAdapterConfig,
    LivoxScanDiagnostics,
)
from .mapless_static_dynamic_filter import MaplessStaticDynamicFilter
from .motion_bootstrap_tracker import MotionBootstrapMultiObstacleTracker
from .forward_passage import ForwardPassageConfig, ForwardPassageController
from .encounter_modes import (
    EncounterMode,
    EncounterModeConfig,
    EncounterModeManager,
    PassageStrategy,
)
from .livox_udp import LivoxUdpReceiver, decode_livox_datagram
from .scout_can import ScoutGuardedCanGateway, ScoutZeroOnlyCanGuard

__all__ = [
    "LivoxPointCloudFrame",
    "LivoxScanAdapter",
    "LivoxScanAdapterConfig",
    "LivoxScanDiagnostics",
    "MaplessStaticDynamicFilter",
    "MotionBootstrapMultiObstacleTracker",
    "ForwardPassageConfig",
    "ForwardPassageController",
    "EncounterMode",
    "EncounterModeConfig",
    "EncounterModeManager",
    "PassageStrategy",
    "LivoxUdpReceiver",
    "decode_livox_datagram",
    "ScoutGuardedCanGateway",
    "ScoutZeroOnlyCanGuard",
]
