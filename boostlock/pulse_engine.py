"""
Compatibility imports for boostlock.pulse_engine.
"""

from boostlock.engine import (
    AdaptiveDutyController,
    AffinityError,
    EngineError,
    EngineMetrics,
    EngineState,
    EngineStateError,
    ExternalLoadMonitor,
    PulseEngine,
    PulseMetrics,
    PulseWorker,
    TargetingMode,
    WaveformGenerator,
    WaveformType,
    precise_sleep,
    tight_alu_burst,
)

__all__ = [
    "EngineError",
    "EngineStateError",
    "AffinityError",
    "EngineState",
    "TargetingMode",
    "WaveformType",
    "precise_sleep",
    "tight_alu_burst",
    "WaveformGenerator",
    "ExternalLoadMonitor",
    "AdaptiveDutyController",
    "PulseMetrics",
    "PulseWorker",
    "EngineMetrics",
    "PulseEngine",
]
