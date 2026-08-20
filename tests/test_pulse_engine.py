"""
Test suite for pulse_engine alias module.
"""

import boostlock.pulse_engine as pe
from boostlock.engine import PulseEngine, PulseWorker, AdaptiveDutyController, EngineState, TargetingMode, WaveformType

def test_pulse_engine_reexports():
    assert pe.PulseEngine is PulseEngine
    assert pe.PulseWorker is PulseWorker
    assert pe.AdaptiveDutyController is AdaptiveDutyController
    assert pe.EngineState is EngineState
    assert pe.TargetingMode is TargetingMode
    assert pe.WaveformType is WaveformType
