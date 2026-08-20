"""
Unit and integration tests for the Micro-Pulse Stimulation Engine (FEAT-03).
"""

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from boostlock.config import BoostLockConfig
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
from boostlock.hardware import CPUInfo, CPUVendor, CoreInfo
from boostlock.sysfs import SysfsController
from boostlock.thermal import ThermalGuard, ThermalReading, ThermalState


# ---------------------------------------------------------------------------
# TestPreciseTiming
# ---------------------------------------------------------------------------

class TestPreciseTiming:
    """Tests for high-resolution timing, hybrid sleep/spin, and ALU burst."""

    def test_precise_sleep_zero_and_negative(self):
        """Zero and negative sleep durations return immediately."""
        t0 = time.perf_counter()
        precise_sleep(0.0)
        precise_sleep(-0.01)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01

    def test_precise_sleep_sub_millisecond_spin(self):
        """Sub-millisecond duration uses busy spin with high accuracy."""
        target_s = 0.0003  # 300 us
        t0 = time.perf_counter()
        precise_sleep(target_s, spin_threshold_s=0.001)
        elapsed = time.perf_counter() - t0
        assert elapsed >= target_s * 0.85
        assert elapsed < 0.005

    def test_precise_sleep_hybrid(self):
        """Durations above spin threshold use hybrid sleep + spin."""
        target_s = 0.003  # 3 ms
        t0 = time.perf_counter()
        precise_sleep(target_s, spin_threshold_s=0.0005)
        elapsed = time.perf_counter() - t0
        assert elapsed >= target_s * 0.90
        assert elapsed < target_s + 0.015

    def test_tight_alu_burst(self):
        """ALU burst executes register operations without error."""
        result = tight_alu_burst(iterations=500)
        assert isinstance(result, int)
        assert result != 0


# ---------------------------------------------------------------------------
# TestWaveformGenerator
# ---------------------------------------------------------------------------

class TestWaveformGenerator:
    """Tests for stimulation waveforms: square, triangular, jittered, sine."""

    def test_square_waveform(self):
        gen = WaveformGenerator(WaveformType.SQUARE, min_duty_pct=5.0, max_duty_pct=50.0)
        assert gen.compute_duty(base_duty_pct=15.0, time_s=0.0) == 15.0
        assert gen.compute_duty(base_duty_pct=15.0, time_s=1.5) == 15.0

    def test_triangular_waveform(self):
        gen = WaveformGenerator(
            WaveformType.TRIANGULAR,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            period_s=2.0,
        )
        d0 = gen.compute_duty(base_duty_pct=20.0, time_s=0.0)
        d_mid = gen.compute_duty(base_duty_pct=20.0, time_s=0.5)  # phase 0.25 -> up ramp
        d_down = gen.compute_duty(base_duty_pct=20.0, time_s=1.5) # phase 0.75 -> down ramp
        d_end = gen.compute_duty(base_duty_pct=20.0, time_s=2.0)

        assert 5.0 <= d0 <= 50.0
        assert d_mid >= d0
        assert 5.0 <= d_down <= 50.0
        assert 5.0 <= d_end <= 50.0

    def test_jittered_waveform(self):
        gen = WaveformGenerator(
            WaveformType.JITTERED,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            jitter_pct=5.0,
        )
        samples = [gen.compute_duty(base_duty_pct=20.0, time_s=float(i)) for i in range(50)]
        assert all(5.0 <= s <= 50.0 for s in samples)
        assert len(set(samples)) > 1

    def test_stochastic_waveform_alias(self):
        gen = WaveformGenerator(
            WaveformType.STOCHASTIC,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            jitter_pct=5.0,
        )
        samples = [gen.compute_duty(base_duty_pct=20.0, time_s=float(i)) for i in range(50)]
        assert all(5.0 <= s <= 50.0 for s in samples)

    def test_sine_waveform(self):
        gen = WaveformGenerator(
            WaveformType.SINE,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            period_s=1.0,
        )
        d_0 = gen.compute_duty(base_duty_pct=25.0, time_s=0.0)
        d_quarter = gen.compute_duty(base_duty_pct=25.0, time_s=0.25)
        d_three_quarter = gen.compute_duty(base_duty_pct=25.0, time_s=0.75)

        assert 5.0 <= d_0 <= 50.0
        assert d_quarter > d_three_quarter

    def test_waveform_default_time(self):
        gen = WaveformGenerator(WaveformType.SINE, min_duty_pct=5.0, max_duty_pct=50.0)
        d = gen.compute_duty(base_duty_pct=20.0)
        assert 5.0 <= d <= 50.0

    def test_waveform_bounds_clamping(self):
        gen = WaveformGenerator(WaveformType.SQUARE, min_duty_pct=10.0, max_duty_pct=40.0)
        assert gen.compute_duty(base_duty_pct=2.0, time_s=0.0) == 10.0
        assert gen.compute_duty(base_duty_pct=90.0, time_s=0.0) == 40.0

    def test_waveform_string_types(self):
        gen1 = WaveformGenerator("triangular", min_duty_pct=5.0, max_duty_pct=50.0)
        assert gen1.waveform_type == WaveformType.TRIANGULAR

        gen2 = WaveformGenerator("unknown_type", min_duty_pct=5.0, max_duty_pct=50.0)
        assert gen2.compute_duty(base_duty_pct=20.0, time_s=0.0) == 20.0

        gen3 = WaveformGenerator(12345, min_duty_pct=5.0, max_duty_pct=50.0)
        assert gen3.waveform_type == WaveformType.SQUARE


# ---------------------------------------------------------------------------
# TestExternalLoadMonitor
# ---------------------------------------------------------------------------

class TestExternalLoadMonitor:
    """Tests for detecting external system CPU load."""

    def test_load_monitor_proc_stat_parsing(self, tmp_path):
        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("cpu  100 0 50 850 0 0 0 0" + chr(10))

        monitor = ExternalLoadMonitor(proc_stat_path=proc_stat)
        load1 = monitor.get_external_load_pct()
        assert load1 == 0.0

        # Sample 2: total 1200 (+200), idle 950 (+100) -> busy 100/200 = 50%
        proc_stat.write_text("cpu  150 0 100 950 0 0 0 0" + chr(10))
        load2 = monitor.get_external_load_pct()
        assert pytest.approx(load2, 0.1) == 50.0

    def test_load_monitor_missing_file_fallback(self, tmp_path):
        non_existent = tmp_path / "non_existent_stat"
        monitor = ExternalLoadMonitor(proc_stat_path=non_existent)
        assert monitor.get_external_load_pct() == 0.0

    def test_load_monitor_non_cpu_line(self, tmp_path):
        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("intr 12345 6789" + chr(10))
        monitor = ExternalLoadMonitor(proc_stat_path=proc_stat)
        assert monitor.get_external_load_pct() == 0.0

    def test_load_monitor_malformed_fields(self, tmp_path):
        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("cpu  100 200" + chr(10))
        monitor = ExternalLoadMonitor(proc_stat_path=proc_stat)
        assert monitor.get_external_load_pct() == 0.0

    def test_load_monitor_exception_handling(self, tmp_path):
        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("cpu  invalid data" + chr(10))
        monitor = ExternalLoadMonitor(proc_stat_path=proc_stat)
        assert monitor.get_external_load_pct() == 0.0

    def test_load_monitor_zero_delta_edge_case(self, tmp_path):
        proc_stat = tmp_path / "proc_stat"
        proc_stat.write_text("cpu  100 0 50 850 0 0 0 0" + chr(10))
        monitor = ExternalLoadMonitor(proc_stat_path=proc_stat)
        monitor.get_external_load_pct()
        assert monitor.get_external_load_pct() == 0.0


# ---------------------------------------------------------------------------
# TestAdaptiveDutyController
# ---------------------------------------------------------------------------

class TestAdaptiveDutyController:
    """Tests for closed-loop PID and adaptive step duty cycle controller."""

    def test_duty_increase_when_below_target(self):
        controller = AdaptiveDutyController(
            target_freq_khz=4000000,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            duty_step_pct=2.0,
        )
        assert controller.current_duty_pct == 5.0

        new_duty = controller.update(current_freq_khz=3500000)
        assert new_duty > 5.0
        assert new_duty <= 50.0

    def test_duty_decrease_when_at_or_above_target(self):
        controller = AdaptiveDutyController(
            target_freq_khz=4000000,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            duty_step_pct=2.0,
            initial_duty_pct=25.0,
        )
        new_duty = controller.update(current_freq_khz=4000000)
        assert new_duty < 25.0
        assert new_duty >= 5.0

    def test_duty_clamping(self):
        controller = AdaptiveDutyController(
            target_freq_khz=4000000,
            min_duty_pct=10.0,
            max_duty_pct=30.0,
            duty_step_pct=15.0,
            initial_duty_pct=20.0,
        )
        d1 = controller.update(current_freq_khz=1000000)
        d2 = controller.update(current_freq_khz=1000000)
        assert d2 <= 30.0

        for _ in range(10):
            controller.update(current_freq_khz=5000000)
        assert controller.current_duty_pct >= 10.0

    def test_thermal_clamp_scaling(self):
        controller = AdaptiveDutyController(
            target_freq_khz=4000000,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            initial_duty_pct=30.0,
        )
        duty_clamped = controller.update(current_freq_khz=3000000, thermal_clamp_factor=0.5)
        assert duty_clamped <= 20.0

        duty_zero = controller.update(current_freq_khz=3000000, thermal_clamp_factor=0.0)
        assert duty_zero == 0.0

    def test_external_load_yield(self):
        controller = AdaptiveDutyController(
            target_freq_khz=4000000,
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            load_yield_threshold_pct=20.0,
            initial_duty_pct=20.0,
        )
        duty = controller.update(current_freq_khz=3500000, external_load_pct=40.0)
        assert duty == 0.0

    def test_controller_reset(self):
        controller = AdaptiveDutyController(initial_duty_pct=25.0)
        controller.update(current_freq_khz=3000000)
        controller.reset()
        assert controller.current_duty_pct == 25.0


# ---------------------------------------------------------------------------
# TestCoreTargeting
# ---------------------------------------------------------------------------

class TestCoreTargeting:
    """Tests for core targeting and affinity selection."""

    def test_targeting_all_cores(self):
        engine = PulseEngine(targeting_mode=TargetingMode.ALL_CORES, core_list=[0, 1, 2, 3])
        assert engine.target_cores == [0, 1, 2, 3]

    def test_targeting_per_ccx(self):
        cpu_info = CPUInfo(
            vendor=CPUVendor.AMD,
            cores=[
                CoreInfo(cpu_id=0, physical_core_id=0),
                CoreInfo(cpu_id=1, physical_core_id=0),
                CoreInfo(cpu_id=2, physical_core_id=1),
                CoreInfo(cpu_id=3, physical_core_id=1),
            ],
            core_to_threads={0: [0, 1], 1: [2, 3]},
        )
        engine = PulseEngine(
            targeting_mode=TargetingMode.PER_CCX,
            target_ccx=1,
            cpu_info=cpu_info,
        )
        assert engine.target_cores == [2, 3]

    def test_targeting_affinity_pinned(self):
        engine = PulseEngine(
            targeting_mode=TargetingMode.AFFINITY_PINNED,
            core_list=[1, 3],
        )
        assert engine.target_cores == [1, 3]

    def test_targeting_fallback_when_empty(self):
        engine = PulseEngine(core_list=[])
        assert len(engine.target_cores) >= 1

    def test_targeting_fallback_when_sysfs_empty(self):
        mock_sysfs = MagicMock(spec=SysfsController)
        mock_sysfs.get_online_cpus.side_effect = Exception("sysfs failed")
        engine = PulseEngine(core_list=[], sysfs=mock_sysfs)
        assert engine.target_cores == [0]


# ---------------------------------------------------------------------------
# TestPulseWorker
# ---------------------------------------------------------------------------

class TestPulseWorker:
    """Tests for PulseWorker thread lifecycle, affinity, and pulse generation."""

    def test_pulse_worker_spawn_and_lifecycle(self):
        worker = PulseWorker(
            cpu_id=0,
            duty_cycle_pct=10.0,
            pulse_period_us=2000,
        )
        assert worker.is_running is False

        worker.start()
        assert worker.is_running is True

        time.sleep(0.05)

        worker.stop()
        assert worker.is_running is False
        metrics = worker.get_metrics()
        assert metrics.pulses_executed > 0
        assert metrics.cpu_id == 0

    def test_pulse_worker_affinity_setting(self):
        with patch("os.sched_setaffinity") as mock_affinity:
            worker = PulseWorker(cpu_id=2, duty_cycle_pct=10.0)
            worker.start()
            time.sleep(0.02)
            worker.stop()
            mock_affinity.assert_called_with(0, {2})

    def test_pulse_worker_affinity_failure_fallback(self):
        with patch("os.sched_setaffinity", side_effect=OSError("Invalid core")):
            worker = PulseWorker(cpu_id=999, duty_cycle_pct=10.0)
            worker.start()
            time.sleep(0.02)
            worker.stop()
            assert worker.affinity_error is True

    def test_pulse_worker_without_sched_setaffinity(self):
        with patch.object(os, "sched_setaffinity", create=False):
            with patch("boostlock.engine.hasattr", return_value=False):
                worker = PulseWorker(cpu_id=0, duty_cycle_pct=10.0)
                worker.start()
                time.sleep(0.02)
                worker.stop()
                assert worker.is_running is False

    def test_pulse_worker_pause_and_resume(self):
        worker = PulseWorker(cpu_id=0, duty_cycle_pct=20.0, pulse_period_us=2000)
        worker.start()
        time.sleep(0.02)
        c1 = worker.get_metrics().pulses_executed

        worker.pause()
        assert worker.is_paused is True
        time.sleep(0.03)
        c2 = worker.get_metrics().pulses_executed

        worker.resume()
        assert worker.is_paused is False
        time.sleep(0.03)
        c3 = worker.get_metrics().pulses_executed
        worker.stop()

        assert c3 > c2

    def test_pulse_worker_duty_cycle_update(self):
        worker = PulseWorker(cpu_id=0, duty_cycle_pct=5.0)
        worker.set_duty_cycle(25.0)
        assert worker.duty_cycle_pct == 25.0

    def test_pulse_worker_set_cur_freq(self):
        worker = PulseWorker(cpu_id=0)
        worker.set_cur_freq(3900000)
        metrics = worker.get_metrics()
        assert metrics.cur_freq_khz == 3900000

    def test_pulse_worker_zero_duty_sleeps(self):
        worker = PulseWorker(cpu_id=0, duty_cycle_pct=0.0, pulse_period_us=2000)
        worker.start()
        time.sleep(0.02)
        worker.stop()
        metrics = worker.get_metrics()
        assert metrics.duty_cycle_pct == 0.0

    def test_pulse_metrics_to_dict(self):
        pm = PulseMetrics(
            thread_id=123,
            cpu_id=1,
            duty_cycle_pct=15.0,
            pulses_executed=100,
            active_time_s=0.15,
            sleep_time_s=0.85,
            cur_freq_khz=4000000,
            state=EngineState.RUNNING,
        )
        d = pm.to_dict()
        assert d["thread_id"] == 123
        assert d["state"] == "RUNNING"


# ---------------------------------------------------------------------------
# TestPulseEngine
# ---------------------------------------------------------------------------

class TestPulseEngine:
    """Tests for PulseEngine behavior."""

    def test_engine_initialization_defaults(self):
        engine = PulseEngine()
        assert engine.state == EngineState.STOPPED
        assert engine.config.target_frequency_khz == 4000000
        assert engine.targeting_mode == TargetingMode.ALL_CORES

    def test_engine_start_stop_lifecycle(self):
        engine = PulseEngine(core_list=[0, 1])
        assert engine.state == EngineState.STOPPED

        engine.start()
        assert engine.state == EngineState.RUNNING
        assert len(engine.workers) == 2
        assert all(w.is_running for w in engine.workers)

        time.sleep(0.05)

        engine.stop()
        assert engine.state == EngineState.STOPPED
        assert all(not w.is_running for w in engine.workers)

    def test_engine_double_start_and_stop(self):
        engine = PulseEngine(core_list=[0])
        engine.start()
        engine.start()
        assert engine.state == EngineState.RUNNING

        engine.stop()
        engine.stop()
        assert engine.state == EngineState.STOPPED

    def test_engine_pause_and_resume(self):
        engine = PulseEngine(core_list=[0])
        engine.start()
        assert engine.state == EngineState.RUNNING

        engine.pause()
        assert engine.state == EngineState.PAUSED
        assert all(w.is_paused for w in engine.workers)

        engine.resume()
        assert engine.state == EngineState.RUNNING
        assert all(not w.is_paused for w in engine.workers)

        engine.stop()

    def test_engine_pause_when_stopped(self):
        engine = PulseEngine(core_list=[0])
        engine.pause()
        assert engine.state == EngineState.STOPPED
        engine.resume()
        assert engine.state == EngineState.STOPPED

    def test_engine_context_manager(self):
        with PulseEngine(core_list=[0]) as engine:
            assert engine.state == EngineState.RUNNING
            time.sleep(0.02)
        assert engine.state == EngineState.STOPPED

    def test_engine_reconfigure(self):
        engine = PulseEngine(core_list=[0])
        new_config = BoostLockConfig(target_frequency_khz=4200000, min_pulse_duty_pct=10.0)
        engine.reconfigure(new_config)
        assert engine.config.target_frequency_khz == 4200000
        assert engine.config.min_pulse_duty_pct == 10.0

    def test_engine_closed_loop_frequency_sampling(self):
        mock_sysfs = MagicMock(spec=SysfsController)
        mock_sysfs.get_scaling_cur_freq.return_value = 3500000
        mock_sysfs.get_online_cpus.return_value = [0]

        engine = PulseEngine(
            core_list=[0],
            sysfs=mock_sysfs,
            poll_interval_s=0.02,
        )
        # Ensure zero external load in hermetic test
        with patch.object(engine._load_monitor, "get_external_load_pct", return_value=0.0):
            engine.start()
            time.sleep(0.08)

            metrics = engine.get_metrics()
            assert metrics.overall_duty_cycle_pct > engine.config.min_pulse_duty_pct
            assert metrics.target_frequency_khz == 4000000
            assert metrics.average_frequency_khz == 3500000

            engine.stop()

    def test_engine_closed_loop_read_frequency_tuple(self):
        mock_sysfs = MagicMock(spec=SysfsController)
        mock_sysfs.get_scaling_cur_freq.return_value = 3600000
        mock_sysfs.get_online_cpus.return_value = [0]

        engine = PulseEngine(
            core_list=[0],
            sysfs=mock_sysfs,
            poll_interval_s=0.02,
        )
        with patch.object(engine._load_monitor, "get_external_load_pct", return_value=0.0):
            engine.start()
            time.sleep(0.08)
            metrics = engine.get_metrics()
            assert metrics.average_frequency_khz == 3600000
            engine.stop()

    def test_engine_closed_loop_read_frequency_exception_resilience(self):
        mock_sysfs = MagicMock(spec=SysfsController)
        mock_sysfs.get_scaling_cur_freq.side_effect = OSError("sysfs read error")
        mock_sysfs.get_online_cpus.return_value = [0]

        engine = PulseEngine(
            core_list=[0],
            sysfs=mock_sysfs,
            poll_interval_s=0.02,
        )
        engine.start()
        time.sleep(0.05)
        assert engine.state == EngineState.RUNNING
        engine.stop()

    def test_engine_closed_loop_general_loop_exception(self):
        mock_sysfs = MagicMock(spec=SysfsController)
        mock_sysfs.get_online_cpus.return_value = [0]

        engine = PulseEngine(
            core_list=[0],
            sysfs=mock_sysfs,
            poll_interval_s=0.02,
        )
        with patch.object(engine._load_monitor, "get_external_load_pct", side_effect=RuntimeError("Simulated crash")):
            engine.start()
            time.sleep(0.05)
            assert engine.state == EngineState.RUNNING
            engine.stop()

    def test_engine_thermal_guard_integration(self):
        mock_thermal = MagicMock(spec=ThermalGuard)
        mock_thermal.calculate_duty_clamp.return_value = 0.0
        mock_thermal.state = ThermalState.CRITICAL
        mock_thermal.is_tripped = True
        mock_thermal.clamp_factor = 0.0

        engine = PulseEngine(
            core_list=[0],
            thermal_guard=mock_thermal,
            poll_interval_s=0.02,
        )
        engine.start()
        time.sleep(0.06)

        metrics = engine.get_metrics()
        assert metrics.thermal_clamped is True
        assert metrics.overall_duty_cycle_pct == 0.0

        engine.stop()

    def test_engine_round_robin_staggering(self):
        engine = PulseEngine(
            targeting_mode=TargetingMode.ROUND_ROBIN,
            core_list=[0, 1, 2, 3],
            stagger_window_ms=50,
        )
        engine.start()
        time.sleep(0.06)
        metrics = engine.get_metrics()
        assert metrics.targeting_mode == TargetingMode.ROUND_ROBIN
        engine.stop()

    def test_engine_get_metrics_and_status(self):
        engine = PulseEngine(core_list=[0])
        engine.start()
        time.sleep(0.03)

        metrics = engine.get_metrics()
        assert isinstance(metrics, EngineMetrics)
        assert metrics.state == EngineState.RUNNING
        assert 0 in metrics.worker_metrics

        status = engine.get_status()
        assert isinstance(status, dict)
        assert status["state"] == "RUNNING"
        assert "overall_duty_cycle_pct" in status
        assert "worker_metrics" in status

        json_str = json.dumps(status)
        assert json_str is not None

        engine.stop()

    def test_engine_exceptions_hierarchy(self):
        assert issubclass(EngineStateError, EngineError)
        assert issubclass(AffinityError, EngineError)


# ---------------------------------------------------------------------------
# TestPulseEngineAliases
# ---------------------------------------------------------------------------

def test_pulse_engine_module_alias():
    """Verify boostlock.pulse_engine exports identical symbols as boostlock.engine."""
    import boostlock.pulse_engine as pe
    assert hasattr(pe, "PulseEngine")
    assert hasattr(pe, "PulseWorker")
    assert hasattr(pe, "AdaptiveDutyController")
    assert hasattr(pe, "EngineState")
    assert pe.PulseEngine is PulseEngine
