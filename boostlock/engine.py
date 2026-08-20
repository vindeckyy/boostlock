"""
Micro-pulse workers and the duty-cycle controller.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from boostlock.config import BoostLockConfig
from boostlock.hardware import CPUInfo, detect_cpu_info
from boostlock.sysfs import SysfsController
from boostlock.thermal import ThermalGuard, ThermalReading, ThermalState

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Base exception for pulse engine subsystem."""
    pass


class EngineStateError(EngineError):
    """Raised on invalid engine lifecycle state transitions."""
    pass


class AffinityError(EngineError):
    """Raised when setting CPU affinity fails unexpectedly."""
    pass


class EngineState(str, Enum):
    """Operational state of the micro-pulse stimulation engine."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    THROTTLED = "THROTTLED"


class TargetingMode(str, Enum):
    """CPU core targeting and scheduling topology mode."""
    ALL_CORES = "all_cores"
    PER_CCX = "per_ccx"
    ROUND_ROBIN = "round_robin"
    AFFINITY_PINNED = "affinity_pinned"


class WaveformType(str, Enum):
    """Waveform shaping for micro-pulse stimulation."""
    SQUARE = "square"
    TRIANGULAR = "triangular"
    JITTERED = "jittered"
    STOCHASTIC = "stochastic"
    SINE = "sine"


def precise_sleep(duration_s: float, spin_threshold_s: float = 0.0005) -> None:
    """
    High-resolution hybrid sleep and busy-spin pause.
    Sleeps for the bulk of duration to yield CPU to OS scheduler without burning power,
    then executes a tight spin loop for sub-millisecond precision.
    """
    if duration_s <= 0.0:
        return

    deadline = time.perf_counter() + duration_s

    # If duration exceeds spin threshold, perform coarse OS sleep first
    if duration_s > spin_threshold_s:
        time.sleep(duration_s - spin_threshold_s)

    # Tight busy spin until deadline
    while time.perf_counter() < deadline:
        pass


def tight_alu_burst(iterations: int = 1000) -> int:
    """
    Tight ALU and register micro-operations loop.
    Exercises CPU integer and logic pipeline to prevent core downclocking.
    """
    acc = 0x55555555
    for i in range(iterations):
        acc = (acc ^ (i * 2654435761)) & 0xFFFFFFFF
    return acc


class WaveformGenerator:
    """
    Generates time-varying duty cycle waveforms (square, triangular, jittered, sine).
    Prevents harmonic resonance and distributes thermal load over time.
    """

    def __init__(
        self,
        waveform_type: Union[WaveformType, str] = WaveformType.SQUARE,
        min_duty_pct: float = 5.0,
        max_duty_pct: float = 50.0,
        period_s: float = 1.0,
        jitter_pct: float = 2.0,
    ) -> None:
        if isinstance(waveform_type, WaveformType):
            self.waveform_type = waveform_type
        elif isinstance(waveform_type, str):
            try:
                self.waveform_type = WaveformType(waveform_type.lower())
            except ValueError:
                self.waveform_type = WaveformType.SQUARE
        else:
            self.waveform_type = WaveformType.SQUARE

        self.min_duty_pct = min_duty_pct
        self.max_duty_pct = max_duty_pct
        self.period_s = max(0.001, period_s)
        self.jitter_pct = jitter_pct

    def compute_duty(self, base_duty_pct: float, time_s: Optional[float] = None) -> float:
        """Compute instantaneous duty cycle according to configured waveform."""
        t = time.time() if time_s is None else time_s
        duty = base_duty_pct

        if self.waveform_type == WaveformType.SQUARE:
            duty = base_duty_pct

        elif self.waveform_type == WaveformType.TRIANGULAR:
            phase = (t % self.period_s) / self.period_s
            if phase < 0.5:
                ramp = phase * 2.0
            else:
                ramp = (1.0 - phase) * 2.0
            span = max(0.0, self.max_duty_pct - base_duty_pct)
            duty = base_duty_pct + (span * ramp)

        elif self.waveform_type in (WaveformType.JITTERED, WaveformType.STOCHASTIC):
            jitter = random.uniform(-self.jitter_pct, self.jitter_pct)
            duty = base_duty_pct + jitter

        elif self.waveform_type == WaveformType.SINE:
            span = (self.max_duty_pct - self.min_duty_pct) / 2.0
            mid = (self.max_duty_pct + self.min_duty_pct) / 2.0
            duty = mid + (span * math.sin(2.0 * math.pi * (t / self.period_s)))

        return max(self.min_duty_pct, min(self.max_duty_pct, duty))


class ExternalLoadMonitor:
    """
    Monitors host system CPU load from /proc/stat to yield stimulation when
    user workloads run, preventing CPU contention.
    """

    def __init__(self, proc_stat_path: Union[str, Path] = "/proc/stat") -> None:
        self.proc_stat_path = Path(proc_stat_path)
        self._prev_idle: Optional[int] = None
        self._prev_total: Optional[int] = None

    def get_external_load_pct(self) -> float:
        """
        Sample CPU utilization percentage across all cores since last sample.
        Returns 0.0 if unable to sample or on initial sample.
        """
        if not self.proc_stat_path.is_file():
            return 0.0

        try:
            with open(self.proc_stat_path, "r", encoding="utf-8") as f:
                first_line = f.readline()

            if not first_line.startswith("cpu "):
                return 0.0

            fields = [int(x) for x in first_line.split()[1:]]
            if len(fields) < 4:
                return 0.0

            idle_ticks = fields[3] + (fields[4] if len(fields) > 4 else 0)
            total_ticks = sum(fields)

            if self._prev_idle is None or self._prev_total is None:
                self._prev_idle = idle_ticks
                self._prev_total = total_ticks
                return 0.0

            delta_total = total_ticks - self._prev_total
            delta_idle = idle_ticks - self._prev_idle

            self._prev_idle = idle_ticks
            self._prev_total = total_ticks

            if delta_total <= 0:
                return 0.0

            busy_pct = 100.0 * (1.0 - (delta_idle / float(delta_total)))
            return max(0.0, min(100.0, busy_pct))
        except Exception as exc:
            logger.debug(f"Failed to read external load from {self.proc_stat_path}: {exc}")
            return 0.0


class AdaptiveDutyController:
    """
    Closed-loop feedback controller that modulates micro-pulse duty cycle
    based on real-time frequency feedback, thermal clamp factor, and external workload.
    """

    def __init__(
        self,
        target_freq_khz: int = 4000000,
        min_duty_pct: float = 5.0,
        max_duty_pct: float = 50.0,
        duty_step_pct: float = 2.0,
        load_yield_threshold_pct: float = 25.0,
        initial_duty_pct: Optional[float] = None,
    ) -> None:
        self.target_freq_khz = target_freq_khz
        self.min_duty_pct = min_duty_pct
        self.max_duty_pct = max_duty_pct
        self.duty_step_pct = duty_step_pct
        self.load_yield_threshold_pct = load_yield_threshold_pct

        self._initial_duty_pct = (
            initial_duty_pct if initial_duty_pct is not None else min_duty_pct
        )
        self._current_duty_pct = self._initial_duty_pct
        self._integral_error = 0.0
        self._last_error = 0.0

    @property
    def current_duty_pct(self) -> float:
        return self._current_duty_pct

    def update(
        self,
        current_freq_khz: int,
        thermal_clamp_factor: float = 1.0,
        external_load_pct: float = 0.0,
    ) -> float:
        """Calculate new duty cycle for next stimulation interval."""
        if external_load_pct >= self.load_yield_threshold_pct:
            return 0.0

        if thermal_clamp_factor <= 0.0:
            return 0.0

        error_khz = self.target_freq_khz - current_freq_khz

        if error_khz > 20000:
            self._current_duty_pct = min(
                self.max_duty_pct, self._current_duty_pct + self.duty_step_pct
            )
        else:
            decay = self.duty_step_pct * 0.5
            self._current_duty_pct = max(
                self.min_duty_pct, self._current_duty_pct - decay
            )

        scaled_duty = self._current_duty_pct * thermal_clamp_factor
        return max(0.0, min(self.max_duty_pct, scaled_duty))

    def reset(self) -> None:
        """Reset controller state to initial configuration."""
        self._current_duty_pct = self._initial_duty_pct
        self._integral_error = 0.0
        self._last_error = 0.0


@dataclass
class PulseMetrics:
    """Real-time metrics for an individual pulse worker thread."""
    thread_id: int
    cpu_id: int
    duty_cycle_pct: float = 0.0
    pulses_executed: int = 0
    active_time_s: float = 0.0
    sleep_time_s: float = 0.0
    cur_freq_khz: Optional[int] = None
    target_freq_khz: int = 4000000
    state: EngineState = EngineState.STOPPED

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["state"] = self.state.value if isinstance(self.state, EngineState) else self.state
        return res


class PulseWorker(threading.Thread):
    """
    Worker thread pinned to a specific CPU core executing high-frequency micro-pulses.
    """

    def __init__(
        self,
        cpu_id: int,
        duty_cycle_pct: float = 5.0,
        pulse_period_us: int = 2000,
        spin_threshold_s: float = 0.0005,
        waveform_generator: Optional[WaveformGenerator] = None,
    ) -> None:
        super().__init__(name=f"PulseWorker-CPU{cpu_id}", daemon=True)
        self.cpu_id = cpu_id
        self._duty_cycle_pct = duty_cycle_pct
        self.pulse_period_us = pulse_period_us
        self.spin_threshold_s = spin_threshold_s
        self.waveform_generator = waveform_generator

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._pulses_executed = 0
        self._active_time_s = 0.0
        self._sleep_time_s = 0.0
        self._cur_freq_khz: Optional[int] = None
        self._target_freq_khz: int = 4000000
        self.affinity_error: bool = False
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def duty_cycle_pct(self) -> float:
        with self._lock:
            return self._duty_cycle_pct

    def set_duty_cycle(self, duty_pct: float) -> None:
        with self._lock:
            self._duty_cycle_pct = max(0.0, min(100.0, duty_pct))

    def set_cur_freq(self, freq_khz: Optional[int]) -> None:
        with self._lock:
            self._cur_freq_khz = freq_khz

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self.resume()
        if self.is_alive():
            self.join(timeout=1.0)
        with self._lock:
            self._is_running = False

    def get_metrics(self) -> PulseMetrics:
        with self._lock:
            state = EngineState.RUNNING if self._is_running else EngineState.STOPPED
            if self._pause_event.is_set():
                state = EngineState.PAUSED
            return PulseMetrics(
                thread_id=self.ident or 0,
                cpu_id=self.cpu_id,
                duty_cycle_pct=round(self._duty_cycle_pct, 2),
                pulses_executed=self._pulses_executed,
                active_time_s=round(self._active_time_s, 4),
                sleep_time_s=round(self._sleep_time_s, 4),
                cur_freq_khz=self._cur_freq_khz,
                target_freq_khz=self._target_freq_khz,
                state=state,
            )

    def run(self) -> None:
        # Set CPU affinity
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {self.cpu_id})
        except Exception as exc:
            self.affinity_error = True
            logger.warning(
                f"Failed to set CPU affinity for worker {self.name} to CPU {self.cpu_id}: {exc}"
            )

        with self._lock:
            self._is_running = True

        period_s = self.pulse_period_us / 1_000_000.0

        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                with self._lock:
                    base_duty = self._duty_cycle_pct

                if base_duty <= 0.0:
                    time.sleep(period_s)
                    continue

                effective_duty = (
                    self.waveform_generator.compute_duty(base_duty)
                    if self.waveform_generator
                    else base_duty
                )

                active_duration_s = (period_s * effective_duty) / 100.0
                sleep_duration_s = max(0.0, period_s - active_duration_s)

                # Active pulse phase: ALU burst
                t0 = time.perf_counter()
                if active_duration_s > 0.0:
                    deadline = t0 + active_duration_s
                    while time.perf_counter() < deadline:
                        tight_alu_burst(iterations=100)
                t1 = time.perf_counter()
                actual_active_s = t1 - t0

                # Inactive phase: precise sleep
                t2 = time.perf_counter()
                if sleep_duration_s > 0.0:
                    precise_sleep(sleep_duration_s, spin_threshold_s=self.spin_threshold_s)
                t3 = time.perf_counter()
                actual_sleep_s = t3 - t2

                with self._lock:
                    self._pulses_executed += 1
                    self._active_time_s += actual_active_s
                    self._sleep_time_s += actual_sleep_s

        finally:
            with self._lock:
                self._is_running = False


@dataclass
class EngineMetrics:
    """Aggregated real-time metrics for the PulseEngine."""
    state: EngineState
    overall_duty_cycle_pct: float
    target_frequency_khz: int
    average_frequency_khz: Optional[float]
    active_workers: int
    total_pulses: int
    thermal_clamped: bool
    external_load_pct: float
    targeting_mode: TargetingMode
    waveform: WaveformType
    worker_metrics: Dict[int, PulseMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["state"] = self.state.value if isinstance(self.state, EngineState) else self.state
        res["targeting_mode"] = (
            self.targeting_mode.value
            if isinstance(self.targeting_mode, TargetingMode)
            else self.targeting_mode
        )
        res["waveform"] = (
            self.waveform.value
            if isinstance(self.waveform, WaveformType)
            else self.waveform
        )
        res["worker_metrics"] = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.worker_metrics.items()
        }
        return res


class PulseEngine:
    """
    Manage pulse workers and their controller thread.
    """

    def __init__(
        self,
        config: Optional[BoostLockConfig] = None,
        targeting_mode: TargetingMode = TargetingMode.ALL_CORES,
        waveform_type: WaveformType = WaveformType.SQUARE,
        core_list: Optional[Sequence[int]] = None,
        target_ccx: Optional[int] = None,
        cpu_info: Optional[CPUInfo] = None,
        sysfs: Optional[SysfsController] = None,
        thermal_guard: Optional[ThermalGuard] = None,
        poll_interval_s: Optional[float] = None,
        stagger_window_ms: int = 50,
    ) -> None:
        self.config = config or BoostLockConfig()
        self.targeting_mode = targeting_mode
        self.waveform_type = waveform_type
        self.target_ccx = target_ccx
        self.cpu_info = cpu_info
        self.sysfs = sysfs or SysfsController()
        self.thermal_guard = thermal_guard
        self.poll_interval_s = (
            poll_interval_s
            if poll_interval_s is not None
            else (self.config.poll_interval_ms / 1000.0)
        )
        self.stagger_window_ms = stagger_window_ms

        self._lock = threading.Lock()
        self._state = EngineState.STOPPED
        self._stop_event = threading.Event()
        self._controller_thread: Optional[threading.Thread] = None

        self._workers: List[PulseWorker] = []
        self._target_cores = self._resolve_target_cores(core_list)

        self._duty_controller = AdaptiveDutyController(
            target_freq_khz=self.config.target_frequency_khz,
            min_duty_pct=self.config.min_pulse_duty_pct,
            max_duty_pct=self.config.max_pulse_duty_pct,
            duty_step_pct=self.config.duty_step_pct,
        )
        self._load_monitor = ExternalLoadMonitor()
        self._waveform_gen = WaveformGenerator(
            waveform_type=self.waveform_type,
            min_duty_pct=self.config.min_pulse_duty_pct,
            max_duty_pct=self.config.max_pulse_duty_pct,
        )

        self._overall_duty_pct = self.config.min_pulse_duty_pct
        self._external_load_pct = 0.0
        self._thermal_clamped = False

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    @property
    def target_cores(self) -> List[int]:
        with self._lock:
            return list(self._target_cores)

    @property
    def workers(self) -> List[PulseWorker]:
        with self._lock:
            return list(self._workers)

    def _resolve_target_cores(self, core_list: Optional[Sequence[int]]) -> List[int]:
        """Resolve target CPU cores based on targeting mode and topology."""
        if core_list is not None and len(core_list) > 0:
            return list(core_list)

        if self.targeting_mode == TargetingMode.PER_CCX and self.target_ccx is not None:
            info = self.cpu_info or detect_cpu_info()
            if hasattr(info, "core_to_threads") and self.target_ccx in info.core_to_threads:
                return info.core_to_threads[self.target_ccx]

        try:
            online = self.sysfs.get_online_cpus()
            if online:
                return online
        except Exception:
            pass

        return [0]

    def start(self) -> None:
        """Start pulse stimulation workers and closed-loop controller thread."""
        with self._lock:
            if self._state == EngineState.RUNNING:
                return

            self._state = EngineState.STARTING
            self._stop_event.clear()

            # Create workers
            self._workers = [
                PulseWorker(
                    cpu_id=cpu_id,
                    duty_cycle_pct=self._overall_duty_pct,
                    waveform_generator=self._waveform_gen,
                )
                for cpu_id in self._target_cores
            ]

            for worker in self._workers:
                worker.start()

            # Start closed loop controller thread
            self._controller_thread = threading.Thread(
                target=self._controller_loop,
                name="PulseEngineController",
                daemon=True,
            )
            self._controller_thread.start()
            self._state = EngineState.RUNNING

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop all workers and closed loop controller."""
        with self._lock:
            if self._state == EngineState.STOPPED:
                return
            self._stop_event.set()

        # Stop workers
        for worker in self._workers:
            worker.stop()

        # Join controller thread
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=timeout_s)

        with self._lock:
            self._state = EngineState.STOPPED
            self._workers.clear()

    def pause(self) -> None:
        """Pause all stimulation workers."""
        with self._lock:
            if self._state != EngineState.RUNNING:
                return
            self._state = EngineState.PAUSED
            for worker in self._workers:
                worker.pause()

    def resume(self) -> None:
        """Resume paused stimulation workers."""
        with self._lock:
            if self._state != EngineState.PAUSED:
                return
            self._state = EngineState.RUNNING
            for worker in self._workers:
                worker.resume()

    def reconfigure(self, config: BoostLockConfig) -> None:
        """Reconfigure engine parameters dynamically."""
        with self._lock:
            self.config = config
            self._duty_controller.target_freq_khz = config.target_frequency_khz
            self._duty_controller.min_duty_pct = config.min_pulse_duty_pct
            self._duty_controller.max_duty_pct = config.max_pulse_duty_pct
            self._duty_controller.duty_step_pct = config.duty_step_pct

    def _controller_loop(self) -> None:
        """Closed-loop feedback control thread loop."""
        round_robin_idx = 0

        while not self._stop_event.is_set():
            try:
                # 1. External load check
                ext_load = self._load_monitor.get_external_load_pct()
                self._external_load_pct = ext_load

                # 2. Thermal guard clamp check
                clamp_factor = 1.0
                if self.thermal_guard:
                    clamp_factor = getattr(self.thermal_guard, "clamp_factor", 1.0)
                    is_tripped = getattr(self.thermal_guard, "is_tripped", False)
                    self._thermal_clamped = (clamp_factor < 1.0 or is_tripped)
                else:
                    self._thermal_clamped = False

                # 3. Read core frequencies
                sampled_freqs: List[int] = []
                for worker in self.workers:
                    try:
                        cur = self.sysfs.get_scaling_cur_freq(worker.cpu_id)
                        if cur is not None:
                            worker.set_cur_freq(cur)
                            sampled_freqs.append(cur)
                    except Exception:
                        pass

                avg_freq = (
                    int(sum(sampled_freqs) / len(sampled_freqs))
                    if sampled_freqs
                    else self.config.target_frequency_khz
                )

                # 4. Adaptive duty cycle update
                new_duty = self._duty_controller.update(
                    current_freq_khz=avg_freq,
                    thermal_clamp_factor=clamp_factor,
                    external_load_pct=ext_load,
                )
                self._overall_duty_pct = new_duty

                # 5. Core targeting & round-robin distribution
                workers = self.workers
                if self.targeting_mode == TargetingMode.ROUND_ROBIN and workers:
                    active_worker = workers[round_robin_idx % len(workers)]
                    round_robin_idx += 1
                    for w in workers:
                        w.set_duty_cycle(new_duty if w == active_worker else 0.0)
                else:
                    for w in workers:
                        w.set_duty_cycle(new_duty)

            except Exception as exc:
                logger.error(f"Error in PulseEngine controller loop: {exc}")

            self._stop_event.wait(timeout=self.poll_interval_s)

    def get_metrics(self) -> EngineMetrics:
        """Snapshot current operational and performance metrics."""
        with self._lock:
            workers = list(self._workers)
            state = self._state
            duty_pct = self._overall_duty_pct
            ext_load = self._external_load_pct
            thermal_clamped = self._thermal_clamped

        worker_metrics_map: Dict[int, PulseMetrics] = {}
        total_pulses = 0
        freqs: List[int] = []

        for w in workers:
            m = w.get_metrics()
            worker_metrics_map[w.cpu_id] = m
            total_pulses += m.pulses_executed
            if m.cur_freq_khz is not None:
                freqs.append(m.cur_freq_khz)

        avg_freq = (sum(freqs) / len(freqs)) if freqs else None

        return EngineMetrics(
            state=state,
            overall_duty_cycle_pct=round(duty_pct, 2),
            target_frequency_khz=self.config.target_frequency_khz,
            average_frequency_khz=avg_freq,
            active_workers=len([w for w in workers if w.is_running and not w.is_paused]),
            total_pulses=total_pulses,
            thermal_clamped=thermal_clamped,
            external_load_pct=round(ext_load, 2),
            targeting_mode=self.targeting_mode,
            waveform=self.waveform_type,
            worker_metrics=worker_metrics_map,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return serializable status dictionary."""
        return self.get_metrics().to_dict()

    def __enter__(self) -> PulseEngine:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
