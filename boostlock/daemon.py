"""
Background service for BoostLock.

It saves CPU state, applies the configured profile, and starts the monitor,
pulse engine, and IPC server.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from boostlock.config import BoostLockConfig, ConfigValidationError
from boostlock.hardware import CPUInfo, detect_cpu_info
from boostlock.ipc import IPCClient, IPCServer
from boostlock.pm_qos import PMQoSController
from boostlock.protocol import Command, Request, Response
from boostlock.pulse_engine import EngineState, PulseEngine
from boostlock.state import StateSnapshotManager
from boostlock.sysfs import SysfsController
from boostlock.thermal import ThermalGuard, ThermalReading, ThermalState

logger = logging.getLogger(__name__)

DEFAULT_PID_DIR = Path("/var/run/boostlock")
FALLBACK_PID_DIR = Path(tempfile.gettempdir()) / "boostlock"
PID_FILENAME = "boostlock.pid"


class DaemonError(Exception):
    """Base exception for BoostLock daemon operations."""
    pass


class DaemonRunningError(DaemonError):
    """Raised when an active BoostLock daemon instance is already running."""
    pass


class PIDFileError(DaemonError):
    """Raised when PID file management operations fail."""
    pass


class DaemonState(str, Enum):
    """High-level operational lifecycle states of the BoostLock daemon."""
    STOPPED = "STOPPED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    THROTTLED = "THROTTLED"
    ERROR = "ERROR"


def resolve_default_pid_path() -> Path:
    """Determine the optimal writable PID file path."""
    try:
        DEFAULT_PID_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DEFAULT_PID_DIR / ".pid_write_test"
        test_file.touch()
        test_file.unlink(missing_ok=True)
        return DEFAULT_PID_DIR / PID_FILENAME
    except (PermissionError, OSError):
        FALLBACK_PID_DIR.mkdir(parents=True, exist_ok=True)
        return FALLBACK_PID_DIR / PID_FILENAME


class PIDFileManager:
    """
    Manages process lock and PID file using POSIX advisory file locking (fcntl.flock).
    Ensures strictly single-instance execution and cleans up stale locks.
    """

    def __init__(self, pid_path: Optional[Union[str, Path]] = None) -> None:
        if pid_path is not None:
            self.pid_path = Path(pid_path).resolve()
        else:
            self.pid_path = resolve_default_pid_path()
        self._fd: Optional[int] = None
        self._is_locked: bool = False

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    def read_pid(self) -> Optional[int]:
        """Read PID from PID file if it exists."""
        if not self.pid_path.is_file():
            return None
        try:
            content = self.pid_path.read_text(encoding="utf-8").strip()
            return int(content) if content else None
        except Exception:
            return None

    def is_daemon_running(self) -> bool:
        """Check whether the process referenced by the PID file is actively running."""
        pid = self.read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def acquire(self) -> bool:
        """
        Acquire exclusive flock on the PID file and write current PID.
        Raises DaemonRunningError if another instance holds the lock.
        """
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd = os.open(
                str(self.pid_path),
                os.O_RDWR | os.O_CREAT,
                0o644,
            )
        except Exception as exc:
            raise PIDFileError(f"Failed to open PID file at {self.pid_path}: {exc}") from exc

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            existing_pid = self.read_pid()
            os.close(fd)
            if existing_pid is not None and self.is_daemon_running():
                raise DaemonRunningError(
                    f"BoostLock daemon is already running (PID {existing_pid})"
                ) from exc
            raise DaemonRunningError(
                f"BoostLock PID file at {self.pid_path} is locked by another process"
            ) from exc

        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            pid_str = f"{os.getpid()}\n"
            os.write(fd, pid_str.encode("utf-8"))
            os.fsync(fd)
        except Exception as exc:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except Exception:
                pass
            raise PIDFileError(f"Failed to write PID to {self.pid_path}: {exc}") from exc

        self._fd = fd
        self._is_locked = True
        logger.info(f"Acquired PID lock on {self.pid_path} for PID {os.getpid()}")
        return True

    def release(self) -> None:
        """Release flock, close file descriptor, and remove PID file."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception as exc:
                logger.warning(f"Error releasing PID lock fd: {exc}")
            self._fd = None

        self._is_locked = False
        try:
            self.pid_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to remove PID file {self.pid_path}: {exc}")


class BoostLockDaemon:
    """
    Main BoostLock background daemon supervisor and coordinator.
    """

    def __init__(
        self,
        config: Optional[BoostLockConfig] = None,
        sysfs_root: Union[str, Path] = "/sys",
        dma_latency_path: Union[str, Path] = "/dev/cpu_dma_latency",
        pid_file: Optional[Union[str, Path]] = None,
        socket_path: Optional[Union[str, Path]] = None,
        snapshot_file: Optional[Union[str, Path]] = None,
        cpu_info: Optional[CPUInfo] = None,
    ) -> None:
        self.config = config or BoostLockConfig()
        self.config.validate()

        self.sysfs_root = Path(sysfs_root).resolve()
        self.dma_latency_path = Path(dma_latency_path)
        self.pid_file = Path(pid_file).resolve() if pid_file else Path(self.config.pid_file)
        self.socket_path = Path(socket_path).resolve() if socket_path else Path(self.config.socket_path)
        self.snapshot_file = Path(snapshot_file).resolve() if snapshot_file else Path(self.config.snapshot_path)
        self.cpu_info = cpu_info

        self._lock = threading.Lock()
        self._state = DaemonState.STOPPED
        self._is_locked_boost = False
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None

        # Subsystems
        self.sysfs = SysfsController(sysfs_root=self.sysfs_root)
        self.pm_qos = PMQoSController(
            device_path=self.dma_latency_path,
            sysfs_root=self.sysfs_root,
            target_latency_us=self.config.dma_latency_us,
        )
        self.state_manager = StateSnapshotManager(
            sysfs_controller=self.sysfs,
            snapshot_file=self.snapshot_file,
            restore_on_exit=False,
        )
        self.thermal_guard = ThermalGuard(
            sysfs_root=self.sysfs_root,
            config=self.config,
            on_warning=self._on_thermal_warning,
            on_tripwire=self._on_thermal_tripwire,
            on_recovery=self._on_thermal_recovery,
        )
        self.pulse_engine = PulseEngine(
            config=self.config,
            sysfs=self.sysfs,
            thermal_guard=self.thermal_guard,
            cpu_info=self.cpu_info,
        )
        self.ipc_server = IPCServer(
            socket_path=self.socket_path,
            handler=self.handle_ipc_request,
        )
        self.pid_manager = PIDFileManager(pid_path=self.pid_file)

        self._signal_handlers_registered = False
        self._orig_signals: Dict[int, Any] = {}

    @property
    def state(self) -> DaemonState:
        with self._lock:
            return self._state

    @property
    def is_locked_boost(self) -> bool:
        with self._lock:
            return self._is_locked_boost

    def start(self) -> None:
        """
        Initialize and engage all subsystems in order:
        1. Acquire PID lock
        2. Create system state snapshot
        3. Register POSIX signal handlers
        4. Apply sysfs performance configuration
        5. Lock PM QoS CPU DMA latency (0 us)
        6. Start ThermalGuard monitoring
        7. Start PulseEngine stimulation
        8. Start Unix domain socket IPC server
        """
        with self._lock:
            if self._state == DaemonState.RUNNING:
                return
            self._state = DaemonState.INITIALIZING

        try:
            # 1. PID file
            self.pid_manager.acquire()

            # 2. Capture state snapshot
            self.state_manager.create_snapshot()

            # 3. Register signal handlers
            self._register_signals()

            # 4. Sysfs governor and boost optimization
            self._apply_sysfs_boost_profile()

            # 5. Lock PM QoS
            self.pm_qos.lock(self.config.dma_latency_us)

            # 6. ThermalGuard
            self.thermal_guard.start()

            # 7. PulseEngine
            self.pulse_engine.start()

            # 8. IPC Server
            self.ipc_server.start()

            with self._lock:
                self._state = DaemonState.RUNNING
                self._is_locked_boost = True
                self._start_time = time.time()
                self._stop_event.clear()

            logger.info("BoostLock daemon started successfully")

        except Exception as exc:
            logger.exception(f"Failed to start BoostLock daemon: {exc}")
            self._emergency_rollback(exc)
            raise DaemonError(f"Daemon startup failed: {exc}") from exc

    def stop(self) -> None:
        """
        Gracefully stop all subsystems and restore initial system state.
        """
        with self._lock:
            if self._state == DaemonState.STOPPED:
                return
            self._state = DaemonState.STOPPED
            self._is_locked_boost = False
            self._stop_event.set()

        logger.info("Stopping BoostLock daemon...")

        # 1. Stop PulseEngine
        try:
            self.pulse_engine.stop()
        except Exception as e:
            logger.error(f"Error stopping PulseEngine: {e}")

        # 2. Stop ThermalGuard
        try:
            self.thermal_guard.stop()
        except Exception as e:
            logger.error(f"Error stopping ThermalGuard: {e}")

        # 3. Unlock PM QoS
        try:
            self.pm_qos.unlock()
        except Exception as e:
            logger.error(f"Error unlocking PM QoS: {e}")

        # 4. Restore initial sysfs state
        try:
            self.state_manager.restore()
        except Exception as e:
            logger.error(f"Error restoring system state: {e}")

        # 5. Unregister signals
        self._unregister_signals()

        # 6. Stop IPC Server
        try:
            self.ipc_server.stop()
        except Exception as e:
            logger.error(f"Error stopping IPC server: {e}")

        # 7. Release PID lock
        try:
            self.pid_manager.release()
        except Exception as e:
            logger.error(f"Error releasing PID lock: {e}")

        logger.info("BoostLock daemon stopped and system state restored")

    def pause(self) -> None:
        """Pause pulse engine stimulation workers."""
        with self._lock:
            if self._state != DaemonState.RUNNING:
                return
            self.pulse_engine.pause()
            self._state = DaemonState.PAUSED

    def resume(self) -> None:
        """Resume paused pulse engine stimulation workers."""
        with self._lock:
            if self._state != DaemonState.PAUSED:
                return
            self.pulse_engine.resume()
            self._state = DaemonState.RUNNING

    def lock(self) -> None:
        """Engage hardware boost clock locking."""
        with self._lock:
            if self._is_locked_boost:
                return
            self._apply_sysfs_boost_profile()
            self.pm_qos.lock(self.config.dma_latency_us)
            self.pulse_engine.start()
            self._is_locked_boost = True
            if self._state != DaemonState.THROTTLED:
                self._state = DaemonState.RUNNING

    def unlock(self) -> None:
        """Disengage boost pinning without stopping the daemon."""
        with self._lock:
            if not self._is_locked_boost:
                return
            self.pulse_engine.stop()
            self.pm_qos.unlock()
            self._is_locked_boost = False

    def reconfigure(self, new_config_dict: Dict[str, Any]) -> BoostLockConfig:
        """Apply dynamic configuration updates to all subsystems."""
        merged = self.config.to_dict()
        merged.update(new_config_dict)
        new_config = BoostLockConfig.from_dict(merged)
        new_config.validate()

        with self._lock:
            self.config = new_config
            self.thermal_guard.thermal_warn_c = new_config.thermal_warn_c
            self.thermal_guard.thermal_limit_c = new_config.thermal_limit_c
            self.thermal_guard.thermal_recover_c = new_config.thermal_recover_c
            self.pulse_engine.reconfigure(new_config)

        return self.config

    def get_status(self) -> Dict[str, Any]:
        """Aggregate high-level status from all subsystems."""
        with self._lock:
            state_val = self._state.value
            is_locked = self._is_locked_boost
            uptime = (time.time() - self._start_time) if self._start_time else 0.0

        cpus = self.sysfs.get_online_cpus()
        # Keep sysfs reads minimal and fault-tolerant for IPC responsiveness
        try:
            cpu_states = self.sysfs.read_all_cpus_state()
        except Exception:
            cpu_states = {}
        try:
            thermal_status = self.thermal_guard.get_status()
        except Exception:
            thermal_status = {}
        try:
            # Use lightweight pulse status to avoid GIL contention and large JSON (>4096) that causes IPC fragmentation
            # Full worker_metrics are available via METRICS command, STATUS should be fast and small
            pm = self.pulse_engine.get_metrics()
            pulse_status = {
                "state": pm.state.value if hasattr(pm.state, "value") else str(pm.state),
                "overall_duty_cycle_pct": pm.overall_duty_cycle_pct,
                "target_frequency_khz": pm.target_frequency_khz,
                "average_frequency_khz": pm.average_frequency_khz,
                "active_workers": pm.active_workers,
                "total_pulses": pm.total_pulses,
                "thermal_clamped": pm.thermal_clamped,
                "external_load_pct": pm.external_load_pct,
                "targeting_mode": pm.targeting_mode.value if hasattr(pm.targeting_mode, "value") else str(pm.targeting_mode),
                "waveform": pm.waveform.value if hasattr(pm.waveform, "value") else str(pm.waveform),
            }
        except Exception:
            pulse_status = {}
        # per_cpu flat mapping for CLI
        per_cpu = {}
        try:
            for cpu_id, state in cpu_states.items():
                per_cpu[str(cpu_id)] = {
                    "cur_freq_khz": state.get("scaling_cur_freq") or state.get("cur_freq_khz") or 0,
                    "governor": state.get("governor"),
                }
        except Exception:
            per_cpu = {}
        # duty_cycle: pulse_status has overall_duty_cycle_pct
        duty = None
        try:
            if isinstance(pulse_status, dict):
                if "overall_duty_cycle_pct" in pulse_status:
                    duty = pulse_status["overall_duty_cycle_pct"] / 100.0
                elif "current_duty_pct" in pulse_status:
                    duty = pulse_status["current_duty_pct"] / 100.0
                else:
                    duty = pulse_status.get("duty_cycle")
        except Exception:
            duty = None
        temp_c = None
        try:
            if isinstance(thermal_status, dict):
                temp_c = thermal_status.get("current_temp_c")
        except Exception:
            temp_c = None
        return {
            "state": state_val,
            "boost_state": state_val,
            "is_locked_boost": is_locked,
            "pid": os.getpid(),
            "uptime_seconds": round(uptime, 2),
            "target_frequency_khz": self.config.target_frequency_khz,
            "target_freq_khz": self.config.target_frequency_khz,
            "governor": self.config.governor,
            "epp": self.config.epp,
            "thermal": thermal_status,
            "temperature_c": temp_c,
            "duty_cycle": duty,
            "pm_qos": {
                "is_locked": self.pm_qos.is_locked,
                "target_latency_us": self.pm_qos.target_latency_us,
                "using_fallback": self.pm_qos.using_fallback,
            },
            "pm_qos_active": self.pm_qos.is_locked,
            "pulse_engine": pulse_status,
            "online_cpus_count": len(cpus),
            "cpu_states": {str(k): v for k, v in cpu_states.items()},
            "per_cpu": per_cpu,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Aggregate operational and frequency metrics."""
        engine_metrics = self.pulse_engine.get_metrics().to_dict()
        thermal_status = self.thermal_guard.get_status()
        cpu_states = self.sysfs.read_all_cpus_state()

        freqs = {
            str(cpu_id): s.get("scaling_cur_freq")
            for cpu_id, s in cpu_states.items()
            if s.get("scaling_cur_freq") is not None
        }

        return {
            "timestamp": time.time(),
            "engine": engine_metrics,
            "thermal": thermal_status,
            "temperatures": self.thermal_guard.get_all_temperatures(),
            "cpus": {
                "frequencies_khz": freqs,
                "governors": {str(k): v.get("governor") for k, v in cpu_states.items()},
            },
        }

    def handle_ipc_request(self, req: Request) -> Response:
        """Dispatch IPC requests to daemon handlers."""
        cmd = req.command

        try:
            if cmd == Command.PING:
                return Response.ok(
                    {
                        "status": self.state.value,
                        "pid": os.getpid(),
                        "is_locked_boost": self.is_locked_boost,
                    },
                    request_id=req.request_id,
                )

            elif cmd == Command.STATUS:
                return Response.ok(self.get_status(), request_id=req.request_id)

            elif cmd == Command.START:
                self.start()
                return Response.ok({"state": self.state.value}, request_id=req.request_id)

            elif cmd == Command.STOP:
                # Schedule stop in background thread to allow response delivery
                def async_stop():
                    time.sleep(0.05)
                    self.stop()

                threading.Thread(target=async_stop, daemon=True).start()
                return Response.ok({"stopped": True}, request_id=req.request_id)

            elif cmd == Command.PAUSE:
                self.pause()
                return Response.ok({"state": self.state.value}, request_id=req.request_id)

            elif cmd == Command.RESUME:
                self.resume()
                return Response.ok({"state": self.state.value}, request_id=req.request_id)

            elif cmd == Command.LOCK:
                self.lock()
                return Response.ok(
                    {"locked": True, "state": self.state.value},
                    request_id=req.request_id,
                )

            elif cmd == Command.UNLOCK:
                self.unlock()
                return Response.ok(
                    {"unlocked": True, "state": self.state.value},
                    request_id=req.request_id,
                )

            elif cmd == Command.RECONFIGURE:
                updated = self.reconfigure(req.args)
                return Response.ok(updated.to_dict(), request_id=req.request_id)

            elif cmd == Command.METRICS:
                return Response.ok(self.get_metrics(), request_id=req.request_id)

            elif cmd == Command.CONFIG:
                return Response.ok(self.config.to_dict(), request_id=req.request_id)

            return Response.fail(f"Unhandled command: {cmd}", request_id=req.request_id)

        except Exception as exc:
            logger.exception(f"Error handling IPC command {cmd}: {exc}")
            return Response.fail(
                error=str(exc),
                error_type=exc.__class__.__name__,
                request_id=req.request_id,
            )

    def run(self) -> None:
        """Run daemon synchronously until stop is signaled."""
        self.start()
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping daemon...")
        finally:
            self.stop()

    def daemonize(self) -> None:
        """Fork process into background daemon (POSIX double-fork)."""
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            raise DaemonError(f"First fork failed: {e}") from e

        os.chdir("/")
        os.setsid()
        os.umask(0o022)

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            raise DaemonError(f"Second fork failed: {e}") from e

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "r") as devnull_r, open(os.devnull, "a+") as devnull_w:
            os.dup2(devnull_r.fileno(), 0)
            os.dup2(devnull_w.fileno(), 1)
            os.dup2(devnull_w.fileno(), 2)

    def _apply_sysfs_boost_profile(self) -> None:
        """Configure kernel cpufreq governors, frequencies, boost flags, and EPP."""
        online_cpus = self.sysfs.get_online_cpus()

        # Set scaling governor to performance
        self.sysfs.set_scaling_governor(self.config.governor, cpus=online_cpus)

        # Set boost and CPB
        self.sysfs.enable_all_boost()

        # Set scaling min frequency - clamp to max available to avoid EINVAL when target (4.0 GHz boost) exceeds scaling_max (3.0 GHz base)
        try:
            max_khz = self.sysfs.get_scaling_max_freq(online_cpus[0]) if online_cpus else None
        except Exception:
            max_khz = None
        # Use target if within available range, else use max base freq
        try:
            available = self.sysfs.get_available_frequencies(online_cpus[0]) if online_cpus else []
        except Exception:
            available = []
        if max_khz and self.config.target_frequency_khz > max_khz:
            # Check if target is a valid boost freq (via boost flag), else clamp to max
            # For acpi-cpufreq on this Ryzen, boost is via cpb flag, min should be max base
            min_freq_to_set = max_khz
        else:
            min_freq_to_set = self.config.target_frequency_khz
        self.sysfs.set_scaling_min_freq(min_freq_to_set, cpus=online_cpus)

        # Set Energy Performance Preference
        self.sysfs.set_energy_performance_preference(self.config.epp, cpus=online_cpus)

    def _on_thermal_warning(self, reading: ThermalReading) -> None:
        """Callback invoked when temperature enters warning zone."""
        logger.warning(
            f"Temperature is {reading.current_temp_c:.1f}C. "
            f"Pulse-duty clamp is {reading.clamp_factor:.2f}."
        )

    def _on_thermal_tripwire(self, reading: ThermalReading) -> None:
        """Callback invoked when emergency thermal tripwire triggers."""
        logger.critical(
            f"Temperature {reading.current_temp_c:.1f}C reached the "
            f"{self.config.thermal_limit_c}C limit. Pausing pulse stimulation."
        )
        with self._lock:
            self._state = DaemonState.THROTTLED
            try:
                self.pulse_engine.pause()
            except Exception as e:
                logger.error(f"Error pausing pulse engine on tripwire: {e}")

    def _on_thermal_recovery(self, reading: ThermalReading) -> None:
        """Callback invoked when temperature recovers below hysteretic recovery floor."""
        logger.info(
            f"Temperature {reading.current_temp_c:.1f}C is below the "
            f"{self.config.thermal_recover_c}C recovery limit. Resuming pulse stimulation."
        )
        with self._lock:
            if self._state == DaemonState.THROTTLED:
                self._state = DaemonState.RUNNING
                try:
                    self.pulse_engine.resume()
                except Exception as e:
                    logger.error(f"Error resuming pulse engine on recovery: {e}")

    def _emergency_rollback(self, exc: Exception) -> None:
        """Perform emergency cleanup and state restoration on startup failure."""
        with self._lock:
            self._state = DaemonState.ERROR
        try:
            self.pulse_engine.stop()
        except Exception:
            pass
        try:
            self.thermal_guard.stop()
        except Exception:
            pass
        try:
            self.pm_qos.unlock()
        except Exception:
            pass
        try:
            self.state_manager.restore()
        except Exception:
            pass
        try:
            self.ipc_server.stop()
        except Exception:
            pass
        try:
            self.pid_manager.release()
        except Exception:
            pass
        self._unregister_signals()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Signal handler for graceful shutdown on SIGINT/SIGTERM."""
        logger.info(f"Signal {signum} intercepted by BoostLock daemon, executing clean shutdown...")
        self.stop()

    def _register_signals(self) -> None:
        """Register signal handlers for clean rollback on termination."""
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            try:
                self._orig_signals[sig] = signal.signal(sig, self._signal_handler)
            except (ValueError, OSError) as exc:
                logger.debug(f"Could not register handler for signal {sig}: {exc}")
        self._signal_handlers_registered = True

    def _unregister_signals(self) -> None:
        """Restore original signal handlers."""
        if not self._signal_handlers_registered or threading.current_thread() is not threading.main_thread():
            return
        for sig, orig in list(self._orig_signals.items()):
            try:
                if orig is not None:
                    signal.signal(sig, orig)
            except (ValueError, OSError):
                pass
        self._orig_signals.clear()
        self._signal_handlers_registered = False
