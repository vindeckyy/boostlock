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
from boostlock.sysfs import PolicyApplyAction, PolicyApplyPlan, SysfsController, SysfsError
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
        self._pm_qos_skip_reason: Optional[str] = None

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

            # 2. Build the exact reversible policy change set and snapshot it.
            policy_plan = self._build_policy_apply_plan()
            self.state_manager.create_snapshot(policy_plan.actions)

            # 3. Register signal handlers
            self._register_signals()

            # 4. Apply the full cpufreq and PM QoS transaction.
            self._apply_startup_policy_transaction(policy_plan)

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

        except SysfsError as exc:
            logger.exception(f"Failed to start BoostLock daemon: {exc}")
            rollback_errors = self._emergency_rollback(
                exc,
                final_state=DaemonState.STOPPED,
                restore_state=False,
            )
            message = f"Daemon startup failed: {exc}"
            if rollback_errors:
                message += f"; rollback failed: {'; '.join(rollback_errors)}"
            raise DaemonError(message) from exc
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
            self._apply_startup_policy_transaction()
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
        policies, text_status = self._policy_status(is_locked)
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
                "skipped_reason": self._pm_qos_skip_reason,
            },
            "pm_qos_active": self.pm_qos.is_locked,
            "pulse_engine": pulse_status,
            "policies": policies,
            "text_status": text_status,
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
                result = updated.to_dict()
                result["policy_targets"] = self.resolve_policy_targets()
                return Response.ok(result, request_id=req.request_id)

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

    def resolve_policy_targets(self) -> Dict[str, Dict[str, int]]:
        """Return each policy's independently clamped target frequency."""
        policies, _ = self._policy_status(self._is_locked_boost)
        return {
            policy_id: {"effective_target_khz": details["effective_target_khz"]}
            for policy_id, details in policies.items()
            if details["effective_target_khz"] is not None
        }

    def _policy_status(self, is_locked: bool) -> tuple[Dict[str, Dict[str, Any]], str]:
        """Build policy-level status without treating one unavailable node as fatal."""
        try:
            policies = self.sysfs.discover_cpufreq_policies()
            plan = self._build_policy_apply_plan()
        except Exception as exc:
            return {}, f"Policy status unavailable: {exc}"

        actions_by_policy: Dict[str, Dict[str, str]] = {}
        for action in plan.actions:
            if action.policy_id == "pm_qos":
                continue
            actions_by_policy.setdefault(action.policy_id, {})[action.control] = action.value

        requested_target = self.config.target_frequency_khz
        result: Dict[str, Dict[str, Any]] = {}
        text_lines: List[str] = []
        for policy in policies:
            planned = actions_by_policy.get(policy.identifier, {})
            effective = self._policy_effective_target(planned)
            skipped = dict(policy.skipped_controls)
            skipped.update(plan.skipped_controls.get(policy.identifier, {}))
            clamp_reason = self._policy_clamp_reason(policy, requested_target, effective)
            applied = dict(planned) if is_locked else {}
            result[policy.identifier] = {
                "identifier": policy.identifier,
                "member_cpus": list(policy.cpus),
                "driver": policy.driver,
                "requested_target": requested_target,
                "effective_target_khz": effective,
                "clamp_reason": clamp_reason,
                "applied_controls": applied,
                "skipped_controls": skipped,
            }
            controls = ", ".join(sorted(applied)) or "none"
            skipped_text = ", ".join(sorted(skipped)) or "none"
            target_text = str(effective) if effective is not None else "unavailable"
            text_lines.append(
                f"{policy.identifier} cpus={policy.cpus} driver={policy.driver or 'unknown'} "
                f"requested={requested_target} effective={target_text} "
                f"clamp={clamp_reason or 'none'} applied={controls} skipped={skipped_text}"
            )
        return result, "\n".join(text_lines)

    @staticmethod
    def _policy_effective_target(planned_controls: Dict[str, str]) -> Optional[int]:
        """Read the requested policy target from its planned min-frequency action."""
        target = planned_controls.get("active_min_frequency")
        try:
            return int(target) if target is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _policy_clamp_reason(
        policy: Any,
        requested_target: Any,
        effective_target: Optional[int],
    ) -> Optional[str]:
        """Explain how one policy converted the configured request into a target."""
        if effective_target is None:
            return "policy frequency bounds unavailable"
        if requested_target == "auto":
            return "automatic policy maximum"
        if not isinstance(requested_target, int):
            return "invalid requested target"

        lower_bounds = [
            value
            for value in (policy.hardware_min_khz, policy.active_min_khz)
            if isinstance(value, int) and value > 0
        ]
        upper_bounds = [
            value
            for value in (policy.hardware_max_khz, policy.active_max_khz)
            if isinstance(value, int) and value > 0
        ]
        if not lower_bounds or not upper_bounds:
            return "policy frequency bounds unavailable"
        lower_bound = max(lower_bounds)
        upper_bound = min(upper_bounds)
        if requested_target < lower_bound:
            return f"raised to policy minimum {lower_bound}"
        if requested_target > upper_bound:
            return f"clamped to policy maximum {upper_bound}"
        return None

    def _apply_sysfs_boost_profile(self) -> None:
        """Apply only the reversible cpufreq portion of the policy plan."""
        self.sysfs.execute_policy_apply_plan(self._build_policy_apply_plan())

    def _apply_startup_policy_transaction(
        self,
        plan: Optional[PolicyApplyPlan] = None,
    ) -> None:
        """Apply cpufreq and one optional PM QoS route as one transaction."""
        plan = plan or self._build_policy_apply_plan()
        route, route_paths = self.pm_qos.select_planned_route()
        fallback_snapshots: Dict[Path, str] = {}

        if route == "device":
            plan.actions.append(
                PolicyApplyAction(
                    "pm_qos",
                    "device",
                    self.dma_latency_path.resolve(),
                    str(self.config.dma_latency_us),
                    "release",
                )
            )
            self._pm_qos_skip_reason = None
        elif route == "cpuidle":
            for path in route_paths:
                original_value = self.sysfs._read_path(path)
                if original_value is None:
                    continue
                resolved = path.resolve()
                fallback_snapshots[resolved] = original_value
                plan.actions.append(
                    PolicyApplyAction(
                        "pm_qos",
                        "cpuidle",
                        resolved,
                        "1",
                        original_value,
                    )
                )
            route = "cpuidle" if fallback_snapshots else None
            self._pm_qos_skip_reason = None if route else "no usable PM QoS route"
        else:
            self._pm_qos_skip_reason = "no usable PM QoS route"

        plan.preflight_paths = self._unique_plan_paths(
            [action.path for action in plan.actions]
        )
        self.sysfs.execute_policy_apply_plan(
            plan,
            open_for_write=self._transaction_opener(route),
            writer=self._transaction_writer(route, fallback_snapshots),
        )
        if route == "cpuidle":
            self.pm_qos.activate_planned_fallback(fallback_snapshots)

    def _build_policy_apply_plan(self) -> PolicyApplyPlan:
        """Build the policy-owned cpufreq part of a daemon transaction."""
        target = self.config.target_frequency_khz
        target_khz = None if target == "auto" else int(target)
        return self.sysfs.build_policy_apply_plan(
            target_khz=target_khz,
            governor=self.config.governor,
            boost=True,
            cpb=True,
            energy_performance_preference=self.config.epp,
        )

    def _transaction_opener(self, route: Optional[str]):
        """Return a write-open check that understands the DMA-latency descriptor."""
        device_path = self.dma_latency_path.resolve()

        def open_for_write(path: Path) -> Any:
            if route == "device" and path.resolve() == device_path:
                return self.pm_qos.open_device_for_preflight()
            return self.sysfs._open_path_for_write(path)

        return open_for_write

    def _transaction_writer(
        self,
        route: Optional[str],
        fallback_snapshots: Dict[Path, str],
    ):
        """Write plan actions and compensate the PM QoS descriptor when required."""
        device_path = self.dma_latency_path.resolve()
        fallback_paths = set(fallback_snapshots)

        def write(path: Path, value: str) -> None:
            resolved = path.resolve()
            if route == "device" and resolved == device_path:
                if value == "release":
                    self.pm_qos.release_strict()
                    return
                try:
                    self.pm_qos.lock(int(value), allow_fallback=False)
                except Exception as exc:
                    try:
                        self.pm_qos.release_strict()
                    except Exception as rollback_exc:
                        raise SysfsError(f"{exc}; rollback failed: {rollback_exc}") from exc
                    raise
                return
            if route == "cpuidle" and resolved in fallback_paths:
                subpath = str(resolved.relative_to(self.sysfs_root))
                self.pm_qos.sysfs._write_file(subpath, value)
                return
            self.sysfs._write_absolute_path(resolved, value)

        return write

    @staticmethod
    def _unique_plan_paths(paths: Sequence[Path]) -> List[Path]:
        """Deduplicate plan paths after resolving any aliases."""
        unique: List[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return unique

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

    def _emergency_rollback(
        self,
        exc: Exception,
        final_state: DaemonState = DaemonState.ERROR,
        restore_state: bool = True,
    ) -> List[str]:
        """Perform emergency cleanup and state restoration on startup failure."""
        rollback_errors: List[str] = []
        cleanups = [
            ("pulse engine", self.pulse_engine.stop),
            ("thermal guard", self.thermal_guard.stop),
            ("PM QoS", self.pm_qos.unlock),
            ("IPC server", self.ipc_server.stop),
            ("PID lock", self.pid_manager.release),
        ]
        if restore_state:
            cleanups.insert(3, ("state restore", self.state_manager.restore))
        for name, cleanup in cleanups:
            try:
                cleanup()
            except Exception as rollback_exc:
                rollback_errors.append(f"{name}: {rollback_exc}")
        self._unregister_signals()
        with self._lock:
            self._state = final_state
            self._is_locked_boost = False
            self._stop_event.set()
        return rollback_errors

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
