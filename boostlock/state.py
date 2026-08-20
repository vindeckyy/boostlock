"""
System state snapshot, atomic persistence, signal handlers, and crash rollback manager.

The manager records CPU settings and restores them during normal cleanup.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import os
import signal
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from boostlock.sysfs import SysfsController, SysfsError

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_DIR = Path("/var/run/boostlock")
FALLBACK_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "boostlock"
SNAPSHOT_FILENAME = "snapshot.json"


class StateError(Exception):
    """Base exception for system state snapshot and rollback operations."""
    pass


class StateSnapshotError(StateError):
    """Raised when capturing system state fails."""
    pass


class StateRestoreError(StateError):
    """Raised when restoring system state fails."""
    pass


@dataclass
class CPUStateSnapshot:
    """Snapshot of cpufreq and cpuidle attributes for a single logical CPU."""

    cpu_id: int
    governor: Optional[str] = None
    scaling_min_freq: Optional[int] = None
    scaling_max_freq: Optional[int] = None
    epp: Optional[str] = None
    epb: Optional[int] = None
    cpb: Optional[bool] = None
    cpuidle_states: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CPUStateSnapshot:
        return cls(
            cpu_id=int(data["cpu_id"]),
            governor=data.get("governor"),
            scaling_min_freq=data.get("scaling_min_freq"),
            scaling_max_freq=data.get("scaling_max_freq"),
            epp=data.get("epp"),
            epb=data.get("epb"),
            cpb=data.get("cpb"),
            cpuidle_states=dict(data.get("cpuidle_states", {})),
        )


@dataclass
class SystemStateSnapshot:
    """Global system CPU frequency and idle configuration state snapshot."""

    global_boost: Optional[bool] = None
    cpus: Dict[int, CPUStateSnapshot] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    hostname: str = field(default_factory=socket.gethostname)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_boost": self.global_boost,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "cpus": {str(cpu_id): snap.to_dict() for cpu_id, snap in self.cpus.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SystemStateSnapshot:
        cpus_data = data.get("cpus", {})
        cpus = {
            int(cpu_id): CPUStateSnapshot.from_dict(snap_data)
            for cpu_id, snap_data in cpus_data.items()
        }
        return cls(
            global_boost=data.get("global_boost"),
            cpus=cpus,
            timestamp=float(data.get("timestamp", time.time())),
            hostname=str(data.get("hostname", "")),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> SystemStateSnapshot:
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Union[str, Path]) -> None:
        """Atomically persist snapshot to disk with .tmp staging file."""
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(self.to_json(), encoding="utf-8")
            tmp_path.replace(target_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise StateSnapshotError(f"Failed to write snapshot to {target_path}: {exc}") from exc

    @classmethod
    def load(cls, path: Union[str, Path]) -> SystemStateSnapshot:
        """Load and deserialize snapshot from disk."""
        target_path = Path(path).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"Snapshot file not found: {target_path}")
        content = target_path.read_text(encoding="utf-8")
        return cls.from_json(content)


def resolve_default_snapshot_path() -> Path:
    """Determine the optimal writable snapshot file path."""
    try:
        DEFAULT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DEFAULT_SNAPSHOT_DIR / ".write_test"
        test_file.touch()
        test_file.unlink(missing_ok=True)
        return DEFAULT_SNAPSHOT_DIR / SNAPSHOT_FILENAME
    except (PermissionError, OSError):
        FALLBACK_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        return FALLBACK_SNAPSHOT_DIR / SNAPSHOT_FILENAME


class StateSnapshotManager:
    """
    Manages taking, persisting, and atomically restoring system CPU state snapshots.
    Integrates POSIX signal interception and atexit triggers for foolproof recovery.
    """

    def __init__(
        self,
        sysfs_controller: Optional[SysfsController] = None,
        snapshot_file: Optional[Union[str, Path]] = None,
        restore_on_exit: bool = False,
    ) -> None:
        self.sysfs = sysfs_controller or SysfsController()
        if snapshot_file is not None:
            self.snapshot_file = Path(snapshot_file).resolve()
        else:
            self.snapshot_file = resolve_default_snapshot_path()

        self.restore_on_exit = restore_on_exit
        self.snapshot: Optional[SystemStateSnapshot] = None

        self._signal_handlers_registered: bool = False
        self._previous_signal_handlers: Dict[int, Any] = {}
        self._atexit_registered: bool = False

    def create_snapshot(self) -> SystemStateSnapshot:
        """Capture current CPU configuration and persist snapshot to disk."""
        online_cpus = self.sysfs.get_online_cpus()
        cpu_snapshots: Dict[int, CPUStateSnapshot] = {}

        for cpu in online_cpus:
            # Capture cpuidle states
            cpuidle_states: Dict[str, int] = {}
            cpu_idle_dir = self.sysfs._resolve_path(f"devices/system/cpu/cpu{cpu}/cpuidle")
            if cpu_idle_dir.is_dir():
                for state_entry in sorted(cpu_idle_dir.iterdir()):
                    if state_entry.is_dir() and state_entry.name.startswith("state"):
                        disable_val = self.sysfs._read_file(
                            f"devices/system/cpu/cpu{cpu}/cpuidle/{state_entry.name}/disable"
                        )
                        if disable_val is not None:
                            try:
                                cpuidle_states[state_entry.name] = int(disable_val)
                            except ValueError:
                                cpuidle_states[state_entry.name] = 0

            cpu_snap = CPUStateSnapshot(
                cpu_id=cpu,
                governor=self.sysfs.get_scaling_governor(cpu),
                scaling_min_freq=self.sysfs.get_scaling_min_freq(cpu),
                scaling_max_freq=self.sysfs.get_scaling_max_freq(cpu),
                epp=self.sysfs.get_energy_performance_preference(cpu),
                epb=self.sysfs.get_energy_perf_bias(cpu),
                cpb=self.sysfs.get_cpb(cpu),
                cpuidle_states=cpuidle_states,
            )
            cpu_snapshots[cpu] = cpu_snap

        global_boost = self.sysfs.get_boost()

        snapshot = SystemStateSnapshot(
            global_boost=global_boost,
            cpus=cpu_snapshots,
            timestamp=time.time(),
            hostname=socket.gethostname(),
        )

        try:
            snapshot.save(self.snapshot_file)
            logger.info(f"System state snapshot saved to {self.snapshot_file}")
        except Exception as exc:
            logger.warning(f"Failed to persist state snapshot to disk: {exc}")

        self.snapshot = snapshot
        return snapshot

    def restore(
        self,
        snapshot: Optional[SystemStateSnapshot] = None,
        delete_snapshot_file: bool = True,
    ) -> bool:
        """
        Restore CPU scaling and idle configurations from snapshot or disk file.
        Falls back to safe defaults if snapshot is missing or corrupted.
        """
        target_snapshot = snapshot or self.snapshot

        if target_snapshot is None and self.snapshot_file.exists():
            try:
                target_snapshot = SystemStateSnapshot.load(self.snapshot_file)
            except Exception as exc:
                logger.warning(
                    f"Corrupted or unreadable snapshot at {self.snapshot_file} ({exc}). Using safe fallback."
                )
                return self.restore_fallback()

        if target_snapshot is None:
            logger.warning("No snapshot available to restore. Executing safe fallback restore.")
            return self.restore_fallback()

        errors: List[str] = []

        # 1. Restore global boost
        if target_snapshot.global_boost is not None:
            try:
                self.sysfs.set_boost(target_snapshot.global_boost)
            except Exception as exc:
                errors.append(f"Failed to restore global boost: {exc}")

        # 2. Restore per-CPU configurations
        for cpu_id, cpu_snap in target_snapshot.cpus.items():
            # A. Restore frequencies with strict ordering to avoid min > max kernel errors
            try:
                curr_min = self.sysfs.get_scaling_min_freq(cpu_id, default=0) or 0
                curr_max = self.sysfs.get_scaling_max_freq(cpu_id, default=0) or 0

                snap_min = cpu_snap.scaling_min_freq
                snap_max = cpu_snap.scaling_max_freq

                if snap_min is not None and snap_max is not None:
                    if snap_min > curr_max:
                        # Expanding frequency upper bound first
                        self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq", str(snap_max), optional=True)
                        self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_min_freq", str(snap_min), optional=True)
                    else:
                        # Lowering frequency lower bound first
                        self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_min_freq", str(snap_min), optional=True)
                        self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq", str(snap_max), optional=True)
                elif snap_min is not None:
                    self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_min_freq", str(snap_min), optional=True)
                elif snap_max is not None:
                    self.sysfs._write_file(f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq", str(snap_max), optional=True)

            except Exception as exc:
                errors.append(f"Failed to restore frequency bounds on CPU {cpu_id}: {exc}")

            # B. Restore governor
            if cpu_snap.governor:
                try:
                    self.sysfs.set_scaling_governor(cpu_snap.governor, cpus=[cpu_id])
                except Exception as exc:
                    errors.append(f"Failed to restore governor '{cpu_snap.governor}' on CPU {cpu_id}: {exc}")

            # C. Restore EPP / EPB
            if cpu_snap.epp:
                try:
                    self.sysfs.set_energy_performance_preference(cpu_snap.epp, cpus=[cpu_id])
                except Exception as exc:
                    logger.debug(f"Non-critical: Failed to restore EPP on CPU {cpu_id}: {exc}")

            if cpu_snap.epb is not None:
                try:
                    self.sysfs.set_energy_perf_bias(cpu_snap.epb, cpus=[cpu_id])
                except Exception as exc:
                    logger.debug(f"Non-critical: Failed to restore EPB on CPU {cpu_id}: {exc}")

            # D. Restore CPB
            if cpu_snap.cpb is not None:
                try:
                    self.sysfs.set_cpb(cpu_snap.cpb, cpus=[cpu_id])
                except Exception as exc:
                    logger.debug(f"Non-critical: Failed to restore CPB on CPU {cpu_id}: {exc}")

            # E. Restore cpuidle disable states
            for state_name, disable_val in cpu_snap.cpuidle_states.items():
                try:
                    self.sysfs._write_file(
                        f"devices/system/cpu/cpu{cpu_id}/cpuidle/{state_name}/disable",
                        str(disable_val),
                        optional=True,
                    )
                except Exception as exc:
                    logger.debug(f"Non-critical: Failed to restore cpuidle {state_name} on CPU {cpu_id}: {exc}")

        if delete_snapshot_file and self.snapshot_file.exists():
            try:
                self.snapshot_file.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug(f"Could not remove snapshot file {self.snapshot_file}: {exc}")

        self.snapshot = None

        if errors:
            logger.warning(f"State restore completed with {len(errors)} warnings: {'; '.join(errors)}")
        else:
            logger.info("System state successfully restored to initial snapshot.")

        return True

    def restore_fallback(self) -> bool:
        """Apply safe fallback configurations when no snapshot exists."""
        logger.info("Executing safe hardware fallback restoration.")
        online_cpus = self.sysfs.get_online_cpus()

        for cpu in online_cpus:
            # Query available governors, fallback to schedutil/ondemand/powersave
            avail = self.sysfs.get_available_governors(cpu)
            chosen_gov = "powersave"
            for preferred in ("schedutil", "ondemand", "powersave"):
                if preferred in avail:
                    chosen_gov = preferred
                    break

            try:
                self.sysfs.set_scaling_governor(chosen_gov, cpus=[cpu])
            except Exception as exc:
                logger.warning(f"Fallback: Failed to set governor {chosen_gov} on CPU {cpu}: {exc}")

            # Reset min freq to hardware min
            hw_min = self.sysfs.get_cpuinfo_min_freq(cpu, default=1000000)
            if hw_min:
                try:
                    self.sysfs.set_scaling_min_freq(hw_min, cpus=[cpu])
                except Exception as exc:
                    logger.debug(f"Fallback: Failed to reset min freq on CPU {cpu}: {exc}")

            # Reset max freq to hardware max
            hw_max = self.sysfs.get_cpuinfo_max_freq(cpu, default=None)
            if hw_max:
                try:
                    self.sysfs.set_scaling_max_freq(hw_max, cpus=[cpu])
                except Exception as exc:
                    logger.debug(f"Fallback: Failed to reset max freq on CPU {cpu}: {exc}")

            # Re-enable all cpuidle states
            cpu_idle_dir = self.sysfs._resolve_path(f"devices/system/cpu/cpu{cpu}/cpuidle")
            if cpu_idle_dir.is_dir():
                for state_entry in sorted(cpu_idle_dir.iterdir()):
                    if state_entry.is_dir() and state_entry.name.startswith("state"):
                        self.sysfs._write_file(
                            f"devices/system/cpu/cpu{cpu}/cpuidle/{state_entry.name}/disable",
                            "0",
                            optional=True,
                        )

        # Restore boost switches
        try:
            self.sysfs.set_boost(True)
        except Exception:
            pass

        return True

    def register_signal_handlers(self) -> None:
        """Register signal handlers for SIGINT, SIGTERM, SIGHUP, and SIGQUIT."""
        if self._signal_handlers_registered:
            return

        target_signals = [signal.SIGINT, signal.SIGTERM]
        for sig_name in ("SIGHUP", "SIGQUIT"):
            if hasattr(signal, sig_name):
                target_signals.append(getattr(signal, sig_name))

        for sig in target_signals:
            try:
                prev = signal.signal(sig, self._signal_handler)
                self._previous_signal_handlers[sig] = prev
            except (ValueError, OSError) as exc:
                logger.debug(f"Could not register handler for signal {sig}: {exc}")

        self._signal_handlers_registered = True

    def unregister_signal_handlers(self) -> None:
        """Restore previous signal handlers."""
        if not self._signal_handlers_registered:
            return

        for sig, prev in list(self._previous_signal_handlers.items()):
            try:
                if prev is not None:
                    signal.signal(sig, prev)
            except (ValueError, OSError):
                pass

        self._previous_signal_handlers.clear()
        self._signal_handlers_registered = False

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Internal signal handler to execute rollback before process termination."""
        sig_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
        logger.warning(f"Intercepted termination signal {sig_name} ({signum}). Initiating rollback.")
        self.restore(delete_snapshot_file=True)

        prev_handler = self._previous_signal_handlers.get(signum)
        if (
            prev_handler is not None
            and prev_handler not in (signal.SIG_DFL, signal.SIG_IGN, signal.default_int_handler)
            and callable(prev_handler)
        ):
            prev_handler(signum, frame)
        else:
            sys.exit(128 + signum)

    def register_atexit(self) -> None:
        """Register atexit cleanup hook."""
        if not self._atexit_registered:
            atexit.register(self._atexit_callback)
            self._atexit_registered = True

    def unregister_atexit(self) -> None:
        """Unregister atexit cleanup hook."""
        if self._atexit_registered:
            atexit.unregister(self._atexit_callback)
            self._atexit_registered = False

    def _atexit_callback(self) -> None:
        """Callback executed on normal or abnormal Python process exit."""
        if self.snapshot is not None:
            logger.info("Atexit hook triggered: executing state rollback.")
            self.restore(delete_snapshot_file=True)

    def __enter__(self) -> StateSnapshotManager:
        self.create_snapshot()
        self.register_signal_handlers()
        self.register_atexit()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.unregister_signal_handlers()
        self.unregister_atexit()
        if exc_type is not None or self.restore_on_exit:
            self.restore(delete_snapshot_file=True)
