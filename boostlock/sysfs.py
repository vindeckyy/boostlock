"""
Sysfs abstraction layer for Linux CPU frequency scaling, governors, boost, and EPP.
"""

from __future__ import annotations

import errno
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class SysfsError(Exception):
    """Base exception for sysfs operations."""
    pass


class SysfsPermissionError(SysfsError, PermissionError):
    """Raised when sysfs read/write fails due to insufficient permissions."""
    pass


class SysfsNotFoundError(SysfsError, FileNotFoundError):
    """Raised when a required sysfs path does not exist."""
    pass


class SysfsCorruptError(SysfsError, ValueError):
    """Raised when sysfs file contains invalid or corrupted data."""
    pass


class CapabilityState(str, Enum):
    """Whether a cpufreq control can be used on a discovered policy."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNUSABLE = "unusable"


@dataclass
class CpufreqPolicy:
    """Inventory of one Linux cpufreq policy and its exposed controls."""

    identifier: str
    path: Path
    cpus: List[int]
    driver: Optional[str]
    governor: Optional[str]
    available_governors: List[str]
    current_freq_khz: Optional[int]
    active_min_khz: Optional[int]
    active_max_khz: Optional[int]
    hardware_min_khz: Optional[int]
    hardware_max_khz: Optional[int]
    boost: Optional[str] = None
    cpb: Optional[str] = None
    energy_performance_preference: Optional[str] = None
    energy_perf_bias: Optional[str] = None
    capabilities: Dict[str, CapabilityState] = field(default_factory=dict)
    writable_paths: Dict[str, Path] = field(default_factory=dict)
    skipped_controls: Dict[str, str] = field(default_factory=dict)
    state: CapabilityState = CapabilityState.UNUSABLE
    usable: bool = False

    @property
    def governors(self) -> List[str]:
        """Return the governors advertised by this policy."""
        return self.available_governors


@dataclass(frozen=True)
class PolicyApplyAction:
    """One reversible cpufreq write prepared from a policy inventory."""

    policy_id: str
    control: str
    path: Path
    value: str
    original_value: str


@dataclass
class PolicyApplyPlan:
    """A complete, preflightable set of policy-owned sysfs writes."""

    actions: List[PolicyApplyAction]
    skipped_controls: Dict[str, Dict[str, str]]
    preflight_paths: List[Path]


def parse_cpu_range(range_str: str) -> List[int]:
    """Parse CPU range string like '0-3,5,7-8' into sorted list of integers."""
    cpus: set[int] = set()
    range_str = range_str.strip()
    if not range_str:
        return []

    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
                cpus.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.add(int(part))
            except ValueError:
                continue
    return sorted(cpus)


class SysfsController:
    """Provides high-level, typed, mockable access to Linux kernel CPU sysfs nodes."""

    def __init__(self, sysfs_root: Union[str, Path] = "/sys") -> None:
        self.sysfs_root = Path(sysfs_root).resolve()

    def _resolve_path(self, subpath: str) -> Path:
        """Resolve a relative subpath against sysfs root."""
        clean_subpath = subpath.lstrip("/")
        return self.sysfs_root / clean_subpath

    def _read_file(self, subpath: str) -> Optional[str]:
        """Read content from sysfs file. Returns None if file does not exist."""
        path = self._resolve_path(subpath)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8").strip()
        except PermissionError as exc:
            raise SysfsPermissionError(
                f"Permission denied reading {path}. Run with sudo/root privileges."
            ) from exc
        except OSError as exc:
            logger.debug(f"Error reading sysfs path {path}: {exc}")
            return None

    def _write_file(self, subpath: str, value: str, optional: bool = False) -> bool:
        """
        Write string value to sysfs file.
        Returns True if successful, False if file doesn't exist and optional=True.
        """
        path = self._resolve_path(subpath)
        if not path.exists():
            if optional:
                return False
            raise SysfsNotFoundError(f"Sysfs file not found: {path}")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{value.strip()}\n")
            return True
        except PermissionError as exc:
            raise SysfsPermissionError(
                f"Permission denied writing '{value}' to {path}. "
                "Ensure boostlock is executed with root/sudo privileges."
            ) from exc
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
                raise SysfsPermissionError(
                    f"Permission denied (errno {exc.errno}) writing to {path}."
                ) from exc
            if optional:
                logger.debug(f"Non-critical write failed for {path}: {exc}")
                return False
            raise SysfsError(f"Failed to write '{value}' to {path}: {exc}") from exc

    def _read_int(self, subpath: str, default: Optional[int] = None) -> Optional[int]:
        """Read integer value from sysfs file."""
        content = self._read_file(subpath)
        if content is None:
            return default
        try:
            return int(content)
        except ValueError:
            logger.warning(f"Sysfs file {subpath} contained non-integer value: {content!r}")
            return default

    # -------------------------------------------------------------------------
    # CPU Topology & Discovery
    # -------------------------------------------------------------------------

    def get_online_cpus(self) -> List[int]:
        """Return list of online logical CPU IDs."""
        online_str = self._read_file("devices/system/cpu/online")
        if online_str:
            cpus = parse_cpu_range(online_str)
            if cpus:
                return cpus

        # Fallback: scan cpu[0-9]+ directories
        cpu_base = self._resolve_path("devices/system/cpu")
        if cpu_base.is_dir():
            cpus = []
            for entry in cpu_base.iterdir():
                if entry.is_dir() and re.match(r"^cpu\d+$", entry.name):
                    try:
                        cpus.append(int(entry.name[3:]))
                    except ValueError:
                        pass
            if cpus:
                return sorted(cpus)
        return [0]

    def discover_cpufreq_policies(self, require_usable: bool = False) -> List[CpufreqPolicy]:
        """Discover global cpufreq policies, falling back to per-CPU aliases."""
        policy_root = self._resolve_path("devices/system/cpu/cpufreq")
        candidates: List[tuple[str, Path]] = []
        if policy_root.is_dir():
            candidates = [
                (entry.name, entry)
                for entry in sorted(policy_root.iterdir(), key=lambda entry: entry.name)
                if entry.is_dir() and re.match(r"^policy\d+$", entry.name)
            ]
            candidates.sort(key=lambda candidate: int(candidate[0][6:]))

        if not candidates:
            for cpu in self.get_online_cpus():
                path = self._resolve_path(f"devices/system/cpu/cpu{cpu}/cpufreq")
                if path.is_dir():
                    candidates.append((f"cpu{cpu}", path))

        policies: List[CpufreqPolicy] = []
        seen_paths: set[Path] = set()
        for fallback_identifier, path in candidates:
            resolved_path = path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            identifier = resolved_path.name if re.match(r"^policy\d+$", resolved_path.name) else fallback_identifier
            policies.append(self._inventory_cpufreq_policy(identifier, resolved_path))

        if require_usable and not any(policy.usable for policy in policies):
            raise SysfsError("No usable policy was discovered")
        return policies

    def _inventory_cpufreq_policy(self, identifier: str, path: Path) -> CpufreqPolicy:
        """Read one policy directory without assuming a processor architecture."""
        members = self._read_policy_cpus(path)
        if not members:
            members = self._policy_cpus_from_aliases(path)

        node_values = {
            "governor": self._read_path(path / "scaling_governor"),
            "available_governors": self._read_path(path / "scaling_available_governors"),
            "current_frequency": self._read_path(path / "scaling_cur_freq"),
            "active_min_frequency": self._read_path(path / "scaling_min_freq"),
            "active_max_frequency": self._read_path(path / "scaling_max_freq"),
            "hardware_min_frequency": self._read_path(path / "cpuinfo_min_freq"),
            "hardware_max_frequency": self._read_path(path / "cpuinfo_max_freq"),
        }
        capabilities = {
            name: self._capability_state(path / filename, value)
            for name, filename, value in (
                ("governor", "scaling_governor", node_values["governor"]),
                ("available_governors", "scaling_available_governors", node_values["available_governors"]),
                ("current_frequency", "scaling_cur_freq", node_values["current_frequency"]),
                ("active_min_frequency", "scaling_min_freq", node_values["active_min_frequency"]),
                ("active_max_frequency", "scaling_max_freq", node_values["active_max_frequency"]),
                ("hardware_min_frequency", "cpuinfo_min_freq", node_values["hardware_min_frequency"]),
                ("hardware_max_frequency", "cpuinfo_max_freq", node_values["hardware_max_frequency"]),
            )
        }
        for name in (
            "current_frequency",
            "active_min_frequency",
            "active_max_frequency",
            "hardware_min_frequency",
            "hardware_max_frequency",
        ):
            if capabilities[name] == CapabilityState.AVAILABLE and self._parse_optional_int(node_values[name]) is None:
                capabilities[name] = CapabilityState.UNUSABLE
        for name in ("governor", "available_governors"):
            if capabilities[name] == CapabilityState.AVAILABLE and not node_values[name]:
                capabilities[name] = CapabilityState.UNUSABLE

        control_paths = {
            "boost": self._boost_path(path),
            "cpb": path / "cpb",
            "energy_performance_preference": path / "energy_performance_preference",
            "energy_perf_bias": self._energy_perf_bias_path(path, members),
        }
        controls: Dict[str, Optional[str]] = {}
        skipped_controls: Dict[str, str] = {}
        for name, control_path in control_paths.items():
            value = self._read_path(control_path) if control_path is not None else None
            controls[name] = value
            capabilities[name] = self._capability_state(control_path, value)
            if capabilities[name] == CapabilityState.AVAILABLE and not value:
                capabilities[name] = CapabilityState.UNUSABLE
            if capabilities[name] != CapabilityState.AVAILABLE:
                skipped_controls[name] = self._skip_reason(capabilities[name])

        writable_paths = {
            name: candidate
            for name, candidate in {
                "governor": path / "scaling_governor",
                "active_min_frequency": path / "scaling_min_freq",
                "active_max_frequency": path / "scaling_max_freq",
                **{name: candidate for name, candidate in control_paths.items() if candidate is not None},
            }.items()
            if candidate.is_file()
        }
        required = ("governor", "active_min_frequency", "active_max_frequency")
        usable = bool(members) and all(capabilities[name] == CapabilityState.AVAILABLE for name in required)
        return CpufreqPolicy(
            identifier=identifier,
            path=path,
            cpus=members,
            driver=self._read_path(path / "scaling_driver"),
            governor=node_values["governor"],
            available_governors=(node_values["available_governors"] or "").split(),
            current_freq_khz=self._parse_optional_int(node_values["current_frequency"]),
            active_min_khz=self._parse_optional_int(node_values["active_min_frequency"]),
            active_max_khz=self._parse_optional_int(node_values["active_max_frequency"]),
            hardware_min_khz=self._parse_optional_int(node_values["hardware_min_frequency"]),
            hardware_max_khz=self._parse_optional_int(node_values["hardware_max_frequency"]),
            boost=controls["boost"],
            cpb=controls["cpb"],
            energy_performance_preference=controls["energy_performance_preference"],
            energy_perf_bias=controls["energy_perf_bias"],
            capabilities=capabilities,
            writable_paths=writable_paths,
            skipped_controls=skipped_controls,
            state=CapabilityState.AVAILABLE if usable else CapabilityState.UNUSABLE,
            usable=usable,
        )

    def _read_policy_cpus(self, path: Path) -> List[int]:
        """Read the kernel-provided membership list for a policy."""
        for name in ("related_cpus", "affected_cpus"):
            value = self._read_path(path / name)
            if value:
                return parse_cpu_range(re.sub(r"\s+", ",", value))
        return []

    def _policy_cpus_from_aliases(self, policy_path: Path) -> List[int]:
        """Infer policy membership only for older per-CPU cpufreq layouts."""
        members = []
        for cpu in self.get_online_cpus():
            alias = self._resolve_path(f"devices/system/cpu/cpu{cpu}/cpufreq")
            if alias.is_dir() and alias.resolve() == policy_path:
                members.append(cpu)
        return members

    def _boost_path(self, policy_path: Path) -> Optional[Path]:
        """Return a policy-local boost control or the shared cpufreq control."""
        local = policy_path / "boost"
        if local.is_file():
            return local
        shared = self._resolve_path("devices/system/cpu/cpufreq/boost")
        return shared if shared.is_file() else None

    def _energy_perf_bias_path(self, policy_path: Path, cpus: Sequence[int]) -> Optional[Path]:
        """Find EPB where the active driver exposes it for this policy."""
        local = policy_path / "energy_perf_bias"
        if local.is_file():
            return local
        if not cpus:
            return None
        power_node = self._resolve_path(f"devices/system/cpu/cpu{cpus[0]}/power/energy_perf_bias")
        return power_node if power_node.is_file() else None

    @staticmethod
    def _parse_optional_int(value: Optional[str]) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _skip_reason(state: CapabilityState) -> str:
        if state == CapabilityState.UNAVAILABLE:
            return "node unavailable"
        return "node unusable"

    def _read_path(self, path: Optional[Path]) -> Optional[str]:
        """Read an absolute sysfs path using the controller's error handling."""
        if path is None or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8").strip()
        except PermissionError as exc:
            raise SysfsPermissionError(
                f"Permission denied reading {path}. Run with sudo/root privileges."
            ) from exc
        except OSError as exc:
            logger.debug(f"Error reading sysfs path {path}: {exc}")
            return None

    @staticmethod
    def _capability_state(path: Optional[Path], value: Optional[str]) -> CapabilityState:
        if path is None or not path.is_file():
            return CapabilityState.UNAVAILABLE
        return CapabilityState.AVAILABLE if value is not None else CapabilityState.UNUSABLE

    # -------------------------------------------------------------------------
    # Policy Apply Planning
    # -------------------------------------------------------------------------

    def build_policy_apply_plan(
        self,
        target_khz: Optional[int],
        governor: Optional[str] = None,
        pm_qos_device: Optional[Path] = None,
        cpuidle_fallback_paths: Sequence[Path] = (),
        *,
        boost: Optional[bool] = None,
        cpb: Optional[bool] = None,
        energy_performance_preference: Optional[str] = None,
    ) -> PolicyApplyPlan:
        """Plan reversible writes for each usable policy without mutating sysfs."""
        actions: List[PolicyApplyAction] = []
        skipped_controls: Dict[str, Dict[str, str]] = {}
        planned_boost_paths: set[Path] = set()
        optional_requests = (
            ("boost", None if boost is None else "1" if boost else "0"),
            ("cpb", None if cpb is None else "1" if cpb else "0"),
            ("energy_performance_preference", energy_performance_preference),
            ("energy_perf_bias", None),
        )

        for policy in self.discover_cpufreq_policies(require_usable=True):
            policy_skips = dict(policy.skipped_controls)
            if not policy.usable:
                policy_skips["policy"] = "policy unusable"
                for control, _ in optional_requests:
                    if policy.capabilities[control] == CapabilityState.AVAILABLE:
                        policy_skips[control] = "policy unusable"
                skipped_controls[policy.identifier] = policy_skips
                continue

            effective_target = self._effective_policy_target(policy, target_khz)
            if effective_target is None:
                policy_skips["active_min_frequency"] = "frequency limits unavailable"
            else:
                if governor is not None:
                    if governor in policy.available_governors:
                        actions.append(self._planned_policy_action(policy, "governor", governor))
                    else:
                        policy_skips["governor"] = "governor unavailable"

                actions.append(
                    self._planned_policy_action(
                        policy,
                        "active_min_frequency",
                        str(effective_target),
                    )
                )

            for control, value in optional_requests:
                if policy.capabilities[control] != CapabilityState.AVAILABLE:
                    continue
                if value is None:
                    policy_skips[control] = "no configured value"
                    continue
                path = policy.writable_paths[control].resolve()
                if control == "boost" and path in planned_boost_paths:
                    policy_skips[control] = "shared control already planned"
                    continue
                actions.append(self._planned_policy_action(policy, control, value))
                if control == "boost":
                    planned_boost_paths.add(path)
            if policy_skips:
                skipped_controls[policy.identifier] = policy_skips

        preflight_paths = self._unique_paths(
            [action.path for action in actions]
            + ([Path(pm_qos_device)] if pm_qos_device is not None else [])
            + [Path(path) for path in cpuidle_fallback_paths]
        )
        return PolicyApplyPlan(actions, skipped_controls, preflight_paths)

    def preflight_policy_apply_plan(
        self,
        plan: PolicyApplyPlan,
        open_for_write: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        """Open every planned path for write access before the first mutation."""
        opener = open_for_write or self._open_path_for_write
        for path in plan.preflight_paths:
            try:
                handle = opener(path)
            except OSError as exc:
                raise SysfsError(f"Cannot open {path} for write: {exc}") from exc
            try:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()
            except OSError as exc:
                raise SysfsError(f"Cannot close write check for {path}: {exc}") from exc

    def execute_policy_apply_plan(
        self,
        plan: PolicyApplyPlan,
        open_for_write: Optional[Callable[[Path], Any]] = None,
        writer: Optional[Callable[[Path, str], None]] = None,
    ) -> None:
        """Preflight and execute a plan, compensating completed writes on failure."""
        self.preflight_policy_apply_plan(plan, open_for_write=open_for_write)
        write = writer or self._write_absolute_path
        completed: List[PolicyApplyAction] = []

        try:
            for action in plan.actions:
                write(action.path, action.value)
                completed.append(action)
        except Exception as exc:
            rollback_errors: List[str] = []
            for action in reversed(completed):
                try:
                    write(action.path, action.original_value)
                except Exception as rollback_exc:
                    rollback_errors.append(f"{action.path}: {rollback_exc}")

            message = f"Failed to apply {action.policy_id} {action.control} at {action.path}: {exc}"
            if rollback_errors:
                message += f"; rollback failed: {'; '.join(rollback_errors)}"
            raise SysfsError(message) from exc

    def _planned_policy_action(
        self,
        policy: CpufreqPolicy,
        control: str,
        value: str,
    ) -> PolicyApplyAction:
        """Snapshot a discovered writable path before scheduling its mutation."""
        path = policy.writable_paths.get(control)
        if path is None:
            raise SysfsError(f"Policy {policy.identifier} has no writable {control}")
        original_value = self._read_path(path)
        if original_value is None:
            raise SysfsError(f"Cannot snapshot {path} before mutation")
        return PolicyApplyAction(policy.identifier, control, path, value, original_value)

    @staticmethod
    def _effective_policy_target(policy: CpufreqPolicy, target_khz: Optional[int]) -> Optional[int]:
        """Resolve one request inside a policy's currently valid frequency interval."""
        lower_bounds = [
            limit
            for limit in (policy.hardware_min_khz, policy.active_min_khz)
            if limit is not None and limit > 0
        ]
        upper_bounds = [
            limit
            for limit in (policy.hardware_max_khz, policy.active_max_khz)
            if limit is not None and limit > 0
        ]
        if not lower_bounds or not upper_bounds:
            return None

        lower_bound = max(lower_bounds)
        upper_bound = min(upper_bounds)
        if lower_bound > upper_bound:
            return None
        requested = upper_bound if target_khz is None else target_khz
        return min(max(requested, lower_bound), upper_bound)

    @staticmethod
    def _unique_paths(paths: Sequence[Path]) -> List[Path]:
        """Return paths once, preserving plan order after symlink resolution."""
        unique: List[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return unique

    @staticmethod
    def _open_path_for_write(path: Path) -> Any:
        """Open a path with write access for preflight without changing its contents."""
        return open(path, "r+", encoding="utf-8")

    @staticmethod
    def _write_absolute_path(path: Path, value: str) -> None:
        """Write one previously discovered path using the normal sysfs text format."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{value.strip()}\n")

    # -------------------------------------------------------------------------
    # Governor Management
    # -------------------------------------------------------------------------

    def get_scaling_governor(self, cpu: int = 0) -> Optional[str]:
        """Get current scaling governor for a CPU."""
        return self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")

    def get_available_governors(self, cpu: int = 0) -> List[str]:
        """Get list of available governors for a CPU."""
        content = self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_governors")
        if not content:
            return []
        return content.split()

    def set_scaling_governor(self, governor: str, cpus: Optional[Sequence[int]] = None) -> None:
        """Set scaling governor across specified CPUs (defaults to all online CPUs)."""
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor", governor)

    # -------------------------------------------------------------------------
    # Frequency Limits & Queries (in kHz)
    # -------------------------------------------------------------------------

    def get_scaling_min_freq(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get current scaling minimum frequency in kHz."""
        return self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_min_freq", default=default)

    def set_scaling_min_freq(self, freq_khz: int, cpus: Optional[Sequence[int]] = None) -> None:
        """Set scaling minimum frequency in kHz across CPUs."""
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_min_freq", str(freq_khz))

    def get_scaling_max_freq(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get current scaling maximum frequency in kHz."""
        return self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq", default=default)

    def set_scaling_max_freq(self, freq_khz: int, cpus: Optional[Sequence[int]] = None) -> None:
        """Set scaling maximum frequency in kHz across CPUs."""
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq", str(freq_khz))

    def get_scaling_cur_freq(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get current scaling frequency in kHz."""
        return self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", default=default)

    def get_cpuinfo_min_freq(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get hardware minimum frequency in kHz."""
        return self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_min_freq", default=default)

    def get_cpuinfo_max_freq(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get hardware maximum base frequency in kHz."""
        return self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq", default=default)

    def get_available_frequencies(self, cpu: int = 0) -> List[int]:
        """Get available scaling frequencies in kHz."""
        content = self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_frequencies")
        if not content:
            return []
        freqs = []
        for item in content.split():
            try:
                freqs.append(int(item))
            except ValueError:
                pass
        return sorted(freqs)

    # -------------------------------------------------------------------------
    # Boost & AMD CPB (Core Performance Boost)
    # -------------------------------------------------------------------------

    def get_boost(self) -> Optional[bool]:
        """Query global or CPU0 boost switch status."""
        val = self._read_file("devices/system/cpu/cpufreq/boost")
        if val is None:
            val = self._read_file("devices/system/cpu/cpu0/cpufreq/boost")
        if val is None:
            val = self._read_file("devices/system/cpu/cpu0/cpufreq/cpb")
        if val is None:
            return None
        return val.strip() == "1"

    def set_boost(self, enable: bool) -> None:
        """Enable or disable global frequency boost."""
        val = "1" if enable else "0"
        # Try global boost first
        self._write_file("devices/system/cpu/cpufreq/boost", val, optional=True)
        # Try per-cpu boost
        for cpu in self.get_online_cpus():
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/boost", val, optional=True)

    def get_cpb(self, cpu: int = 0) -> Optional[bool]:
        """Query AMD Core Performance Boost (CPB) status for a CPU."""
        val = self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/cpb")
        if val is None:
            return None
        return val.strip() == "1"

    def set_cpb(self, enable: bool, cpus: Optional[Sequence[int]] = None) -> None:
        """Set AMD Core Performance Boost (CPB) status for specified CPUs."""
        val = "1" if enable else "0"
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/cpb", val, optional=True)

    def enable_all_boost(self) -> bool:
        """Enable all available boost switches (global boost and per-core cpb)."""
        self.set_boost(True)
        self.set_cpb(True)
        return True

    # -------------------------------------------------------------------------
    # Energy Performance Preference (EPP) & Bias (EPB)
    # -------------------------------------------------------------------------

    def get_energy_performance_preference(self, cpu: int = 0) -> Optional[str]:
        """Get Energy Performance Preference (EPP) string for a CPU."""
        return self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/energy_performance_preference")

    def get_available_energy_performance_preferences(self, cpu: int = 0) -> List[str]:
        """Get available EPP options for a CPU."""
        content = self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/energy_performance_available_preferences")
        if not content:
            return []
        return content.split()

    def set_energy_performance_preference(self, epp: str, cpus: Optional[Sequence[int]] = None) -> None:
        """Set Energy Performance Preference (EPP) across CPUs."""
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/energy_performance_preference", epp, optional=True)

    def get_energy_perf_bias(self, cpu: int = 0, default: Optional[int] = None) -> Optional[int]:
        """Get Energy Performance Bias (EPB) integer for Intel CPUs."""
        val = self._read_int(f"devices/system/cpu/cpu{cpu}/power/energy_perf_bias", default=None)
        if val is None:
            val = self._read_int(f"devices/system/cpu/cpu{cpu}/cpufreq/energy_perf_bias", default=default)
        return val if val is not None else default

    def set_energy_perf_bias(self, epb: int, cpus: Optional[Sequence[int]] = None) -> None:
        """Set Energy Performance Bias (EPB) across CPUs (0=performance, 15=powersave)."""
        target_cpus = self.get_online_cpus() if cpus is None else cpus
        for cpu in target_cpus:
            wrote = self._write_file(f"devices/system/cpu/cpu{cpu}/power/energy_perf_bias", str(epb), optional=True)
            if not wrote:
                self._write_file(f"devices/system/cpu/cpu{cpu}/cpufreq/energy_perf_bias", str(epb), optional=True)

    # -------------------------------------------------------------------------
    # Scaling Driver
    # -------------------------------------------------------------------------

    def get_scaling_driver(self, cpu: int = 0) -> Optional[str]:
        """Get CPU scaling driver name."""
        return self._read_file(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_driver")

    # -------------------------------------------------------------------------
    # State Inspection
    # -------------------------------------------------------------------------

    def read_cpu_state(self, cpu: int) -> Dict[str, Any]:
        """Read full snapshot of cpufreq attributes for a single CPU."""
        return {
            "cpu_id": cpu,
            "governor": self.get_scaling_governor(cpu),
            "scaling_min_freq": self.get_scaling_min_freq(cpu),
            "scaling_max_freq": self.get_scaling_max_freq(cpu),
            "scaling_cur_freq": self.get_scaling_cur_freq(cpu),
            "cpuinfo_min_freq": self.get_cpuinfo_min_freq(cpu),
            "cpuinfo_max_freq": self.get_cpuinfo_max_freq(cpu),
            "driver": self.get_scaling_driver(cpu),
            "cpb": self.get_cpb(cpu),
            "epp": self.get_energy_performance_preference(cpu),
            "epb": self.get_energy_perf_bias(cpu),
        }

    def read_all_cpus_state(self) -> Dict[int, Dict[str, Any]]:
        """Read cpufreq snapshot for all online CPUs."""
        states = {}
        for cpu in self.get_online_cpus():
            states[cpu] = self.read_cpu_state(cpu)
        return states
