"""
Sysfs abstraction layer for Linux CPU frequency scaling, governors, boost, and EPP.
"""

from __future__ import annotations

import errno
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
