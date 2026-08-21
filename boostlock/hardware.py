"""
Hardware detection and CPU topology discovery module for BoostLock.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from boostlock.sysfs import (
    CapabilityState,
    CpufreqPolicy,
    SysfsController,
    SysfsError,
    parse_cpu_range,
)

logger = logging.getLogger(__name__)


class CPUVendor(str, Enum):
    """CPU vendor."""
    AMD = "AuthenticAMD"
    INTEL = "GenuineIntel"
    UNKNOWN = "Unknown"


class ScalingDriver(str, Enum):
    """Scaling driver."""
    ACPI_CPUFREQ = "acpi-cpufreq"
    AMD_PSTATE = "amd-pstate"
    AMD_PSTATE_EPP = "amd-pstate-epp"
    INTEL_PSTATE = "intel_pstate"
    INTEL_CPUFREQ = "intel_cpufreq"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class CoreInfo:
    """One CPU core."""
    cpu_id: int
    physical_core_id: int = 0
    socket_id: int = 0
    online: bool = True
    cur_freq_khz: Optional[int] = None
    min_freq_khz: Optional[int] = None
    max_freq_khz: Optional[int] = None
    base_freq_khz: Optional[int] = None
    boost_freq_khz: Optional[int] = None
    governor: Optional[str] = None
    driver: Optional[str] = None


@dataclass
class CPUInfo:
    """System CPU info."""
    vendor: CPUVendor = CPUVendor.UNKNOWN
    vendor_raw: str = ""
    model_name: str = "Generic CPU"
    family: int = 0
    model: int = 0
    stepping: int = 0
    logical_cpus: int = 1
    physical_cores: int = 1
    sockets: int = 1
    flags: Set[str] = field(default_factory=set)
    base_freq_mhz: float = 2000.0
    max_boost_mhz: float = 2000.0
    min_freq_mhz: float = 800.0
    scaling_driver: ScalingDriver = ScalingDriver.UNKNOWN
    has_cpb: bool = False
    has_boost: bool = False
    has_epp: bool = False
    has_epb: bool = False
    cores: List[CoreInfo] = field(default_factory=list)
    core_to_threads: Dict[int, List[int]] = field(default_factory=dict)
    policies: List[CpufreqPolicy] = field(default_factory=list)


def detect_vendor(vendor_str: str) -> CPUVendor:
    """Map vendor string to enum."""
    if "AuthenticAMD" in vendor_str:
        return CPUVendor.AMD
    if "GenuineIntel" in vendor_str:
        return CPUVendor.INTEL
    return CPUVendor.UNKNOWN


def parse_driver(driver_str: Optional[str]) -> ScalingDriver:
    """Map driver string to enum."""
    if not driver_str:
        return ScalingDriver.UNKNOWN
    driver_clean = driver_str.strip().lower()
    for d in ScalingDriver:
        if d.value == driver_clean:
            return d
    return ScalingDriver.GENERIC


def parse_proc_cpuinfo(content: str) -> Dict[str, Any]:
    """Parse /proc/cpuinfo."""
    processors: List[Dict[str, str]] = []
    current_proc: Dict[str, str] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line:
            if current_proc:
                processors.append(current_proc)
                current_proc = {}
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            current_proc[key.strip()] = val.strip()

    if current_proc:
        processors.append(current_proc)

    if not processors:
        return {
            "vendor_id": "Unknown",
            "model_name": "Generic CPU",
            "family": 0,
            "model": 0,
            "stepping": 0,
            "logical_count": 1,
            "physical_cores": 1,
            "sockets": 1,
            "flags": set(),
            "mhz": 2000.0,
        }

    p0 = processors[0]
    vendor_id = p0.get("vendor_id", "Unknown")
    model_name = p0.get("model name", p0.get("model_name", "Generic CPU"))

    try:
        family = int(p0.get("cpu family", "0"))
    except ValueError:
        family = 0

    try:
        model = int(p0.get("model", "0"))
    except ValueError:
        model = 0

    try:
        stepping = int(p0.get("stepping", "0"))
    except ValueError:
        stepping = 0

    try:
        mhz = float(p0.get("cpu MHz", "2000.0"))
    except ValueError:
        mhz = 2000.0

    flags: Set[str] = set()
    for p in processors:
        if "flags" in p:
            flags.update(p["flags"].split())

    # Count distinct physical cores and sockets
    physical_cores_set = set()
    sockets_set = set()
    for p in processors:
        core_id = p.get("core id", "0")
        phys_id = p.get("physical id", "0")
        sockets_set.add(phys_id)
        physical_cores_set.add(f"{phys_id}:{core_id}")

    try:
        cpu_cores_attr = int(p0.get("cpu cores", str(len(physical_cores_set) or 1)))
    except ValueError:
        cpu_cores_attr = len(physical_cores_set) or 1

    physical_count = max(cpu_cores_attr, len(physical_cores_set), 1)

    return {
        "vendor_id": vendor_id,
        "model_name": model_name,
        "family": family,
        "model": model,
        "stepping": stepping,
        "logical_count": len(processors),
        "physical_cores": physical_count,
        "sockets": len(sockets_set) or 1,
        "flags": flags,
        "mhz": mhz,
    }


def lookup_boost_frequency(
    model_name: str,
    base_mhz: float,
    flags: Set[str],
    sysfs_root: Optional[str] = None,
) -> float:
    """Get boost limit if known, else policy max."""
    del model_name, flags
    if not sysfs_root:
        return base_mhz

    controller = SysfsController(sysfs_root)
    boost_limits_khz: List[int] = []
    for policy in controller.discover_cpufreq_policies():
        try:
            value = controller._read_path(policy.path / "scaling_boost_frequencies")
        except SysfsError as exc:
            logger.debug("Could not read scaling_boost_frequencies: %s", exc)
            continue
        if not value:
            continue
        boost_limits_khz.extend(
            frequency
            for item in value.split()
            if (frequency := SysfsController._parse_optional_int(item)) is not None
        )
    return max(boost_limits_khz) / 1000.0 if boost_limits_khz else base_mhz


def detect_cpu_info(
    proc_cpuinfo_path: str = "/proc/cpuinfo",
    sysfs_root: str = "/sys",
) -> CPUInfo:
    """
    Perform full system CPU hardware detection and topology discovery.
    """
    controller = SysfsController(sysfs_root)
    proc_path = Path(proc_cpuinfo_path)

    if proc_path.is_file():
        try:
            content = proc_path.read_text(encoding="utf-8")
            parsed = parse_proc_cpuinfo(content)
        except Exception as exc:
            logger.warning(f"Failed to parse {proc_cpuinfo_path}: {exc}")
            parsed = parse_proc_cpuinfo("")
    else:
        parsed = parse_proc_cpuinfo("")

    vendor_raw = parsed["vendor_id"]
    vendor = detect_vendor(vendor_raw)
    model_name = parsed["model_name"]
    flags = parsed["flags"]

    policies = controller.discover_cpufreq_policies()
    primary_policy = next((policy for policy in policies if policy.usable), policies[0] if policies else None)

    # Online logical CPUs
    online_cpus = controller.get_online_cpus()
    logical_count = max(len(online_cpus), parsed["logical_count"], 1)

    # Scaling driver
    raw_driver = primary_policy.driver if primary_policy else controller.get_scaling_driver(0)
    scaling_driver = parse_driver(raw_driver)

    # Base and min frequencies from sysfs or proc
    max_base_khz = (
        primary_policy.hardware_max_khz
        if primary_policy is not None
        else controller.get_cpuinfo_max_freq(0)
    )
    min_khz = (
        primary_policy.hardware_min_khz
        if primary_policy is not None
        else controller.get_cpuinfo_min_freq(0)
    )

    if max_base_khz and max_base_khz > 0:
        base_freq_mhz = max_base_khz / 1000.0
    else:
        base_freq_mhz = parsed["mhz"]

    if min_khz and min_khz > 0:
        min_freq_mhz = min_khz / 1000.0
    else:
        min_freq_mhz = 800.0

    # Max boost clock calculation
    max_boost_mhz = lookup_boost_frequency(model_name, base_freq_mhz, flags, sysfs_root=sysfs_root)

    def has_available_control(name: str) -> bool:
        return any(
            policy.capabilities.get(name) == CapabilityState.AVAILABLE
            for policy in policies
        )

    has_cpb = has_available_control("cpb")
    has_boost = has_available_control("boost") or has_cpb
    has_epp = has_available_control("energy_performance_preference")
    has_epb = has_available_control("energy_perf_bias")

    # Build per-core topology and thread mapping
    cores: List[CoreInfo] = []
    core_to_threads: Dict[int, List[int]] = {}
    physical_cores_set = set()
    policies_by_cpu = {
        cpu: policy
        for policy in policies
        for cpu in policy.cpus
    }

    for cpu_id in online_cpus:
        # Read topology
        core_id_val = controller._read_int(f"devices/system/cpu/cpu{cpu_id}/topology/core_id", default=None)
        phys_pkg_val = controller._read_int(f"devices/system/cpu/cpu{cpu_id}/topology/physical_package_id", default=0)

        physical_core_id = core_id_val if core_id_val is not None else (cpu_id // 2 if logical_count > parsed["physical_cores"] else cpu_id)
        socket_id = phys_pkg_val if phys_pkg_val is not None else 0

        physical_cores_set.add((socket_id, physical_core_id))
        core_to_threads.setdefault(physical_core_id, []).append(cpu_id)
        policy = policies_by_cpu.get(cpu_id)

        core_info = CoreInfo(
            cpu_id=cpu_id,
            physical_core_id=physical_core_id,
            socket_id=socket_id,
            online=True,
            cur_freq_khz=(policy.current_freq_khz if policy else controller.get_scaling_cur_freq(cpu_id)),
            min_freq_khz=(policy.active_min_khz if policy else controller.get_scaling_min_freq(cpu_id)),
            max_freq_khz=(policy.active_max_khz if policy else controller.get_scaling_max_freq(cpu_id)),
            base_freq_khz=int(base_freq_mhz * 1000),
            boost_freq_khz=int(max_boost_mhz * 1000),
            governor=(policy.governor if policy else controller.get_scaling_governor(cpu_id)),
            driver=(policy.driver if policy else raw_driver),
        )
        cores.append(core_info)

    physical_count = len(physical_cores_set) or parsed["physical_cores"]

    return CPUInfo(
        vendor=vendor,
        vendor_raw=vendor_raw,
        model_name=model_name,
        family=parsed["family"],
        model=parsed["model"],
        stepping=parsed["stepping"],
        logical_cpus=logical_count,
        physical_cores=physical_count,
        sockets=parsed["sockets"],
        flags=flags,
        base_freq_mhz=base_freq_mhz,
        max_boost_mhz=max_boost_mhz,
        min_freq_mhz=min_freq_mhz,
        scaling_driver=scaling_driver,
        has_cpb=has_cpb,
        has_boost=has_boost,
        has_epp=has_epp,
        has_epb=has_epb,
        cores=cores,
        core_to_threads=core_to_threads,
        policies=policies,
    )
