"""
Hardware detection and CPU topology discovery module for BoostLock.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from boostlock.sysfs import SysfsController, parse_cpu_range

logger = logging.getLogger(__name__)


class CPUVendor(str, Enum):
    """CPU Hardware Vendor."""
    AMD = "AuthenticAMD"
    INTEL = "GenuineIntel"
    UNKNOWN = "Unknown"


class ScalingDriver(str, Enum):
    """CPU Frequency Scaling Driver."""
    ACPI_CPUFREQ = "acpi-cpufreq"
    AMD_PSTATE = "amd-pstate"
    AMD_PSTATE_EPP = "amd-pstate-epp"
    INTEL_PSTATE = "intel_pstate"
    INTEL_CPUFREQ = "intel_cpufreq"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class CoreInfo:
    """Detailed information for a single logical CPU core."""
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
    """System-wide CPU topology, capabilities, and frequency characteristics."""
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


# Known boost frequencies database (Model Substring -> Max Boost MHz)
KNOWN_BOOST_FREQUENCIES: Dict[str, float] = {
    # AMD Ryzen 4000 series (Renoir)
    "4600H": 4000.0,
    "4600HS": 4000.0,
    "4600U": 4000.0,
    "4600G": 4200.0,
    "4700U": 4100.0,
    "4800H": 4200.0,
    "4800HS": 4200.0,
    "4800U": 4200.0,
    "4900H": 4400.0,
    "4900HS": 4300.0,
    # AMD Ryzen 5000 series (Cezanne / Vermeer / Lucienne)
    "5500U": 4000.0,
    "5600U": 4200.0,
    "5600H": 4200.0,
    "5600G": 4400.0,
    "5600X": 4600.0,
    "5700U": 4300.0,
    "5800H": 4400.0,
    "5800HS": 4400.0,
    "5800U": 4400.0,
    "5800X": 4700.0,
    "5800X3D": 4500.0,
    "5900HX": 4600.0,
    "5900HS": 4600.0,
    "5900X": 4800.0,
    "5950X": 4900.0,
    # AMD Ryzen 6000 / 7000 / 8000 series
    "6600H": 4500.0,
    "6800H": 4700.0,
    "6800U": 4700.0,
    "6900HX": 4900.0,
    "7600X": 5300.0,
    "7700X": 5400.0,
    "7800X3D": 5000.0,
    "7900X": 5600.0,
    "7950X": 5700.0,
    "7950X3D": 5700.0,
    "8845HS": 5100.0,
    # Intel Core 10th - 14th Gen
    "10750H": 5000.0,
    "10850H": 5100.0,
    "10875H": 5100.0,
    "10900K": 5300.0,
    "11700K": 5000.0,
    "11800H": 4600.0,
    "11900K": 5300.0,
    "12700H": 4700.0,
    "12700K": 5000.0,
    "12900H": 5000.0,
    "12900K": 5200.0,
    "13600K": 5100.0,
    "13700H": 5000.0,
    "13700K": 5400.0,
    "13900H": 5400.0,
    "13900K": 5800.0,
    "14700K": 5600.0,
    "14900K": 6000.0,
}


def detect_vendor(vendor_str: str) -> CPUVendor:
    """Map raw vendor ID string to CPUVendor enum."""
    if "AuthenticAMD" in vendor_str:
        return CPUVendor.AMD
    if "GenuineIntel" in vendor_str:
        return CPUVendor.INTEL
    return CPUVendor.UNKNOWN


def parse_driver(driver_str: Optional[str]) -> ScalingDriver:
    """Map raw driver string to ScalingDriver enum."""
    if not driver_str:
        return ScalingDriver.UNKNOWN
    driver_clean = driver_str.strip().lower()
    for d in ScalingDriver:
        if d.value == driver_clean:
            return d
    return ScalingDriver.GENERIC


def parse_proc_cpuinfo(content: str) -> Dict[str, Any]:
    """Parse `/proc/cpuinfo` content into structured dictionary."""
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
    """
    Determine max boost clock frequency in MHz.
    Checks:
    1. Sysfs scaling_boost_frequencies if available
    2. Known CPU model database
    3. Model name @ X.XXGHz clock extraction
    4. Hardware capability heuristics
    """
    # 1. Check sysfs scaling_boost_frequencies
    if sysfs_root:
        ctrl = SysfsController(sysfs_root)
        boost_freqs_file = ctrl._resolve_path("devices/system/cpu/cpu0/cpufreq/scaling_boost_frequencies")
        if boost_freqs_file.is_file():
            try:
                content = boost_freqs_file.read_text(encoding="utf-8").strip()
                if content:
                    boost_ints = [int(x) for x in content.split() if x.isdigit()]
                    if boost_ints:
                        return max(boost_ints) / 1000.0
            except Exception as exc:
                logger.debug(f"Could not read scaling_boost_frequencies: {exc}")

    # 2. Check known CPU database by token/substring
    for key, boost_val in KNOWN_BOOST_FREQUENCIES.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, model_name, re.IGNORECASE):
            return boost_val

    # 3. Check if model name has explicit boost or base frequency (e.g. '@ 2.60GHz')
    m = re.search(r"@\s*([\d\.]+)\s*GHz", model_name, re.IGNORECASE)
    parsed_ghz_mhz = float(m.group(1)) * 1000.0 if m else base_mhz

    # 4. Heuristic fallback based on CPB / Turbo flags
    if "cpb" in flags or "ida" in flags or "hwp" in flags:
        return max(parsed_ghz_mhz * 1.30, base_mhz)

    return max(parsed_ghz_mhz, base_mhz)


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

    # Online logical CPUs
    online_cpus = controller.get_online_cpus()
    logical_count = max(len(online_cpus), parsed["logical_count"], 1)

    # Scaling driver
    raw_driver = controller.get_scaling_driver(0)
    scaling_driver = parse_driver(raw_driver)

    # Base and min frequencies from sysfs or proc
    max_base_khz = controller.get_cpuinfo_max_freq(0)
    min_khz = controller.get_cpuinfo_min_freq(0)

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

    # Check hardware boost flags
    has_cpb = "cpb" in flags or controller.get_cpb(0) is not None
    has_boost = (
        controller.get_boost() is not None
        or has_cpb
        or "ida" in flags
        or "hwp" in flags
    )
    has_epp = (
        controller.get_energy_performance_preference(0) is not None
        or "hwp_epp" in flags
    )
    has_epb = (
        controller.get_energy_perf_bias(0) is not None
        or "epb" in flags
    )

    # Build per-core topology and thread mapping
    cores: List[CoreInfo] = []
    core_to_threads: Dict[int, List[int]] = {}
    physical_cores_set = set()

    for cpu_id in online_cpus:
        # Read topology
        core_id_val = controller._read_int(f"devices/system/cpu/cpu{cpu_id}/topology/core_id", default=None)
        phys_pkg_val = controller._read_int(f"devices/system/cpu/cpu{cpu_id}/topology/physical_package_id", default=0)

        physical_core_id = core_id_val if core_id_val is not None else (cpu_id // 2 if logical_count > parsed["physical_cores"] else cpu_id)
        socket_id = phys_pkg_val if phys_pkg_val is not None else 0

        physical_cores_set.add((socket_id, physical_core_id))
        core_to_threads.setdefault(physical_core_id, []).append(cpu_id)

        core_info = CoreInfo(
            cpu_id=cpu_id,
            physical_core_id=physical_core_id,
            socket_id=socket_id,
            online=True,
            cur_freq_khz=controller.get_scaling_cur_freq(cpu_id),
            min_freq_khz=controller.get_scaling_min_freq(cpu_id),
            max_freq_khz=controller.get_scaling_max_freq(cpu_id),
            base_freq_khz=int(base_freq_mhz * 1000),
            boost_freq_khz=int(max_boost_mhz * 1000),
            governor=controller.get_scaling_governor(cpu_id),
            driver=raw_driver,
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
    )
