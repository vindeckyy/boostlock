"""
Thermal safety watchdog, hwmon/thermal_zone sensor discovery, and cooling clamp controller.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from boostlock.config import BoostLockConfig

logger = logging.getLogger(__name__)


class ThermalError(Exception):
    """Base exception for thermal subsystem errors."""
    pass


class SensorReadError(ThermalError):
    """Raised when reading from a thermal sensor fails."""
    pass


class ThermalTripwireError(ThermalError):
    """Raised when the emergency thermal tripwire limit is breached."""
    pass


class ThermalState(str, Enum):
    """Thermal operating zones for BoostLock."""
    NORMAL = "NORMAL"        # T < T_warn: Normal operation, full boost allowed (clamp = 1.0)
    WARNING = "WARNING"      # T_warn <= T < T_trip: Duty cycle throttled proportionally
    THROTTLED = "THROTTLED"  # Actively throttled or recovering from tripwire below T_recover
    CRITICAL = "CRITICAL"    # T >= T_trip: Emergency tripwire triggered, boost suspended


class SensorType(str, Enum):
    """Types of thermal and power monitoring interfaces."""
    HWMON = "hwmon"
    THERMAL_ZONE = "thermal_zone"
    POWERCAP = "powercap"


@dataclass
class ThermalSensor:
    """Represents a discovered hardware thermal sensor."""

    sensor_id: str
    name: str
    path: Path
    sensor_type: SensorType = SensorType.HWMON
    label: Optional[str] = None
    is_cpu: bool = True
    critical_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None

    def read_temp_c(self) -> Optional[float]:
        """
        Read temperature in degrees Celsius.
        Returns None if file is missing, unreadable, or contains invalid data.
        """
        if not self.path.is_file():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            val = float(raw)
            # Linux sysfs sensors report in millidegrees Celsius (e.g., 55000 = 55.0C)
            if abs(val) > 1000.0:
                temp_c = val / 1000.0
            else:
                temp_c = val

            # Basic sanity check (-40C to 150C)
            if -40.0 <= temp_c <= 150.0:
                return temp_c
            logger.debug(f"Sensor {self.sensor_id} reported out-of-bounds temp: {temp_c}C")
            return None
        except Exception as exc:
            logger.debug(f"Failed to read sensor {self.sensor_id} at {self.path}: {exc}")
            return None


class SpikeFilter:
    """
    Rolling window median filter and rate-of-change limiter.
    Filters out transient hardware reading glitches and single-sample sensor anomalies.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_temp_c: float = -20.0,
        max_temp_c: float = 130.0,
        max_rate_of_change: float = 30.0,
    ) -> None:
        self.window_size = max(1, window_size)
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.max_rate_of_change = max_rate_of_change
        self.history: List[float] = []

    def add_sample(self, val: float) -> float:
        """Add a raw temperature reading and return filtered temperature."""
        # Reject physical impossibilities
        if val < self.min_temp_c or val > self.max_temp_c:
            if self.history:
                return self.history[-1]
            return max(self.min_temp_c, min(self.max_temp_c, val))

        self.history.append(val)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return float(statistics.median(self.history))

    def reset(self) -> None:
        """Clear filter history."""
        self.history.clear()


@dataclass
class RAPLReading:
    """Snapshot of RAPL energy and computed power draw."""

    name: str
    energy_uj: Optional[int]
    power_w: Optional[float]
    timestamp: float


class RAPLMonitor:
    """
    Monitors Intel / AMD RAPL (Running Average Power Limit) package energy and wattage.
    """

    def __init__(self, sysfs_root: Union[str, Path] = "/sys") -> None:
        self.sysfs_root = Path(sysfs_root).resolve()
        self.rapl_dir = self.sysfs_root / "class" / "powercap" / "intel-rapl"
        self._last_energy_uj: Optional[int] = None
        self._last_timestamp: float = 0.0
        self._max_energy_range_uj: Optional[int] = None
        self._pkg_dir: Optional[Path] = None
        self._init_package_dir()

    def _init_package_dir(self) -> None:
        """Locate package-0 RAPL directory."""
        if not self.rapl_dir.is_dir():
            # Try alternate path /sys/devices/virtual/powercap/intel-rapl
            alt = self.sysfs_root / "devices" / "virtual" / "powercap" / "intel-rapl"
            if alt.is_dir():
                self.rapl_dir = alt
            else:
                return

        # Look for intel-rapl:0 or package-0
        for entry in sorted(self.rapl_dir.glob("intel-rapl:*")):
            if entry.is_dir():
                self._pkg_dir = entry
                self._read_max_range()
                break

    def _read_max_range(self) -> None:
        if not self._pkg_dir:
            return
        range_file = self._pkg_dir / "max_energy_range_uj"
        if range_file.is_file():
            try:
                self._max_energy_range_uj = int(range_file.read_text(encoding="utf-8").strip())
            except Exception:
                self._max_energy_range_uj = None

    def is_available(self) -> bool:
        """Check if RAPL energy monitoring is available."""
        if not self._pkg_dir:
            self._init_package_dir()
        if not self._pkg_dir:
            return False
        return (self._pkg_dir / "energy_uj").is_file()

    def read_power(self) -> Optional[RAPLReading]:
        """
        Read current RAPL energy and compute power in Watts since last reading.
        Returns None if RAPL is unavailable or unreadable.
        """
        if not self.is_available() or not self._pkg_dir:
            return None

        energy_file = self._pkg_dir / "energy_uj"
        name_file = self._pkg_dir / "name"

        pkg_name = "package-0"
        if name_file.is_file():
            try:
                pkg_name = name_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        try:
            current_energy = int(energy_file.read_text(encoding="utf-8").strip())
        except Exception as exc:
            logger.debug(f"Failed to read RAPL energy_uj: {exc}")
            return None

        now = time.time()
        power_w: Optional[float] = None

        if self._last_energy_uj is not None and self._last_timestamp > 0:
            dt = now - self._last_timestamp
            if dt > 0.0001:
                delta_uj = current_energy - self._last_energy_uj
                if delta_uj < 0 and self._max_energy_range_uj:
                    # Wraparound occurred
                    delta_uj += self._max_energy_range_uj
                if delta_uj >= 0:
                    power_w = (delta_uj / 1_000_000.0) / dt

        self._last_energy_uj = current_energy
        self._last_timestamp = now

        return RAPLReading(
            name=pkg_name,
            energy_uj=current_energy,
            power_w=power_w,
            timestamp=now,
        )


@dataclass
class ThermalReading:
    """Snapshot of complete thermal status."""

    timestamp: float
    current_temp_c: float
    sensors: Dict[str, float]
    state: ThermalState
    clamp_factor: float
    is_tripped: bool
    package_power_w: Optional[float] = None


def discover_sensors(sysfs_root: Union[str, Path] = "/sys") -> List[ThermalSensor]:
    """
    Discover all thermal sensors in hwmon and thermal_zone sysfs trees.
    CPU-specific sensors (k10temp, coretemp, acpitz, etc.) are prioritized first.
    """
    root = Path(sysfs_root).resolve()
    discovered: List[ThermalSensor] = []

    cpu_driver_names = {"k10temp", "coretemp", "zenpower", "acpitz", "cpu_thermal", "soc_thermal"}
    cpu_label_keywords = ["tctl", "tdie", "package", "core", "cpu", "x86_pkg"]

    # 1. Scan /sys/class/hwmon/hwmon*
    hwmon_base = root / "class" / "hwmon"
    if hwmon_base.is_dir():
        for hwmon_dir in sorted(hwmon_base.glob("hwmon*")):
            if not hwmon_dir.is_dir():
                continue

            name = "unknown"
            name_file = hwmon_dir / "name"
            if name_file.is_file():
                try:
                    name = name_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            for temp_input in sorted(hwmon_dir.glob("temp*_input")):
                stem = temp_input.name.split("_")[0]  # e.g., 'temp1'
                sensor_id = f"{hwmon_dir.name}_{stem}"

                label = None
                label_file = hwmon_dir / f"{stem}_label"
                if label_file.is_file():
                    try:
                        label = label_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass

                crit_c = None
                crit_file = hwmon_dir / f"{stem}_crit"
                if crit_file.is_file():
                    try:
                        crit_c = float(crit_file.read_text(encoding="utf-8").strip()) / 1000.0
                    except Exception:
                        pass

                max_c = None
                max_file = hwmon_dir / f"{stem}_max"
                if max_file.is_file():
                    try:
                        max_c = float(max_file.read_text(encoding="utf-8").strip()) / 1000.0
                    except Exception:
                        pass

                # Determine if this is a CPU sensor
                is_cpu = False
                if name.lower() in cpu_driver_names:
                    is_cpu = True
                elif label:
                    is_cpu = any(kw in label.lower() for kw in cpu_label_keywords)

                discovered.append(
                    ThermalSensor(
                        sensor_id=sensor_id,
                        name=name,
                        path=temp_input,
                        sensor_type=SensorType.HWMON,
                        label=label,
                        is_cpu=is_cpu,
                        critical_temp_c=crit_c,
                        max_temp_c=max_c,
                    )
                )

    # 2. Scan /sys/class/thermal/thermal_zone*
    thermal_base = root / "class" / "thermal"
    if thermal_base.is_dir():
        for zone_dir in sorted(thermal_base.glob("thermal_zone*")):
            if not zone_dir.is_dir():
                continue

            temp_file = zone_dir / "temp"
            if not temp_file.is_file():
                continue

            type_file = zone_dir / "type"
            zone_type = "thermal_zone"
            if type_file.is_file():
                try:
                    zone_type = type_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            is_cpu = any(kw in zone_type.lower() for kw in cpu_label_keywords + list(cpu_driver_names))

            discovered.append(
                ThermalSensor(
                    sensor_id=zone_dir.name,
                    name=zone_type,
                    path=temp_file,
                    sensor_type=SensorType.THERMAL_ZONE,
                    label=zone_type,
                    is_cpu=is_cpu,
                )
            )

    # Sort CPU sensors first
    discovered.sort(key=lambda s: (0 if s.is_cpu else 1, s.sensor_id))
    return discovered


class ThermalGuard:
    """
    Thermal safety guard, continuous temperature monitor, and closed-loop duty cycle clamp.
    """

    def __init__(
        self,
        config: Optional[BoostLockConfig] = None,
        sysfs_root: Union[str, Path] = "/sys",
        thermal_warn_c: Optional[float] = None,
        thermal_limit_c: Optional[float] = None,
        thermal_recover_c: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        on_warning: Optional[Callable[[float], None]] = None,
        on_tripwire: Optional[Callable[[], None]] = None,
        on_recovery: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or BoostLockConfig()
        self.sysfs_root = Path(sysfs_root).resolve()

        self.thermal_warn_c = (
            thermal_warn_c if thermal_warn_c is not None else self.config.thermal_warn_c
        )
        self.thermal_limit_c = (
            thermal_limit_c if thermal_limit_c is not None else self.config.thermal_limit_c
        )
        self.thermal_recover_c = (
            thermal_recover_c if thermal_recover_c is not None else self.config.thermal_recover_c
        )
        self.poll_interval_s = (
            poll_interval_s
            if poll_interval_s is not None
            else (self.config.poll_interval_ms / 1000.0)
        )

        self.on_warning = on_warning
        self.on_tripwire = on_tripwire
        self.on_recovery = on_recovery

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self._filter = SpikeFilter(window_size=5)
        self._rapl = RAPLMonitor(sysfs_root=self.sysfs_root)
        self._sensors: List[ThermalSensor] = []

        self._state: ThermalState = ThermalState.NORMAL
        self._is_tripped: bool = False
        self._clamp_factor: float = 1.0
        self._current_temp_c: float = 0.0
        self._last_reading: Optional[ThermalReading] = None

        self.discover_sensors()

    @property
    def state(self) -> ThermalState:
        with self._lock:
            return self._state

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._is_tripped

    @property
    def clamp_factor(self) -> float:
        with self._lock:
            return self._clamp_factor

    @property
    def current_temp_c(self) -> float:
        with self._lock:
            return self._current_temp_c

    @property
    def sensors(self) -> List[ThermalSensor]:
        return list(self._sensors)

    @property
    def is_running(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def discover_sensors(self) -> List[ThermalSensor]:
        """Discover available thermal sensors."""
        self._sensors = discover_sensors(self.sysfs_root)
        return self._sensors

    def get_cpu_temperature(self) -> float:
        """
        Poll active CPU sensors and return highest recorded temperature.
        Returns safe baseline (50.0C) if no sensors are discoverable or readable.
        """
        cpu_sensors = [s for s in self._sensors if s.is_cpu]
        target_sensors = cpu_sensors if cpu_sensors else self._sensors

        temps: List[float] = []
        for sensor in target_sensors:
            t = sensor.read_temp_c()
            if t is not None:
                temps.append(t)

        if not temps:
            # Fallback if sensors could not be read
            return 50.0

        max_raw = max(temps)
        return self._filter.add_sample(max_raw)

    def get_all_temperatures(self) -> Dict[str, float]:
        """Read all available thermal sensors into a dictionary {sensor_id: temp_c}."""
        res: Dict[str, float] = {}
        for s in self._sensors:
            t = s.read_temp_c()
            if t is not None:
                res[s.sensor_id] = t
        return res

    def update_state(self, temp_c: Optional[float] = None) -> ThermalReading:
        """
        Evaluate temperature against warning, tripwire, and recovery thresholds,
        updating internal state, clamp factor, and invoking callbacks.
        """
        now = time.time()
        sensor_dict: Dict[str, float] = {}

        with self._lock:
            if temp_c is None:
                sensor_dict = self.get_all_temperatures()
                temp = self.get_cpu_temperature()
            else:
                temp = temp_c

            self._current_temp_c = temp

            prev_state = self._state
            prev_tripped = self._is_tripped

            # State transition logic
            if temp >= self.thermal_limit_c:
                self._state = ThermalState.CRITICAL
                self._is_tripped = True
                self._clamp_factor = 0.0
                if not prev_tripped and self.on_tripwire:
                    try:
                        self.on_tripwire()
                    except Exception as exc:
                        logger.error(f"Error in on_tripwire callback: {exc}")

            elif self._is_tripped:
                # In tripwire latch mode: must cool down below thermal_recover_c
                if temp <= self.thermal_recover_c:
                    self._is_tripped = False
                    self._state = ThermalState.NORMAL
                    self._clamp_factor = 1.0
                    if self.on_recovery:
                        try:
                            self.on_recovery()
                        except Exception as exc:
                            logger.error(f"Error in on_recovery callback: {exc}")
                else:
                    # Still in recovery hysteresis
                    self._state = ThermalState.THROTTLED
                    self._clamp_factor = 0.0

            elif temp >= self.thermal_warn_c:
                self._state = ThermalState.WARNING
                # Linear proportional clamp between warn and limit:
                # At T_warn -> 1.0, at T_limit -> 0.0
                span = max(0.0001, self.thermal_limit_c - self.thermal_warn_c)
                self._clamp_factor = max(0.0, min(1.0, (self.thermal_limit_c - temp) / span))

                if self.on_warning:
                    try:
                        self.on_warning(self._clamp_factor)
                    except Exception as exc:
                        logger.error(f"Error in on_warning callback: {exc}")

            else:
                self._state = ThermalState.NORMAL
                self._clamp_factor = 1.0

            # RAPL reading
            rapl_data = self._rapl.read_power()
            pkg_power_w = rapl_data.power_w if rapl_data else None

            reading = ThermalReading(
                timestamp=now,
                current_temp_c=self._current_temp_c,
                sensors=sensor_dict,
                state=self._state,
                clamp_factor=self._clamp_factor,
                is_tripped=self._is_tripped,
                package_power_w=pkg_power_w,
            )
            self._last_reading = reading
            return reading

    def calculate_duty_clamp(self, requested_duty: float, temp_c: Optional[float] = None) -> float:
        """
        Calculate duty cycle after applying thermal throttling clamp factor.
        Returns 0.0 if critical or tripped.
        """
        if temp_c is not None:
            self.update_state(temp_c)

        with self._lock:
            if self._is_tripped or self._state == ThermalState.CRITICAL:
                return 0.0
            return max(0.0, requested_duty * self._clamp_factor)

    def get_status(self) -> Dict[str, Any]:
        """Return human-readable and serializable thermal status dictionary."""
        with self._lock:
            return {
                "state": self._state.value,
                "current_temp_c": round(self._current_temp_c, 2),
                "clamp_factor": round(self._clamp_factor, 3),
                "is_tripped": self._is_tripped,
                "thermal_warn_c": self.thermal_warn_c,
                "thermal_limit_c": self.thermal_limit_c,
                "thermal_recover_c": self.thermal_recover_c,
                "sensors_count": len(self._sensors),
            }

    def _monitor_loop(self) -> None:
        """Continuous background monitor loop."""
        while not self._stop_event.is_set():
            try:
                self.update_state()
            except Exception as exc:
                logger.error(f"Error in thermal monitor loop: {exc}")
            self._stop_event.wait(self.poll_interval_s)

    def start(self) -> None:
        """Start background thermal monitoring thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ThermalGuardMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop background thermal monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for background monitoring thread to terminate."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

    def __enter__(self) -> ThermalGuard:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"ThermalGuard(state={self.state.value}, temp={self.current_temp_c:.1f}C, "
            f"clamp={self.clamp_factor:.2f}, tripped={self.is_tripped})"
        )
