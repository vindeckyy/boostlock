"""
Unit and integration tests for Thermal Safety Guard (FEAT-04).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from boostlock.config import BoostLockConfig
from boostlock.thermal import (
    RAPLMonitor,
    RAPLReading,
    SensorReadError,
    SensorType,
    SpikeFilter,
    ThermalError,
    ThermalGuard,
    ThermalReading,
    ThermalSensor,
    ThermalState,
    ThermalTripwireError,
    discover_sensors,
)


class MockSysfsThermalTree:
    """Helper to construct realistic mock hwmon, thermal_zone, and powercap trees."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hwmon_dir = root / "class" / "hwmon"
        self.thermal_dir = root / "class" / "thermal"
        self.powercap_dir = root / "class" / "powercap"

        self.hwmon_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_dir.mkdir(parents=True, exist_ok=True)
        self.powercap_dir.mkdir(parents=True, exist_ok=True)

    def add_hwmon(
        self,
        index: int,
        name: str,
        temps: Dict[int, Dict[str, str]],
    ) -> Path:
        """
        Add a mock hwmon directory.
        temps: mapping temp_index -> {'input': '55000', 'label': 'Tctl', 'crit': '95000', 'max': '85000'}
        """
        h_dir = self.hwmon_dir / f"hwmon{index}"
        h_dir.mkdir(parents=True, exist_ok=True)
        (h_dir / "name").write_text(name + "\n", encoding="utf-8")

        for t_idx, attrs in temps.items():
            for attr_name, val in attrs.items():
                file_name = f"temp{t_idx}_{attr_name}"
                (h_dir / file_name).write_text(str(val) + "\n", encoding="utf-8")
        return h_dir

    def add_thermal_zone(
        self,
        index: int,
        zone_type: str,
        temp_millic: int,
    ) -> Path:
        """Add a mock thermal_zone directory."""
        z_dir = self.thermal_dir / f"thermal_zone{index}"
        z_dir.mkdir(parents=True, exist_ok=True)
        (z_dir / "type").write_text(zone_type + "\n", encoding="utf-8")
        (z_dir / "temp").write_text(str(temp_millic) + "\n", encoding="utf-8")
        return z_dir

    def add_powercap_rapl(
        self,
        name: str = "package-0",
        energy_uj: int = 100000000,
        max_energy_range_uj: int = 262143328850,
        base_dir: Optional[Path] = None,
    ) -> Path:
        """Add a mock intel-rapl powercap package directory."""
        target_base = base_dir if base_dir else (self.powercap_dir / "intel-rapl")
        rapl_pkg = target_base / "intel-rapl:0"
        rapl_pkg.mkdir(parents=True, exist_ok=True)
        (rapl_pkg / "name").write_text(name + "\n", encoding="utf-8")
        (rapl_pkg / "energy_uj").write_text(str(energy_uj) + "\n", encoding="utf-8")
        (rapl_pkg / "max_energy_range_uj").write_text(
            str(max_energy_range_uj) + "\n", encoding="utf-8"
        )
        (rapl_pkg / "enabled").write_text("1\n", encoding="utf-8")
        return rapl_pkg


@pytest.fixture
def temp_sysfs(tmp_path: Path) -> MockSysfsThermalTree:
    return MockSysfsThermalTree(tmp_path)


class TestThermalExceptions:
    """Tests for exception hierarchy."""

    def test_exception_inheritance(self) -> None:
        assert issubclass(SensorReadError, ThermalError)
        assert issubclass(ThermalTripwireError, ThermalError)
        err = ThermalTripwireError("Emergency trip")
        assert str(err) == "Emergency trip"


class TestThermalSensor:
    """Tests for individual ThermalSensor representation and reading."""

    def test_read_millidegrees_and_degrees(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "temp1_input"
        temp_file.write_text("55000\n", encoding="utf-8")
        sensor = ThermalSensor(
            sensor_id="hwmon0_temp1",
            name="k10temp",
            path=temp_file,
        )
        assert sensor.read_temp_c() == 55.0

        # Input already in degrees (e.g. 48.5)
        temp_file.write_text("48.5\n", encoding="utf-8")
        assert sensor.read_temp_c() == 48.5

    def test_read_out_of_bounds(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "temp1_input"
        sensor = ThermalSensor(
            sensor_id="hwmon0_temp1",
            name="k10temp",
            path=temp_file,
        )
        # Below -40C
        temp_file.write_text("-50000\n", encoding="utf-8")
        assert sensor.read_temp_c() is None

        # Above 150C
        temp_file.write_text("180000\n", encoding="utf-8")
        assert sensor.read_temp_c() is None


class TestSpikeFilter:
    """Tests for the rolling median and spike rejection filter."""

    def test_filter_normal_series(self) -> None:
        filter_ = SpikeFilter(window_size=5, min_temp_c=0.0, max_temp_c=120.0)
        readings = [50.0, 52.0, 51.0, 53.0, 54.0]
        filtered = [filter_.add_sample(r) for r in readings]
        assert filtered[-1] == 52.0  # Median of [50, 51, 52, 53, 54]

    def test_filter_rejects_out_of_bounds(self) -> None:
        filter_ = SpikeFilter(window_size=3, min_temp_c=10.0, max_temp_c=115.0)
        # First sample out of bounds with empty history
        assert filter_.add_sample(250.0) == 115.0
        filter_.reset()
        assert filter_.add_sample(-50.0) == 10.0

        # Sample with non-empty history
        filter_.add_sample(50.0)
        assert filter_.add_sample(-10.0) == 50.0
        assert filter_.add_sample(250.0) == 50.0
        assert filter_.add_sample(60.0) == 55.0

    def test_filter_spike_damping(self) -> None:
        filter_ = SpikeFilter(window_size=5, max_rate_of_change=15.0)
        filter_.add_sample(50.0)
        filter_.add_sample(50.0)
        filter_.add_sample(50.0)
        # Single aberrant spike
        result = filter_.add_sample(95.0)
        assert result < 70.0  # Median filter suppresses single sample jump

    def test_filter_reset(self) -> None:
        filter_ = SpikeFilter(window_size=3)
        filter_.add_sample(60.0)
        filter_.reset()
        assert len(filter_.history) == 0
        assert filter_.add_sample(70.0) == 70.0


class TestSensorDiscovery:
    """Tests for discovering hwmon, thermal_zone, and powercap nodes."""

    def test_discover_amd_k10temp(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_hwmon(
            index=0,
            name="k10temp",
            temps={
                1: {"input": "62125", "label": "Tctl", "crit": "105000", "max": "95000"},
                2: {"input": "58500", "label": "Tdie"},
            },
        )
        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) == 2
        s1 = next(s for s in sensors if s.label == "Tctl")
        assert s1.name == "k10temp"
        assert s1.sensor_type == SensorType.HWMON
        assert s1.is_cpu is True
        assert s1.critical_temp_c == 105.0
        assert s1.max_temp_c == 95.0
        assert s1.read_temp_c() == 62.125

        s2 = next(s for s in sensors if s.label == "Tdie")
        assert s2.is_cpu is True
        assert s2.read_temp_c() == 58.5

    def test_discover_intel_coretemp(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_hwmon(
            index=1,
            name="coretemp",
            temps={
                1: {"input": "45000", "label": "Package id 0", "max": "100000", "crit": "105000"},
                2: {"input": "43000", "label": "Core 0"},
                3: {"input": "44000", "label": "Core 1"},
            },
        )
        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) == 3
        pkg = next(s for s in sensors if s.label == "Package id 0")
        assert pkg.is_cpu is True
        assert pkg.max_temp_c == 100.0
        assert pkg.critical_temp_c == 105.0
        assert pkg.read_temp_c() == 45.0

    def test_discover_thermal_zones(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_thermal_zone(0, "acpitz", 55000)
        temp_sysfs.add_thermal_zone(1, "x86_pkg_temp", 58000)
        temp_sysfs.add_thermal_zone(2, "wireless", 35000)

        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) == 3
        z0 = next(s for s in sensors if s.sensor_id == "thermal_zone0")
        assert z0.name == "acpitz"
        assert z0.is_cpu is True
        assert z0.read_temp_c() == 55.0

        z1 = next(s for s in sensors if s.sensor_id == "thermal_zone1")
        assert z1.is_cpu is True
        assert z1.read_temp_c() == 58.0

        z2 = next(s for s in sensors if s.sensor_id == "thermal_zone2")
        assert z2.is_cpu is False

    def test_discover_mixed_and_cpu_prioritization(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_hwmon(
            index=0,
            name="amdgpu",
            temps={1: {"input": "42000", "label": "edge"}},
        )
        temp_sysfs.add_hwmon(
            index=1,
            name="nvme",
            temps={1: {"input": "38000", "label": "Composite"}},
        )
        temp_sysfs.add_hwmon(
            index=2,
            name="k10temp",
            temps={1: {"input": "65000", "label": "Tctl"}},
        )
        temp_sysfs.add_thermal_zone(0, "acpitz", 60000)

        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) == 4
        # First sensor should be CPU sensor (k10temp or acpitz)
        assert sensors[0].is_cpu is True

    def test_discover_empty_sysfs(self, tmp_path: Path) -> None:
        empty_root = tmp_path / "empty_sys"
        empty_root.mkdir()
        sensors = discover_sensors(empty_root)
        assert sensors == []

    def test_discover_non_directory_or_missing_attribute_files(self, temp_sysfs: MockSysfsThermalTree) -> None:
        # Put a regular file inside class/hwmon
        (temp_sysfs.hwmon_dir / "hwmon_not_a_dir").write_text("dummy", encoding="utf-8")

        # Add hwmon without name file or unreadable attributes
        h_dir = temp_sysfs.hwmon_dir / "hwmon99"
        h_dir.mkdir(parents=True, exist_ok=True)
        (h_dir / "temp1_input").write_text("52000\n", encoding="utf-8")
        (h_dir / "temp1_crit").write_text("corrupted_crit\n", encoding="utf-8")
        (h_dir / "temp1_max").write_text("corrupted_max\n", encoding="utf-8")

        # Put a regular file inside class/thermal
        (temp_sysfs.thermal_dir / "thermal_zone_not_a_dir").write_text("dummy", encoding="utf-8")

        # Add thermal zone without temp file
        z_dir = temp_sysfs.thermal_dir / "thermal_zone99"
        z_dir.mkdir(parents=True, exist_ok=True)

        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) >= 1
        s = next(s for s in sensors if s.sensor_id == "hwmon99_temp1")
        assert s.name == "unknown"
        assert s.critical_temp_c is None
        assert s.max_temp_c is None

    def test_discover_attribute_read_exceptions(self, temp_sysfs: MockSysfsThermalTree) -> None:
        h_dir = temp_sysfs.add_hwmon(
            index=0,
            name="test_hwmon",
            temps={1: {"input": "55000", "label": "test_lbl"}},
        )
        z_dir = temp_sysfs.add_thermal_zone(0, "acpitz", 55000)

        # Mock Path.read_text to raise for label and type
        original_read_text = Path.read_text

        def mock_read_text(path_obj: Path, *args, **kwargs):
            if path_obj.name.endswith("_label") or path_obj.name == "type":
                raise OSError("Simulated read error")
            return original_read_text(path_obj, *args, **kwargs)

        with patch("pathlib.Path.read_text", side_effect=mock_read_text):
            sensors = discover_sensors(temp_sysfs.root)
            assert len(sensors) >= 2

    def test_sensor_read_errors(self, temp_sysfs: MockSysfsThermalTree) -> None:
        h_dir = temp_sysfs.add_hwmon(
            index=0,
            name="corrupt_hwmon",
            temps={1: {"input": "invalid_number"}},
        )
        sensors = discover_sensors(temp_sysfs.root)
        assert len(sensors) == 1
        # Reading corrupt value returns None
        assert sensors[0].read_temp_c() is None

        # Remove file to test missing file
        (h_dir / "temp1_input").unlink()
        assert sensors[0].read_temp_c() is None


class TestRAPLMonitor:
    """Tests for RAPL powercap energy and wattage monitor."""

    def test_rapl_monitor_power_calculation(self, temp_sysfs: MockSysfsThermalTree) -> None:
        rapl_pkg = temp_sysfs.add_powercap_rapl(
            name="package-0",
            energy_uj=10_000_000,  # 10 Joules
        )
        rapl = RAPLMonitor(sysfs_root=temp_sysfs.root)
        assert rapl.is_available() is True

        # First sample establishes baseline
        reading1 = rapl.read_power()
        assert reading1 is not None
        assert reading1.energy_uj == 10_000_000
        assert reading1.power_w is None  # Needs at least 2 readings for power

        # Second sample after time delta
        (rapl_pkg / "energy_uj").write_text("25000000\n", encoding="utf-8")  # +15 Joules
        with patch("time.time", return_value=rapl._last_timestamp + 1.0):
            reading2 = rapl.read_power()
            assert reading2 is not None
            assert reading2.power_w == pytest.approx(15.0, abs=0.1)

    def test_rapl_monitor_wraparound(self, temp_sysfs: MockSysfsThermalTree) -> None:
        max_range = 100_000_000  # 100 Joules max
        rapl_pkg = temp_sysfs.add_powercap_rapl(
            name="package-0",
            energy_uj=90_000_000,
            max_energy_range_uj=max_range,
        )
        rapl = RAPLMonitor(sysfs_root=temp_sysfs.root)
        rapl.read_power()

        # Wraparound: energy drops to 10_000_000 (+20 Joules delta: (100-90) + 10)
        (rapl_pkg / "energy_uj").write_text("10000000\n", encoding="utf-8")
        mocked_now = rapl._last_timestamp + 2.0
        with patch("time.time", return_value=mocked_now):
            reading = rapl.read_power()
            assert reading is not None
            assert reading.power_w == pytest.approx(10.0, abs=0.1)  # 20J / 2s = 10W

    def test_rapl_alternate_sysfs_path(self, tmp_path: Path) -> None:
        # sys/devices/virtual/powercap/intel-rapl
        virtual_rapl = tmp_path / "devices" / "virtual" / "powercap" / "intel-rapl"
        virtual_rapl.mkdir(parents=True, exist_ok=True)
        pkg = virtual_rapl / "intel-rapl:0"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "energy_uj").write_text("5000000\n", encoding="utf-8")
        (pkg / "max_energy_range_uj").write_text("invalid_int\n", encoding="utf-8")

        rapl = RAPLMonitor(sysfs_root=tmp_path)
        assert rapl.is_available() is True
        reading = rapl.read_power()
        assert reading is not None
        assert reading.energy_uj == 5000000

    def test_rapl_unavailable(self, tmp_path: Path) -> None:
        rapl = RAPLMonitor(sysfs_root=tmp_path)
        assert rapl.is_available() is False
        assert rapl.read_power() is None

    def test_rapl_permission_or_read_error(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_powercap_rapl(name="package-0", energy_uj=5000)
        rapl = RAPLMonitor(sysfs_root=temp_sysfs.root)
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Permission denied")):
            assert rapl.read_power() is None


class TestThermalGuard:
    """Tests for ThermalGuard state transitions, hysteresis, and clamp calculations."""

    def test_initialization_defaults(self, temp_sysfs: MockSysfsThermalTree) -> None:
        guard = ThermalGuard(sysfs_root=temp_sysfs.root)
        assert guard.thermal_warn_c == 75.0
        assert guard.thermal_limit_c == 85.0
        assert guard.thermal_recover_c == 70.0
        assert guard.state == ThermalState.NORMAL
        assert guard.is_tripped is False
        assert guard.clamp_factor == 1.0
        assert isinstance(guard.sensors, list)

    def test_initialization_with_config(self, temp_sysfs: MockSysfsThermalTree) -> None:
        config = BoostLockConfig(
            thermal_warn_c=80.0,
            thermal_limit_c=90.0,
            thermal_recover_c=72.0,
            poll_interval_ms=50,
        )
        guard = ThermalGuard(config=config, sysfs_root=temp_sysfs.root)
        assert guard.thermal_warn_c == 80.0
        assert guard.thermal_limit_c == 90.0
        assert guard.thermal_recover_c == 72.0
        assert guard.poll_interval_s == 0.05

    def test_state_transitions_rising_temperature(self, temp_sysfs: MockSysfsThermalTree) -> None:
        temp_sysfs.add_hwmon(
            index=0,
            name="k10temp",
            temps={1: {"input": "50000", "label": "Tctl"}},
        )
        on_warn = MagicMock()
        on_trip = MagicMock()
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            on_warning=on_warn,
            on_tripwire=on_trip,
        )

        # 1. Normal Zone (50C < 75C)
        reading = guard.update_state(50.0)
        assert reading.state == ThermalState.NORMAL
        assert reading.clamp_factor == 1.0
        assert guard.is_tripped is False
        on_warn.assert_not_called()
        on_trip.assert_not_called()

        # 2. Warning Zone (80C: midpoint between 75C and 85C)
        # Linear clamp: (85 - 80) / (85 - 75) = 0.5
        reading = guard.update_state(80.0)
        assert reading.state == ThermalState.WARNING
        assert reading.clamp_factor == pytest.approx(0.5, abs=0.01)
        assert guard.is_tripped is False
        on_warn.assert_called_once_with(pytest.approx(0.5, abs=0.01))

        # 3. Critical Zone (85C >= 85C limit)
        reading = guard.update_state(85.5)
        assert reading.state == ThermalState.CRITICAL
        assert reading.clamp_factor == 0.0
        assert guard.is_tripped is True
        on_trip.assert_called_once()

    def test_hysteresis_and_recovery(self, temp_sysfs: MockSysfsThermalTree) -> None:
        on_recovery = MagicMock()
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            thermal_warn_c=75.0,
            thermal_limit_c=85.0,
            thermal_recover_c=70.0,
            on_recovery=on_recovery,
        )

        # Trip the guard
        guard.update_state(88.0)
        assert guard.state == ThermalState.CRITICAL
        assert guard.is_tripped is True

        # Cooled down to 82C (below limit 85C, but above recover 70C) -> Still throttled/tripped!
        reading1 = guard.update_state(82.0)
        assert reading1.state == ThermalState.THROTTLED
        assert reading1.clamp_factor == 0.0
        assert guard.is_tripped is True
        on_recovery.assert_not_called()

        # Cooled down to 72C (below warn 75C, but above recover 70C) -> Still throttled/tripped!
        reading2 = guard.update_state(72.0)
        assert reading2.state == ThermalState.THROTTLED
        assert reading2.clamp_factor == 0.0
        assert guard.is_tripped is True
        on_recovery.assert_not_called()

        # Cooled down to 69.5C (below recover 70C) -> Full recovery!
        reading3 = guard.update_state(69.5)
        assert reading3.state == ThermalState.NORMAL
        assert reading3.clamp_factor == 1.0
        assert guard.is_tripped is False
        on_recovery.assert_called_once()

    def test_callback_exceptions_do_not_crash_update_state(
        self, temp_sysfs: MockSysfsThermalTree
    ) -> None:
        bad_warn = MagicMock(side_effect=RuntimeError("warn failure"))
        bad_trip = MagicMock(side_effect=RuntimeError("trip failure"))
        bad_rec = MagicMock(side_effect=RuntimeError("rec failure"))

        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            thermal_warn_c=75.0,
            thermal_limit_c=85.0,
            thermal_recover_c=70.0,
            on_warning=bad_warn,
            on_tripwire=bad_trip,
            on_recovery=bad_rec,
        )

        # Warning
        guard.update_state(80.0)
        # Critical
        guard.update_state(90.0)
        # Recovery
        guard.update_state(65.0)
        assert guard.state == ThermalState.NORMAL

    def test_warning_span_zero_clamp(self, temp_sysfs: MockSysfsThermalTree) -> None:
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            thermal_warn_c=85.0,
            thermal_limit_c=85.0,
        )
        guard.update_state(85.0)
        assert guard.clamp_factor == 0.0

    def test_calculate_duty_clamp(self, temp_sysfs: MockSysfsThermalTree) -> None:
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            thermal_warn_c=75.0,
            thermal_limit_c=85.0,
            thermal_recover_c=70.0,
        )

        # Base duty cycle 30.0%
        # At 60C (Normal): 30.0 * 1.0 = 30.0
        assert guard.calculate_duty_clamp(30.0, temp_c=60.0) == 30.0

        # At 80C (Warning): 30.0 * 0.5 = 15.0
        assert guard.calculate_duty_clamp(30.0, temp_c=80.0) == pytest.approx(15.0, abs=0.1)

        # At 86C (Critical): 0.0
        assert guard.calculate_duty_clamp(30.0, temp_c=86.0) == 0.0

        # While tripped, even if temp is 72C: 0.0
        assert guard.calculate_duty_clamp(30.0, temp_c=72.0) == 0.0

    def test_get_cpu_temperature_and_sensor_reading(
        self, temp_sysfs: MockSysfsThermalTree
    ) -> None:
        temp_sysfs.add_hwmon(
            index=0,
            name="k10temp",
            temps={
                1: {"input": "65000", "label": "Tctl"},
                2: {"input": "61000", "label": "Tdie"},
            },
        )
        guard = ThermalGuard(sysfs_root=temp_sysfs.root)
        temp = guard.get_cpu_temperature()
        assert temp == pytest.approx(65.0, abs=0.1)

        all_temps = guard.get_all_temperatures()
        assert "hwmon0_temp1" in all_temps
        assert all_temps["hwmon0_temp1"] == pytest.approx(65.0, abs=0.1)

    def test_get_cpu_temperature_non_cpu_sensors_only(
        self, temp_sysfs: MockSysfsThermalTree
    ) -> None:
        temp_sysfs.add_hwmon(
            index=0,
            name="amdgpu",
            temps={1: {"input": "42000", "label": "edge"}},
        )
        guard = ThermalGuard(sysfs_root=temp_sysfs.root)
        temp = guard.get_cpu_temperature()
        assert temp == pytest.approx(42.0, abs=0.1)

    def test_get_cpu_temperature_fallback_when_no_sensors(
        self, tmp_path: Path
    ) -> None:
        empty_root = tmp_path / "empty_sys"
        empty_root.mkdir()
        guard = ThermalGuard(sysfs_root=empty_root)
        # Should return safe fallback temperature (e.g. 50.0C) without crashing
        assert guard.get_cpu_temperature() == 50.0
        assert guard.get_all_temperatures() == {}

    def test_background_monitor_thread_and_lifecycle(
        self, temp_sysfs: MockSysfsThermalTree
    ) -> None:
        h_dir = temp_sysfs.add_hwmon(
            index=0,
            name="k10temp",
            temps={1: {"input": "55000", "label": "Tctl"}},
        )
        on_trip = MagicMock()
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            poll_interval_s=0.02,
            on_tripwire=on_trip,
        )

        with guard:
            # Test starting when already running (idempotent)
            guard.start()

            time.sleep(0.06)
            assert guard.state == ThermalState.NORMAL

            # Simulate heat spike past limit
            (h_dir / "temp1_input").write_text("90000\n", encoding="utf-8")
            time.sleep(0.06)
            assert guard.is_tripped is True
            assert guard.state == ThermalState.CRITICAL
            assert on_trip.call_count >= 1

            # Test join
            guard.join(timeout=0.01)

        # Check thread clean shutdown
        assert not guard.is_running

    def test_monitor_loop_exception_resilience(
        self, temp_sysfs: MockSysfsThermalTree
    ) -> None:
        guard = ThermalGuard(
            sysfs_root=temp_sysfs.root,
            poll_interval_s=0.02,
        )
        with patch.object(guard, "update_state", side_effect=[RuntimeError("monitor error"), None]):
            guard.start()
            time.sleep(0.06)
            guard.stop()

    def test_to_dict_and_repr(self, temp_sysfs: MockSysfsThermalTree) -> None:
        guard = ThermalGuard(sysfs_root=temp_sysfs.root)
        status = guard.get_status()
        assert "state" in status
        assert "current_temp_c" in status
        assert "clamp_factor" in status
        assert "is_tripped" in status
        assert "sensors_count" in status
        assert repr(guard).startswith("ThermalGuard(")

def test_rapl_read_max_range_no_pkg_dir(tmp_path: Path) -> None:
    rapl = RAPLMonitor(sysfs_root=tmp_path)
    rapl._pkg_dir = None
    rapl._read_max_range()
    assert rapl._max_energy_range_uj is None
