import errno
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from boostlock.sysfs import (
    SysfsController,
    SysfsCorruptError,
    SysfsError,
    SysfsNotFoundError,
    SysfsPermissionError,
    parse_cpu_range,
)


class TestSysfs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="boostlock_sysfs_test_")
        self.sysfs_root = Path(self.temp_dir)
        self.cpu_dir = self.sysfs_root / "devices" / "system" / "cpu"
        self.cpu_dir.mkdir(parents=True, exist_ok=True)
        (self.cpu_dir / "online").write_text("0-3\n")

        self.cpufreq_global = self.cpu_dir / "cpufreq"
        self.cpufreq_global.mkdir(parents=True, exist_ok=True)
        (self.cpufreq_global / "boost").write_text("0\n")

        for i in range(4):
            core_cf = self.cpu_dir / f"cpu{i}" / "cpufreq"
            core_cf.mkdir(parents=True, exist_ok=True)
            (core_cf / "scaling_governor").write_text("ondemand\n")
            (core_cf / "scaling_available_governors").write_text("performance powersave ondemand schedutil\n")
            (core_cf / "scaling_min_freq").write_text("1400000\n")
            (core_cf / "scaling_max_freq").write_text("3000000\n")
            (core_cf / "scaling_cur_freq").write_text("2200000\n")
            (core_cf / "cpuinfo_min_freq").write_text("1400000\n")
            (core_cf / "cpuinfo_max_freq").write_text("3000000\n")
            (core_cf / "scaling_available_frequencies").write_text("1400000 2000000 3000000\n")
            (core_cf / "scaling_driver").write_text("acpi-cpufreq\n")
            (core_cf / "cpb").write_text("0\n")
            (core_cf / "energy_performance_preference").write_text("balance_performance\n")
            (core_cf / "energy_performance_available_preferences").write_text("default performance balance_performance balance_power power\n")
            (core_cf / "energy_perf_bias").write_text("6\n")

        self.ctrl = SysfsController(sysfs_root=str(self.sysfs_root))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_cpu_range(self):
        self.assertEqual(parse_cpu_range(""), [])
        self.assertEqual(parse_cpu_range("   "), [])
        self.assertEqual(parse_cpu_range("0-3, 5, 7-8"), [0, 1, 2, 3, 5, 7, 8])
        self.assertEqual(parse_cpu_range("0,,2, "), [0, 2])
        # Invalid range elements
        self.assertEqual(parse_cpu_range("0-abc,def-5,xyz,1-2"), [1, 2])

    def test_get_online_cpus(self):
        cpus = self.ctrl.get_online_cpus()
        self.assertEqual(cpus, [0, 1, 2, 3])

        # Test complex ranges e.g. "0,2-3"
        (self.cpu_dir / "online").write_text("0,2-3\n")
        self.assertEqual(self.ctrl.get_online_cpus(), [0, 2, 3])

    def test_get_online_cpus_fallback_scan(self):
        # Remove online file
        (self.cpu_dir / "online").unlink()
        # Add non-cpu dirs
        (self.cpu_dir / "cpufreq").mkdir(exist_ok=True)
        (self.cpu_dir / "power").mkdir(exist_ok=True)
        (self.cpu_dir / "cpu_other").mkdir(exist_ok=True)

        cpus = self.ctrl.get_online_cpus()
        self.assertEqual(cpus, [0, 1, 2, 3])

        # Remove all cpu dirs
        for i in range(4):
            shutil.rmtree(self.cpu_dir / f"cpu{i}")
        self.assertEqual(self.ctrl.get_online_cpus(), [0])

    def test_governor_read_write(self):
        self.assertEqual(self.ctrl.get_scaling_governor(0), "ondemand")
        governors = self.ctrl.get_available_governors(0)
        self.assertIn("performance", governors)
        self.assertIn("ondemand", governors)

        # Set governor on specific cpu
        self.ctrl.set_scaling_governor("performance", cpus=[0])
        self.assertEqual(self.ctrl.get_scaling_governor(0), "performance")
        self.assertEqual(self.ctrl.get_scaling_governor(1), "ondemand")

        # Set governor on all cpus
        self.ctrl.set_scaling_governor("performance")
        for i in range(4):
            self.assertEqual(self.ctrl.get_scaling_governor(i), "performance")

    def test_get_available_governors_empty(self):
        gov_file = self.cpu_dir / "cpu0" / "cpufreq" / "scaling_available_governors"
        gov_file.write_text("")
        self.assertEqual(self.ctrl.get_available_governors(0), [])

    def test_frequency_read_write(self):
        self.assertEqual(self.ctrl.get_scaling_min_freq(0), 1400000)
        self.assertEqual(self.ctrl.get_scaling_max_freq(0), 3000000)
        self.assertEqual(self.ctrl.get_scaling_cur_freq(0), 2200000)
        self.assertEqual(self.ctrl.get_cpuinfo_min_freq(0), 1400000)
        self.assertEqual(self.ctrl.get_cpuinfo_max_freq(0), 3000000)
        self.assertEqual(self.ctrl.get_available_frequencies(0), [1400000, 2000000, 3000000])

        # Set min freq
        self.ctrl.set_scaling_min_freq(3000000, cpus=[0, 1])
        self.assertEqual(self.ctrl.get_scaling_min_freq(0), 3000000)
        self.assertEqual(self.ctrl.get_scaling_min_freq(1), 3000000)
        self.assertEqual(self.ctrl.get_scaling_min_freq(2), 1400000)

        # Set max freq on all
        self.ctrl.set_scaling_max_freq(4000000)
        for i in range(4):
            self.assertEqual(self.ctrl.get_scaling_max_freq(i), 4000000)

    def test_available_frequencies_empty_or_corrupt(self):
        freq_file = self.cpu_dir / "cpu0" / "cpufreq" / "scaling_available_frequencies"
        freq_file.write_text("")
        self.assertEqual(self.ctrl.get_available_frequencies(0), [])

        freq_file.write_text("1400000 bad_freq 3000000\n")
        self.assertEqual(self.ctrl.get_available_frequencies(0), [1400000, 3000000])

    def test_boost_and_cpb(self):
        self.assertFalse(self.ctrl.get_boost())
        self.assertFalse(self.ctrl.get_cpb(0))

        # Enable global boost
        self.ctrl.set_boost(True)
        self.assertTrue(self.ctrl.get_boost())

        # Enable CPB per core
        self.ctrl.set_cpb(True, cpus=[0, 1])
        self.assertTrue(self.ctrl.get_cpb(0))
        self.assertTrue(self.ctrl.get_cpb(1))
        self.assertFalse(self.ctrl.get_cpb(2))

        # Enable all boost
        self.ctrl.enable_all_boost()
        self.assertTrue(self.ctrl.get_boost())
        for i in range(4):
            self.assertTrue(self.ctrl.get_cpb(i))

    def test_energy_performance_preference_and_bias(self):
        self.assertEqual(self.ctrl.get_energy_performance_preference(0), "balance_performance")
        self.assertIn("performance", self.ctrl.get_available_energy_performance_preferences(0))

        self.ctrl.set_energy_performance_preference("performance")
        for i in range(4):
            self.assertEqual(self.ctrl.get_energy_performance_preference(i), "performance")

        self.assertEqual(self.ctrl.get_energy_perf_bias(0), 6)
        self.ctrl.set_energy_perf_bias(0, cpus=[0, 2])
        self.assertEqual(self.ctrl.get_energy_perf_bias(0), 0)
        self.assertEqual(self.ctrl.get_energy_perf_bias(1), 6)
        self.assertEqual(self.ctrl.get_energy_perf_bias(2), 0)

    def test_get_available_energy_performance_preferences_empty(self):
        epp_avail = self.cpu_dir / "cpu0" / "cpufreq" / "energy_performance_available_preferences"
        epp_avail.write_text("")
        self.assertEqual(self.ctrl.get_available_energy_performance_preferences(0), [])

    def test_read_cpu_state_and_all_cpus_state(self):
        state0 = self.ctrl.read_cpu_state(0)
        self.assertEqual(state0["governor"], "ondemand")
        self.assertEqual(state0["scaling_min_freq"], 1400000)
        self.assertEqual(state0["scaling_max_freq"], 3000000)
        self.assertEqual(state0["cpb"], False)

        all_states = self.ctrl.read_all_cpus_state()
        self.assertEqual(len(all_states), 4)
        self.assertIn(0, all_states)
        self.assertIn(3, all_states)

    def test_sysfs_permission_error_handling(self):
        target_file = self.cpu_dir / "cpu0" / "cpufreq" / "scaling_governor"

        # Make file read-only to simulate EACCES
        os.chmod(target_file, 0o444)
        try:
            with self.assertRaises(SysfsPermissionError):
                self.ctrl.set_scaling_governor("performance", cpus=[0])
        finally:
            os.chmod(target_file, 0o666)

    def test_sysfs_permission_error_on_read(self):
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Permission denied")):
            with self.assertRaises(SysfsPermissionError):
                self.ctrl.get_scaling_governor(0)

    def test_sysfs_oserror_on_read(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("I/O error")):
            val = self.ctrl.get_scaling_governor(0)
            self.assertIsNone(val)

    def test_absolute_path_oserror_on_read(self):
        path = self.cpu_dir / "cpu0" / "cpufreq" / "scaling_governor"
        with patch("pathlib.Path.read_text", side_effect=OSError("I/O error")):
            self.assertIsNone(self.ctrl._read_path(path))

    def test_sysfs_not_found_on_mandatory_write(self):
        with self.assertRaises(SysfsNotFoundError):
            self.ctrl._write_file("devices/system/cpu/cpu0/cpufreq/nonexistent_file", "1", optional=False)

    def test_sysfs_write_os_errors(self):
        # Test EROFS (Read-only file system)
        err = OSError()
        err.errno = errno.EROFS
        with patch("builtins.open", side_effect=err):
            with self.assertRaises(SysfsPermissionError):
                self.ctrl._write_file("devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance")

        # Test other non-permission OSError on non-optional write
        err2 = OSError("Hardware fault")
        err2.errno = errno.EIO
        with patch("builtins.open", side_effect=err2):
            with self.assertRaises(SysfsError):
                self.ctrl._write_file("devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance", optional=False)

        # Test write error on optional write
        with patch("builtins.open", side_effect=err2):
            result = self.ctrl._write_file("devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance", optional=True)
            self.assertFalse(result)

    def test_sysfs_missing_files_handling(self):
        # Removing optional EPP file should return None, not crash
        epp_file = self.cpu_dir / "cpu0" / "cpufreq" / "energy_performance_preference"
        if epp_file.exists():
            epp_file.unlink()

        self.assertIsNone(self.ctrl.get_energy_performance_preference(0))
        # Setting EPP when file doesn't exist should not raise fatal exception if optional
        self.ctrl.set_energy_performance_preference("performance", cpus=[0])

    def test_sysfs_corrupt_content_handling(self):
        # Write corrupt string into scaling_cur_freq
        freq_file = self.cpu_dir / "cpu0" / "cpufreq" / "scaling_cur_freq"
        freq_file.write_text("corrupted_non_int\n")

        # Reading non-numeric frequency should return None or default fallback
        val = self.ctrl.get_scaling_cur_freq(0, default=0)
        self.assertEqual(val, 0)


if __name__ == "__main__":
    unittest.main()


class TestSysfsCoverageGaps(unittest.TestCase):
    def test_get_online_cpus_unparseable_cpu_dirname(self):
        temp_dir = tempfile.mkdtemp(prefix="sysfs_cov_")
        try:
            cpu_base = Path(temp_dir) / "devices" / "system" / "cpu"
            cpu_base.mkdir(parents=True, exist_ok=True)
            # Create a dir that matches regex cpu\d+ but int(name[3:]) raises ValueError (mocked or custom)
            (cpu_base / "cpu0").mkdir()
            ctrl = SysfsController(sysfs_root=temp_dir)
            with patch("builtins.int", side_effect=[ValueError("invalid"), 0]):
                cpus = ctrl.get_online_cpus()
                self.assertIsInstance(cpus, list)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
