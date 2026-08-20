import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from boostlock.hardware import (
    CPUInfo,
    CPUVendor,
    CoreInfo,
    ScalingDriver,
    detect_cpu_info,
    detect_vendor,
    lookup_boost_frequency,
    parse_driver,
    parse_proc_cpuinfo,
)

SAMPLE_AMD_CPUINFO = """processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 23
model		: 96
model name	: AMD Ryzen 5 4600H with Radeon Graphics
stepping	: 1
microcode	: 0x860010d
cpu MHz		: 2970.592
cache size	: 512 KB
physical id	: 0
siblings	: 12
core id		: 0
cpu cores	: 6
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 16
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca

""" + "\n\n".join([f"""processor	: {i}
vendor_id	: AuthenticAMD
cpu family	: 23
model		: 96
model name	: AMD Ryzen 5 4600H with Radeon Graphics
stepping	: 1
microcode	: 0x860010d
cpu MHz		: 2970.592
cache size	: 512 KB
physical id	: 0
siblings	: 12
core id		: {i // 2}
cpu cores	: 6
flags		: fpu msr cpb
""" for i in range(1, 12)])

SAMPLE_INTEL_CPUINFO = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
stepping	: 2
microcode	: 0xf0
cpu MHz		: 2600.000
cache size	: 12288 KB
physical id	: 0
siblings	: 12
core id		: 0
cpu cores	: 6
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities

""" + "\n\n".join([f"""processor	: {i}
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
stepping	: 2
microcode	: 0xf0
cpu MHz		: 2600.000
cache size	: 12288 KB
physical id	: 0
siblings	: 12
core id		: {i // 2}
cpu cores	: 6
flags		: fpu msr hwp hwp_epp ida
""" for i in range(1, 12)])


class TestHardware(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="boostlock_hw_test_")
        self.proc_cpuinfo = os.path.join(self.temp_dir, "cpuinfo")
        self.sysfs_root = os.path.join(self.temp_dir, "sys")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _setup_mock_sysfs(self, vendor="amd", num_cpus=12, base_khz=3000000, min_khz=1400000, driver="acpi-cpufreq"):
        cpu_dir = Path(self.sysfs_root) / "devices" / "system" / "cpu"
        cpu_dir.mkdir(parents=True, exist_ok=True)
        (cpu_dir / "online").write_text(f"0-{num_cpus - 1}\n")

        cpufreq_global = cpu_dir / "cpufreq"
        cpufreq_global.mkdir(parents=True, exist_ok=True)
        (cpufreq_global / "boost").write_text("0\n")

        for i in range(num_cpus):
            core_dir = cpu_dir / f"cpu{i}"
            topo_dir = core_dir / "topology"
            topo_dir.mkdir(parents=True, exist_ok=True)
            (topo_dir / "core_id").write_text(f"{i // 2}\n")
            (topo_dir / "physical_package_id").write_text("0\n")

            cf_dir = core_dir / "cpufreq"
            cf_dir.mkdir(parents=True, exist_ok=True)
            (cf_dir / "cpuinfo_max_freq").write_text(f"{base_khz}\n")
            (cf_dir / "cpuinfo_min_freq").write_text(f"{min_khz}\n")
            (cf_dir / "scaling_cur_freq").write_text(f"{base_khz}\n")
            (cf_dir / "scaling_min_freq").write_text(f"{min_khz}\n")
            (cf_dir / "scaling_max_freq").write_text(f"{base_khz}\n")
            (cf_dir / "scaling_driver").write_text(f"{driver}\n")
            (cf_dir / "scaling_governor").write_text("performance\n")
            (cf_dir / "scaling_available_governors").write_text("performance powersave schedutil\n")

            if vendor == "amd":
                (cf_dir / "cpb").write_text("0\n")
            elif vendor == "intel":
                (cf_dir / "energy_performance_preference").write_text("balance_performance\n")
                (cf_dir / "energy_perf_bias").write_text("6\n")

    def test_vendor_detection(self):
        self.assertEqual(detect_vendor("AuthenticAMD"), CPUVendor.AMD)
        self.assertEqual(detect_vendor("GenuineIntel"), CPUVendor.INTEL)
        self.assertEqual(detect_vendor("UnknownVendor"), CPUVendor.UNKNOWN)

    def test_parse_driver(self):
        self.assertEqual(parse_driver(None), ScalingDriver.UNKNOWN)
        self.assertEqual(parse_driver(""), ScalingDriver.UNKNOWN)
        self.assertEqual(parse_driver("acpi-cpufreq"), ScalingDriver.ACPI_CPUFREQ)
        self.assertEqual(parse_driver("amd-pstate"), ScalingDriver.AMD_PSTATE)
        self.assertEqual(parse_driver("amd-pstate-epp"), ScalingDriver.AMD_PSTATE_EPP)
        self.assertEqual(parse_driver("intel_pstate"), ScalingDriver.INTEL_PSTATE)
        self.assertEqual(parse_driver("intel_cpufreq"), ScalingDriver.INTEL_CPUFREQ)
        self.assertEqual(parse_driver("custom_scaling_driver"), ScalingDriver.GENERIC)

    def test_parse_proc_cpuinfo_amd(self):
        with open(self.proc_cpuinfo, "w") as f:
            f.write(SAMPLE_AMD_CPUINFO)

        parsed = parse_proc_cpuinfo(SAMPLE_AMD_CPUINFO)
        self.assertEqual(parsed["vendor_id"], "AuthenticAMD")
        self.assertEqual(parsed["model_name"], "AMD Ryzen 5 4600H with Radeon Graphics")
        self.assertEqual(parsed["logical_count"], 12)
        self.assertEqual(parsed["physical_cores"], 6)
        self.assertIn("cpb", parsed["flags"])

    def test_parse_proc_cpuinfo_intel(self):
        parsed = parse_proc_cpuinfo(SAMPLE_INTEL_CPUINFO)
        self.assertEqual(parsed["vendor_id"], "GenuineIntel")
        self.assertIn("i7-10750H", parsed["model_name"])
        self.assertEqual(parsed["logical_count"], 12)
        self.assertEqual(parsed["physical_cores"], 6)
        self.assertIn("hwp_epp", parsed["flags"])

    def test_parse_proc_cpuinfo_invalid_fields(self):
        corrupt_cpuinfo = """processor : 0
vendor_id : AuthenticAMD
cpu family : invalid_fam
model : invalid_mod
stepping : invalid_step
cpu MHz : invalid_mhz
cpu cores : invalid_cores
"""
        parsed = parse_proc_cpuinfo(corrupt_cpuinfo)
        self.assertEqual(parsed["family"], 0)
        self.assertEqual(parsed["model"], 0)
        self.assertEqual(parsed["stepping"], 0)
        self.assertEqual(parsed["mhz"], 2000.0)
        self.assertEqual(parsed["physical_cores"], 1)

    def test_detect_cpu_info_amd_ryzen_4600h(self):
        with open(self.proc_cpuinfo, "w") as f:
            f.write(SAMPLE_AMD_CPUINFO)

        self._setup_mock_sysfs(vendor="amd", num_cpus=12, base_khz=3000000, min_khz=1400000, driver="acpi-cpufreq")

        info: CPUInfo = detect_cpu_info(proc_cpuinfo_path=self.proc_cpuinfo, sysfs_root=self.sysfs_root)

        self.assertEqual(info.vendor, CPUVendor.AMD)
        self.assertEqual(info.model_name, "AMD Ryzen 5 4600H with Radeon Graphics")
        self.assertEqual(info.logical_cpus, 12)
        self.assertEqual(info.physical_cores, 6)
        self.assertEqual(info.base_freq_mhz, 3000.0)
        self.assertEqual(info.max_boost_mhz, 4000.0)
        self.assertEqual(info.min_freq_mhz, 1400.0)
        self.assertEqual(info.scaling_driver, ScalingDriver.ACPI_CPUFREQ)
        self.assertTrue(info.has_cpb)
        self.assertTrue(info.has_boost)
        self.assertEqual(len(info.cores), 12)
        self.assertEqual(len(info.core_to_threads), 6)
        self.assertEqual(info.core_to_threads[0], [0, 1])

    def test_detect_cpu_info_intel_i7_10750h(self):
        with open(self.proc_cpuinfo, "w") as f:
            f.write(SAMPLE_INTEL_CPUINFO)

        self._setup_mock_sysfs(vendor="intel", num_cpus=12, base_khz=2600000, min_khz=800000, driver="intel_pstate")

        info: CPUInfo = detect_cpu_info(proc_cpuinfo_path=self.proc_cpuinfo, sysfs_root=self.sysfs_root)

        self.assertEqual(info.vendor, CPUVendor.INTEL)
        self.assertIn("i7-10750H", info.model_name)
        self.assertEqual(info.logical_cpus, 12)
        self.assertEqual(info.physical_cores, 6)
        self.assertEqual(info.base_freq_mhz, 2600.0)
        self.assertEqual(info.max_boost_mhz, 5000.0)
        self.assertEqual(info.scaling_driver, ScalingDriver.INTEL_PSTATE)
        self.assertTrue(info.has_epp)
        self.assertTrue(info.has_epb)

    def test_lookup_boost_frequency(self):
        # Known model lookup
        self.assertEqual(lookup_boost_frequency("AMD Ryzen 5 4600H with Radeon Graphics", 3000.0, {"cpb"}), 4000.0)
        self.assertEqual(lookup_boost_frequency("AMD Ryzen 7 4800H with Radeon Graphics", 2900.0, {"cpb"}), 4200.0)
        self.assertEqual(lookup_boost_frequency("AMD Ryzen 7 5800H", 3200.0, {"cpb"}), 4400.0)
        self.assertEqual(lookup_boost_frequency("Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz", 2600.0, {"ida"}), 5000.0)

        # Fallback heuristic when model is unknown
        fallback = lookup_boost_frequency("Unknown Custom CPU @ 2.50GHz", 2500.0, set())
        self.assertEqual(fallback, 2500.0)

        # Fallback heuristic with cpb flag
        fallback_boost = lookup_boost_frequency("Unknown Custom CPU", 2000.0, {"cpb"})
        self.assertAlmostEqual(fallback_boost, 2600.0)

    def test_lookup_boost_frequency_from_sysfs(self):
        # Create scaling_boost_frequencies file in sysfs
        boost_file = Path(self.sysfs_root) / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_boost_frequencies"
        boost_file.parent.mkdir(parents=True, exist_ok=True)
        boost_file.write_text("3200000 4200000 4400000\n")

        freq = lookup_boost_frequency("Unknown CPU", 3000.0, set(), sysfs_root=self.sysfs_root)
        self.assertEqual(freq, 4400.0)

        # Empty file
        boost_file.write_text("\n")
        freq2 = lookup_boost_frequency("Unknown CPU", 3000.0, set(), sysfs_root=self.sysfs_root)
        self.assertEqual(freq2, 3000.0)

    def test_lookup_boost_frequency_sysfs_exception(self):
        boost_file = Path(self.sysfs_root) / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_boost_frequencies"
        boost_file.parent.mkdir(parents=True, exist_ok=True)
        boost_file.write_text("4000000\n")
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Permission denied")):
            freq = lookup_boost_frequency("Unknown CPU", 3000.0, set(), sysfs_root=self.sysfs_root)
            self.assertEqual(freq, 3000.0)

    def test_detect_cpu_info_real_host(self):
        info = detect_cpu_info()
        self.assertGreater(info.logical_cpus, 0)
        self.assertGreater(info.physical_cores, 0)
        self.assertGreater(info.base_freq_mhz, 0)
        self.assertGreater(info.max_boost_mhz, 0)

    def test_detect_cpu_info_missing_or_corrupt_files(self):
        empty_proc = os.path.join(self.temp_dir, "nonexistent_cpuinfo")
        empty_sys = os.path.join(self.temp_dir, "empty_sys")
        os.makedirs(empty_sys, exist_ok=True)

        info = detect_cpu_info(proc_cpuinfo_path=empty_proc, sysfs_root=empty_sys)
        self.assertEqual(info.vendor, CPUVendor.UNKNOWN)
        self.assertGreaterEqual(info.logical_cpus, 1)

    def test_detect_cpu_info_proc_read_exception(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("Read error")):
            with open(self.proc_cpuinfo, "w") as f:
                f.write("test")
            info = detect_cpu_info(proc_cpuinfo_path=self.proc_cpuinfo, sysfs_root=self.sysfs_root)
            self.assertEqual(info.vendor, CPUVendor.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
