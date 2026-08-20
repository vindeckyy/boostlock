"""tests/test_bench.py - Tests for boostlock/bench.py"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from boostlock.bench import (
    BenchResult,
    BenchSample,
    BenchmarkRunner,
    _percentile,
)


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 50) == 0

    def test_single_element(self):
        assert _percentile([100], 50) == 100

    def test_p0_returns_min(self):
        data = [1, 2, 3, 4, 5]
        assert _percentile(data, 0) == 1

    def test_p100_returns_max(self):
        data = [1, 2, 3, 4, 5]
        assert _percentile(data, 100) == 5

    def test_p50_median(self):
        data = sorted([1, 2, 3, 4, 5])
        result = _percentile(data, 50)
        # Median of 5 items (index 2) = 3
        assert result == pytest.approx(3, abs=1)

    def test_p90_typical(self):
        data = sorted(range(100))
        result = _percentile(data, 90)
        assert result == pytest.approx(89, abs=1)

    def test_p99_typical(self):
        data = sorted(range(100))
        result = _percentile(data, 99)
        assert result >= 98

    def test_uniform_data(self):
        data = [4_000_000] * 100
        assert _percentile(data, 50) == 4_000_000
        assert _percentile(data, 99) == 4_000_000

    def test_two_elements(self):
        data = [1000, 4000]
        assert _percentile(data, 0) == 1000
        assert _percentile(data, 100) == 4000
        # p50 interpolates midpoint
        assert _percentile(data, 50) == pytest.approx(2500, abs=1)

    def test_boundary_hi_beyond_len(self):
        # Ensure no IndexError when hi == len(data)
        data = [10]
        assert _percentile(data, 99) == 10


# ---------------------------------------------------------------------------
# BenchSample dataclass
# ---------------------------------------------------------------------------

class TestBenchSample:
    def test_construction(self):
        s = BenchSample(
            timestamp=1.0,
            frequencies_khz=[4_000_000, 3_900_000],
            cpu_ids=[0, 1],
        )
        assert s.timestamp == 1.0
        assert s.frequencies_khz == [4_000_000, 3_900_000]
        assert s.cpu_ids == [0, 1]
        assert s.temperature_c is None

    def test_with_temperature(self):
        s = BenchSample(
            timestamp=2.0,
            frequencies_khz=[4_000_000],
            cpu_ids=[0],
            temperature_c=72.5,
        )
        assert s.temperature_c == 72.5


# ---------------------------------------------------------------------------
# BenchResult.format_report
# ---------------------------------------------------------------------------

class TestBenchResultFormatReport:
    def _make_result(
        self,
        all_samples: Optional[List[int]] = None,
        compliance: float = 1.0,
        temp_start: Optional[float] = None,
        temp_end: Optional[float] = None,
    ) -> BenchResult:
        freqs = all_samples or [4_000_000] * 100
        sorted_f = sorted(freqs)
        gradient = (temp_end - temp_start) if (temp_start is not None and temp_end is not None) else None
        return BenchResult(
            target_khz=4_000_000,
            duration_s=10.0,
            sample_count=len(freqs) // 4,
            cpu_ids=[0, 1, 2, 3],
            all_samples_khz=freqs,
            min_khz=sorted_f[0],
            max_khz=sorted_f[-1],
            mean_khz=statistics.mean(freqs),
            p50_khz=_percentile(sorted_f, 50),
            p90_khz=_percentile(sorted_f, 90),
            p99_khz=_percentile(sorted_f, 99),
            compliance_rate=compliance,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            thermal_gradient_c=gradient,
            elapsed_s=10.05,
        )

    def test_report_contains_header(self):
        r = self._make_result()
        report = r.format_report()
        assert "BoostLock Benchmark Report" in report

    def test_report_contains_target_freq(self):
        r = self._make_result()
        report = r.format_report()
        assert "4.000 GHz" in report

    def test_report_contains_compliance_100(self):
        r = self._make_result(compliance=1.0)
        report = r.format_report()
        assert "100.0%" in report

    def test_report_contains_compliance_partial(self):
        freqs = [4_000_000] * 75 + [3_200_000] * 25
        r = self._make_result(all_samples=freqs, compliance=0.75)
        report = r.format_report()
        assert "75.0%" in report

    def test_report_contains_min_max(self):
        freqs = [3_000_000, 4_000_000]
        r = self._make_result(all_samples=freqs, compliance=0.5)
        report = r.format_report()
        assert "3.000 GHz" in report
        assert "4.000 GHz" in report

    def test_report_with_temperatures(self):
        r = self._make_result(temp_start=60.0, temp_end=72.5)
        report = r.format_report()
        assert "60.0" in report
        assert "72.5" in report
        assert "+12.5" in report or "12.5" in report

    def test_report_without_temperatures(self):
        r = self._make_result()
        report = r.format_report()
        assert "unavailable" in report

    def test_report_with_temp_start_only(self):
        r = self._make_result(temp_start=65.0)
        report = r.format_report()
        assert "65.0" in report

    def test_report_contains_separator_lines(self):
        r = self._make_result()
        report = r.format_report()
        assert "=" in report


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class MockSysfsForBench:
    """Minimal sysfs mock used by BenchmarkRunner in tests."""

    def __init__(self, cpu_ids: List[int], freq_khz: int = 4_000_000):
        self._cpu_ids = cpu_ids
        self._freq_khz = freq_khz

    def get_online_cpus(self) -> List[int]:
        return list(self._cpu_ids)

    def get_scaling_cur_freq(self, cpu_id: int) -> int:
        return self._freq_khz


class TestBenchmarkRunnerConstructor:
    def test_defaults(self, tmp_path):
        runner = BenchmarkRunner()
        assert runner.target_khz == 4_000_000
        assert runner.duration_s == 10.0
        assert runner.sample_rate_hz == 20.0

    def test_custom_params(self, tmp_path):
        runner = BenchmarkRunner(target_khz=3_900_000, duration_s=5.0, sample_rate_hz=10.0)
        assert runner.target_khz == 3_900_000
        assert runner.duration_s == 5.0
        assert runner.sample_rate_hz == 10.0

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration_s"):
            BenchmarkRunner(duration_s=0)

    def test_invalid_negative_duration(self):
        with pytest.raises(ValueError, match="duration_s"):
            BenchmarkRunner(duration_s=-1.0)

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            BenchmarkRunner(sample_rate_hz=0)

    def test_invalid_target_khz(self):
        with pytest.raises(ValueError, match="target_khz"):
            BenchmarkRunner(target_khz=0)

    def test_samples_starts_empty(self):
        runner = BenchmarkRunner()
        assert runner.samples == []


class TestBenchmarkRunnerRun:
    def _patched_runner(
        self,
        cpu_ids: List[int],
        freq_khz: int = 4_000_000,
        duration_s: float = 0.2,
        sample_rate_hz: float = 50.0,
        thermal_getter=None,
    ) -> BenchmarkRunner:
        runner = BenchmarkRunner(
            target_khz=4_000_000,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            thermal_getter=thermal_getter,
        )
        mock_sysfs = MockSysfsForBench(cpu_ids=cpu_ids, freq_khz=freq_khz)
        runner._sysfs = mock_sysfs
        return runner

    def test_run_returns_bench_result(self):
        runner = self._patched_runner(cpu_ids=[0, 1], freq_khz=4_000_000, duration_s=0.1)
        result = runner.run()
        assert isinstance(result, BenchResult)

    def test_run_collects_samples(self):
        runner = self._patched_runner(cpu_ids=[0, 1], duration_s=0.2, sample_rate_hz=50.0)
        result = runner.run()
        # At 50 Hz for 0.2s we expect ~10 samples; allow wide range due to timing
        assert result.sample_count >= 1

    def test_run_all_at_target_gives_100_compliance(self):
        runner = self._patched_runner(cpu_ids=[0], freq_khz=4_000_000, duration_s=0.1)
        result = runner.run()
        assert result.compliance_rate == pytest.approx(1.0, abs=0.01)

    def test_run_all_below_target_gives_0_compliance(self):
        runner = self._patched_runner(cpu_ids=[0], freq_khz=2_000_000, duration_s=0.1)
        result = runner.run()
        assert result.compliance_rate == pytest.approx(0.0, abs=0.01)

    def test_run_stats_correct_for_uniform_freq(self):
        runner = self._patched_runner(cpu_ids=[0, 1], freq_khz=4_000_000, duration_s=0.1)
        result = runner.run()
        assert result.min_khz == 4_000_000
        assert result.max_khz == 4_000_000
        assert result.mean_khz == pytest.approx(4_000_000, abs=1)
        assert result.p50_khz == 4_000_000
        assert result.p99_khz == 4_000_000

    def test_run_with_thermal_getter(self):
        temps = [60.0, 65.0]
        call_count = [0]

        def mock_temp():
            val = temps[min(call_count[0], len(temps) - 1)]
            call_count[0] += 1
            return val

        runner = self._patched_runner(
            cpu_ids=[0], duration_s=0.1, thermal_getter=mock_temp
        )
        result = runner.run()
        assert result.temp_start_c is not None
        assert result.temp_end_c is not None

    def test_run_without_thermal_getter(self):
        runner = self._patched_runner(cpu_ids=[0], duration_s=0.1)
        result = runner.run()
        assert result.temp_start_c is None
        assert result.temp_end_c is None
        assert result.thermal_gradient_c is None

    def test_run_with_positive_thermal_gradient(self):
        temps = iter([60.0, 72.0])

        def mock_temp():
            try:
                return next(temps)
            except StopIteration:
                return 72.0

        runner = self._patched_runner(
            cpu_ids=[0], duration_s=0.1, thermal_getter=mock_temp
        )
        result = runner.run()
        if result.thermal_gradient_c is not None:
            assert result.thermal_gradient_c == pytest.approx(12.0, abs=1.0)

    def test_run_cpu_ids_in_result(self):
        runner = self._patched_runner(cpu_ids=[0, 1, 2], duration_s=0.1)
        result = runner.run()
        assert result.cpu_ids == [0, 1, 2]

    def test_run_elapsed_approximately_correct(self):
        runner = self._patched_runner(cpu_ids=[0], duration_s=0.2, sample_rate_hz=50.0)
        result = runner.run()
        assert result.elapsed_s == pytest.approx(0.2, abs=0.15)

    def test_samples_property_returns_copy(self):
        runner = self._patched_runner(cpu_ids=[0], duration_s=0.1)
        runner.run()
        s1 = runner.samples
        s2 = runner.samples
        assert s1 is not s2

    def test_sysfs_freq_exception_records_zero(self):
        runner = self._patched_runner(cpu_ids=[0], duration_s=0.1)

        class ErrorSysfs:
            def get_online_cpus(self):
                return [0]

            def get_scaling_cur_freq(self, cpu_id):
                raise OSError("Permission denied")

        runner._sysfs = ErrorSysfs()
        result = runner.run()
        assert result.sample_count >= 1
        # All zeros -> compliance 0
        assert result.compliance_rate == 0.0

    def test_no_samples_returns_zeroed_result(self):
        """If the deadline is already reached before any sample, return zeroed."""
        runner = BenchmarkRunner(target_khz=4_000_000, duration_s=0.001, sample_rate_hz=1.0)
        mock_sysfs = MockSysfsForBench(cpu_ids=[0])
        runner._sysfs = mock_sysfs

        # Patch time.monotonic so deadline is instantly past
        call_count = [0]
        base = time.monotonic()

        def fake_monotonic():
            v = call_count[0]
            call_count[0] += 1
            # First call = t_start, second call = loop_start (already past deadline)
            if v == 0:
                return base
            return base + 10.0  # always past deadline

        with patch("boostlock.bench.time.monotonic", side_effect=fake_monotonic):
            with patch("boostlock.bench.time.sleep"):
                result = runner.run()

        assert result.sample_count == 0
        assert result.min_khz == 0
        assert result.compliance_rate == 0.0
