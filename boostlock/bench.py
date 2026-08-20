"""
boostlock/bench.py - Boost clock stability benchmark suite.

Samples per-core frequencies at a configurable sample rate, computes
statistical metrics (p50/p90/p99, mean, min, max, compliance ratio),
and produces a formatted stability report.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .sysfs import SysfsController


@dataclass
class BenchSample:
    """One instantaneous sample of all online-CPU frequencies."""

    timestamp: float
    """Wall-clock time of the sample."""
    frequencies_khz: List[int]
    """Per-CPU current frequency in kHz (index = logical CPU id)."""
    cpu_ids: List[int]
    """Logical CPU IDs corresponding to *frequencies_khz*."""
    temperature_c: Optional[float] = None
    """Package temperature at sample time, if available."""


@dataclass
class BenchResult:
    """Aggregate statistics from a completed benchmark run."""

    target_khz: int
    """Boost target frequency in kHz (e.g. 4_000_000 for 4.0 GHz)."""
    duration_s: float
    """Requested benchmark duration in seconds."""
    sample_count: int
    """Total samples collected."""
    cpu_ids: List[int]
    """Logical CPUs sampled."""

    # Per-sample aggregate stats (all CPUs merged)
    all_samples_khz: List[int] = field(default_factory=list)
    """Flat list of every (cpu, sample) frequency value."""

    min_khz: int = 0
    max_khz: int = 0
    mean_khz: float = 0.0
    p50_khz: int = 0
    p90_khz: int = 0
    p99_khz: int = 0

    compliance_rate: float = 0.0
    """Fraction of samples at or above *target_khz* (0.0-1.0)."""

    temp_start_c: Optional[float] = None
    temp_end_c: Optional[float] = None
    thermal_gradient_c: Optional[float] = None
    """Temperature rise (end - start) in C, if available."""

    elapsed_s: float = 0.0
    """Actual elapsed wall time of the benchmark run."""

    def format_report(self) -> str:
        """Return a human-readable multi-line benchmark report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  BoostLock Benchmark Report")
        lines.append("=" * 60)
        lines.append(f"  Target frequency : {self.target_khz / 1_000_000:.3f} GHz")
        lines.append(f"  Duration         : {self.duration_s:.1f}s requested  "
                     f"/ {self.elapsed_s:.2f}s actual")
        lines.append(f"  Samples collected: {self.sample_count} "
                     f"({len(self.cpu_ids)} CPUs x {self.sample_count // max(len(self.cpu_ids), 1)} avg)")
        lines.append("")
        lines.append("  Frequency distribution (all cores combined):")
        lines.append(f"    Min   : {self.min_khz / 1_000_000:.3f} GHz")
        lines.append(f"    p50   : {self.p50_khz / 1_000_000:.3f} GHz")
        lines.append(f"    p90   : {self.p90_khz / 1_000_000:.3f} GHz")
        lines.append(f"    p99   : {self.p99_khz / 1_000_000:.3f} GHz")
        lines.append(f"    Mean  : {self.mean_khz / 1_000_000:.3f} GHz")
        lines.append(f"    Max   : {self.max_khz / 1_000_000:.3f} GHz")
        lines.append("")
        compliance_pct = self.compliance_rate * 100.0
        bar_width = 30
        filled = int(round(compliance_pct / 100.0 * bar_width))
        bar = "#" * filled + "." * (bar_width - filled)
        lines.append(f"  Boost compliance : [{bar}] {compliance_pct:.1f}%")
        if self.thermal_gradient_c is not None:
            lines.append(f"  Thermal gradient : +{self.thermal_gradient_c:+.1f}C "
                         f"({self.temp_start_c:.1f}C -> {self.temp_end_c:.1f}C)")
        elif self.temp_start_c is not None:
            lines.append(f"  Temperature      : {self.temp_start_c:.1f}C "
                         f"(thermal sensor unavailable at end)")
        else:
            lines.append("  Temperature      : sensor unavailable")
        lines.append("=" * 60)
        return "\n".join(lines)


def _percentile(sorted_data: List[int], pct: float) -> int:
    """Return the *pct*-th percentile of a pre-sorted list (0-100)."""
    if not sorted_data:
        return 0
    if pct <= 0:
        return sorted_data[0]
    if pct >= 100:
        return sorted_data[-1]
    n = len(sorted_data)
    rank = pct / 100.0 * (n - 1)
    lo = int(rank)
    hi = lo + 1
    frac = rank - lo
    if hi >= n:
        return sorted_data[-1]
    return int(sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac)


class BenchmarkRunner:
    """
    Samples per-core *scaling_cur_freq* at *sample_rate_hz* for *duration_s*
    seconds and aggregates statistical metrics.

    Parameters
    ----------
    target_khz:
        Boost goal frequency in kHz (default 4_000_000 = 4.0 GHz).
    duration_s:
        How long to run the benchmark.
    sample_rate_hz:
        How many times per second to sample all online CPUs.
    sysfs_root:
        Override for the sysfs root (used in testing).
    thermal_getter:
        Optional callable() -> Optional[float] that returns current package
        temperature in C.  When provided, start/end temps and thermal
        gradient are recorded.
    """

    def __init__(
        self,
        target_khz: int = 4_000_000,
        duration_s: float = 10.0,
        sample_rate_hz: float = 20.0,
        sysfs_root: Optional[Path] = None,
        thermal_getter=None,
    ) -> None:
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
        if target_khz <= 0:
            raise ValueError(f"target_khz must be positive, got {target_khz}")

        self.target_khz = target_khz
        self.duration_s = duration_s
        self.sample_rate_hz = sample_rate_hz
        self._sysfs = SysfsController(sysfs_root=sysfs_root if sysfs_root is not None else Path("/sys"))
        self._thermal_getter = thermal_getter
        self._samples: List[BenchSample] = []

    @property
    def samples(self) -> List[BenchSample]:
        return list(self._samples)

    def _get_temperature(self) -> Optional[float]:
        if self._thermal_getter is not None:
            return self._thermal_getter()
        return None

    def _take_sample(self) -> BenchSample:
        cpu_ids = self._sysfs.get_online_cpus()
        freqs: List[int] = []
        for cpu in cpu_ids:
            try:
                f = self._sysfs.get_scaling_cur_freq(cpu)
                freqs.append(f if f is not None else 0)
            except Exception:
                freqs.append(0)
        return BenchSample(
            timestamp=time.monotonic(),
            frequencies_khz=freqs,
            cpu_ids=cpu_ids,
            temperature_c=self._get_temperature(),
        )

    def run(self) -> BenchResult:
        """
        Execute the benchmark loop and return a populated *BenchResult*.
        This is a blocking call that runs for approximately *duration_s* seconds.
        """
        interval_s = 1.0 / self.sample_rate_hz
        self._samples = []

        temp_start = self._get_temperature()
        t_start = time.monotonic()
        deadline = t_start + self.duration_s

        while True:
            loop_start = time.monotonic()
            if loop_start >= deadline:
                break
            sample = self._take_sample()
            self._samples.append(sample)
            elapsed_in_loop = time.monotonic() - loop_start
            sleep_for = interval_s - elapsed_in_loop
            if sleep_for > 0:
                time.sleep(sleep_for)

        t_end = time.monotonic()
        elapsed = t_end - t_start
        temp_end = self._get_temperature()

        return self._compute_result(temp_start, temp_end, elapsed)

    def _compute_result(
        self,
        temp_start: Optional[float],
        temp_end: Optional[float],
        elapsed_s: float,
    ) -> BenchResult:
        # Flatten all (cpu, sample) frequency values
        all_freqs: List[int] = []
        cpu_ids: List[int] = self._samples[0].cpu_ids if self._samples else []

        for s in self._samples:
            all_freqs.extend(s.frequencies_khz)

        if not all_freqs:
            # No samples at all - return zeroed result
            return BenchResult(
                target_khz=self.target_khz,
                duration_s=self.duration_s,
                sample_count=0,
                cpu_ids=cpu_ids,
                elapsed_s=elapsed_s,
                temp_start_c=temp_start,
                temp_end_c=temp_end,
            )

        sorted_freqs = sorted(all_freqs)
        compliance = sum(1 for f in all_freqs if f >= self.target_khz) / len(all_freqs)

        thermal_gradient: Optional[float] = None
        if temp_start is not None and temp_end is not None:
            thermal_gradient = temp_end - temp_start

        return BenchResult(
            target_khz=self.target_khz,
            duration_s=self.duration_s,
            sample_count=len(self._samples),
            cpu_ids=cpu_ids,
            all_samples_khz=all_freqs,
            min_khz=sorted_freqs[0],
            max_khz=sorted_freqs[-1],
            mean_khz=statistics.mean(all_freqs),
            p50_khz=_percentile(sorted_freqs, 50),
            p90_khz=_percentile(sorted_freqs, 90),
            p99_khz=_percentile(sorted_freqs, 99),
            compliance_rate=compliance,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            thermal_gradient_c=thermal_gradient,
            elapsed_s=elapsed_s,
        )
