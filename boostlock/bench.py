"""
CPU frequency benchmark helpers.

Samples CPU frequencies and formats a report with percentiles and compliance.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .sysfs import SysfsController


@dataclass
class BenchSample:
    """One freq sample."""

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
    """Bench results."""

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
    """All sampled values."""

    min_khz: int = 0
    max_khz: int = 0
    mean_khz: float = 0.0
    p50_khz: int = 0
    p90_khz: int = 0
    p99_khz: int = 0

    compliance_rate: float = 0.0
    """Share at or above target."""

    temp_start_c: Optional[float] = None
    temp_end_c: Optional[float] = None
    thermal_gradient_c: Optional[float] = None
    """Temp delta if available."""

    elapsed_s: float = 0.0
    """Wall time."""
    policy_id: Optional[str] = None
    """Policy for this result."""
    policy_results: Dict[str, BenchResult] = field(default_factory=dict)
    """Per-policy breakdown."""

    def format_report(self) -> str:
        """Format a bench report."""
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
        if self.policy_results:
            lines.append("")
            lines.append("  Per-policy compliance:")
            for policy_id, result in sorted(self.policy_results.items()):
                lines.append(
                    f"    {policy_id}: {result.target_khz / 1_000_000:.3f} GHz "
                    f"target, {result.compliance_rate * 100.0:.1f}%"
                )
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
    """Percentile of sorted data."""
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
        target_khz: Union[int, str] = 4_000_000,
        duration_s: float = 10.0,
        sample_rate_hz: float = 20.0,
        sysfs_root: Optional[Path] = None,
        thermal_getter=None,
    ) -> None:
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
        if target_khz != "auto" and (
            not isinstance(target_khz, int) or target_khz <= 0
        ):
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

        policy_targets = self.resolve_policy_targets()
        aggregate_target = (
            self.target_khz
            if isinstance(self.target_khz, int)
            else max(
                (details["effective_target_khz"] for details in policy_targets.values()),
                default=4_000_000,
            )
        )
        result = self._compute_result(temp_start, temp_end, elapsed, aggregate_target)
        result.policy_results = self._policy_results(
            policy_targets,
            temp_start,
            temp_end,
            elapsed,
        )
        return result

    def resolve_policy_targets(self) -> Dict[str, Dict[str, Any]]:
        """Resolve each discovered policy to its own effective benchmark target."""
        try:
            policies = self._sysfs.discover_cpufreq_policies()
        except (AttributeError, OSError):
            policies = []

        resolved: Dict[str, Dict[str, Any]] = {}
        for policy in policies:
            target, reason = self._effective_policy_target(policy)
            if target is None:
                continue
            resolved[policy.identifier] = {
                "member_cpus": list(policy.cpus),
                "driver": policy.driver,
                "requested_target": self.target_khz,
                "effective_target_khz": target,
                "clamp_reason": reason,
            }

        if resolved:
            return resolved
        target = self.target_khz if isinstance(self.target_khz, int) else 4_000_000
        try:
            cpu_ids = self._sysfs.get_online_cpus()
        except Exception:
            cpu_ids = []
        return {
            f"cpu{cpu_id}": {
                "member_cpus": [cpu_id],
                "driver": None,
                "requested_target": self.target_khz,
                "effective_target_khz": target,
                "clamp_reason": "policy discovery unavailable",
            }
            for cpu_id in cpu_ids
        }

    def _effective_policy_target(self, policy: Any) -> tuple[Optional[int], Optional[str]]:
        """Clamp one requested target within the limits exposed by that policy."""
        lower_bounds = [
            value
            for value in (policy.hardware_min_khz, policy.active_min_khz)
            if isinstance(value, int) and value > 0
        ]
        upper_bounds = [
            value
            for value in (policy.hardware_max_khz, policy.active_max_khz)
            if isinstance(value, int) and value > 0
        ]
        if not lower_bounds or not upper_bounds:
            return None, "policy frequency bounds unavailable"
        lower_bound = max(lower_bounds)
        upper_bound = min(upper_bounds)
        if lower_bound > upper_bound:
            return None, "policy frequency bounds are incompatible"
        if self.target_khz == "auto":
            return upper_bound, "automatic policy maximum"
        assert isinstance(self.target_khz, int)
        if self.target_khz < lower_bound:
            return lower_bound, f"raised to policy minimum {lower_bound}"
        if self.target_khz > upper_bound:
            return upper_bound, f"clamped to policy maximum {upper_bound}"
        return self.target_khz, None

    def _policy_results(
        self,
        policy_targets: Dict[str, Dict[str, Any]],
        temp_start: Optional[float],
        temp_end: Optional[float],
        elapsed_s: float,
    ) -> Dict[str, BenchResult]:
        """Calculate policy-local samples against each policy-local target."""
        results: Dict[str, BenchResult] = {}
        for policy_id, details in policy_targets.items():
            members = set(details["member_cpus"])
            samples = [
                BenchSample(
                    timestamp=sample.timestamp,
                    frequencies_khz=[
                        frequency
                        for cpu_id, frequency in zip(sample.cpu_ids, sample.frequencies_khz)
                        if cpu_id in members
                    ],
                    cpu_ids=[cpu_id for cpu_id in sample.cpu_ids if cpu_id in members],
                    temperature_c=sample.temperature_c,
                )
                for sample in self._samples
            ]
            results[policy_id] = self._compute_result(
                temp_start,
                temp_end,
                elapsed_s,
                details["effective_target_khz"],
                samples=samples,
                policy_id=policy_id,
            )
        return results

    def _compute_result(
        self,
        temp_start: Optional[float],
        temp_end: Optional[float],
        elapsed_s: float,
        target_khz: Optional[int] = None,
        *,
        samples: Optional[Sequence[BenchSample]] = None,
        policy_id: Optional[str] = None,
    ) -> BenchResult:
        # Flatten all (cpu, sample) frequency values
        all_freqs: List[int] = []
        sample_set = list(samples) if samples is not None else self._samples
        cpu_ids: List[int] = sample_set[0].cpu_ids if sample_set else []
        resolved_target = target_khz if target_khz is not None else self.target_khz
        if not isinstance(resolved_target, int):
            raise ValueError("A numeric target is required to compute benchmark compliance")

        for s in sample_set:
            all_freqs.extend(s.frequencies_khz)

        if not all_freqs:
            # No samples at all - return zeroed result
            return BenchResult(
                target_khz=resolved_target,
                duration_s=self.duration_s,
                sample_count=len(sample_set),
                cpu_ids=cpu_ids,
                elapsed_s=elapsed_s,
                temp_start_c=temp_start,
                temp_end_c=temp_end,
                policy_id=policy_id,
            )

        sorted_freqs = sorted(all_freqs)
        compliance = sum(1 for f in all_freqs if f >= resolved_target) / len(all_freqs)

        thermal_gradient: Optional[float] = None
        if temp_start is not None and temp_end is not None:
            thermal_gradient = temp_end - temp_start

        return BenchResult(
            target_khz=resolved_target,
            duration_s=self.duration_s,
            sample_count=len(sample_set),
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
            policy_id=policy_id,
        )
