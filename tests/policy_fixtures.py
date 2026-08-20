"""Reusable cpufreq-policy sysfs fixtures for policy contract tests."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PolicySpec:
    """The writable and optional nodes exposed by one Linux cpufreq policy."""

    name: str
    cpus: tuple[int, ...]
    driver: str = "generic-cpufreq"
    hardware_min_khz: int = 800_000
    hardware_max_khz: int = 3_000_000
    active_min_khz: int = 800_000
    active_max_khz: int = 3_000_000
    current_khz: int = 2_000_000
    related_cpus: str | None = None
    affected_cpus: str | None = None
    optional_nodes: Mapping[str, str] = field(default_factory=dict)
    missing_nodes: tuple[str, ...] = ()


class WriteOpenAudit:
    """Records planned write-opens and rejects selected paths before mutation."""

    def __init__(self, rejected_paths: Iterable[Path] = ()) -> None:
        self.rejected_paths = {Path(path).resolve() for path in rejected_paths}
        self.opened_paths: list[Path] = []

    def __call__(self, path: str | Path):
        resolved = Path(path).resolve()
        self.opened_paths.append(resolved)
        if resolved in self.rejected_paths:
            raise PermissionError(f"write-open rejected for {resolved}")
        return io.StringIO()


class PolicySysfsFixture:
    """Build a sysfs tree with policy directories and CPU cpufreq aliases."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "sys"
        self.cpu_root = self.root / "devices" / "system" / "cpu"
        self.policy_root = self.cpu_root / "cpufreq"
        self.policy_root.mkdir(parents=True, exist_ok=True)
        self.specs: dict[str, PolicySpec] = {}
        self.write_rejections: set[Path] = set()
        (self.policy_root / "boost").write_text("0\n")

    def add_policy(
        self,
        spec: PolicySpec,
        *,
        aliases: bool = True,
        global_policy: bool = True,
    ) -> Path:
        if not spec.cpus:
            raise ValueError("A policy fixture requires at least one CPU")
        if spec.related_cpus is not None and spec.affected_cpus is not None:
            raise ValueError("related_cpus and affected_cpus are mutually exclusive")

        policy_dir = (self.policy_root if global_policy else self.cpu_root) / spec.name
        policy_dir.mkdir(parents=True)
        nodes = {
            "scaling_driver": spec.driver,
            "related_cpus": spec.related_cpus or " ".join(map(str, spec.cpus)),
            "scaling_governor": "schedutil",
            "scaling_available_governors": "performance schedutil powersave",
            "scaling_min_freq": str(spec.active_min_khz),
            "scaling_max_freq": str(spec.active_max_khz),
            "scaling_cur_freq": str(spec.current_khz),
            "cpuinfo_min_freq": str(spec.hardware_min_khz),
            "cpuinfo_max_freq": str(spec.hardware_max_khz),
        }
        if spec.affected_cpus is not None:
            nodes.pop("related_cpus")
            nodes["affected_cpus"] = spec.affected_cpus
        nodes.update(spec.optional_nodes)
        for name, value in nodes.items():
            if name not in spec.missing_nodes:
                (policy_dir / name).write_text(f"{value}\n")

        self.specs[spec.name] = spec
        for cpu in spec.cpus:
            cpu_dir = self.cpu_root / f"cpu{cpu}"
            cpu_dir.mkdir(parents=True, exist_ok=True)
            if aliases:
                target = f"../cpufreq/{spec.name}" if global_policy else f"../{spec.name}"
                (cpu_dir / "cpufreq").symlink_to(target)
            else:
                alias = cpu_dir / "cpufreq"
                alias.mkdir()
                for node, value in nodes.items():
                    if node not in spec.missing_nodes:
                        (alias / node).write_text(f"{value}\n")
        self._write_online_file()
        return policy_dir

    def add_cpuidle(self, cpus: Iterable[int], states: Iterable[int] = (1, 2)) -> list[Path]:
        paths: list[Path] = []
        for cpu in cpus:
            for state in states:
                disable = self.cpu_root / f"cpu{cpu}" / "cpuidle" / f"state{state}" / "disable"
                disable.parent.mkdir(parents=True, exist_ok=True)
                disable.write_text("0\n")
                paths.append(disable)
        return paths

    def reject_write_open(self, *paths: Path) -> None:
        self.write_rejections.update(path.resolve() for path in paths)

    def write_opener(self) -> WriteOpenAudit:
        return WriteOpenAudit(self.write_rejections)

    def policy_path(self, policy: str, node: str) -> Path:
        return self.policy_root / policy / node

    def _write_online_file(self) -> None:
        cpus = sorted(cpu for spec in self.specs.values() for cpu in spec.cpus)
        (self.cpu_root / "online").write_text(" ".join(map(str, cpus)) + "\n")
