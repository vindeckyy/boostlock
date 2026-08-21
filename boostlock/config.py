"""
Configuration management, validation, and serialization for BoostLock.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union


class ConfigValidationError(ValueError):
    """Raised when configuration parameters fail validation constraints."""
    pass


@dataclass
class BoostLockConfig:
    """BoostLock config."""

    target_frequency_khz: Union[int, Literal["auto"]] = 4000000
    # kHz target or "auto"
    min_pulse_duty_pct: float = 5.0
    max_pulse_duty_pct: float = 50.0
    duty_step_pct: float = 2.0
    thermal_limit_c: float = 100.0
    thermal_warn_c: float = 90.0
    thermal_recover_c: float = 85.0
    poll_interval_ms: int = 100
    dma_latency_us: int = 0
    governor: str = "performance"
    epp: str = "performance"
    pid_file: str = "/var/run/boostlock/boostlock.pid"
    socket_path: str = "/var/run/boostlock/boostlock.sock"
    snapshot_path: str = "/var/run/boostlock/snapshot.json"
    log_file: str = "/var/log/boostlock.log"

    @property
    def pulse_duty_cycle(self) -> float:
        """Alias for min_pulse_duty_pct."""
        return self.min_pulse_duty_pct / 100.0

    @pulse_duty_cycle.setter
    def pulse_duty_cycle(self, value: float) -> None:
        # Accept 0-1 fraction (from CLI) or 0-100 percent
        if value <= 1.0:
            pct = value * 100.0
        else:
            pct = value
        self.min_pulse_duty_pct = float(pct)
        if self.max_pulse_duty_pct < self.min_pulse_duty_pct:
            self.max_pulse_duty_pct = float(pct)

    def validate(self) -> None:
        """Validate config."""
        target = self.target_frequency_khz
        if target == "auto":
            pass
        elif (
            not isinstance(target, int)
            or isinstance(target, bool)
            or target <= 0
            or target > 10_000_000
        ):
            raise ConfigValidationError(
                "target_frequency_khz must be 'auto' or an integer between 1 and "
                f"10000000 kHz, got {target}"
            )

        if not (50.0 <= self.thermal_limit_c <= 115.0):
            raise ConfigValidationError(
                f"thermal_limit_c must be between 50.0C and 115.0C, got {self.thermal_limit_c}"
            )

        if not (30.0 <= self.thermal_warn_c < self.thermal_limit_c):
            raise ConfigValidationError(
                f"thermal_warn_c ({self.thermal_warn_c}C) must be >= 30.0C and strictly < thermal_limit_c ({self.thermal_limit_c}C)"
            )

        if not (20.0 <= self.thermal_recover_c < self.thermal_warn_c):
            raise ConfigValidationError(
                f"thermal_recover_c ({self.thermal_recover_c}C) must be >= 20.0C and strictly < thermal_warn_c ({self.thermal_warn_c}C)"
            )

        if not (0.0 <= self.min_pulse_duty_pct <= 100.0):
            raise ConfigValidationError(
                f"min_pulse_duty_pct must be between 0.0 and 100.0, got {self.min_pulse_duty_pct}"
            )

        if not (0.0 <= self.max_pulse_duty_pct <= 100.0):
            raise ConfigValidationError(
                f"max_pulse_duty_pct must be between 0.0 and 100.0, got {self.max_pulse_duty_pct}"
            )

        if self.min_pulse_duty_pct > self.max_pulse_duty_pct:
            raise ConfigValidationError(
                f"min_pulse_duty_pct ({self.min_pulse_duty_pct}) cannot exceed max_pulse_duty_pct ({self.max_pulse_duty_pct})"
            )

        if not (0.1 <= self.duty_step_pct <= 50.0):
            raise ConfigValidationError(
                f"duty_step_pct must be between 0.1 and 50.0, got {self.duty_step_pct}"
            )

        if self.poll_interval_ms < 10:
            raise ConfigValidationError(
                f"poll_interval_ms must be at least 10 ms, got {self.poll_interval_ms}"
            )

        if self.dma_latency_us < 0:
            raise ConfigValidationError(
                f"dma_latency_us must be non-negative, got {self.dma_latency_us}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoostLockConfig:
        """From dict, ignore unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_json(self, path: Optional[str] = None) -> str:
        """To JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str) -> BoostLockConfig:
        """From JSON."""
        if os.path.isfile(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        return cls.from_dict(data)
