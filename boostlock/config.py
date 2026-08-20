"""
Configuration management, validation, and serialization for BoostLock.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigValidationError(ValueError):
    """Raised when configuration parameters fail validation constraints."""
    pass


@dataclass
class BoostLockConfig:
    """Configuration options for BoostLock daemon and operations."""

    target_frequency_khz: int = 4000000  # 4.0 GHz target frequency
    min_pulse_duty_pct: float = 5.0      # Minimum pulse stimulation duty cycle (%)
    max_pulse_duty_pct: float = 50.0     # Maximum pulse stimulation duty cycle (%)
    duty_step_pct: float = 2.0           # Step size for duty cycle adjustment (%)
    thermal_limit_c: float = 100.0        # Tripwire limit: emergency disengagement (C)
    thermal_warn_c: float = 90.0         # Warning limit: proportional duty cycle throttling (C)
    thermal_recover_c: float = 85.0      # Recovery hysteresis floor: re-engage boost (C)
    poll_interval_ms: int = 100          # Sensor & closed-loop poll interval (ms)
    dma_latency_us: int = 0              # PM QoS latency constraint in microseconds (0 = prevent C2+)
    governor: str = "performance"        # CPU frequency scaling governor
    epp: str = "performance"             # Energy Performance Preference
    pid_file: str = "/var/run/boostlock/boostlock.pid"
    socket_path: str = "/var/run/boostlock/boostlock.sock"
    snapshot_path: str = "/var/run/boostlock/snapshot.json"
    log_file: str = "/var/log/boostlock.log"

    @property
    def pulse_duty_cycle(self) -> float:
        """Legacy alias for min_pulse_duty_pct as 0-1 fraction."""
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
        """Validate configuration settings and raise ConfigValidationError on invalid values."""
        if self.target_frequency_khz <= 0 or self.target_frequency_khz > 10_000_000:
            raise ConfigValidationError(
                f"target_frequency_khz must be between 1 and 10000000 kHz, got {self.target_frequency_khz}"
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
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoostLockConfig:
        """Create BoostLockConfig from a dictionary, ignoring extraneous keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize configuration to JSON string or write to file path."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str) -> BoostLockConfig:
        """Load configuration from JSON string or file path."""
        if os.path.isfile(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        return cls.from_dict(data)
