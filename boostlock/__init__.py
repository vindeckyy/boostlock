"""
BoostLock: 24/7 Sustained CPU Boost Clock Management System.
"""

from boostlock.config import BoostLockConfig, ConfigValidationError
from boostlock.hardware import (
    CPUInfo,
    CPUVendor,
    CoreInfo,
    ScalingDriver,
    detect_cpu_info,
)
from boostlock.sysfs import (
    SysfsController,
    SysfsCorruptError,
    SysfsError,
    SysfsNotFoundError,
    SysfsPermissionError,
)

__version__ = "0.1.0"
__all__ = [
    "BoostLockConfig",
    "ConfigValidationError",
    "CPUInfo",
    "CPUVendor",
    "CoreInfo",
    "ScalingDriver",
    "detect_cpu_info",
    "SysfsController",
    "SysfsError",
    "SysfsPermissionError",
    "SysfsNotFoundError",
    "SysfsCorruptError",
    "__version__",
]
