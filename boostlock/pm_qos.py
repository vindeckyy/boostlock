"""
PM QoS CPU DMA-latency management.

Controls `/dev/cpu_dma_latency` and can fall back to cpuidle sysfs settings.
"""

from __future__ import annotations

import errno
import logging
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from boostlock.sysfs import SysfsController, SysfsError, SysfsPermissionError

logger = logging.getLogger(__name__)

DEFAULT_CPU_DMA_LATENCY_PATH = "/dev/cpu_dma_latency"
INT32_MAX = 2147483647


class PMQoSError(Exception):
    """Base exception for PM QoS operations."""
    pass


class PMQoSPermissionError(PMQoSError, PermissionError):
    """Raised when opening or writing to /dev/cpu_dma_latency fails due to permissions."""
    pass


class PMQoSNotFoundError(PMQoSError, FileNotFoundError):
    """Raised when /dev/cpu_dma_latency does not exist and fallback is disabled."""
    pass


class PMQoSLockError(PMQoSError):
    """Raised when PM QoS lock cannot be acquired or modified."""
    pass


class PMQoSController:
    """
    Manages PM QoS CPU DMA latency constraint to prevent deep CPU C-states.

    When opened and held open with a target latency of 0 microseconds, the Linux
    kernel's cpuidle governor will not select idle states with exit latency > 0,
    effectively pinning CPU cores in active execution (C0) or shallow idle (C1).
    """

    def __init__(
        self,
        device_path: Union[str, Path] = DEFAULT_CPU_DMA_LATENCY_PATH,
        sysfs_root: Union[str, Path] = "/sys",
        target_latency_us: int = 0,
        enable_sysfs_fallback: bool = True,
        min_fallback_state_index: int = 1,
    ) -> None:
        self.device_path = Path(device_path)
        self.sysfs_root = Path(sysfs_root).resolve()
        self.sysfs = SysfsController(sysfs_root=self.sysfs_root)
        self.enable_sysfs_fallback = enable_sysfs_fallback
        self.min_fallback_state_index = min_fallback_state_index

        self._validate_latency(target_latency_us)
        self._target_latency_us = target_latency_us

        self._fd: Optional[int] = None
        self._is_locked: bool = False
        self._using_fallback: bool = False
        self._fallback_saved_states: Dict[str, str] = {}

    @property
    def target_latency_us(self) -> int:
        return self._target_latency_us

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def fd(self) -> Optional[int]:
        return self._fd

    def _validate_latency(self, latency_us: int) -> None:
        if latency_us < 0:
            raise ValueError(f"target_latency_us must be non-negative, got {latency_us}")
        if latency_us > INT32_MAX:
            raise ValueError(f"target_latency_us exceeds maximum 32-bit integer ({INT32_MAX}), got {latency_us}")

    def lock(
        self,
        target_latency_us: Optional[int] = None,
        *,
        allow_fallback: bool = True,
    ) -> bool:
        """
        Acquire PM QoS lock by opening /dev/cpu_dma_latency and writing latency target.
        Falls back to cpuidle sysfs state disabling if device is unavailable.
        """
        if target_latency_us is not None:
            self._validate_latency(target_latency_us)
            self._target_latency_us = target_latency_us

        if self._is_locked:
            if not self._using_fallback and self._fd is not None:
                # Update existing lock latency
                try:
                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                    except OSError:
                        pass
                    payload = struct.pack("i", self._target_latency_us)
                    os.write(self._fd, payload)
                    return True
                except OSError as exc:
                    logger.warning(f"Failed to update PM QoS latency on open fd: {exc}")
            return True

        fd: Optional[int] = None

        # Attempt to open device node
        try:
            flags = os.O_RDWR
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC

            fd = os.open(str(self.device_path), flags)
            payload = struct.pack("i", self._target_latency_us)
            try:
                os.write(fd, payload)
            except OSError as exc:
                try:
                    os.close(fd)
                except OSError as close_exc:
                    raise PMQoSError(
                        f"Failed to write target latency to {self.device_path}: {exc}; "
                        f"failed to close descriptor: {close_exc}"
                    ) from exc
                raise

            self._fd = fd
            self._is_locked = True
            self._using_fallback = False
            logger.info(
                f"Acquired PM QoS DMA latency lock ({self._target_latency_us} us) on {self.device_path}"
            )
            return True

        except PermissionError as exc:
            if not self.enable_sysfs_fallback or not allow_fallback:
                raise PMQoSPermissionError(
                    f"Permission denied accessing {self.device_path}. Run with sudo/root privileges."
                ) from exc
            logger.warning(
                f"Permission denied on {self.device_path}, falling back to cpuidle sysfs disable: {exc}"
            )
            return self._lock_via_sysfs_fallback()

        except FileNotFoundError as exc:
            if not self.enable_sysfs_fallback or not allow_fallback:
                raise PMQoSNotFoundError(
                    f"PM QoS device node not found at {self.device_path} and fallback is disabled."
                ) from exc
            logger.info(
                f"Device {self.device_path} not found, using cpuidle sysfs fallback."
            )
            return self._lock_via_sysfs_fallback()

        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
                if not self.enable_sysfs_fallback or not allow_fallback:
                    raise PMQoSPermissionError(
                        f"Permission error (errno {exc.errno}) accessing {self.device_path}."
                    ) from exc
                logger.warning(
                    f"Permission error on {self.device_path}, using cpuidle sysfs fallback."
                )
                return self._lock_via_sysfs_fallback()

            if not self.enable_sysfs_fallback or not allow_fallback:
                raise PMQoSError(
                    f"Failed to write target latency to {self.device_path}: {exc}"
                ) from exc

            logger.warning(
                f"Error writing to {self.device_path} ({exc}), attempting cpuidle sysfs fallback."
            )
            return self._lock_via_sysfs_fallback()

    def select_planned_route(self) -> Tuple[Optional[str], List[Path]]:
        """Choose a preflightable DMA device route or a cpuidle fallback route."""
        if self.device_path.exists():
            return "device", [self.device_path.resolve()]
        fallback_paths = self.discover_fallback_paths()
        return ("cpuidle", fallback_paths) if fallback_paths else (None, [])

    def open_device_for_preflight(self) -> Any:
        """Open the DMA-latency device with write access without changing it."""
        flags = os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(str(self.device_path), flags)
        return os.fdopen(fd, "rb+", buffering=0)

    def discover_fallback_paths(self) -> List[Path]:
        """Return cpuidle disable nodes eligible for a planned fallback action."""
        paths: List[Path] = []
        if not self.enable_sysfs_fallback:
            return paths
        for cpu in self.sysfs.get_online_cpus():
            cpu_dir = self.sysfs._resolve_path(f"devices/system/cpu/cpu{cpu}/cpuidle")
            if not cpu_dir.is_dir():
                continue
            for state_entry in sorted(cpu_dir.iterdir()):
                if not state_entry.is_dir() or not state_entry.name.startswith("state"):
                    continue
                try:
                    state_idx = int(state_entry.name[5:])
                except ValueError:
                    continue
                if state_idx < self.min_fallback_state_index:
                    continue
                disable_path = state_entry / "disable"
                if self.sysfs._read_path(disable_path) is not None:
                    paths.append(disable_path.resolve())
        return paths

    def activate_planned_fallback(self, saved_states: Dict[Path, str]) -> None:
        """Record a fallback already applied by the shared transaction executor."""
        self._fallback_saved_states = {
            str(path.resolve().relative_to(self.sysfs_root)): value
            for path, value in saved_states.items()
        }
        self._is_locked = True
        self._using_fallback = True

    def _lock_via_sysfs_fallback(self) -> bool:
        """Fallback mechanism: disable deep C-states via sysfs cpuidle nodes."""
        self._fallback_saved_states.clear()

        try:
            cpus = self.sysfs.get_online_cpus()
            for cpu in cpus:
                cpu_dir = self.sysfs._resolve_path(f"devices/system/cpu/cpu{cpu}/cpuidle")
                if not cpu_dir.is_dir():
                    continue

                for state_entry in sorted(cpu_dir.iterdir()):
                    if not state_entry.is_dir() or not state_entry.name.startswith("state"):
                        continue

                    try:
                        state_idx = int(state_entry.name[5:])
                    except ValueError:
                        continue

                    if state_idx < self.min_fallback_state_index:
                        continue

                    disable_path = f"devices/system/cpu/cpu{cpu}/cpuidle/{state_entry.name}/disable"
                    current_val = self.sysfs._read_file(disable_path)
                    if current_val is not None:
                        self._fallback_saved_states[disable_path] = current_val
                        self.sysfs._write_file(disable_path, "1")

            self._is_locked = True
            self._using_fallback = True
            logger.info("PM QoS lock active via cpuidle sysfs fallback.")
            return True

        except (SysfsPermissionError, PermissionError) as exc:
            raise PMQoSPermissionError(
                "Permission denied disabling cpuidle states via sysfs. Run with root privileges."
            ) from exc
        except Exception as exc:
            self.unlock()
            raise PMQoSError(f"Failed to configure cpuidle sysfs fallback: {exc}") from exc

    def unlock(self) -> None:
        """Release PM QoS lock and restore any modified cpuidle states."""
        self._release(strict=False)

    def release_strict(self) -> None:
        """Release a transaction-owned PM QoS descriptor without hiding close errors."""
        self._release(strict=True)

    def _release(self, strict: bool) -> None:
        """Release PM QoS state, optionally preserving a descriptor close failure."""
        close_error: Optional[OSError] = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError as exc:
                if strict:
                    close_error = exc
                else:
                    logger.debug(f"Error closing PM QoS fd {self._fd}: {exc}")
            finally:
                self._fd = None

        if self._using_fallback:
            for subpath, original_val in self._fallback_saved_states.items():
                try:
                    self.sysfs._write_file(subpath, original_val, optional=True)
                except Exception as exc:
                    logger.warning(f"Failed to restore cpuidle state at {subpath}: {exc}")
            self._fallback_saved_states.clear()
            self._using_fallback = False

        self._is_locked = False
        logger.info("PM QoS DMA latency lock released.")
        if close_error is not None:
            raise close_error

    def release(self) -> None:
        """Alias for unlock()."""
        self.unlock()

    def get_current_latency(self) -> Optional[int]:
        """Read current latency value from device node if supported, or return None."""
        if not self.device_path.exists():
            return None
        try:
            with open(self.device_path, "rb") as f:
                data = f.read(4)
                if len(data) == 4:
                    return struct.unpack("i", data)[0]
        except Exception as exc:
            logger.debug(f"Unable to read latency from {self.device_path}: {exc}")
        return None

    def __enter__(self) -> PMQoSController:
        self.lock()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.unlock()
