"""
State rollback and restore management for BoostLock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from boostlock.state import (
    CPUStateSnapshot,
    StateError,
    StateRestoreError,
    StateSnapshotError,
    StateSnapshotManager,
    SystemStateSnapshot,
    resolve_default_snapshot_path,
)
from boostlock.sysfs import SysfsController


class RollbackManager(StateSnapshotManager):
    """Alias for StateSnapshotManager for semantic rollback management."""
    pass


def create_state_snapshot(
    sysfs_controller: Optional[SysfsController] = None,
    snapshot_file: Optional[Union[str, Path]] = None,
) -> SystemStateSnapshot:
    """Convenience function to capture and persist current system CPU state."""
    manager = StateSnapshotManager(
        sysfs_controller=sysfs_controller,
        snapshot_file=snapshot_file,
    )
    return manager.create_snapshot()


def restore_system_state(
    sysfs_controller: Optional[SysfsController] = None,
    snapshot_file: Optional[Union[str, Path]] = None,
    delete_snapshot_file: bool = True,
) -> bool:
    """Convenience function to restore system state from snapshot or fallback."""
    manager = StateSnapshotManager(
        sysfs_controller=sysfs_controller,
        snapshot_file=snapshot_file,
    )
    return manager.restore(delete_snapshot_file=delete_snapshot_file)


__all__ = [
    "RollbackManager",
    "StateSnapshotManager",
    "SystemStateSnapshot",
    "CPUStateSnapshot",
    "StateError",
    "StateSnapshotError",
    "StateRestoreError",
    "create_state_snapshot",
    "restore_system_state",
    "resolve_default_snapshot_path",
]
