"""
Tests for rollback module compatibility and convenience functions.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from boostlock.rollback import (
    RollbackManager,
    StateSnapshotManager,
    SystemStateSnapshot,
    restore_system_state,
    create_state_snapshot,
)
from boostlock.sysfs import SysfsController


@pytest.fixture
def mock_sysfs(tmp_path: Path) -> Path:
    """Create a minimal mock sysfs tree."""
    sysfs = tmp_path / "sys"
    for cpu_id in (0, 1):
        cpu_dir = sysfs / "devices" / "system" / "cpu" / f"cpu{cpu_id}"
        cpufreq_dir = cpu_dir / "cpufreq"
        cpufreq_dir.mkdir(parents=True, exist_ok=True)
        (cpufreq_dir / "scaling_governor").write_text("schedutil\n")
        (cpufreq_dir / "scaling_min_freq").write_text("1400000\n")
        (cpufreq_dir / "scaling_max_freq").write_text("3000000\n")
        (cpufreq_dir / "scaling_cur_freq").write_text("1400000\n")
        (cpufreq_dir / "cpuinfo_min_freq").write_text("1400000\n")
        (cpufreq_dir / "cpuinfo_max_freq").write_text("3000000\n")
        (cpufreq_dir / "cpb").write_text("1\n")

    online_file = sysfs / "devices" / "system" / "cpu" / "online"
    online_file.parent.mkdir(parents=True, exist_ok=True)
    online_file.write_text("0-1\n")

    global_cpufreq = sysfs / "devices" / "system" / "cpu" / "cpufreq"
    global_cpufreq.mkdir(parents=True, exist_ok=True)
    (global_cpufreq / "boost").write_text("1\n")

    return sysfs


def test_rollback_manager_alias() -> None:
    """Test that RollbackManager is an alias or subclass of StateSnapshotManager."""
    assert issubclass(RollbackManager, StateSnapshotManager)


def test_convenience_functions(mock_sysfs: Path, tmp_path: Path) -> None:
    """Test create_state_snapshot and restore_system_state helper functions."""
    snap_file = tmp_path / "convenience_snap.json"
    sysfs = SysfsController(sysfs_root=mock_sysfs)

    snap = create_state_snapshot(sysfs_controller=sysfs, snapshot_file=snap_file)
    assert isinstance(snap, SystemStateSnapshot)
    assert snap.cpus[0].governor == "schedutil"
    assert snap_file.exists()

    sysfs.set_scaling_governor("performance")
    assert sysfs.get_scaling_governor(0) == "performance"

    # Restore using convenience function
    restored = restore_system_state(sysfs_controller=sysfs, snapshot_file=snap_file)
    assert restored is True
    assert sysfs.get_scaling_governor(0) == "schedutil"
