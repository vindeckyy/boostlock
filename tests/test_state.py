"""
Tests for system state snapshot capture, persistence, and safe rollback manager.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.fixture
def mock_sysfs(tmp_path: Path) -> Path:
    """Create a fully populated mock sysfs tree with 2 CPUs."""
    sysfs = tmp_path / "sys"
    for cpu_id in (0, 1):
        cpu_dir = sysfs / "devices" / "system" / "cpu" / f"cpu{cpu_id}"
        cpufreq_dir = cpu_dir / "cpufreq"
        cpufreq_dir.mkdir(parents=True, exist_ok=True)
        (cpufreq_dir / "scaling_governor").write_text("powersave\n")
        (cpufreq_dir / "scaling_min_freq").write_text("1400000\n")
        (cpufreq_dir / "scaling_max_freq").write_text("3000000\n")
        (cpufreq_dir / "scaling_cur_freq").write_text("1400000\n")
        (cpufreq_dir / "cpuinfo_min_freq").write_text("1400000\n")
        (cpufreq_dir / "cpuinfo_max_freq").write_text("3000000\n")
        (cpufreq_dir / "scaling_available_governors").write_text("powersave performance schedutil ondemand\n")
        (cpufreq_dir / "cpb").write_text("0\n")
        (cpufreq_dir / "energy_performance_preference").write_text("balance_power\n")

        power_dir = cpu_dir / "power"
        power_dir.mkdir(parents=True, exist_ok=True)
        (power_dir / "energy_perf_bias").write_text("6\n")

        # cpuidle states
        for state_id in range(3):
            state_dir = cpu_dir / "cpuidle" / f"state{state_id}"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "disable").write_text("0\n")

    online_file = sysfs / "devices" / "system" / "cpu" / "online"
    online_file.parent.mkdir(parents=True, exist_ok=True)
    online_file.write_text("0-1\n")

    global_cpufreq = sysfs / "devices" / "system" / "cpu" / "cpufreq"
    global_cpufreq.mkdir(parents=True, exist_ok=True)
    (global_cpufreq / "boost").write_text("0\n")

    return sysfs


class TestSystemStateSnapshot:
    """Test suite for SystemStateSnapshot data structures."""

    def test_cpu_state_snapshot_dict_roundtrip(self) -> None:
        """Test CPUStateSnapshot serialization to/from dict."""
        cpu_snap = CPUStateSnapshot(
            cpu_id=0,
            governor="powersave",
            scaling_min_freq=1400000,
            scaling_max_freq=3000000,
            epp="balance_power",
            epb=6,
            cpb=False,
            cpuidle_states={"state0": 0, "state1": 0, "state2": 1},
        )

        d = cpu_snap.to_dict()
        reconstructed = CPUStateSnapshot.from_dict(d)
        assert reconstructed == cpu_snap
        assert reconstructed.cpu_id == 0
        assert reconstructed.governor == "powersave"
        assert reconstructed.cpuidle_states["state2"] == 1

    def test_system_state_snapshot_json_roundtrip(self, tmp_path: Path) -> None:
        """Test SystemStateSnapshot serialization to/from JSON file."""
        cpu0 = CPUStateSnapshot(
            cpu_id=0,
            governor="powersave",
            scaling_min_freq=1400000,
            scaling_max_freq=3000000,
            epp="default",
            epb=0,
            cpb=True,
            cpuidle_states={"state0": 0, "state1": 0},
        )
        sys_snap = SystemStateSnapshot(
            global_boost=False,
            cpus={0: cpu0},
            timestamp=123456789.0,
            hostname="testhost",
        )

        json_str = sys_snap.to_json()
        reconstructed = SystemStateSnapshot.from_json(json_str)
        assert reconstructed.global_boost is False
        assert reconstructed.hostname == "testhost"
        assert 0 in reconstructed.cpus
        assert reconstructed.cpus[0].governor == "powersave"

        # File save & load
        snap_file = tmp_path / "snapshot.json"
        sys_snap.save(snap_file)
        loaded = SystemStateSnapshot.load(snap_file)
        assert loaded.global_boost == sys_snap.global_boost
        assert loaded.cpus[0].scaling_min_freq == 1400000

    def test_system_state_snapshot_save_error(self, tmp_path: Path) -> None:
        """Test error handling when snapshot file cannot be saved."""
        sys_snap = SystemStateSnapshot()
        snap_file = tmp_path / "snapshot.json"

        with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
            with pytest.raises(StateSnapshotError, match="Failed to write snapshot"):
                sys_snap.save(snap_file)

    def test_system_state_snapshot_load_non_existent(self, tmp_path: Path) -> None:
        """Test loading non-existent snapshot file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Snapshot file not found"):
            SystemStateSnapshot.load(tmp_path / "missing.json")

    def test_resolve_default_snapshot_path(self, tmp_path: Path) -> None:
        """Test resolve_default_snapshot_path with both normal and fallback permissions."""
        with patch("boostlock.state.DEFAULT_SNAPSHOT_DIR", tmp_path / "run_test"):
            p = resolve_default_snapshot_path()
            assert "snapshot.json" in str(p)

        with patch("boostlock.state.DEFAULT_SNAPSHOT_DIR", Path("/root/forbidden_dir")):
            p_fallback = resolve_default_snapshot_path()
            assert "snapshot.json" in str(p_fallback)


class TestStateSnapshotManager:
    """Test suite for StateSnapshotManager."""

    def test_create_snapshot(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test capturing full state snapshot from mock sysfs."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        snapshot = manager.create_snapshot()
        assert snapshot is not None
        assert snapshot.global_boost is False
        assert len(snapshot.cpus) == 2
        assert snapshot.cpus[0].governor == "powersave"
        assert snapshot.cpus[0].scaling_min_freq == 1400000
        assert snapshot.cpus[0].scaling_max_freq == 3000000
        assert snapshot.cpus[0].epp == "balance_power"
        assert snapshot.cpus[0].epb == 6
        assert snapshot.cpus[0].cpb is False
        assert snapshot.cpus[0].cpuidle_states == {"state0": 0, "state1": 0, "state2": 0}

        # Snapshot file should have been written
        assert snap_file.exists()
        loaded = SystemStateSnapshot.load(snap_file)
        assert loaded.cpus[0].governor == "powersave"

    def test_create_snapshot_default_path(self, mock_sysfs: Path) -> None:
        """Test creating snapshot with default resolved path."""
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl)
        assert manager.snapshot_file is not None

    def test_create_snapshot_save_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test that snapshot save failure logs warning and does not crash."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        with patch.object(SystemStateSnapshot, "save", side_effect=StateSnapshotError("Save failed")):
            snap = manager.create_snapshot()
            assert snap is not None
            assert manager.snapshot is not None

    def test_create_snapshot_corrupt_cpuidle_disable(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test handling unparseable cpuidle disable value."""
        (mock_sysfs / "devices" / "system" / "cpu" / "cpu0" / "cpuidle" / "state0" / "disable").write_text("invalid\n")
        (mock_sysfs / "devices" / "system" / "cpu" / "cpu0" / "cpuidle" / "non_state").mkdir(exist_ok=True)

        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=tmp_path / "snap.json")
        snap = manager.create_snapshot()
        assert snap.cpus[0].cpuidle_states["state0"] == 0

    def test_restore_system_state(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test modifying sysfs and restoring state back to exact values."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        # 1. Take snapshot of initial state (powersave, 1400000, 3000000, boost=0, cpb=0)
        manager.create_snapshot()

        # 2. Modify system state (simulate boostlock activation)
        sysfs_ctrl.set_scaling_governor("performance")
        sysfs_ctrl.set_scaling_max_freq(4000000)
        sysfs_ctrl.set_scaling_min_freq(3000000)
        sysfs_ctrl.set_boost(True)
        sysfs_ctrl.set_cpb(True)
        sysfs_ctrl.set_energy_performance_preference("performance")
        sysfs_ctrl.set_energy_perf_bias(0)

        # Disable cpuidle states
        for cpu in (0, 1):
            for st in (1, 2):
                (mock_sysfs / "devices" / "system" / "cpu" / f"cpu{cpu}" / "cpuidle" / f"state{st}" / "disable").write_text("1\n")

        # Verify modifications took place
        assert sysfs_ctrl.get_scaling_governor(0) == "performance"
        assert sysfs_ctrl.get_scaling_min_freq(0) == 3000000
        assert sysfs_ctrl.get_boost() is True
        assert sysfs_ctrl.get_cpb(0) is True

        # 3. Restore state
        manager.restore(delete_snapshot_file=True)

        # 4. Verify 100% restored
        for cpu in (0, 1):
            assert sysfs_ctrl.get_scaling_governor(cpu) == "powersave"
            assert sysfs_ctrl.get_scaling_min_freq(cpu) == 1400000
            assert sysfs_ctrl.get_scaling_max_freq(cpu) == 3000000
            assert sysfs_ctrl.get_energy_performance_preference(cpu) == "balance_power"
            assert sysfs_ctrl.get_energy_perf_bias(cpu) == 6
            assert sysfs_ctrl.get_cpb(cpu) is False
            for st in (0, 1, 2):
                dis = (mock_sysfs / "devices" / "system" / "cpu" / f"cpu{cpu}" / "cpuidle" / f"state{st}" / "disable").read_text().strip()
                assert dis == "0"

        assert sysfs_ctrl.get_boost() is False
        assert not snap_file.exists()

    def test_restore_frequency_ordering(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """
        Test that restoring frequencies sets max before min or min before max appropriately
        to avoid kernel validation failures (min > max).
        """
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        # Case 1: snapshot min > current max (expand max first)
        manager.create_snapshot()
        sysfs_ctrl.set_scaling_max_freq(1000000)
        sysfs_ctrl.set_scaling_min_freq(1000000)
        manager.restore()
        assert sysfs_ctrl.get_scaling_min_freq(0) == 1400000
        assert sysfs_ctrl.get_scaling_max_freq(0) == 3000000

        # Case 2: snapshot max < current min (lower min first)
        manager.create_snapshot()
        sysfs_ctrl.set_scaling_max_freq(4000000)
        sysfs_ctrl.set_scaling_min_freq(3500000)
        manager.restore()
        assert sysfs_ctrl.get_scaling_min_freq(0) == 1400000
        assert sysfs_ctrl.get_scaling_max_freq(0) == 3000000

    def test_restore_partial_snapshot_attributes(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test restore when snapshot has only min or only max freq, or missing epp/epb/cpb."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        cpu_snap_min_only = CPUStateSnapshot(cpu_id=0, scaling_min_freq=1500000)
        cpu_snap_max_only = CPUStateSnapshot(cpu_id=1, scaling_max_freq=2500000)
        snap = SystemStateSnapshot(cpus={0: cpu_snap_min_only, 1: cpu_snap_max_only})

        manager.restore(snapshot=snap)
        assert sysfs_ctrl.get_scaling_min_freq(0) == 1500000
        assert sysfs_ctrl.get_scaling_max_freq(1) == 2500000

    def test_restore_boost_exception_handling(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test handling of exception during boost restoration."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()

        with patch.object(sysfs_ctrl, "set_boost", side_effect=OSError("Boost write error")):
            # Should record error and continue
            assert manager.restore() is True

    def test_restore_freq_exception_handling(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test handling of exception during frequency restoration."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()

        with patch.object(sysfs_ctrl, "get_scaling_min_freq", side_effect=OSError("Read error")):
            assert manager.restore() is True

    def test_restore_epp_epb_cpb_idle_exceptions(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test non-critical exceptions during EPP, EPB, CPB, and cpuidle restoration."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()

        with patch.object(sysfs_ctrl, "set_energy_performance_preference", side_effect=OSError("EPP error")), \
             patch.object(sysfs_ctrl, "set_energy_perf_bias", side_effect=OSError("EPB error")), \
             patch.object(sysfs_ctrl, "set_cpb", side_effect=OSError("CPB error")), \
             patch.object(sysfs_ctrl, "_write_file", side_effect=OSError("cpuidle error")):
            assert manager.restore() is True

    def test_restore_snapshot_unlink_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test exception when deleting snapshot file during restore."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()

        with patch.object(Path, "unlink", side_effect=PermissionError("Unlink error")):
            assert manager.restore(delete_snapshot_file=True) is True

    def test_restore_with_missing_snapshot_fallback(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test that restore falls back safely if no snapshot file or in-memory snapshot exists."""
        snap_file = tmp_path / "non_existent_snapshot.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        # Set to performance
        sysfs_ctrl.set_scaling_governor("performance")
        sysfs_ctrl.set_scaling_min_freq(3000000)

        # Restore with fallback
        manager.restore()

        # Governor should be set to available fallback (e.g. schedutil or ondemand or powersave)
        assert sysfs_ctrl.get_scaling_governor(0) in ("schedutil", "ondemand", "powersave")
        assert sysfs_ctrl.get_scaling_min_freq(0) == 1400000

    def test_restore_with_corrupted_snapshot_file(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test restore when snapshot file contains corrupted JSON."""
        snap_file = tmp_path / "corrupt_snapshot.json"
        snap_file.write_text("INVALID JSON CONTENT { [")

        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        sysfs_ctrl.set_scaling_governor("performance")

        # restore should catch JSONDecodeError, log warning, and execute fallback
        manager.restore()
        assert sysfs_ctrl.get_scaling_governor(0) in ("schedutil", "ondemand", "powersave")

    def test_restore_fallback_governor_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test fallback restoration when governor setting raises exception."""
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=tmp_path / "none.json")

        with patch.object(sysfs_ctrl, "set_scaling_governor", side_effect=OSError("Gov write error")):
            assert manager.restore_fallback() is True

        with patch.object(sysfs_ctrl, "set_scaling_min_freq", side_effect=OSError("Min freq error")), \
             patch.object(sysfs_ctrl, "set_scaling_max_freq", side_effect=OSError("Max freq error")), \
             patch.object(sysfs_ctrl, "set_boost", side_effect=OSError("Boost error")):
            assert manager.restore_fallback() is True

    def test_context_manager_rollback_on_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test that StateSnapshotManager as context manager restores state if exception occurs."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            with manager:
                sysfs_ctrl.set_scaling_governor("performance")
                assert sysfs_ctrl.get_scaling_governor(0) == "performance"
                raise RuntimeError("Simulated crash")

        # State should be rolled back
        assert sysfs_ctrl.get_scaling_governor(0) == "powersave"

    def test_context_manager_clean_exit(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test context manager with restore_on_exit=True."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(
            sysfs_controller=sysfs_ctrl,
            snapshot_file=snap_file,
            restore_on_exit=True,
        )

        with manager:
            sysfs_ctrl.set_scaling_governor("performance")

        # Restored on exit
        assert sysfs_ctrl.get_scaling_governor(0) == "powersave"

    def test_signal_handlers_registration_and_trigger(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test registering signal handlers and triggering rollback on signal."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()
        sysfs_ctrl.set_scaling_governor("performance")

        # Register signal handlers
        manager.register_signal_handlers()
        # Double registration should be idempotent
        manager.register_signal_handlers()

        # Trigger mock signal handler directly
        assert manager._signal_handler is not None
        with pytest.raises(SystemExit):
            manager._signal_handler(signal.SIGINT, None)

        # Check state was restored
        assert sysfs_ctrl.get_scaling_governor(0) == "powersave"

        # Cleanup signal handlers
        manager.unregister_signal_handlers()
        # Double unregister should be safe
        manager.unregister_signal_handlers()

    def test_signal_handlers_with_custom_previous_handler(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test that custom previous signal handler is invoked after rollback."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        custom_handler = MagicMock()
        manager._previous_signal_handlers[signal.SIGTERM] = custom_handler

        manager._signal_handler(signal.SIGTERM, None)
        custom_handler.assert_called_once_with(signal.SIGTERM, None)

    def test_signal_handler_registration_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test handling of exception when registering signal handlers."""
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=tmp_path / "snap.json")

        with patch("signal.signal", side_effect=ValueError("signal only works in main thread")):
            manager.register_signal_handlers()
            assert manager._signal_handlers_registered is True
            manager.unregister_signal_handlers()

    def test_signal_handler_unregistration_exception(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test handling of exception when unregistering signal handlers."""
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=tmp_path / "snap.json")
        manager.register_signal_handlers()

        with patch("signal.signal", side_effect=ValueError("signal error")):
            manager.unregister_signal_handlers()
            assert manager._signal_handlers_registered is False

    def test_atexit_handler(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test atexit handler execution."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()
        sysfs_ctrl.set_scaling_governor("performance")

        manager.register_atexit()
        # Double register should be idempotent
        manager.register_atexit()

        # Call atexit callback manually
        manager._atexit_callback()

        assert sysfs_ctrl.get_scaling_governor(0) == "powersave"
        manager.unregister_atexit()
        manager.unregister_atexit()

    def test_restore_resilient_to_individual_failures(self, mock_sysfs: Path, tmp_path: Path) -> None:
        """Test that failure on restoring one attribute/CPU doesn't halt restoration of others."""
        snap_file = tmp_path / "test_snap.json"
        sysfs_ctrl = SysfsController(sysfs_root=mock_sysfs)
        manager = StateSnapshotManager(sysfs_controller=sysfs_ctrl, snapshot_file=snap_file)

        manager.create_snapshot()

        # Mock _write_file to fail for cpu0 governor only
        original_write = sysfs_ctrl._write_file

        def failing_write(subpath: str, value: str, optional: bool = False) -> bool:
            if "cpu0/cpufreq/scaling_governor" in subpath:
                raise PermissionError("Mock failure on cpu0")
            return original_write(subpath, value, optional)

        with patch.object(sysfs_ctrl, "_write_file", side_effect=failing_write):
            # Should not raise uncaught error, but restore what it can
            manager.restore()

        # cpu1 governor should still have been restored
        assert sysfs_ctrl.get_scaling_governor(1) == "powersave"

    def test_save_snapshot_unlink_temp_file_on_replace_error(self, tmp_path: Path) -> None:
        """Test that temporary snapshot file is unlinked if replace raises an error."""
        snap = SystemStateSnapshot(
            timestamp=12345.0,
            cpus={},
        )
        target_path = tmp_path / "final_snapshot.json"
        with patch.object(Path, "replace", side_effect=OSError("Replace failed")):
            with pytest.raises(StateSnapshotError, match="Failed to write snapshot"):
                snap.save(target_path)
        assert not (tmp_path / "final_snapshot.tmp").exists()
