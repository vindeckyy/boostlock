"""
Tests for PM QoS /dev/cpu_dma_latency DMA latency lock and sysfs cpuidle fallback.
"""

from __future__ import annotations

import errno
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from boostlock.pm_qos import (
    PMQoSController,
    PMQoSError,
    PMQoSLockError,
    PMQoSNotFoundError,
    PMQoSPermissionError,
)
from boostlock.sysfs import SysfsController, SysfsPermissionError


@pytest.fixture
def mock_sysfs_tree(tmp_path: Path) -> Path:
    """Create a mock sysfs hierarchy with CPU cpufreq and cpuidle nodes."""
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

        # cpuidle states
        for state_id in range(3):
            state_dir = cpu_dir / "cpuidle" / f"state{state_id}"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "name").write_text(f"C{state_id}\n")
            (state_dir / "latency").write_text(f"{state_id * 10}\n")
            (state_dir / "disable").write_text("0\n")

    online_file = sysfs / "devices" / "system" / "cpu" / "online"
    online_file.parent.mkdir(parents=True, exist_ok=True)
    online_file.write_text("0-1\n")

    global_cpufreq = sysfs / "devices" / "system" / "cpu" / "cpufreq"
    global_cpufreq.mkdir(parents=True, exist_ok=True)
    (global_cpufreq / "boost").write_text("1\n")

    return sysfs


class TestPMQoSController:
    """Test suite for PMQoSController."""

    def test_lock_and_unlock_with_file(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test locking PM QoS using a mock /dev/cpu_dma_latency file."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=0,
        )

        assert not controller.is_locked
        assert not controller.using_fallback
        assert controller.fd is None

        # Lock
        success = controller.lock()
        assert success is True
        assert controller.is_locked
        assert not controller.using_fallback
        assert controller.target_latency_us == 0
        assert controller.fd is not None

        # Verify 4-byte int 0 was written
        data = dev_node.read_bytes()
        assert len(data) == 4
        assert struct.unpack("i", data)[0] == 0

        # Unlock
        controller.unlock()
        assert not controller.is_locked
        assert controller.fd is None

    def test_lock_custom_target_latency(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test locking with custom latency value."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=100,
        )

        controller.lock()
        data = dev_node.read_bytes()
        assert struct.unpack("i", data)[0] == 100
        controller.unlock()

    def test_update_locked_latency(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test updating target latency when already locked."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=0,
        )
        controller.lock()
        assert controller.target_latency_us == 0

        # Update while locked
        controller.lock(target_latency_us=250)
        assert controller.target_latency_us == 250
        data = dev_node.read_bytes()
        assert len(data) == 4
        assert struct.unpack("i", data)[0] == 250

        controller.unlock()

    def test_update_locked_latency_oserror(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test OSError during latency update on already open fd."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=0,
        )
        controller.lock()

        with patch("os.write", side_effect=OSError("Disk error")):
            # Should log warning and still return True
            assert controller.lock(target_latency_us=150) is True

        controller.unlock()

    def test_context_manager(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test PMQoSController as a context manager."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=0,
        )

        with controller as qos:
            assert qos.is_locked
            assert struct.unpack("i", dev_node.read_bytes())[0] == 0

        assert not controller.is_locked

    def test_double_lock_and_unlock_idempotent(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test locking when already locked and unlocking when unlocked."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            target_latency_us=0,
        )

        controller.lock()
        assert controller.is_locked
        controller.lock()
        assert controller.is_locked

        controller.release()
        controller.unlock()
        assert not controller.is_locked

    def test_invalid_latency_value(self, tmp_path: Path) -> None:
        """Test validation of latency values."""
        dev_node = tmp_path / "cpu_dma_latency"
        with pytest.raises(ValueError, match="target_latency_us must be non-negative"):
            PMQoSController(device_path=dev_node, target_latency_us=-1)

        with pytest.raises(ValueError, match="exceeds maximum 32-bit integer"):
            PMQoSController(device_path=dev_node, target_latency_us=2**31)

        controller = PMQoSController(device_path=dev_node, target_latency_us=10)
        with pytest.raises(ValueError):
            controller.lock(target_latency_us=-5)

    def test_permission_error_on_device(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test permission error handling on opening device node when fallback is disabled."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        with patch("os.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PMQoSPermissionError, match="Permission denied"):
                controller.lock()

    def test_missing_device_node_no_fallback_raises(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test missing device node with fallback disabled raises PMQoSNotFoundError."""
        dev_node = tmp_path / "non_existent_device"
        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        with pytest.raises(PMQoSNotFoundError, match="PM QoS device node not found"):
            controller.lock()

    def test_oserror_eacces_handling(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test OSError with EACCES errno when opening device node."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        err = OSError("Permission error")
        err.errno = errno.EACCES

        # Fallback disabled -> raises PMQoSPermissionError
        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )
        with patch("os.open", side_effect=err):
            with pytest.raises(PMQoSPermissionError):
                controller.lock()

        # Fallback enabled -> succeeds via fallback
        controller_fb = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )
        with patch("os.open", side_effect=err):
            assert controller_fb.lock() is True
            assert controller_fb.using_fallback is True
            controller_fb.unlock()

    def test_oserror_generic_handling(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test generic OSError when opening device node."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        err = OSError(errno.EIO, "I/O error")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )
        with patch("os.open", side_effect=err):
            with pytest.raises(PMQoSError, match="Failed to write target latency"):
                controller.lock()

        controller_fb = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )
        with patch("os.open", side_effect=err):
            assert controller_fb.lock() is True
            assert controller_fb.using_fallback is True
            controller_fb.unlock()

    def test_sysfs_cpuidle_fallback_activation(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test that missing /dev/cpu_dma_latency falls back to sysfs cpuidle disable."""
        non_existent_dev = tmp_path / "non_existent_dma_latency"
        controller = PMQoSController(
            device_path=non_existent_dev,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
            min_fallback_state_index=1,
        )

        assert not controller.is_locked
        success = controller.lock()
        assert success is True
        assert controller.is_locked
        assert controller.using_fallback is True

        # Check that state1 and state2 are disabled (disable = '1') and state0 remains '0'
        for cpu_id in (0, 1):
            s0 = (mock_sysfs_tree / "devices" / "system" / "cpu" / f"cpu{cpu_id}" / "cpuidle" / "state0" / "disable").read_text().strip()
            s1 = (mock_sysfs_tree / "devices" / "system" / "cpu" / f"cpu{cpu_id}" / "cpuidle" / "state1" / "disable").read_text().strip()
            s2 = (mock_sysfs_tree / "devices" / "system" / "cpu" / f"cpu{cpu_id}" / "cpuidle" / "state2" / "disable").read_text().strip()
            assert s0 == "0"
            assert s1 == "1"
            assert s2 == "1"

        # Unlock and verify restored to '0'
        controller.unlock()
        assert not controller.is_locked
        for cpu_id in (0, 1):
            s1 = (mock_sysfs_tree / "devices" / "system" / "cpu" / f"cpu{cpu_id}" / "cpuidle" / "state1" / "disable").read_text().strip()
            s2 = (mock_sysfs_tree / "devices" / "system" / "cpu" / f"cpu{cpu_id}" / "cpuidle" / "state2" / "disable").read_text().strip()
            assert s1 == "0"
            assert s2 == "0"

    def test_sysfs_cpuidle_fallback_skips_invalid_entries(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test fallback skips non-state directories or unparseable names."""
        cpu0_idle = mock_sysfs_tree / "devices" / "system" / "cpu" / "cpu0" / "cpuidle"
        (cpu0_idle / "driver").mkdir(exist_ok=True)
        (cpu0_idle / "state_invalid").mkdir(exist_ok=True)
        (cpu0_idle / "dummy_file.txt").write_text("test")

        non_existent_dev = tmp_path / "non_existent_dma_latency"
        controller = PMQoSController(
            device_path=non_existent_dev,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
            min_fallback_state_index=1,
        )
        assert controller.lock() is True
        controller.unlock()

    def test_discover_planned_fallback_paths_excludes_invalid_and_shallow_entries(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test planned fallback discovery returns only eligible disable nodes."""
        cpu0_idle = mock_sysfs_tree / "devices" / "system" / "cpu" / "cpu0" / "cpuidle"
        (cpu0_idle / "metadata").mkdir()
        (cpu0_idle / "stateinvalid").mkdir()

        controller = PMQoSController(
            device_path=tmp_path / "missing_dma",
            sysfs_root=mock_sysfs_tree,
            min_fallback_state_index=1,
        )

        paths = controller.discover_fallback_paths()

        assert paths == [
            mock_sysfs_tree / "devices/system/cpu/cpu0/cpuidle/state1/disable",
            mock_sysfs_tree / "devices/system/cpu/cpu0/cpuidle/state2/disable",
            mock_sysfs_tree / "devices/system/cpu/cpu1/cpuidle/state1/disable",
            mock_sysfs_tree / "devices/system/cpu/cpu1/cpuidle/state2/disable",
        ]

    def test_discover_planned_fallback_paths_returns_empty_when_disabled(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test planned fallback discovery honors the fallback setting."""
        controller = PMQoSController(
            device_path=tmp_path / "missing_dma",
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        assert controller.discover_fallback_paths() == []

    def test_activate_planned_fallback_records_states_for_unlock(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test a transaction-applied fallback is restored through the controller."""
        disable_path = (
            mock_sysfs_tree
            / "devices/system/cpu/cpu0/cpuidle/state1/disable"
        )
        disable_path.write_text("1\n")
        controller = PMQoSController(
            device_path=tmp_path / "missing_dma",
            sysfs_root=mock_sysfs_tree,
        )

        controller.activate_planned_fallback({disable_path: "0"})

        assert controller.is_locked
        assert controller.using_fallback
        controller.unlock()
        assert disable_path.read_text() == "0\n"

    def test_sysfs_cpuidle_fallback_permission_error(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test fallback failure when sysfs cpuidle write fails with permission error."""
        non_existent_dev = tmp_path / "non_existent_dma_latency"
        controller = PMQoSController(
            device_path=non_existent_dev,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )

        with patch.object(controller.sysfs, "_write_file", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PMQoSPermissionError):
                controller.lock()

        with patch.object(controller.sysfs, "_write_file", side_effect=SysfsPermissionError("Sysfs denied")):
            with pytest.raises(PMQoSPermissionError):
                controller.lock()

    def test_sysfs_cpuidle_fallback_unexpected_exception(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test fallback failure on unexpected exception."""
        non_existent_dev = tmp_path / "non_existent_dma_latency"
        controller = PMQoSController(
            device_path=non_existent_dev,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )

        with patch.object(controller.sysfs, "get_online_cpus", side_effect=RuntimeError("Unexpected")):
            with pytest.raises(PMQoSError, match="Failed to configure cpuidle sysfs fallback"):
                controller.lock()

    def test_unlock_oserror_closing_fd(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test handling of OSError when closing fd."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
        )
        controller.lock()

        with patch("os.close", side_effect=OSError("Bad file descriptor")):
            # Should not raise exception
            controller.unlock()
            assert controller.fd is None

    def test_unlock_fallback_restore_exception(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test handling of exception when restoring fallback cpuidle states."""
        non_existent_dev = tmp_path / "non_existent_dma_latency"
        controller = PMQoSController(
            device_path=non_existent_dev,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )
        controller.lock()

        with patch.object(controller.sysfs, "_write_file", side_effect=OSError("Write failed")):
            # unlock should log warning and finish cleanly
            controller.unlock()
            assert not controller.is_locked

    def test_write_os_error_handling(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test handling of OSError during os.write to device node."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        with patch("os.write", side_effect=OSError(22, "Invalid argument")):
            with pytest.raises(PMQoSError, match="Failed to write target latency"):
                controller.lock()

    def test_write_failure_after_open_closes_unassigned_descriptor(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test a failed initial DMA write closes its local descriptor."""
        controller = PMQoSController(
            device_path=tmp_path / "cpu_dma_latency",
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        with patch("os.open", return_value=73), \
             patch("os.write", side_effect=OSError("write failed")), \
             patch("os.close") as close:
            with pytest.raises(PMQoSError, match="write failed"):
                controller.lock(allow_fallback=False)

        close.assert_called_once_with(73)
        assert controller.fd is None
        assert not controller.is_locked

    def test_write_failure_reports_local_descriptor_close_error(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test a post-open write failure retains its descriptor close failure."""
        controller = PMQoSController(
            device_path=tmp_path / "cpu_dma_latency",
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=False,
        )

        with patch("os.open", return_value=73), \
             patch("os.write", side_effect=OSError("write failed")), \
             patch("os.close", side_effect=OSError("close failed")):
            with pytest.raises(PMQoSError, match="write failed.*close failed"):
                controller.lock(allow_fallback=False)

        assert controller.fd is None
        assert not controller.is_locked

    def test_strict_release_propagates_descriptor_close_error(
        self,
        tmp_path: Path,
        mock_sysfs_tree: Path,
    ) -> None:
        """Test transaction release exposes a descriptor close error."""
        controller = PMQoSController(
            device_path=tmp_path / "cpu_dma_latency",
            sysfs_root=mock_sysfs_tree,
        )
        controller._fd = 73
        controller._is_locked = True

        with patch("os.close", side_effect=OSError("close failed")):
            with pytest.raises(OSError, match="close failed"):
                controller.release_strict()

        assert controller.fd is None
        assert not controller.is_locked

    def test_get_current_latency_from_file(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test reading current latency from device node."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(struct.pack("i", 2000))

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
        )
        assert controller.get_current_latency() == 2000

    def test_get_current_latency_when_missing(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test reading latency when file is missing returns None."""
        dev_node = tmp_path / "missing_latency"
        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
        )
        assert controller.get_current_latency() is None

    def test_get_current_latency_on_read_exception(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test reading latency when file read raises exception returns None."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
        )
        with patch("builtins.open", side_effect=PermissionError("Denied")):
            assert controller.get_current_latency() is None

    def test_lock_already_locked_lseek_oserror(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test updating latency on already-locked controller when lseek raises OSError."""
        dev_node = tmp_path / "cpu_dma_latency"
        dev_node.write_bytes(b"\x00\x00\x00\x00")

        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
        )
        controller.lock()

        with patch("os.lseek", side_effect=OSError(errno.ESPIPE, "Illegal seek")):
            # Update target latency
            success = controller.lock(target_latency_us=50)
            assert success is True
            assert controller.target_latency_us == 50
        controller.unlock()

    def test_lock_permission_error_with_sysfs_fallback(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test PermissionError on device open triggering sysfs fallback when enabled."""
        dev_node = tmp_path / "cpu_dma_latency"
        controller = PMQoSController(
            device_path=dev_node,
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )
        with patch("os.open", side_effect=PermissionError("Access denied")):
            success = controller.lock()
            assert success is True
            assert controller.is_locked
            assert controller.using_fallback
        controller.unlock()

    def test_lock_sysfs_fallback_missing_cpuidle_dir(self, tmp_path: Path) -> None:
        """Test fallback when cpuidle directory does not exist for a CPU."""
        sysfs = tmp_path / "sys"
        cpu_dir = sysfs / "devices" / "system" / "cpu" / "cpu0"
        (cpu_dir / "cpufreq").mkdir(parents=True, exist_ok=True)
        (cpu_dir / "cpufreq" / "scaling_governor").write_text("powersave\n")
        (sysfs / "devices" / "system" / "cpu" / "online").write_text("0\n")

        controller = PMQoSController(
            device_path=tmp_path / "missing_dev",
            sysfs_root=sysfs,
            enable_sysfs_fallback=True,
        )
        controller.lock()
        assert controller.is_locked
        assert controller.using_fallback
        controller.unlock()

    def test_lock_sysfs_fallback_sysfs_permission_error(self, tmp_path: Path, mock_sysfs_tree: Path) -> None:
        """Test fallback raising SysfsPermissionError converts to PMQoSPermissionError."""
        controller = PMQoSController(
            device_path=tmp_path / "missing_dev",
            sysfs_root=mock_sysfs_tree,
            enable_sysfs_fallback=True,
        )
        with patch.object(controller.sysfs, "_write_file", side_effect=SysfsPermissionError("Permission denied")):
            with pytest.raises(PMQoSPermissionError):
                controller.lock()
