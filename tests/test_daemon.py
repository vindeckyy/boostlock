"""
Unit tests for BoostLock Daemon, PID file manager, and system integration (FEAT-05).
"""

import fcntl
import os
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import pytest

from boostlock.config import BoostLockConfig, ConfigValidationError
from boostlock.daemon import (
    BoostLockDaemon,
    DaemonError,
    DaemonRunningError,
    DaemonState,
    PIDFileError,
    PIDFileManager,
    resolve_default_pid_path,
)
from boostlock.ipc import IPCClient
from boostlock.protocol import Command, Request, Response
from boostlock.state import StateSnapshotManager
from boostlock.sysfs import SysfsController
from boostlock.pm_qos import PMQoSController
from boostlock.thermal import ThermalGuard, ThermalReading, ThermalState


@pytest.fixture(autouse=True)
def preserve_signals():
    orig = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            orig[sig] = signal.getsignal(sig)
        except Exception:
            pass
    yield
    for sig, handler in orig.items():
        try:
            if handler is not None:
                signal.signal(sig, handler)
        except Exception:
            pass


@pytest.fixture
def mock_env(tmp_path):
    """Set up complete mock sysfs, pm_qos, pid, socket, and snapshot paths."""
    sysfs_root = tmp_path / "sys"
    sysfs_root.mkdir(parents=True, exist_ok=True)

    # CPU 0 and CPU 1 cpufreq
    for cpu in range(2):
        cpudir = sysfs_root / f"devices/system/cpu/cpu{cpu}/cpufreq"
        cpudir.mkdir(parents=True, exist_ok=True)
        (cpudir / "scaling_governor").write_text("schedutil\n")
        (cpudir / "scaling_available_governors").write_text("performance powersave schedutil ondemand\n")
        (cpudir / "scaling_min_freq").write_text("1400000\n")
        (cpudir / "scaling_max_freq").write_text("4000000\n")
        (cpudir / "scaling_cur_freq").write_text("2400000\n")
        (cpudir / "cpuinfo_min_freq").write_text("1400000\n")
        (cpudir / "cpuinfo_max_freq").write_text("4000000\n")
        (cpudir / "cpuinfo_base_freq").write_text("3000000\n")
        (cpudir / "energy_performance_preference").write_text("balance_performance\n")
        (cpudir / "cpb").write_text("1\n")

    # Global cpufreq & cpus
    global_cpufreq = sysfs_root / "devices/system/cpu/cpufreq"
    global_cpufreq.mkdir(parents=True, exist_ok=True)
    (global_cpufreq / "boost").write_text("1\n")
    (sysfs_root / "devices/system/cpu/online").write_text("0-1\n")

    # Thermal sensor
    hwmon0 = sysfs_root / "class/hwmon/hwmon0"
    hwmon0.mkdir(parents=True, exist_ok=True)
    (hwmon0 / "name").write_text("k10temp\n")
    (hwmon0 / "temp1_input").write_text("55000\n")
    (hwmon0 / "temp1_label").write_text("Tctl\n")

    # PM QoS device node
    dma_lat = tmp_path / "dev_cpu_dma_latency"
    dma_lat.write_bytes(b"\xff\xff\xff\x7f")

    # Paths
    pid_file = tmp_path / "boostlock.pid"
    sock_path = tmp_path / "boostlock.sock"
    snap_file = tmp_path / "snapshot.json"

    config = BoostLockConfig(
        target_frequency_khz=4000000,
        thermal_limit_c=85.0,
        thermal_warn_c=75.0,
        thermal_recover_c=70.0,
        poll_interval_ms=50,
        min_pulse_duty_pct=5.0,
        max_pulse_duty_pct=30.0,
        pid_file=str(pid_file),
        socket_path=str(sock_path),
        snapshot_path=str(snap_file),
    )

    return {
        "tmp_path": tmp_path,
        "sysfs_root": sysfs_root,
        "dma_latency_path": dma_lat,
        "pid_file": pid_file,
        "socket_path": sock_path,
        "snapshot_file": snap_file,
        "config": config,
    }


class TestPIDFileManager:
    """Tests for PID file locking, lifecycle, and stale detection."""

    def test_acquire_and_release(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)

        assert not mgr.is_locked
        assert mgr.acquire()
        assert mgr.is_locked
        assert pid_path.exists()
        assert int(pid_path.read_text().strip()) == os.getpid()

        mgr.release()
        assert not mgr.is_locked
        assert not pid_path.exists()

    def test_acquire_already_running_raises_error(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr1 = PIDFileManager(pid_path)
        assert mgr1.acquire()

        mgr2 = PIDFileManager(pid_path)
        with pytest.raises(DaemonRunningError) as exc_info:
            mgr2.acquire()
        assert "already running" in str(exc_info.value)
        assert str(os.getpid()) in str(exc_info.value)

        mgr1.release()

    def test_stale_pid_cleanup(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        # Write non-existent PID (e.g., PID 99999999)
        pid_path.write_text("99999999\n")

        mgr = PIDFileManager(pid_path)
        # Should detect stale PID, acquire lock, and overwrite PID file
        assert mgr.acquire()
        assert mgr.is_locked
        assert int(pid_path.read_text().strip()) == os.getpid()
        mgr.release()

    def test_read_pid_and_is_running(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)

        assert mgr.read_pid() is None
        assert not mgr.is_daemon_running()

        mgr.acquire()
        assert mgr.read_pid() == os.getpid()
        assert mgr.is_daemon_running()
        mgr.release()

    def test_read_pid_corrupt_content(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("invalid-non-numeric-content\n")
        mgr = PIDFileManager(pid_path)
        assert mgr.read_pid() is None

    def test_is_daemon_running_permission_and_os_errors(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345\n")
        mgr = PIDFileManager(pid_path)

        with patch("os.kill", side_effect=PermissionError("Permission denied")):
            assert mgr.is_daemon_running() is True

        with patch("os.kill", side_effect=OSError("Generic OS error")):
            assert mgr.is_daemon_running() is False

    def test_acquire_open_failure(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)
        with patch("os.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PIDFileError) as exc_info:
                mgr.acquire()
            assert "Failed to open PID file" in str(exc_info.value)

    def test_acquire_write_failure(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)
        with patch("os.write", side_effect=OSError("Disk full")):
            with pytest.raises(PIDFileError) as exc_info:
                mgr.acquire()
            assert "Failed to write PID" in str(exc_info.value)

    def test_acquire_locked_by_dead_or_unreadable_pid(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)
        # Mock flock throwing BlockingIOError and read_pid returning None
        with patch("fcntl.flock", side_effect=BlockingIOError("Locked")):
            with patch.object(mgr, "read_pid", return_value=None):
                with pytest.raises(DaemonRunningError) as exc_info:
                    mgr.acquire()
                assert "is locked by another process" in str(exc_info.value)

    def test_release_exceptions_handled_gracefully(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        mgr = PIDFileManager(pid_path)
        mgr.acquire()

        with patch("fcntl.flock", side_effect=OSError("Unlock error")), \
             patch.object(Path, "unlink", side_effect=OSError("Unlink error")):
            mgr.release()
            assert not mgr.is_locked

    def test_pid_file_parent_directory_creation(self, tmp_path):
        nested_pid = tmp_path / "nested" / "dir" / "boostlock.pid"
        mgr = PIDFileManager(nested_pid)
        assert mgr.acquire()
        assert nested_pid.exists()
        mgr.release()

    def test_resolve_default_pid_path(self):
        p = resolve_default_pid_path()
        assert p.name == "boostlock.pid"
        assert p.parent.exists()

    def test_resolve_default_pid_path_fallback_on_permission_error(self):
        with patch.object(Path, "mkdir", side_effect=[PermissionError("No access"), None]):
            fallback_p = resolve_default_pid_path()
            assert fallback_p.name == "boostlock.pid"
            assert "boostlock" in str(fallback_p)


class TestBoostLockDaemonLifecycle:
    """Tests for daemon startup, subsystem coordination, IPC requests, and teardown."""

    def test_daemon_full_start_and_stop(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        assert daemon.state == DaemonState.STOPPED

        daemon.start()
        try:
            assert daemon.state == DaemonState.RUNNING
            assert daemon.is_locked_boost is True
            assert mock_env["pid_file"].exists()
            assert mock_env["socket_path"].exists()
            assert mock_env["snapshot_file"].exists()

            # Verify sysfs governors set to performance
            sysfs = SysfsController(sysfs_root=mock_env["sysfs_root"])
            assert sysfs.get_scaling_governor(0) == "performance"
            assert sysfs.get_scaling_governor(1) == "performance"
            assert sysfs.get_scaling_min_freq(0) == 4000000

            # Verify PM QoS locked
            assert daemon.pm_qos.is_locked is True

            # Verify ThermalGuard & PulseEngine running
            assert daemon.thermal_guard.is_running is True
            assert daemon.pulse_engine.state.value == "RUNNING"

            # Connect via IPC client and test status & ping
            client = IPCClient(socket_path=mock_env["socket_path"])
            ping_res = client.ping()
            assert ping_res.success is True

            status = client.get_status()
            assert status["state"] == "RUNNING"
            assert status["target_frequency_khz"] == 4000000
            assert status["thermal"]["state"] == "NORMAL"
            assert status["pm_qos"]["is_locked"] is True

            metrics = client.get_metrics()
            assert "thermal" in metrics
            assert "engine" in metrics
            assert "cpus" in metrics

            # Double start should be a no-op
            daemon.start()
            assert daemon.state == DaemonState.RUNNING

        finally:
            daemon.stop()
            assert daemon.state == DaemonState.STOPPED
            assert not mock_env["pid_file"].exists()
            assert not mock_env["socket_path"].exists()
            assert daemon.pm_qos.is_locked is False

            # Verify sysfs state was rolled back to original
            sysfs = SysfsController(sysfs_root=mock_env["sysfs_root"])
            assert sysfs.get_scaling_governor(0) == "schedutil"
            assert sysfs.get_scaling_governor(1) == "schedutil"
            assert sysfs.get_scaling_min_freq(0) == 1400000

    def test_daemon_pause_and_resume(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Pause when not running is a no-op
        daemon.pause()
        assert daemon.state == DaemonState.STOPPED

        # Resume when not paused is a no-op
        daemon.resume()
        assert daemon.state == DaemonState.STOPPED

        daemon.start()
        try:
            client = IPCClient(socket_path=mock_env["socket_path"])
            
            pause_res = client.pause()
            assert pause_res.success is True
            assert daemon.state == DaemonState.PAUSED
            assert daemon.pulse_engine.state.value == "PAUSED"

            resume_res = client.resume()
            assert resume_res.success is True
            assert daemon.state == DaemonState.RUNNING
            assert daemon.pulse_engine.state.value == "RUNNING"

        finally:
            daemon.stop()

    def test_daemon_unlock_and_lock(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        daemon.start()
        try:
            client = IPCClient(socket_path=mock_env["socket_path"])

            # Lock when already locked is a no-op
            daemon.lock()
            assert daemon.is_locked_boost is True

            # Unlock boost pinning (disengages PM QoS and pulse engine, relaxes sysfs)
            unlock_res = client.unlock()
            assert unlock_res.success is True
            assert daemon.is_locked_boost is False
            assert daemon.pm_qos.is_locked is False
            assert daemon.pulse_engine.state.value == "STOPPED"

            # Double unlock is a no-op
            daemon.unlock()
            assert daemon.is_locked_boost is False

            # Re-lock
            lock_res = client.lock()
            assert lock_res.success is True
            assert daemon.is_locked_boost is True
            assert daemon.pm_qos.is_locked is True
            assert daemon.pulse_engine.state.value == "RUNNING"

        finally:
            daemon.stop()

    def test_daemon_reconfigure(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        daemon.start()
        try:
            client = IPCClient(socket_path=mock_env["socket_path"])

            # Reconfigure with valid parameters
            reconf_res = client.reconfigure({
                "target_frequency_khz": 3800000,
                "thermal_warn_c": 72.0,
                "thermal_limit_c": 82.0,
                "thermal_recover_c": 68.0,
            })
            assert reconf_res.success is True
            assert daemon.config.target_frequency_khz == 3800000
            assert daemon.thermal_guard.thermal_warn_c == 72.0
            assert daemon.thermal_guard.thermal_limit_c == 82.0

            # Reconfigure with invalid parameters returns error
            bad_reconf = client.send_request(Request(
                command=Command.RECONFIGURE,
                args={"thermal_limit_c": 150.0},
            ))
            assert bad_reconf.success is False
            assert "thermal_limit_c must be between" in bad_reconf.error

        finally:
            daemon.stop()

    def test_daemon_ipc_start_command(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Start IPC server manually to test START command via IPC
        daemon.ipc_server.start()
        try:
            client = IPCClient(socket_path=mock_env["socket_path"])
            start_res = client.start()
            assert start_res.success is True
            assert daemon.state == DaemonState.RUNNING
        finally:
            daemon.stop()

    def test_daemon_stop_command_via_ipc(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        daemon.start()
        client = IPCClient(socket_path=mock_env["socket_path"])
        stop_res = client.stop()
        assert stop_res.success is True
        
        # Give thread moment to complete stop
        time.sleep(0.2)
        assert daemon.state == DaemonState.STOPPED
        assert not client.is_daemon_running()

    def test_daemon_stop_handles_subsystem_errors_resiliently(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )
        daemon.start()

        with patch.object(daemon.pulse_engine, "stop", side_effect=RuntimeError("PulseEngine stop fail")), \
             patch.object(daemon.thermal_guard, "stop", side_effect=RuntimeError("ThermalGuard stop fail")), \
             patch.object(daemon.pm_qos, "unlock", side_effect=RuntimeError("PMQoS unlock fail")), \
             patch.object(daemon.state_manager, "restore", side_effect=RuntimeError("State restore fail")), \
             patch.object(daemon.ipc_server, "stop", side_effect=RuntimeError("IPC stop fail")), \
             patch.object(daemon.pid_manager, "release", side_effect=RuntimeError("PID release fail")):
            daemon.stop()
            assert daemon.state == DaemonState.STOPPED

    def test_daemon_thermal_callbacks_and_error_handling(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        daemon.start()
        try:
            # Thermal warning
            warn_reading = ThermalReading(
                timestamp=time.time(),
                current_temp_c=78.0,
                sensors={"Tctl": 78.0},
                state=ThermalState.WARNING,
                clamp_factor=0.7,
                is_tripped=False,
            )
            daemon._on_thermal_warning(warn_reading)

            # Thermal tripwire with pause exception handling
            tripwire_reading = ThermalReading(
                timestamp=time.time(),
                current_temp_c=88.0,
                sensors={"Tctl": 88.0},
                state=ThermalState.CRITICAL,
                clamp_factor=0.0,
                is_tripped=True,
            )
            with patch.object(daemon.pulse_engine, "pause", side_effect=RuntimeError("Pause fail")):
                daemon._on_thermal_tripwire(tripwire_reading)
                assert daemon.state == DaemonState.THROTTLED

            # Thermal recovery with resume exception handling
            recovery_reading = ThermalReading(
                timestamp=time.time(),
                current_temp_c=65.0,
                sensors={"Tctl": 65.0},
                state=ThermalState.NORMAL,
                clamp_factor=1.0,
                is_tripped=False,
            )
            with patch.object(daemon.pulse_engine, "resume", side_effect=RuntimeError("Resume fail")):
                daemon._on_thermal_recovery(recovery_reading)
                assert daemon.state == DaemonState.RUNNING

        finally:
            daemon.stop()

    def test_daemon_startup_exception_triggers_rollback(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Force PulseEngine.start to fail
        with patch.object(daemon.pulse_engine, "start", side_effect=RuntimeError("PulseEngine failed to start")):
            with pytest.raises(DaemonError) as exc_info:
                daemon.start()
            assert "PulseEngine failed to start" in str(exc_info.value)

        assert daemon.state == DaemonState.ERROR
        # Verify rollback was called and original state restored
        sysfs = SysfsController(sysfs_root=mock_env["sysfs_root"])
        assert sysfs.get_scaling_governor(0) == "schedutil"
        assert daemon.pm_qos.is_locked is False
        assert not mock_env["pid_file"].exists()

    def test_daemon_emergency_rollback_handles_exceptions_cleanly(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        with patch.object(daemon.pulse_engine, "stop", side_effect=RuntimeError("PE stop err")), \
             patch.object(daemon.thermal_guard, "stop", side_effect=RuntimeError("TG stop err")), \
             patch.object(daemon.pm_qos, "unlock", side_effect=RuntimeError("QoS unlock err")), \
             patch.object(daemon.state_manager, "restore", side_effect=RuntimeError("Restore err")), \
             patch.object(daemon.ipc_server, "stop", side_effect=RuntimeError("IPC stop err")), \
             patch.object(daemon.pid_manager, "release", side_effect=RuntimeError("PID release err")):
            daemon._emergency_rollback(RuntimeError("Root crash"))
            assert daemon.state == DaemonState.ERROR

    def test_daemon_signal_handling(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        daemon.start()
        try:
            # Trigger signal handler directly
            daemon._signal_handler(signal.SIGTERM, None)
            assert daemon.state == DaemonState.STOPPED
            assert daemon.pm_qos.is_locked is False
        finally:
            daemon.stop()

    def test_daemon_signals_non_main_thread(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Call signal registration from subthread
        def thread_target():
            daemon._register_signals()
            daemon._unregister_signals()

        t = threading.Thread(target=thread_target)
        t.start()
        t.join()

    def test_daemon_signal_registration_exception(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        with patch("signal.signal", side_effect=ValueError("Cannot set signal in thread")):
            daemon._register_signals()
            daemon._unregister_signals()

    def test_daemon_run_blocking_and_keyboard_interrupt(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Normal stop
        def stop_after_delay():
            time.sleep(0.1)
            daemon.stop()

        t = threading.Thread(target=stop_after_delay)
        t.start()
        daemon.run()
        t.join()
        assert daemon.state == DaemonState.STOPPED

        # KeyboardInterrupt handling
        with patch.object(daemon._stop_event, "wait", side_effect=KeyboardInterrupt("Interrupt")):
            daemon.run()
            assert daemon.state == DaemonState.STOPPED

    def test_daemon_daemonize_mock(self, mock_env):
        daemon = BoostLockDaemon(
            config=mock_env["config"],
            sysfs_root=mock_env["sysfs_root"],
            dma_latency_path=mock_env["dma_latency_path"],
            pid_file=mock_env["pid_file"],
            socket_path=mock_env["socket_path"],
            snapshot_file=mock_env["snapshot_file"],
        )

        # Mock child process branches (pid=0)
        with patch("os.fork", return_value=0), \
             patch("os.chdir"), \
             patch("os.setsid"), \
             patch("os.umask"), \
             patch("os.dup2"):
            daemon.daemonize()

        # Mock fork failures
        with patch("os.fork", side_effect=OSError("Fork fail")):
            with pytest.raises(DaemonError) as exc_info:
                daemon.daemonize()
            assert "First fork failed" in str(exc_info.value)

        # Mock second fork failure
        fork_calls = [0, OSError("Second fork fail")]
        with patch("os.fork", side_effect=fork_calls), \
             patch("os.chdir"), \
             patch("os.setsid"), \
             patch("os.umask"):
            with pytest.raises(DaemonError) as exc_info:
                daemon.daemonize()
            assert "Second fork failed" in str(exc_info.value)


class TestSystemdServiceFile:
    """Tests for systemd unit file presence and valid formatting."""

    def test_systemd_unit_file(self):
        unit_file = Path(__file__).resolve().parents[1] / "systemd" / "boostlock.service"
        assert unit_file.exists(), "boostlock.service must exist in systemd/"
        content = unit_file.read_text(encoding="utf-8")

        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "Type=simple" in content
        assert "ExecStart=" in content
        assert "ExecStop=" in content
        assert "ProtectSystem=full" in content
        assert "CAP_SYS_ADMIN" in content
