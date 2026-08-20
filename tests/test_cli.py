"""tests/test_cli.py - Tests for boostlock/cli.py"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, call

import pytest

from boostlock.cli import (
    DEFAULT_SOCKET_PATH,
    SYSTEMD_SERVICE_SRC,
    build_parser,
    cmd_bench,
    cmd_restore,
    cmd_service,
    cmd_start,
    cmd_status,
    cmd_stop,
    main,
    _connect_client,
    _render_status_table,
    _send_command,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_parser_returns_argumentparser(self):
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_requires_subcommand(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_start_parses_defaults(self):
        p = build_parser()
        args = p.parse_args(["start"])
        assert args.command == "start"
        assert args.daemon is False
        assert args.target is None
        assert args.duty is None
        assert args.max_temp is None

    def test_start_daemon_flag(self):
        p = build_parser()
        args = p.parse_args(["start", "--daemon"])
        assert args.daemon is True

    def test_start_with_all_options(self):
        p = build_parser()
        args = p.parse_args(["start", "--target", "4000", "--duty", "15", "--max-temp", "85"])
        assert args.target == 4000.0
        assert args.duty == 15.0
        assert args.max_temp == 85.0

    @pytest.mark.parametrize("value", ["zero", "0"])
    def test_target_rejects_invalid_or_zero_values(self, value):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["start", "--target", value])

    def test_stop_subcommand(self):
        p = build_parser()
        args = p.parse_args(["stop"])
        assert args.command == "stop"

    def test_status_defaults(self):
        p = build_parser()
        args = p.parse_args(["status"])
        assert args.command == "status"
        assert args.watch is False
        assert args.json is False

    def test_status_watch_and_json_flags(self):
        p = build_parser()
        args = p.parse_args(["status", "--watch", "--json"])
        assert args.watch is True
        assert args.json is True

    def test_bench_defaults(self):
        p = build_parser()
        args = p.parse_args(["bench"])
        assert args.command == "bench"
        assert args.duration == 10.0
        assert args.target == "auto"
        assert args.sample_hz == 20.0
        assert args.output is None

    def test_bench_all_options(self):
        p = build_parser()
        args = p.parse_args(["bench", "--duration", "30", "--target", "3800",
                              "--sample-hz", "10", "--output", "/tmp/report.txt"])
        assert args.duration == 30.0
        assert args.target == 3800
        assert args.sample_hz == 10.0
        assert args.output == "/tmp/report.txt"

    def test_restore_subcommand(self):
        p = build_parser()
        args = p.parse_args(["restore"])
        assert args.command == "restore"

    def test_service_install(self):
        p = build_parser()
        args = p.parse_args(["service", "install"])
        assert args.service_action == "install"

    def test_service_uninstall(self):
        p = build_parser()
        args = p.parse_args(["service", "uninstall"])
        assert args.service_action == "uninstall"

    def test_service_status(self):
        p = build_parser()
        args = p.parse_args(["service", "status"])
        assert args.service_action == "status"

    def test_service_invalid_action_exits(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["service", "bogus"])

    def test_global_socket_option(self):
        p = build_parser()
        args = p.parse_args(["--socket", "/tmp/test.sock", "status"])
        assert args.socket == "/tmp/test.sock"


# ---------------------------------------------------------------------------
# _connect_client
# ---------------------------------------------------------------------------

class TestConnectClient:
    def test_returns_client_on_success(self, tmp_path):
        mock_client = MagicMock()
        socket_path = tmp_path / "test.sock"
        with patch("boostlock.cli.IPCClient", return_value=mock_client) as MockIPCClient:
            result = _connect_client(socket_path)
        assert result is mock_client

    def test_returns_none_on_exception(self, tmp_path, capsys):
        socket_path = tmp_path / "nonexistent.sock"
        with patch("boostlock.cli.IPCClient", side_effect=ConnectionRefusedError("refused")):
            result = _connect_client(socket_path)
        assert result is None
        captured = capsys.readouterr()
        assert "Cannot connect" in captured.err


# ---------------------------------------------------------------------------
# _send_command
# ---------------------------------------------------------------------------

class TestSendCommand:
    def _make_client(self, success: bool = True, data=None, error=None):
        from boostlock.protocol import Response
        resp = MagicMock(spec=Response)
        resp.success = success
        resp.data = data
        resp.error = error
        client = MagicMock()
        client.send.return_value = resp
        return client

    def test_returns_data_on_success(self):
        client = self._make_client(success=True, data={"state": "running"})
        result = _send_command(client, "STATUS")
        assert result == {"state": "running"}

    def test_returns_none_on_daemon_error(self, capsys):
        client = self._make_client(success=False, error="Not running")
        result = _send_command(client, "STATUS")
        assert result is None
        captured = capsys.readouterr()
        assert "Not running" in captured.err

    def test_returns_none_on_none_response(self, capsys):
        client = MagicMock()
        client.send.return_value = None
        result = _send_command(client, "STATUS")
        assert result is None

    def test_returns_none_on_exception(self, capsys):
        client = MagicMock()
        client.send.side_effect = OSError("socket broken")
        result = _send_command(client, "STATUS")
        assert result is None
        captured = capsys.readouterr()
        assert "Communication error" in captured.err

    def test_passes_kwargs_as_params(self):
        client = self._make_client(success=True, data={})
        _send_command(client, "RECONFIGURE", target_freq_khz=4_000_000)
        call_args = client.send.call_args
        req = call_args[0][0]
        assert req.params == {"target_freq_khz": 4_000_000}


# ---------------------------------------------------------------------------
# cmd_stop
# ---------------------------------------------------------------------------

class TestCmdStop:
    def test_stop_success(self, tmp_path, capsys):
        args = argparse.Namespace()
        socket_path = tmp_path / "test.sock"
        mock_client = MagicMock()
        from boostlock.protocol import Response
        resp = MagicMock(spec=Response)
        resp.success = True
        resp.data = {}
        mock_client.send.return_value = resp
        with patch("boostlock.cli.IPCClient", return_value=mock_client):
            rc = cmd_stop(args, socket_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "stopped" in captured.out

    def test_stop_no_daemon(self, tmp_path, capsys):
        args = argparse.Namespace()
        socket_path = tmp_path / "test.sock"
        with patch("boostlock.cli.IPCClient", side_effect=ConnectionRefusedError()):
            rc = cmd_stop(args, socket_path)
        assert rc == 1

    def test_stop_daemon_error(self, tmp_path, capsys):
        args = argparse.Namespace()
        socket_path = tmp_path / "test.sock"
        mock_client = MagicMock()
        from boostlock.protocol import Response
        resp = MagicMock(spec=Response)
        resp.success = False
        resp.error = "Already stopped"
        mock_client.send.return_value = resp
        with patch("boostlock.cli.IPCClient", return_value=mock_client):
            rc = cmd_stop(args, socket_path)
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

class TestCmdStatus:
    def _make_status_data(self) -> Dict[str, Any]:
        return {
            "boost_state": "running",
            "governor": "performance",
            "pm_qos_active": True,
            "target_freq_khz": 4_000_000,
            "temperature_c": 65.0,
            "duty_cycle": 0.15,
            "per_cpu": {
                "0": {"cur_freq_khz": 4_000_000},
                "1": {"cur_freq_khz": 3_950_000},
            },
        }

    def test_status_json_output(self, tmp_path, capsys):
        args = argparse.Namespace(watch=False, json=True)
        socket_path = tmp_path / "test.sock"
        data = self._make_status_data()
        mock_client = MagicMock()
        from boostlock.protocol import Response
        resp = MagicMock(spec=Response)
        resp.success = True
        resp.data = data
        mock_client.send.return_value = resp
        with patch("boostlock.cli.IPCClient", return_value=mock_client):
            rc = cmd_status(args, socket_path)
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["governor"] == "performance"

    def test_status_table_output(self, tmp_path, capsys):
        args = argparse.Namespace(watch=False, json=False)
        socket_path = tmp_path / "test.sock"
        data = self._make_status_data()
        mock_client = MagicMock()
        from boostlock.protocol import Response
        resp = MagicMock(spec=Response)
        resp.success = True
        resp.data = data
        mock_client.send.return_value = resp
        with patch("boostlock.cli.IPCClient", return_value=mock_client):
            rc = cmd_status(args, socket_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "BoostLock Status" in captured.out
        assert "performance" in captured.out

    def test_status_no_daemon(self, tmp_path):
        args = argparse.Namespace(watch=False, json=False)
        socket_path = tmp_path / "test.sock"
        with patch("boostlock.cli.IPCClient", side_effect=ConnectionRefusedError()):
            rc = cmd_status(args, socket_path)
        assert rc == 1


# ---------------------------------------------------------------------------
# _render_status_table
# ---------------------------------------------------------------------------

class TestRenderStatusTable:
    def test_renders_without_error(self, capsys):
        data = {
            "boost_state": "running",
            "governor": "performance",
            "pm_qos_active": True,
            "target_freq_khz": 4_000_000,
            "temperature_c": 70.0,
            "duty_cycle": 0.2,
            "per_cpu": {
                "0": {"cur_freq_khz": 4_000_000},
            },
        }
        _render_status_table(data)
        captured = capsys.readouterr()
        assert "BoostLock Status" in captured.out

    def test_renders_without_temperature(self, capsys):
        data = {
            "boost_state": "running",
            "governor": "performance",
            "pm_qos_active": False,
            "target_freq_khz": 4_000_000,
        }
        _render_status_table(data)
        captured = capsys.readouterr()
        assert "unavailable" in captured.out

    def test_renders_without_per_cpu(self, capsys):
        data = {
            "boost_state": "running",
            "governor": "performance",
            "pm_qos_active": True,
            "target_freq_khz": 4_000_000,
            "temperature_c": 65.0,
        }
        _render_status_table(data)
        captured = capsys.readouterr()
        assert "BoostLock Status" in captured.out

    def test_renders_without_duty_cycle(self, capsys):
        data = {
            "boost_state": "stopped",
            "governor": "schedutil",
            "pm_qos_active": False,
            "target_freq_khz": 0,
        }
        _render_status_table(data)
        captured = capsys.readouterr()
        assert "schedutil" in captured.out

    def test_renders_automatic_target(self, capsys):
        _render_status_table({"target_freq_khz": "auto"})
        assert "automatic per-policy" in capsys.readouterr().out

    def test_watch_mode_completes(self, capsys):
        data = {"boost_state": "running", "governor": "performance",
                "pm_qos_active": True, "target_freq_khz": 4_000_000}
        with patch("boostlock.cli.time.sleep"):
            _render_status_table(data, watch=True)
        # Should complete without error


# ---------------------------------------------------------------------------
# cmd_bench
# ---------------------------------------------------------------------------

class TestCmdBench:
    def test_bench_success(self, tmp_path, capsys):
        args = argparse.Namespace(target=4000, duration=0.1, sample_hz=50, output=None)
        mock_result = MagicMock()
        mock_result.format_report.return_value = "=== Bench Report ==="
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        with patch("boostlock.cli.BenchmarkRunner", return_value=mock_runner):
            rc = cmd_bench(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Bench Report" in captured.out

    def test_bench_with_output_file(self, tmp_path, capsys):
        output_file = tmp_path / "report.txt"
        args = argparse.Namespace(target=4000, duration=0.1, sample_hz=50,
                                  output=str(output_file))
        mock_result = MagicMock()
        mock_result.format_report.return_value = "Report Content"
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        with patch("boostlock.cli.BenchmarkRunner", return_value=mock_runner):
            rc = cmd_bench(args)
        assert rc == 0
        assert output_file.exists()
        assert "Report Content" in output_file.read_text()

    def test_bench_error_returns_1(self, capsys):
        args = argparse.Namespace(target=4000, duration=0.1, sample_hz=50, output=None)
        with patch("boostlock.cli.BenchmarkRunner", side_effect=RuntimeError("boom")):
            rc = cmd_bench(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Benchmark error" in captured.err

    def test_bench_uses_default_target_when_none(self, tmp_path, capsys):
        args = argparse.Namespace(target=None, duration=None, sample_hz=None, output=None)
        mock_result = MagicMock()
        mock_result.format_report.return_value = "Report"
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        with patch("boostlock.cli.BenchmarkRunner", return_value=mock_runner) as MockRunner:
            cmd_bench(args)
        call_kwargs = MockRunner.call_args[1]
        assert call_kwargs["target_khz"] == "auto"


# ---------------------------------------------------------------------------
# cmd_restore
# ---------------------------------------------------------------------------

class TestCmdRestore:
    def test_restore_no_snapshot(self, capsys):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        with patch("boostlock.cli.StateSnapshotManager", return_value=mock_mgr):
            rc = cmd_restore(argparse.Namespace())
        assert rc == 0
        captured = capsys.readouterr()
        assert "No snapshot found" in captured.out

    def test_restore_success(self, capsys):
        mock_mgr = MagicMock()
        mock_snap = MagicMock()
        mock_mgr.load.return_value = mock_snap
        with patch("boostlock.cli.StateSnapshotManager", return_value=mock_mgr):
            rc = cmd_restore(argparse.Namespace())
        assert rc == 0
        mock_mgr.restore.assert_called_once_with(mock_snap)
        captured = capsys.readouterr()
        assert "Restore complete" in captured.out

    def test_restore_exception_returns_1(self, capsys):
        with patch("boostlock.cli.StateSnapshotManager", side_effect=OSError("no perms")):
            rc = cmd_restore(argparse.Namespace())
        assert rc == 1
        captured = capsys.readouterr()
        assert "Restore error" in captured.err


# ---------------------------------------------------------------------------
# cmd_service
# ---------------------------------------------------------------------------

class TestCmdService:
    def test_service_source_is_bundled_with_package(self):
        assert SYSTEMD_SERVICE_SRC.parts[-3:] == ("boostlock", "data", "boostlock.service")
        assert SYSTEMD_SERVICE_SRC.exists()

    def test_service_status_calls_systemctl(self, tmp_path):
        args = argparse.Namespace(service_action="status")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("boostlock.cli.subprocess.run", return_value=mock_result) as mock_run:
            rc = cmd_service(args)
        assert rc == 0
        mock_run.assert_called()

    def test_service_install_missing_source(self, tmp_path, capsys):
        args = argparse.Namespace(service_action="install")
        with patch("boostlock.cli.SYSTEMD_SERVICE_SRC", tmp_path / "missing.service"):
            rc = cmd_service(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_service_install_success(self, tmp_path, capsys):
        src = tmp_path / "boostlock.service"
        src.write_text("[Unit]\nDescription=test\n")
        dst = tmp_path / "installed.service"
        args = argparse.Namespace(service_action="install")
        with patch("boostlock.cli.SYSTEMD_SERVICE_SRC", src), \
             patch("boostlock.cli.SYSTEMD_SERVICE_DST", dst), \
             patch("boostlock.cli.subprocess.run"):
            rc = cmd_service(args)
        assert rc == 0
        assert dst.exists()
        assert dst.read_text() == src.read_text()

    def test_service_uninstall_calls_systemctl(self, tmp_path, capsys):
        dst = tmp_path / "boostlock.service"
        dst.write_text("[Unit]\nDescription=test\n")
        args = argparse.Namespace(service_action="uninstall")
        with patch("boostlock.cli.SYSTEMD_SERVICE_DST", dst), \
             patch("boostlock.cli.subprocess.run"):
            rc = cmd_service(args)
        assert rc == 0
        assert not dst.exists()

    def test_service_no_systemctl(self, tmp_path, capsys):
        args = argparse.Namespace(service_action="status")
        with patch("boostlock.cli.subprocess.run", side_effect=FileNotFoundError("no systemctl")):
            rc = cmd_service(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "systemctl not found" in captured.err

    def test_service_unknown_action(self, capsys):
        args = argparse.Namespace(service_action="bogus_action")
        rc = cmd_service(args)
        assert rc == 2
        captured = capsys.readouterr()
        assert "Unknown service action" in captured.err

    def test_service_generic_exception(self, capsys):
        args = argparse.Namespace(service_action="status")
        with patch("boostlock.cli.subprocess.run", side_effect=PermissionError("denied")):
            rc = cmd_service(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Service command failed" in captured.err


# ---------------------------------------------------------------------------
# cmd_start
# ---------------------------------------------------------------------------

class TestCmdStart:
    def test_start_daemon_mode_spawns_process(self, tmp_path, capsys):
        args = argparse.Namespace(daemon=True, target=None, duty=None, max_temp=None)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("boostlock.cli.subprocess.Popen", return_value=mock_proc):
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 0
        captured = capsys.readouterr()
        assert "12345" in captured.out

    def test_start_daemon_spawn_error(self, tmp_path, capsys):
        args = argparse.Namespace(daemon=True, target=None, duty=None, max_temp=None)
        with patch("boostlock.cli.subprocess.Popen", side_effect=OSError("no exec")):
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 1
        captured = capsys.readouterr()
        assert "Failed to start daemon" in captured.err

    def test_start_foreground_mode_runs_daemon(self, tmp_path):
        args = argparse.Namespace(daemon=False, target=4000.0, duty=15.0, max_temp=85.0)
        mock_daemon = MagicMock()
        mock_daemon.run.return_value = None
        with patch("boostlock.cli.BoostLockDaemon", return_value=mock_daemon), \
             patch("boostlock.cli.BoostLockConfig"):
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 0
        mock_daemon.run.assert_called_once()

    def test_start_applies_target_to_configuration(self, tmp_path):
        args = argparse.Namespace(daemon=False, target=3900.0, duty=None, max_temp=None)
        mock_daemon = MagicMock()
        with patch("boostlock.cli.BoostLockDaemon", return_value=mock_daemon) as daemon_class:
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")

        assert rc == 0
        config = daemon_class.call_args.kwargs["config"]
        assert config.target_frequency_khz == 3_900_000

    def test_start_without_target_uses_automatic_policy_mode(self, tmp_path):
        args = argparse.Namespace(daemon=False, target=None, duty=None, max_temp=None)
        mock_daemon = MagicMock()
        with patch("boostlock.cli.BoostLockDaemon", return_value=mock_daemon) as daemon_class:
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")

        assert rc == 0
        config = daemon_class.call_args.kwargs["config"]
        assert config.target_frequency_khz == "auto"

    def test_start_foreground_keyboard_interrupt(self, tmp_path, capsys):
        args = argparse.Namespace(daemon=False, target=None, duty=None, max_temp=None)
        mock_daemon = MagicMock()
        mock_daemon.run.side_effect = KeyboardInterrupt()
        with patch("boostlock.cli.BoostLockDaemon", return_value=mock_daemon), \
             patch("boostlock.cli.BoostLockConfig"):
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 0

    def test_start_foreground_exception(self, tmp_path, capsys):
        args = argparse.Namespace(daemon=False, target=None, duty=None, max_temp=None)
        mock_daemon = MagicMock()
        mock_daemon.run.side_effect = RuntimeError("crash")
        with patch("boostlock.cli.BoostLockDaemon", return_value=mock_daemon), \
             patch("boostlock.cli.BoostLockConfig"):
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 1
        captured = capsys.readouterr()
        assert "Fatal error" in captured.err

    def test_start_daemon_with_options(self, tmp_path, capsys):
        args = argparse.Namespace(daemon=True, target=4000, duty=20, max_temp=80)
        mock_proc = MagicMock()
        mock_proc.pid = 99
        with patch("boostlock.cli.subprocess.Popen", return_value=mock_proc) as MockPopen:
            rc = cmd_start(args, tmp_path / "test.sock", tmp_path / "test.pid")
        assert rc == 0
        cmd_used = MockPopen.call_args[0][0]
        assert "--target" in cmd_used
        assert "--duty" in cmd_used
        assert "--max-temp" in cmd_used
        assert "--socket" in cmd_used
        assert str(tmp_path / "test.sock") in cmd_used


# ---------------------------------------------------------------------------
# main() dispatcher
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_status_dispatches(self, tmp_path, capsys):
        with patch("boostlock.cli.cmd_status", return_value=0) as mock_status, \
             patch("boostlock.cli.DEFAULT_SOCKET_PATH", tmp_path / "test.sock"):
            rc = main(["status"])
        assert rc == 0
        mock_status.assert_called_once()

    def test_main_stop_dispatches(self, tmp_path, capsys):
        with patch("boostlock.cli.cmd_stop", return_value=0) as mock_stop, \
             patch("boostlock.cli.DEFAULT_SOCKET_PATH", tmp_path / "test.sock"):
            rc = main(["stop"])
        assert rc == 0
        mock_stop.assert_called_once()

    def test_main_restore_dispatches(self, capsys):
        with patch("boostlock.cli.cmd_restore", return_value=0) as mock_restore:
            rc = main(["restore"])
        assert rc == 0
        mock_restore.assert_called_once()

    def test_main_bench_dispatches(self, capsys):
        with patch("boostlock.cli.cmd_bench", return_value=0) as mock_bench:
            rc = main(["bench"])
        assert rc == 0
        mock_bench.assert_called_once()

    def test_main_service_dispatches(self, capsys):
        with patch("boostlock.cli.cmd_service", return_value=0) as mock_service:
            rc = main(["service", "status"])
        assert rc == 0
        mock_service.assert_called_once()

    def test_main_start_dispatches(self, tmp_path, capsys):
        with patch("boostlock.cli.cmd_start", return_value=0) as mock_start:
            rc = main(["start"])
        assert rc == 0
        mock_start.assert_called_once()

    def test_main_no_args_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_main_invalid_command_exits(self):
        with pytest.raises(SystemExit):
            main(["notacommand"])

    def test_main_custom_socket(self, tmp_path):
        sock = tmp_path / "custom.sock"
        with patch("boostlock.cli.cmd_status", return_value=0) as mock_status:
            rc = main(["--socket", str(sock), "status"])
        assert rc == 0
        call_args = mock_status.call_args[0]
        assert call_args[1] == sock
