"""
Unit tests for BoostLock Unix Domain Socket IPC Server & Client (FEAT-05).
"""

import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import pytest

from boostlock.ipc import (
    IPCClient,
    IPCConnectionError,
    IPCError,
    IPCPermissionError,
    IPCServer,
    IPCTimeoutError,
)
from boostlock.protocol import (
    Command,
    InvalidMessageError,
    Request,
    Response,
    UnknownCommandError,
)


@pytest.fixture
def temp_socket_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = Path(tmpdir) / "boostlock_test.sock"
        yield sock_path


class TestIPCServerAndClient:
    """End-to-end tests for IPC server and client communication."""

    def test_ping_command(self, temp_socket_path):
        def handler(req: Request) -> Response:
            if req.command == Command.PING:
                return Response.ok(data={"pong": True}, request_id=req.request_id)
            return Response.fail(error="Unhandled", request_id=req.request_id)

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        server.start()
        assert server.is_running
        assert temp_socket_path.exists()

        try:
            client = IPCClient(socket_path=temp_socket_path, timeout_s=2.0)
            assert client.is_daemon_running()

            res = client.ping()
            assert res.success is True
            assert res.data == {"pong": True}
        finally:
            server.stop()
            assert not server.is_running
            assert not temp_socket_path.exists()

    def test_all_client_convenience_methods(self, temp_socket_path):
        def handler(req: Request) -> Response:
            if req.command == Command.PING:
                return Response.ok({"pong": True}, request_id=req.request_id)
            elif req.command == Command.STATUS:
                return Response.ok({"state": "RUNNING", "target": 4000000}, request_id=req.request_id)
            elif req.command == Command.START:
                return Response.ok({"started": True}, request_id=req.request_id)
            elif req.command == Command.STOP:
                return Response.ok({"stopped": True}, request_id=req.request_id)
            elif req.command == Command.PAUSE:
                return Response.ok({"paused": True}, request_id=req.request_id)
            elif req.command == Command.RESUME:
                return Response.ok({"resumed": True}, request_id=req.request_id)
            elif req.command == Command.RECONFIGURE:
                return Response.ok({"reconfigured": req.args}, request_id=req.request_id)
            elif req.command == Command.LOCK:
                return Response.ok({"locked": True}, request_id=req.request_id)
            elif req.command == Command.UNLOCK:
                return Response.ok({"unlocked": True}, request_id=req.request_id)
            elif req.command == Command.METRICS:
                return Response.ok({"total_pulses": 12345}, request_id=req.request_id)
            elif req.command == Command.CONFIG:
                return Response.ok({"governor": "performance"}, request_id=req.request_id)
            return Response.fail("Unknown", request_id=req.request_id)

        with IPCServer(socket_path=temp_socket_path, handler=handler) as server:
            client = IPCClient(socket_path=temp_socket_path, timeout_s=2.0)

            # Test each convenience method
            assert client.ping().success is True
            assert client.get_status() == {"state": "RUNNING", "target": 4000000}
            assert client.start().success is True
            assert client.stop().success is True
            assert client.pause().success is True
            assert client.resume().success is True
            
            reconf_res = client.reconfigure({"target_frequency_khz": 4200000})
            assert reconf_res.success is True
            assert reconf_res.data == {"reconfigured": {"target_frequency_khz": 4200000}}

            assert client.lock().success is True
            assert client.unlock().success is True
            assert client.get_metrics() == {"total_pulses": 12345}
            assert client.get_config() == {"governor": "performance"}

    def test_handler_exception_returns_failure_response(self, temp_socket_path):
        def handler(req: Request) -> Response:
            raise ValueError("Something exploded in daemon handler")

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            client = IPCClient(socket_path=temp_socket_path, timeout_s=2.0)
            res = client.send_request(Request(command=Command.STATUS, request_id="err-1"))
            assert res.success is False
            assert "Something exploded in daemon handler" in res.error
            assert res.error_type == "ValueError"
            assert res.request_id == "err-1"

    def test_concurrent_clients(self, temp_socket_path):
        def handler(req: Request) -> Response:
            time.sleep(0.01)
            return Response.ok({"client_req": req.args.get("num")}, request_id=req.request_id)

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            results = []
            errors = []

            def worker(num: int):
                try:
                    c = IPCClient(socket_path=temp_socket_path, timeout_s=5.0)
                    r = c.send_request(Request(command=Command.STATUS, args={"num": num}))
                    if r.success and r.data.get("client_req") == num:
                        results.append(num)
                    else:
                        errors.append(f"Bad response for {num}: {r}")
                except Exception as e:
                    errors.append(f"Exception for {num}: {e}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(15)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results) == 15


class TestIPCErrorsAndEdgeCases:
    """Tests for IPC error conditions, stale sockets, and timeouts."""

    def test_client_connect_when_server_not_running(self, temp_socket_path):
        client = IPCClient(socket_path=temp_socket_path, timeout_s=1.0)
        assert not client.is_daemon_running()

        with pytest.raises(IPCConnectionError):
            client.send_request(Request(command=Command.PING))

    def test_client_connect_connection_refused(self, temp_socket_path):
        client = IPCClient(socket_path=temp_socket_path)
        with patch.object(socket.socket, "connect", side_effect=ConnectionRefusedError("Refused")):
            with pytest.raises(IPCConnectionError) as exc_info:
                client.send_request(Request(command=Command.PING))
            assert "Connection refused" in str(exc_info.value)

    def test_client_timeout(self, temp_socket_path):
        def handler(req: Request) -> Response:
            time.sleep(1.0)  # Delay longer than client timeout
            return Response.ok()

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            client = IPCClient(socket_path=temp_socket_path, timeout_s=0.1)
            with pytest.raises(IPCTimeoutError):
                client.send_request(Request(command=Command.PING))

    def test_server_stale_socket_cleanup(self, temp_socket_path):
        # Create a dummy stale socket file
        temp_socket_path.touch()
        assert temp_socket_path.exists()

        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        server.start()
        try:
            assert server.is_running
            client = IPCClient(socket_path=temp_socket_path)
            assert client.ping().success is True
        finally:
            server.stop()

    def test_server_double_start_error_if_active(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server1 = IPCServer(socket_path=temp_socket_path, handler=handler)
        server1.start()
        try:
            # Second server should detect active socket and raise IPCError
            server2 = IPCServer(socket_path=temp_socket_path, handler=handler)
            with pytest.raises(IPCError) as exc_info:
                server2.start()
            assert "already in use" in str(exc_info.value)
        finally:
            server1.stop()

    def test_server_start_stop_idempotent(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        server.stop()  # stopping non-running server is a no-op
        assert not server.is_running

        server.start()
        assert server.is_running
        server.start()  # starting already running server is a no-op
        assert server.is_running

        server.stop()
        assert not server.is_running
        server.stop()
        assert not server.is_running

    def test_server_socket_close_exception_on_stop(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        server.start()
        mock_sock = MagicMock()
        mock_sock.close.side_effect = OSError("Close error")
        server._server_socket = mock_sock
        server.stop()
        assert not server.is_running

    def test_server_receives_malformed_json(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(temp_socket_path))
            sock.sendall(b"not json at all\n")
            
            raw_res = sock.recv(4096)
            sock.close()
            assert b"Invalid JSON in request" in raw_res or b"InvalidMessageError" in raw_res

    def test_server_receives_unknown_command(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(temp_socket_path))
            sock.sendall(b'{"command": "nonexistent_command_xyz"}\n')
            
            raw_res = sock.recv(4096)
            sock.close()
            assert b"UnknownCommandError" in raw_res or b"Unknown command" in raw_res

    def test_server_client_empty_recv_disconnect(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(temp_socket_path))
            # Close connection immediately without sending data
            sock.close()
            time.sleep(0.05)

    def test_server_bind_failure(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        with patch.object(socket.socket, "bind", side_effect=OSError("Mock bind error")):
            with pytest.raises(IPCError) as exc_info:
                server.start()
            assert "Failed to bind IPC server socket" in str(exc_info.value)

    def test_server_chmod_warning(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        with patch("os.chmod", side_effect=OSError("Chmod failure")):
            server.start()
            assert server.is_running
            server.stop()

    def test_server_stale_socket_unlink_failure(self, temp_socket_path):
        temp_socket_path.touch()
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        with patch.object(Path, "unlink", side_effect=OSError("Unlink failure")):
            with pytest.raises(IPCError) as exc_info:
                server.start()
            assert "Failed to remove stale socket" in str(exc_info.value)

    def test_server_stop_socket_unlink_warning(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.ok()

        server = IPCServer(socket_path=temp_socket_path, handler=handler)
        server.start()
        with patch.object(Path, "unlink", side_effect=OSError("Unlink stop error")):
            server.stop()
            assert not server.is_running

    def test_client_helper_methods_raise_on_failure_response(self, temp_socket_path):
        def handler(req: Request) -> Response:
            return Response.fail("Service temporarily disabled")

        with IPCServer(socket_path=temp_socket_path, handler=handler):
            client = IPCClient(socket_path=temp_socket_path)
            with pytest.raises(IPCError) as exc_info:
                client.get_status()
            assert "Failed to get status: Service temporarily disabled" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                client.get_metrics()
            assert "Failed to get metrics: Service temporarily disabled" in str(exc_info.value)

            with pytest.raises(IPCError) as exc_info:
                client.get_config()
            assert "Failed to get config: Service temporarily disabled" in str(exc_info.value)

    def test_client_permission_error_handling(self, temp_socket_path):
        client = IPCClient(socket_path=temp_socket_path)
        with patch.object(socket.socket, "connect", side_effect=PermissionError("Permission denied")):
            with pytest.raises(IPCPermissionError):
                client.send_request(Request(command=Command.PING))

    def test_client_connect_socket_timeout(self, temp_socket_path):
        client = IPCClient(socket_path=temp_socket_path)
        with patch.object(socket.socket, "connect", side_effect=socket.timeout("Connect timeout")):
            with pytest.raises(IPCTimeoutError):
                client.send_request(Request(command=Command.PING))

    def test_client_connect_generic_os_error(self, temp_socket_path):
        client = IPCClient(socket_path=temp_socket_path)
        with patch.object(socket.socket, "connect", side_effect=OSError("Generic OS error")):
            with pytest.raises(IPCConnectionError):
                client.send_request(Request(command=Command.PING))

    def test_client_closed_connection_without_data(self, temp_socket_path):
        # Create server that immediately closes accepted connection
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(temp_socket_path))
        sock.listen(1)

        def close_worker():
            conn, _ = sock.accept()
            conn.close()

        t = threading.Thread(target=close_worker, daemon=True)
        t.start()

        try:
            client = IPCClient(socket_path=temp_socket_path)
            with pytest.raises(IPCConnectionError):
                client.send_request(Request(command=Command.PING))
        finally:
            t.join(timeout=1.0)
            sock.close()
            temp_socket_path.unlink(missing_ok=True)

    def test_client_malformed_response_payload(self, temp_socket_path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(temp_socket_path))
        sock.listen(1)

        def reply_bad_json():
            conn, _ = sock.accept()
            conn.recv(1024)
            conn.sendall(b"not valid json\n")
            conn.close()

        t = threading.Thread(target=reply_bad_json, daemon=True)
        t.start()

        try:
            client = IPCClient(socket_path=temp_socket_path)
            with pytest.raises(IPCError) as exc_info:
                client.send_request(Request(command=Command.PING))
            assert "Malformed response" in str(exc_info.value)
        finally:
            t.join(timeout=1.0)
            sock.close()
            temp_socket_path.unlink(missing_ok=True)

    def test_client_response_without_trailing_newline(self, temp_socket_path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(temp_socket_path))
        sock.listen(1)

        def reply_without_newline():
            conn, _ = sock.accept()
            conn.recv(1024)
            conn.sendall(b'{"success": true, "data": {"raw": 1}}')
            conn.close()

        t = threading.Thread(target=reply_without_newline, daemon=True)
        t.start()

        try:
            client = IPCClient(socket_path=temp_socket_path)
            res = client.send_request(Request(command=Command.PING))
            assert res.success is True
            assert res.data == {"raw": 1}
        finally:
            t.join(timeout=1.0)
            sock.close()
            temp_socket_path.unlink(missing_ok=True)
