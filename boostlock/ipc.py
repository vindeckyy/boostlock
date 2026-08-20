"""
Unix Domain Socket IPC Server & Client for BoostLock (FEAT-05).
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from boostlock.protocol import (
    Command,
    InvalidMessageError,
    ProtocolError,
    Request,
    Response,
    UnknownCommandError,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)

logger = logging.getLogger(__name__)


class IPCError(Exception):
    """Base exception for BoostLock IPC operations."""
    pass


class IPCConnectionError(IPCError):
    """Raised when connecting to or communicating with the IPC server fails."""
    pass


class IPCTimeoutError(IPCError, TimeoutError):
    """Raised when an IPC operation times out."""
    pass


class IPCPermissionError(IPCError, PermissionError):
    """Raised when accessing the IPC socket fails due to permissions."""
    pass


class IPCServer:
    """
    Unix Domain Socket Server for daemon IPC.
    Handles concurrent client connections and dispatches requests to a callback handler.
    """

    def __init__(
        self,
        socket_path: Union[str, Path],
        handler: Callable[[Request], Response],
        socket_permissions: int = 0o660,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        self.handler = handler
        self.socket_permissions = socket_permissions

        self._lock = threading.Lock()
        self._server_socket: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def start(self) -> None:
        """Initialize Unix domain socket, bind, and start listener thread."""
        with self._lock:
            if self._is_running:
                return

            self._ensure_clean_socket_path()

            self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_socket.settimeout(0.5)  # allow periodic stop checking
            
            try:
                self._server_socket.bind(str(self.socket_path))
                try:
                    os.chmod(self.socket_path, self.socket_permissions)
                except OSError as e:
                    logger.warning(f"Could not set permissions on socket {self.socket_path}: {e}")

                self._server_socket.listen(128)
            except Exception as exc:
                self._server_socket.close()
                self._server_socket = None
                raise IPCError(f"Failed to bind IPC server socket at {self.socket_path}: {exc}") from exc

            self._stop_event.clear()
            self._is_running = True

            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                name="BoostLockIPCServer",
                daemon=True,
            )
            self._listener_thread.start()
            logger.info(f"IPC server listening on {self.socket_path}")

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop IPC server and remove socket file."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()

        if self._listener_thread is not None:
            self._listener_thread.join(timeout=timeout_s)

        with self._lock:
            if self._server_socket is not None:
                try:
                    self._server_socket.close()
                except Exception:
                    pass
                self._server_socket = None

            if self.socket_path.exists():
                try:
                    self.socket_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove socket file {self.socket_path}: {e}")

        logger.info("IPC server stopped")

    def _ensure_clean_socket_path(self) -> None:
        """Ensure parent dir exists and cleanup stale socket file if not active."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            # Test if another server is actively listening
            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                test_sock.connect(str(self.socket_path))
                test_sock.close()
                raise IPCError(f"IPC socket is already in use by another running instance at {self.socket_path}")
            except (ConnectionRefusedError, FileNotFoundError, socket.error):
                # Stale socket file
                try:
                    self.socket_path.unlink(missing_ok=True)
                except Exception as exc:
                    raise IPCError(f"Failed to remove stale socket {self.socket_path}: {exc}") from exc
            finally:
                try:
                    test_sock.close()
                except Exception:
                    pass

    def _listen_loop(self) -> None:
        """Main connection acceptance loop."""
        while not self._stop_event.is_set():
            try:
                if self._server_socket is None:
                    break
                try:
                    conn, _ = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                )
                client_thread.start()
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.error(f"Error in IPC accept loop: {exc}")

    def _handle_client(self, conn: socket.socket) -> None:
        """Handle incoming client connection request/response."""
        try:
            conn.settimeout(10.0)
            buffer = b""
            while not self._stop_event.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    req_id: Optional[str] = None
                    try:
                        req = decode_request(line)
                        req_id = req.request_id
                        res = self.handler(req)
                    except InvalidMessageError as err:
                        res = Response.fail(error=str(err), error_type="InvalidMessageError", request_id=req_id)
                    except UnknownCommandError as err:
                        res = Response.fail(error=str(err), error_type="UnknownCommandError", request_id=req_id)
                    except Exception as err:
                        logger.exception(f"Unhandled error in IPC request handler: {err}")
                        res = Response.fail(
                            error=str(err),
                            error_type=err.__class__.__name__,
                            request_id=req_id,
                        )

                    conn.sendall(encode_response(res))
                    break
        except Exception as exc:
            logger.debug(f"IPC client handler connection error: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self) -> IPCServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class IPCClient:
    """
    Unix Domain Socket Client for interacting with the BoostLock daemon.
    """

    def __init__(
        self,
        socket_path: Union[str, Path] = "/var/run/boostlock/boostlock.sock",
        timeout_s: float = 5.0,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        self.timeout_s = timeout_s

    def is_daemon_running(self) -> bool:
        """Check if daemon is reachable and responding to PING."""
        if not self.socket_path.exists():
            return False
        try:
            res = self.ping(timeout=1.0)
            return res.success
        except Exception:
            return False

    def send_request(self, request: Request, timeout: Optional[float] = None) -> Response:
        """Send a Request message to the daemon and receive Response."""
        eff_timeout = timeout if timeout is not None else self.timeout_s
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(eff_timeout)

        try:
            sock.connect(str(self.socket_path))
        except FileNotFoundError as exc:
            sock.close()
            raise IPCConnectionError(f"IPC socket not found at {self.socket_path} (is daemon running?)") from exc
        except ConnectionRefusedError as exc:
            sock.close()
            raise IPCConnectionError(f"Connection refused at {self.socket_path} (is daemon running?)") from exc
        except PermissionError as exc:
            sock.close()
            raise IPCPermissionError(f"Permission denied accessing IPC socket at {self.socket_path}") from exc
        except socket.timeout as exc:
            sock.close()
            raise IPCTimeoutError(f"Timed out connecting to {self.socket_path}") from exc
        except Exception as exc:
            sock.close()
            raise IPCConnectionError(f"Failed to connect to IPC socket at {self.socket_path}: {exc}") from exc

        try:
            sock.sendall(encode_request(request))
            buffer = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    line, _ = buffer.split(b"\n", 1)
                    return decode_response(line)

            if buffer:
                return decode_response(buffer)
            raise IPCConnectionError("Server closed connection without responding")
        except socket.timeout as exc:
            raise IPCTimeoutError(f"Timed out waiting for response from {self.socket_path}") from exc
        except (InvalidMessageError, UnknownCommandError) as exc:
            raise IPCError(f"Malformed response received from daemon: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, (IPCConnectionError, IPCTimeoutError, IPCPermissionError, IPCError)):
                raise
            raise IPCConnectionError(f"Communication error with {self.socket_path}: {exc}") from exc
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def ping(self, timeout: Optional[float] = None) -> Response:
        """Send PING command."""
        return self.send_request(Request(command=Command.PING), timeout=timeout)

    def get_status(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch daemon status dictionary."""
        res = self.send_request(Request(command=Command.STATUS), timeout=timeout)
        if not res.success:
            raise IPCError(f"Failed to get status: {res.error}")
        return res.data or {}

    def start(self, timeout: Optional[float] = None) -> Response:
        """Send START command."""
        return self.send_request(Request(command=Command.START), timeout=timeout)

    def stop(self, timeout: Optional[float] = None) -> Response:
        """Send STOP command."""
        return self.send_request(Request(command=Command.STOP), timeout=timeout)

    def pause(self, timeout: Optional[float] = None) -> Response:
        """Send PAUSE command."""
        return self.send_request(Request(command=Command.PAUSE), timeout=timeout)

    def resume(self, timeout: Optional[float] = None) -> Response:
        """Send RESUME command."""
        return self.send_request(Request(command=Command.RESUME), timeout=timeout)

    def reconfigure(self, new_config: Dict[str, Any], timeout: Optional[float] = None) -> Response:
        """Send RECONFIGURE command with new parameters."""
        return self.send_request(Request(command=Command.RECONFIGURE, args=new_config), timeout=timeout)

    def lock(self, timeout: Optional[float] = None) -> Response:
        """Send LOCK command to re-engage boost clock pinning."""
        return self.send_request(Request(command=Command.LOCK), timeout=timeout)

    def unlock(self, timeout: Optional[float] = None) -> Response:
        """Send UNLOCK command to disengage boost pinning."""
        return self.send_request(Request(command=Command.UNLOCK), timeout=timeout)

    def get_metrics(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch detailed metrics dictionary."""
        res = self.send_request(Request(command=Command.METRICS), timeout=timeout)
        if not res.success:
            raise IPCError(f"Failed to get metrics: {res.error}")
        return res.data or {}

    def get_config(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch current configuration dictionary."""
        res = self.send_request(Request(command=Command.CONFIG), timeout=timeout)
        if not res.success:
            raise IPCError(f"Failed to get config: {res.error}")
        return res.data or {}
