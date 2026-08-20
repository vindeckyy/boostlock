"""
IPC commands and request-response encoding.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union


class ProtocolError(Exception):
    """Base exception for IPC protocol errors."""
    pass


class InvalidMessageError(ProtocolError):
    """Raised when an IPC message payload is malformed or invalid."""
    pass


class UnknownCommandError(ProtocolError):
    """Raised when an unrecognized command string is received."""
    pass


class Command(str, Enum):
    """Supported IPC command verbs."""

    PING = "ping"
    STATUS = "status"
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RECONFIGURE = "reconfigure"
    LOCK = "lock"
    UNLOCK = "unlock"
    METRICS = "metrics"
    CONFIG = "config"

    @classmethod
    def from_str(cls, val: Union[str, Command]) -> Command:
        """Parse command string or enum instance into Command."""
        if isinstance(val, Command):
            return val
        cleaned = str(val).strip().lower()
        for cmd in cls:
            if cmd.value == cleaned:
                return cmd
        raise UnknownCommandError(f"Unknown command: '{val}'")


@dataclass(init=False)
class Request:
    """IPC Request message structure."""

    command: Command
    args: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    @property
    def params(self) -> Dict[str, Any]:
        """Legacy alias for args (kept for CLI/dashboard compatibility)."""
        return self.args

    @params.setter
    def params(self, value: Optional[Dict[str, Any]]) -> None:
        self.args = dict(value) if value is not None else {}

    def __init__(
        self,
        command: Command,
        args: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        # Handle legacy params alias
        if params is not None:
            if args is not None:
                raise TypeError("Request accepts either 'args' or 'params', not both") 
            args = params
        if args is None:
            args = {}
        # Allow string command for convenience (CLI passes string)
        if isinstance(command, Command):
            self.command = command
        else:
            try:
                self.command = Command.from_str(command)
            except UnknownCommandError:
                self.command = command  # type: ignore
        self.args = dict(args)
        self.request_id = str(request_id) if request_id is not None else str(uuid.uuid4())
        self.timestamp = float(timestamp) if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert Request to dictionary representation."""
        return {
            "command": self.command.value if isinstance(self.command, Command) else str(self.command),
            "args": self.args,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Request:
        """Deserialize Request from dictionary."""
        if not isinstance(data, dict):
            raise InvalidMessageError("Request payload must be a JSON object (dict)")
        if "command" not in data:
            raise InvalidMessageError("Missing required 'command' field in Request")
        
        raw_cmd = data["command"]
        cmd = Command.from_str(raw_cmd)

        raw_args = data.get("args", {})
        if not isinstance(raw_args, dict):
            raise InvalidMessageError("'args' field in Request must be a JSON object (dict)")

        return cls(
            command=cmd,
            args=raw_args,
            request_id=str(data.get("request_id") or str(uuid.uuid4())),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class Response:
    """IPC Response message structure."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def ok(cls, data: Optional[Any] = None, request_id: Optional[str] = None) -> Response:
        """Construct successful Response."""
        return cls(success=True, data=data, request_id=request_id)

    @classmethod
    def fail(
        cls,
        error: str,
        error_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Response:
        """Construct failure Response."""
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            request_id=request_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Response to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Response:
        """Deserialize Response from dictionary."""
        if not isinstance(data, dict):
            raise InvalidMessageError("Response payload must be a JSON object (dict)")
        if "success" not in data:
            raise InvalidMessageError("Missing required 'success' field in Response")

        return cls(
            success=bool(data["success"]),
            data=data.get("data"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            request_id=data.get("request_id"),
            timestamp=float(data.get("timestamp", time.time())),
        )


def encode_request(req: Request) -> bytes:
    """Serialize Request to newline-terminated UTF-8 JSON bytes."""
    return (json.dumps(req.to_dict()) + "\n").encode("utf-8")


def decode_request(raw_bytes: bytes) -> Request:
    """Deserialize newline-terminated UTF-8 JSON bytes into Request."""
    raw_str = raw_bytes.decode("utf-8").strip()
    if not raw_str:
        raise InvalidMessageError("Empty request payload")
    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise InvalidMessageError(f"Invalid JSON in request: {exc}") from exc
    return Request.from_dict(data)


def encode_response(res: Response) -> bytes:
    """Serialize Response to newline-terminated UTF-8 JSON bytes."""
    return (json.dumps(res.to_dict()) + "\n").encode("utf-8")


def decode_response(raw_bytes: bytes) -> Response:
    """Deserialize newline-terminated UTF-8 JSON bytes into Response."""
    raw_str = raw_bytes.decode("utf-8").strip()
    if not raw_str:
        raise InvalidMessageError("Empty response payload")
    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise InvalidMessageError(f"Invalid JSON in response: {exc}") from exc
    return Response.from_dict(data)
