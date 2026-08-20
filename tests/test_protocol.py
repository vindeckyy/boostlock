"""
Unit tests for BoostLock IPC protocol, command parsing, and message serialization (FEAT-05).
"""

import json
import pytest
import time
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


class TestCommandEnum:
    """Tests for Command enum values and string parsing."""

    def test_command_values(self):
        assert Command.PING.value == "ping"
        assert Command.STATUS.value == "status"
        assert Command.START.value == "start"
        assert Command.STOP.value == "stop"
        assert Command.PAUSE.value == "pause"
        assert Command.RESUME.value == "resume"
        assert Command.RECONFIGURE.value == "reconfigure"
        assert Command.LOCK.value == "lock"
        assert Command.UNLOCK.value == "unlock"
        assert Command.METRICS.value == "metrics"
        assert Command.CONFIG.value == "config"

    def test_command_from_string(self):
        assert Command.from_str("ping") == Command.PING
        assert Command.from_str("PING") == Command.PING
        assert Command.from_str("  Status  ") == Command.STATUS
        assert Command.from_str(Command.STOP) == Command.STOP

    def test_command_from_invalid_string(self):
        with pytest.raises(UnknownCommandError) as exc_info:
            Command.from_str("invalid_command")
        assert "Unknown command: 'invalid_command'" in str(exc_info.value)


class TestRequest:
    """Tests for Request data structure and serialization."""

    def test_request_creation_defaults(self):
        req = Request(command=Command.PING)
        assert req.command == Command.PING
        assert req.args == {}
        assert req.request_id is not None
        assert isinstance(req.timestamp, float)
        assert req.timestamp > 0

    def test_request_creation_with_args(self):
        req = Request(
            command=Command.RECONFIGURE,
            args={"target_frequency_khz": 4200000, "governor": "performance"},
            request_id="req-1234",
            timestamp=1000.0,
        )
        assert req.command == Command.RECONFIGURE
        assert req.args["target_frequency_khz"] == 4200000
        assert req.request_id == "req-1234"
        assert req.timestamp == 1000.0

    def test_request_to_dict_and_from_dict(self):
        req = Request(
            command="status",
            args={"detailed": True},
            request_id="abc-1",
            timestamp=123.456,
        )
        d = req.to_dict()
        assert d["command"] == "status"
        assert d["args"] == {"detailed": True}
        assert d["request_id"] == "abc-1"
        assert d["timestamp"] == 123.456

        req2 = Request.from_dict(d)
        assert req2.command == Command.STATUS
        assert req2.args == {"detailed": True}
        assert req2.request_id == "abc-1"
        assert req2.timestamp == 123.456

    def test_request_from_dict_missing_command(self):
        with pytest.raises(InvalidMessageError):
            Request.from_dict({"args": {}})

    def test_request_from_dict_non_dict_args(self):
        with pytest.raises(InvalidMessageError):
            Request.from_dict({"command": "ping", "args": "invalid"})


class TestResponse:
    """Tests for Response data structure and serialization."""

    def test_response_success(self):
        res = Response.ok(data={"freq": 4000000}, request_id="req-1")
        assert res.success is True
        assert res.data == {"freq": 4000000}
        assert res.error is None
        assert res.error_type is None
        assert res.request_id == "req-1"
        assert isinstance(res.timestamp, float)

    def test_response_error(self):
        res = Response.fail(
            error="Daemon is busy",
            error_type="BusyError",
            request_id="req-2",
        )
        assert res.success is False
        assert res.data is None
        assert res.error == "Daemon is busy"
        assert res.error_type == "BusyError"
        assert res.request_id == "req-2"

    def test_response_to_dict_and_from_dict(self):
        res = Response(
            success=True,
            data={"status": "running"},
            error=None,
            error_type=None,
            request_id="req-3",
            timestamp=500.0,
        )
        d = res.to_dict()
        assert d["success"] is True
        assert d["data"] == {"status": "running"}
        assert d["request_id"] == "req-3"

        res2 = Response.from_dict(d)
        assert res2.success is True
        assert res2.data == {"status": "running"}
        assert res2.request_id == "req-3"
        assert res2.timestamp == 500.0

    def test_response_from_dict_missing_success(self):
        with pytest.raises(InvalidMessageError):
            Response.from_dict({"data": "something"})


class TestFramingAndEncoding:
    """Tests for message encoding and decoding over byte streams."""

    def test_encode_and_decode_request(self):
        req = Request(command=Command.START, args={"force": True}, request_id="id-1")
        encoded = encode_request(req)
        assert encoded.endswith(b"\n")
        
        decoded = decode_request(encoded)
        assert decoded.command == Command.START
        assert decoded.args == {"force": True}
        assert decoded.request_id == "id-1"

    def test_encode_and_decode_response(self):
        res = Response.ok(data={"cpus": [0, 1, 2, 3]}, request_id="id-2")
        encoded = encode_response(res)
        assert encoded.endswith(b"\n")

        decoded = decode_response(encoded)
        assert decoded.success is True
        assert decoded.data == {"cpus": [0, 1, 2, 3]}
        assert decoded.request_id == "id-2"

    def test_decode_invalid_json(self):
        with pytest.raises(InvalidMessageError):
            decode_request(b"not json\n")

        with pytest.raises(InvalidMessageError):
            decode_response(b"{broken json\n")

    def test_decode_non_dict_json(self):
        with pytest.raises(InvalidMessageError):
            decode_request(b"\"just a string\"\n")

        with pytest.raises(InvalidMessageError):
            decode_response(b"[1, 2, 3]\n")

    def test_decode_empty_bytes(self):
        with pytest.raises(InvalidMessageError):
            decode_request(b"")

        with pytest.raises(InvalidMessageError):
            decode_response(b"   \n")
