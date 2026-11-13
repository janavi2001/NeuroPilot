"""Strict client boundary for an official GR00T policy server."""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import msgpack_numpy as mnp
import zmq


class GR00TRobotController:
    """Call GR00T over its official ZeroMQ-style policy-server protocol.

    This class deliberately has no scripted fallback. A server error or timeout
    is raised so a demo cannot claim GR00T control when GR00T is unavailable.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 15000) -> None:
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _pack(value: Any) -> bytes:
        return msgpack.packb(value, default=functools.partial(mnp.encode), use_bin_type=True)

    @staticmethod
    def _unpack(value: bytes) -> Any:
        return msgpack.unpackb(value, object_hook=mnp.decode, raw=False)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(self._pack(request))
            response = self._unpack(self.socket.recv())
        except zmq.error.Again as exc:
            raise TimeoutError(f"GR00T policy server timed out at {self.host}:{self.port}") from exc
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T policy server error: {response['error']}")
        return response

    def ping(self) -> bool:
        return self._call("ping").get("status") == "ok"

    def execute(self, instruction: str, observation: dict[str, Any]) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("GR00T requires a non-empty semantic instruction")
        response = self._call("get_action", {"observation": observation, "options": None})
        action, info = response
        return {"instruction": instruction, "action": action, "info": info}

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
