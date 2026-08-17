"""Tiny stdio MCP server used by the whole-task Luna AndroidWorld agent."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

TOOLS = [
    {
        "name": "androidworld_observe",
        "description": "Read official AndroidWorld state, accessibility XML, and the current screenshot.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "androidworld_act",
        "description": (
            "Execute exactly one native AndroidWorld action. Use {action:{tool,args}}. "
            "For click, long_press, and input_text, x/y are canonical 0..1000 "
            "coordinates, not screenshot pixels; convert using the display size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "object",
                    "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
]


def _rpc_write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _rpc_read() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    if line.lower().startswith(b"content-length:"):
        length = int(line.split(b":", 1)[1].strip())
        while True:
            header = sys.stdin.buffer.readline()
            if header in (b"\r\n", b"\n", b""):
                break
        raw = sys.stdin.buffer.read(length)
    else:
        raw = line
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _socket_call(socket_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(600.0)
        connection.connect(socket_path)
        connection.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    result = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("androidworld_bridge_invalid_response")
    return result


def _tool_result(value: dict[str, Any], *, include_image: bool = False) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}
    ]
    state = value.get("state")
    pixels = state.get("pixels") if isinstance(state, dict) else None
    if include_image and isinstance(pixels, dict):
        path = Path(str(pixels.get("path") or ""))
        if path.is_file():
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "mimeType": str(pixels.get("mime_type") or "image/png"),
                }
            )
    return {"content": content, "isError": bool(value.get("error"))}


def main() -> int:
    socket_path = str(os.environ.get("OMNIFLOW_ANDROIDWORLD_BRIDGE_SOCKET") or "").strip()
    if not socket_path:
        print("OMNIFLOW_ANDROIDWORLD_BRIDGE_SOCKET is required", file=sys.stderr)
        return 2
    while True:
        request = _rpc_read()
        if request is None:
            return 0
        request_id = request.get("id")
        method = str(request.get("method") or "")
        if method == "notifications/initialized":
            continue
        if method == "initialize":
            _rpc_write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": str((request.get("params") or {}).get("protocolVersion") or "2024-11-05"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "androidworld-direct", "version": "1.0"},
                    },
                }
            )
            continue
        if method == "tools/list":
            _rpc_write({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
            continue
        if method != "tools/call":
            if request_id is not None:
                _rpc_write({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method_not_found:{method}"}})
            continue
        params = request.get("params") or {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        try:
            if name == "androidworld_observe":
                result = _tool_result(_socket_call(socket_path, {"op": "observe"}), include_image=True)
            elif name == "androidworld_act":
                action = arguments.get("action") if isinstance(arguments, dict) else None
                if not isinstance(action, dict):
                    raise ValueError("androidworld_act requires action={tool,args}")
                result = _tool_result(_socket_call(socket_path, {"op": "act", "action": action}))
            else:
                raise ValueError(f"unknown_tool:{name}")
        except Exception as error:  # noqa: BLE001
            result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
        _rpc_write({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    raise SystemExit(main())
