"""Thin MCP transport for the existing GUI-agent runtime; no planning loop."""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from jsonschema import validate

from omniflow.core.model import Observation
from omniflow.runtime.control import ExecutionStopped
from omniflow.runtime.engine import OmniFlow
from omniflow.runtime.protocol import observation_payload
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.integrations.android_world.oob_control import OobControlClient
from src.integrations.gui_agent_oob_host import OobGuiAgentHost
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


def _object(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


_STRING = {"type": "string", "minLength": 1, "maxLength": 128}
_TOOLS = {
    "omniflow_tools": ("List the canonical action and registered Function schemas.", _object()),
    "omniflow_status": ("Read session identity, budget, and in-flight state without device I/O.", _object()),
    "omniflow_observe": ("Observe the current device through OOB; return one image and current UI.", _object()),
    "omniflow_execute": (
        "Execute one canonical tool or Function and return to the caller. Retry transport requests with the SAME request_id; never restart a partially executed Function blindly.",
        _object({"session_id": _STRING, "request_id": _STRING, "tool_name": _STRING,
                 "arguments": {"type": "object"}}, ("session_id", "request_id", "tool_name", "arguments"))),
    "omniflow_cancel": ("Close the current session; drain an already-dispatched operation and prohibit the next action.",
                        _object({"session_id": _STRING}, ("session_id",))),
    "omniflow_finish": ("Report task completion. A configured verifier is authoritative; without one the task remains unverified.",
                        _object({"session_id": _STRING, "content": {"type": "string", "minLength": 1}},
                                ("session_id", "content"))),
    "omniflow_start": ("Explicitly begin a NEW task after the previous session closes; this does not resume interrupted work.",
                       _object({"session_id": _STRING}, ("session_id",))),
}


class GuiAgentMcp:
    def __init__(self, runtime: GuiAgentToolRuntime):
        self.runtime = runtime

    async def list_tools(self, _context=None, _params=None):
        from mcp import types
        return types.ListToolsResult(tools=[types.Tool(
            name=name, description=description, input_schema=schema)
            for name, (description, schema) in _TOOLS.items()])

    async def call_tool(self, _context, params):
        from mcp import types
        name, args = params.name, params.arguments or {}
        try:
            if name not in _TOOLS:
                raise ValueError(f"unknown_mcp_tool:{name}")
            validate(args, _TOOLS[name][1])
            if "session_id" in args and args["session_id"] != self.runtime.session_id:
                raise ValueError("gui_agent_session_mismatch")
            image_data = None
            is_error = False
            if name == "omniflow_tools":
                result = {"tools": [tool.to_mcp_tool() for tool in self.runtime.list_tools()]}
            elif name == "omniflow_status":
                result = self.runtime.session_status()
            elif name == "omniflow_observe":
                observation = Observation.from_value(await self.runtime.aobserve())
                image_data = _image(observation)
                result = {"observation": observation_payload(observation)}
            elif name == "omniflow_execute":
                tool = next((item for item in self.runtime.list_tools() if item.name == args["tool_name"]), None)
                if tool is None:
                    raise ValueError("gui_agent_tool_unknown")
                validate(args["arguments"], tool.input_schema)
                execution = await self.runtime.execute_request(**args)
                # Do not repeat a full trace or an inline base64 image in text.
                result = {"name": execution.name, "success": execution.success,
                          "error": execution.error, "feedback": execution.output.get("feedback")}
                is_error = not execution.success
                observation = execution.output.get("observation")
                if observation is None:
                    observation = (execution.output.get("feedback") or {}).get("observation")
                if observation:
                    image_data = _image(Observation.from_value(observation))
            elif name == "omniflow_cancel":
                result = self.runtime.cancel()
            elif name == "omniflow_finish":
                result = await self.runtime.finish(args["content"])
            else:
                result = self.runtime.start_session()
            result = {"session_id": self.runtime.session_id, **result}
            content = [types.TextContent(text=json.dumps(result, ensure_ascii=False))]
            if image_data:
                content.append(types.ImageContent(data=image_data, mime_type="image/png"))
            return types.CallToolResult(content=content, is_error=is_error)
        except (Exception, ExecutionStopped) as error:  # noqa: BLE001 -- MCP error boundary
            if isinstance(error, ExecutionStopped):
                self.runtime.cancel()
            return types.CallToolResult(is_error=True, content=[types.TextContent(text=json.dumps({
                "session_id": self.runtime.session_id, "error": str(error),
                "automatic_retry": False, "session": self.runtime.session_status()}, ensure_ascii=False))])


def _image(observation: Observation) -> str | None:
    path = observation.extra.get("screenshot_path")
    if path and Path(path).is_file():
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return observation.image_base64


def create_server(runtime: GuiAgentToolRuntime):
    from mcp.server.lowlevel import Server
    transport = GuiAgentMcp(runtime)
    return Server("omniflow", version="1.1.0.dev0", on_list_tools=transport.list_tools,
                  on_call_tool=transport.call_tool,
                  instructions="Read omniflow_tools and omniflow_status first. The host owns task planning. Function success is not verified task success.")


@contextmanager
def device_lease(device: str):
    """Prevent two local MCP processes from owning the same configured serial."""
    import fcntl
    digest = hashlib.sha256(device.encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"omniflow-device-{digest}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("gui_agent_device_already_owned") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def build_runtime(*, device: str, store: Path | None = None, adb_path: str = "",
                  evidence_root: Path | None = None, timeout_seconds: float = 600):
    root = evidence_root or Path.cwd() / "data/runtime/gui_agent" / uuid.uuid4().hex
    if store is not None and not store.is_file():
        raise ValueError("explicit_function_store_not_found")
    states = load_transfer_state_catalog(store.parent / "transfer_states.json") if store else {}
    host = OobGuiAgentHost(OobControlClient(None, adb_serial=device, adb_path=adb_path),
                          evidence_root=root, source_states=states)
    # No Memory means an empty in-process store. This path is never saved or
    # populated by the runtime; no compiler, discovery scan, or catalog runs.
    empty_path = root / f"unused-empty-store-{uuid.uuid4().hex}.json"
    flow = OmniFlow(store or empty_path, host=host)
    if flow.store.load_errors:
        raise ValueError(f"function_store_invalid:{flow.store.load_errors}")
    return GuiAgentToolRuntime(host=host, flow=flow, timeout_seconds=timeout_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit OOB device adb serial")
    parser.add_argument("--store", type=Path, help="Explicit v2 Function Store; omit for Memory OFF")
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()

    async def serve():
        from mcp.server.stdio import stdio_server
        runtime = build_runtime(device=args.device, store=args.store, adb_path=args.adb_path,
                                evidence_root=args.evidence_root, timeout_seconds=args.timeout_seconds)
        server = create_server(runtime)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    with device_lease(args.device):
        asyncio.run(serve())


if __name__ == "__main__":
    main()
