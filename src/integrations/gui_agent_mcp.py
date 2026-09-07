"""Thin MCP transport for the existing GUI-agent runtime; no planning loop."""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import uuid

from omniflow.core.model import Observation
from omniflow.runtime.control import ExecutionStopped
from omniflow.runtime.engine import OmniFlow
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.integrations.android_world.oob_control import OobControlClient
from src.integrations.gui_agent_oob_host import OobGuiAgentHost
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


class GuiAgentMcp:
    def __init__(self, runtime: GuiAgentToolRuntime):
        self.runtime = runtime

    async def list_tools(self, _context=None, _params=None):
        from mcp import types
        return types.ListToolsResult(tools=[types.Tool(
            name=tool.name, description=tool.description, input_schema=tool.input_schema)
            for tool in self.runtime.function_tools()])

    async def call_tool(self, _context, params):
        from mcp import types
        name, args = params.name, params.arguments or {}
        try:
            image_data = None
            execution = await self.runtime.call_function_tool(name, args)
            result = {"name": name, "success": execution.success,
                      "error": execution.error, "feedback": execution.output.get("feedback")}
            if name == "omniflow_recall":
                result.update({key: execution.output[key] for key in ("functions", "audit", "task_id")
                               if key in execution.output})
            observation = execution.output.get("observation")
            if observation is None:
                observation = (execution.output.get("feedback") or {}).get("observation")
            if observation:
                image_data = _image(Observation.from_value(observation))
            result = {"session_id": self.runtime.session_id, **result}
            content = [types.TextContent(text=json.dumps(result, ensure_ascii=False))]
            if image_data:
                content.append(types.ImageContent(data=image_data, mime_type="image/png"))
            return types.CallToolResult(content=content, is_error=not execution.success)
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
    return Server("omniflow", version="1.1.0.dev1", on_list_tools=transport.list_tools,
                  on_call_tool=transport.call_tool,
                  instructions="Use omniflow_recall then omniflow_execute. The host owns planning, primitive actions, task completion and cancellation. Function success is not task success.")


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
                  evidence_root: Path | None = None, timeout_seconds: float = 600,
                  host_factory=None):
    root = evidence_root or Path.cwd() / "data/runtime/gui_agent" / uuid.uuid4().hex
    if store is not None and not store.is_file():
        raise ValueError("explicit_function_store_not_found")
    states = load_transfer_state_catalog(store.parent / "transfer_states.json") if store else {}
    if host_factory is None:
        host = OobGuiAgentHost(OobControlClient(None, adb_serial=device, adb_path=adb_path),
                              evidence_root=root, source_states=states)
    else:
        host = host_factory(device=device, evidence_root=root, source_states=states)
        if not all(callable(getattr(host, name, None)) for name in ("observe", "act", "get_state")):
            raise TypeError("host_factory_must_implement_observe_act_get_state")
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
    parser.add_argument("--host-factory", help="Explicit Python module:callable implementing Host; default is OOB")
    args = parser.parse_args()

    async def serve():
        from mcp.server.stdio import stdio_server
        factory = None
        if args.host_factory:
            module, separator, attribute = args.host_factory.partition(":")
            if not separator or not module or not attribute:
                raise ValueError("host_factory_requires_module_colon_callable")
            factory = getattr(importlib.import_module(module), attribute)
            if not callable(factory):
                raise TypeError("host_factory_not_callable")
        runtime = build_runtime(device=args.device, store=args.store, adb_path=args.adb_path,
                                evidence_root=args.evidence_root, timeout_seconds=args.timeout_seconds,
                                host_factory=factory)
        server = create_server(runtime)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    with device_lease(args.device):
        asyncio.run(serve())


if __name__ == "__main__":
    main()
