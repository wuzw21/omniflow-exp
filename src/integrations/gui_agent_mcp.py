"""Thin MCP transport for the existing GUI-agent runtime; no planning loop."""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager, contextmanager
import hashlib
import importlib
from io import BytesIO
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
                try:
                    image_data = _image(Observation.from_value(observation))
                except (OSError, ValueError):
                    # A preview failure must not erase completed action facts.
                    image_data = None
                if result.get("feedback"):
                    result["feedback"] = {**result["feedback"],
                        "observation": {**observation, "image_base64": None}}
            result = {"session_id": self.runtime.session_id, **result}
            content = [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
            if image_data:
                content.append(types.ImageContent(type="image", data=image_data[0], mime_type=image_data[1]))
            return types.CallToolResult(content=content, is_error=not execution.success)
        except (Exception, ExecutionStopped) as error:  # noqa: BLE001 -- MCP error boundary
            if isinstance(error, ExecutionStopped):
                self.runtime.cancel()
            return types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text=json.dumps({
                "session_id": self.runtime.session_id, "error": str(error),
                "automatic_retry": False, "session": self.runtime.session_status()}, ensure_ascii=False))])


def _image(observation: Observation) -> tuple[str, str] | None:
    """Send a bounded preview; original evidence and matcher inputs stay intact."""
    from PIL import Image

    path = observation.extra.get("screenshot_path")
    if path and Path(path).is_file():
        payload = Path(path).read_bytes()
    elif observation.image_base64:
        payload = base64.b64decode(observation.image_base64.split(",")[-1])
    else:
        return None
    with Image.open(BytesIO(payload)) as source:
        preview = source.convert("RGB")
        preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        output = BytesIO()
        preview.save(output, format="JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode("ascii"), "image/jpeg"


def create_server(runtime: GuiAgentToolRuntime):
    from mcp.server.lowlevel import Server
    transport = GuiAgentMcp(runtime)
    return Server("omniflow", version="1.1.0.dev1", on_list_tools=transport.list_tools,
                  on_call_tool=transport.call_tool,
                  instructions="Use omniflow_recall then omniflow_execute. The host owns planning, primitive actions, task completion and cancellation. Function success is not task success.")


@asynccontextmanager
async def runtime_socket(runtime: GuiAgentToolRuntime):
    """Connect a host to an already-owned runtime without creating another device."""
    import anyio
    from mcp import types
    from mcp.shared.message import SessionMessage

    connections = set()

    async def connection(reader, writer):
        task = asyncio.current_task()
        connections.add(task)
        incoming, server_read = anyio.create_memory_object_stream(16)
        server_write, outgoing = anyio.create_memory_object_stream(16)

        async def receive():
            async with incoming:
                while line := await reader.readline():
                    try:
                        message = types.jsonrpc_message_adapter.validate_json(line, by_name=False)
                        await incoming.send(SessionMessage(message))
                    except ValueError as error:
                        await incoming.send(error)

        async def send():
            async with outgoing:
                async for item in outgoing:
                    writer.write((item.message.model_dump_json(by_alias=True, exclude_unset=True) + "\n").encode())
                    await writer.drain()

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(receive)
                group.start_soon(send)
                server = create_server(runtime)
                await server.run(server_read, server_write, server.create_initialization_options())
                group.cancel_scope.cancel()
        finally:
            writer.close()
            await writer.wait_closed()
            connections.discard(task)

    # A private directory keeps the local socket inaccessible to other users.
    with tempfile.TemporaryDirectory(prefix="of-mcp-") as directory:
        path = str(Path(directory) / "runtime.sock")
        listener = await asyncio.start_unix_server(connection, path=path, limit=32 * 1024 * 1024)
        try:
            yield path
        finally:
            listener.close()
            await listener.wait_closed()
            runtime.cancel()
            for task in tuple(connections):
                task.cancel()
            if connections:
                await asyncio.gather(*tuple(connections), return_exceptions=True)


def connect_stdio(path: str) -> None:
    """Byte-only MCP bridge; device, Store and execution remain in the owner."""
    import socket
    import sys
    import threading

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.connect(path)

        def send():
            try:
                while line := sys.stdin.buffer.readline():
                    stream.sendall(line)
                stream.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        threading.Thread(target=send, daemon=True).start()
        while data := stream.recv(65536):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()


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
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--device", help="Explicit OOB device adb serial")
    destination.add_argument("--connect", help="Internal stdio bridge to an existing runtime socket")
    parser.add_argument("--store", type=Path, help="Explicit v2 Function Store; omit for Memory OFF")
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--host-factory", help="Explicit Python module:callable implementing Host; default is OOB")
    args = parser.parse_args()
    if args.connect:
        if args.store or args.host_factory:
            parser.error("--connect uses the owner's Store and Host")
        connect_stdio(args.connect)
        return

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
