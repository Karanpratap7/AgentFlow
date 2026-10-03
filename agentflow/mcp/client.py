"""MCP client with stdio transport, session management and tool discovery.

Two transports:

``StdioTransport``
    Launches a server subprocess and speaks newline-delimited JSON-RPC over its
    stdin/stdout. stderr is drained on a background thread so a chatty server can
    never deadlock the protocol channel.
``InProcessTransport``
    Calls a server object directly. Used by tests and by the embedded server so
    the agent loop can be exercised without spawning processes.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentflow.errors import MCPRuntimeError
from agentflow.mcp.protocol import (
    INITIALIZE,
    INITIALIZED,
    PING,
    PROMPTS_LIST,
    PROTOCOL_VERSION,
    RESOURCES_LIST,
    RESOURCES_READ,
    TOOLS_CALL,
    TOOLS_LIST,
    JSONRPCRequest,
    JSONRPCResponse,
    ToolResult,
)
from agentflow.mcp.server import MCPServer

log = logging.getLogger("agentflow.mcp.client")

CLIENT_INFO = {"name": "agentflow", "version": "0.4.0"}
CLIENT_CAPABILITIES = {"roots": {"listChanged": False}, "sampling": {}}


class Transport(Protocol):
    """Anything that can carry one JSON-RPC payload to a server."""

    def send(self, payload: dict[str, Any]) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


class InProcessTransport:
    """Dispatches straight into an :class:`MCPServer` instance."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.closed = False

    def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.closed:
            raise MCPRuntimeError("transport is closed")
        return self.server.handle_dict(payload)

    def close(self) -> None:
        self.closed = True


@dataclass
class StdioTransport:
    """Newline-delimited JSON-RPC over a child process's stdin/stdout."""

    command: list[str]
    cwd: str | Path | None = None
    env: dict[str, str] | None = None
    timeout_s: float = 30.0
    max_skipped_messages: int = 32
    process: subprocess.Popen[str] | None = field(default=None, init=False)
    _stderr_thread: threading.Thread | None = field(default=None, init=False)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            self.command = shlex.split(self.command)

    def start(self) -> None:
        if self.process is not None:
            return
        environment = {**os.environ, **(self.env or {})}
        try:
            self.process = subprocess.Popen(  # noqa: S603 - command is operator-supplied config
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(self.cwd) if self.cwd else None,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise MCPRuntimeError(f"cannot launch MCP server: {self.command[0]!r}") from exc

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process and self.process.stderr
        try:
            for line in self.process.stderr:
                log.debug("mcp server stderr: %s", line.rstrip())
        except ValueError:  # stderr closed while the parent is tearing down
            pass

    def _reply_to_server_request(self, request: dict[str, Any]) -> None:
        """Answer a server-initiated JSON-RPC request.

        Servers may ask the client for things we do not implement (the filesystem
        server issues ``roots/list``). Ignoring them outright hangs the server, so
        we answer with a spec-compliant ``-32601`` and carry on.
        """
        assert self.process and self.process.stdin
        error = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32601,
                "message": f"AgentFlow client does not implement {request.get('method')!r}",
            },
        }
        try:
            self.process.stdin.write(json.dumps(error) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError):  # pragma: no cover - server already gone
            pass

    def send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.start()
        assert self.process and self.process.stdin and self.process.stdout
        if self.process.poll() is not None:
            raise MCPRuntimeError(
                f"MCP server exited with code {self.process.returncode} before responding"
            )
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

        # A notification has no id and therefore no reply.
        if payload.get("id") is None:
            return None

        expected = payload["id"]
        method = payload.get("method", "?")
        # A blocking readline cannot be interrupted, so a watchdog kills the child if
        # it stops talking. That turns an unbounded hang into a clean, reportable error.
        watchdog = threading.Timer(self.timeout_s, self._kill_on_timeout, args=(method,))
        watchdog.daemon = True
        watchdog.start()
        try:
            return self._read_response(expected, method)
        finally:
            watchdog.cancel()

    def _kill_on_timeout(self, method: str) -> None:
        log.warning("mcp server did not answer %s within %.1fs; killing it", method, self.timeout_s)
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()

    def _read_response(self, expected: Any, method: str) -> dict[str, Any]:
        assert self.process and self.process.stdout
        # The stream is shared with server-initiated traffic, so keep reading until
        # our own response arrives rather than assuming it is the next line.
        for _ in range(self.max_skipped_messages):
            line = self.process.stdout.readline()
            if not line:
                raise MCPRuntimeError(f"MCP server closed stdout without answering {method!r}")
            message = json.loads(line)

            if message.get("method") and message.get("id") is not None:
                log.debug("mcp server request: %s", message["method"])
                self._reply_to_server_request(message)
                continue
            if message.get("method"):
                log.debug("mcp server notification: %s", message["method"])
                continue
            if message.get("id") != expected:
                log.debug("mcp unsolicited response id=%s", message.get("id"))
                continue
            return message

        raise MCPRuntimeError(
            f"no response to {method!r} after {self.max_skipped_messages} interleaved messages"
        )

    def close(self) -> None:
        if self.process is None:
            return
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:  # noqa: BLE001, S110 - best effort teardown
                pass
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            self.process.kill()
        self.process = None

    def __enter__(self) -> StdioTransport:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(slots=True)
class ToolProxy:
    """Client-side handle for one discovered MCP tool."""

    server: str
    definition: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.definition.get("name", ""))

    @property
    def description(self) -> str:
        return str(self.definition.get("description", ""))

    @property
    def input_schema(self) -> dict[str, Any]:
        return dict(self.definition.get("inputSchema") or {"type": "object", "properties": {}})

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.name}"

    def to_tool_schema(self) -> Any:
        """Adapt to the runtime's tool schema so agents can call it directly."""
        from agentflow.llm.types import ToolSchema

        return ToolSchema(
            name=self.qualified_name,
            description=self.description,
            input_schema=self.input_schema,
        )


class MCPClient:
    """Session-aware MCP client with tool discovery and prefixed tool names."""

    def __init__(self, name: str, transport: Transport, *, request_timeout_s: float = 30.0) -> None:
        self.name = name
        self.transport = transport
        self.request_timeout_s = request_timeout_s
        self._next_id = 0
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.protocol_version = PROTOCOL_VERSION
        self._tools: dict[str, ToolProxy] = {}

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def connect(
        cls,
        name: str,
        command: list[str] | str,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPClient:
        client = cls(name, StdioTransport(command=command, cwd=cwd, env=env))
        client.initialize()
        return client

    @classmethod
    def embed(cls, name: str, server: MCPServer) -> MCPClient:
        """Connect to an in-process server. No subprocess, no serialisation."""
        client = cls(name, InProcessTransport(server))
        client.initialize()
        return client

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            INITIALIZE,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": CLIENT_CAPABILITIES,
                "clientInfo": CLIENT_INFO,
            },
        )
        self.server_info = dict(result.get("serverInfo") or {})
        self.capabilities = dict(result.get("capabilities") or {})
        self.protocol_version = str(result.get("protocolVersion", PROTOCOL_VERSION))
        self.notify(INITIALIZED)
        return result

    def supports(self, capability: str) -> bool:
        """Whether the server advertised a capability during ``initialize``.

        The filesystem server, for instance, advertises only ``tools``; asking it for
        ``resources/list`` earns a ``-32601`` rather than an empty list.
        """
        if not self.capabilities:
            return True  # no advertisement: optimistically probe
        return capability in self.capabilities

    def ping(self) -> bool:
        try:
            self.request(PING)
            return True
        except MCPRuntimeError:
            return False

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> MCPClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- transport --------------------------------------------------------
    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        payload = JSONRPCRequest(method=method, params=params or {}, id=self._next_id).to_dict()
        raw = self.transport.send(payload)
        if raw is None:
            raise MCPRuntimeError(f"no response to {method}")
        response = JSONRPCResponse.from_dict(raw)
        if not response.ok:
            error = response.error or {}
            raise MCPRuntimeError(
                f"{method} failed [{error.get('code')}: {error.get('message')}]"
                + (f" data={error.get('data')}" if error.get("data") else "")
            )
        return dict(response.result or {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.transport.send(JSONRPCRequest(method=method, params=params or {}).to_dict())

    # -- discovery --------------------------------------------------------
    def list_tools(self, *, refresh: bool = False) -> list[ToolProxy]:
        if refresh or not self._tools:
            result = self.request(TOOLS_LIST)
            self._tools = {
                str(d.get("name")): ToolProxy(server=self.name, definition=dict(d))
                for d in result.get("tools", [])
            }
        return list(self._tools.values())

    def list_resources(self) -> list[dict[str, Any]]:
        if not self.supports("resources"):
            return []
        return list(self.request(RESOURCES_LIST).get("resources", []))

    def list_prompts(self) -> list[dict[str, Any]]:
        if not self.supports("prompts"):
            return []
        return list(self.request(PROMPTS_LIST).get("prompts", []))

    def read_resource(self, uri: str) -> str:
        result = self.request(RESOURCES_READ, {"uri": uri})
        return "\n".join(str(c.get("text", "")) for c in result.get("contents", []))

    # -- invocation -------------------------------------------------------
    def call_tool(self, tool_name: str, **arguments: Any) -> ToolResult:
        raw = self.request(TOOLS_CALL, {"name": tool_name, "arguments": arguments})
        return ToolResult(
            content=list(raw.get("content", [])),
            is_error=bool(raw.get("isError", False)),
            structured=raw.get("structuredContent"),
        )

    def call_proxy(self, proxy: ToolProxy, **arguments: Any) -> ToolResult:
        return self.call_tool(proxy.name, **arguments)

    def tool_schemas(self) -> list[Any]:
        """Every discovered tool as a runtime :class:`ToolSchema`, for agents."""
        return [p.to_tool_schema() for p in self.list_tools()]

    def index(self) -> dict[str, Any]:
        return {
            "server": self.name,
            "protocol_version": self.protocol_version,
            "server_info": self.server_info,
            "tools": [p.name for p in self.list_tools()],
            "resources": [r.get("uri") for r in self.list_resources()],
            "prompts": [p.get("name") for p in self.list_prompts()],
        }


def _as_argv(value: Any) -> list[str]:
    """Normalise a manifest command/args entry to argv.

    A JSON manifest naturally writes ``"command": "python3"`` as a string.
    ``list("python3")`` would silently produce ``['p', 'y', ...]`` and fail with a
    baffling "cannot launch 'p'".
    """
    if value is None:
        return []
    if isinstance(value, str):
        import shlex

        return shlex.split(value)
    return [str(v) for v in value]


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """One entry in the MCP manifest."""

    name: str
    command: list[str]
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def argv(self) -> list[str]:
        return [*self.command, *self.args]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MCPServerSpec:
        return cls(
            name=payload["name"],
            command=_as_argv(payload.get("command", [])),
            args=_as_argv(payload.get("args", [])),
            cwd=payload.get("cwd"),
            env=dict(payload.get("env", {})),
            description=payload.get("description", ""),
        )


class MCPClientPool:
    """Connects to every server in a manifest and aggregates their tools.

    Agents call tools by ``server.tool`` name, so a tool name collision between
    two servers is impossible by construction.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.clients: dict[str, MCPClient] = {}
        self.errors: dict[str, str] = {}

    @classmethod
    def from_manifest_dict(cls, payload: dict[str, Any]) -> MCPClientPool:
        pool = cls.__new__(cls)
        pool.manifest_path = Path("<inline>")
        pool.clients = {}
        pool.errors = {}
        pool._connect_all(payload.get("mcpServers", payload))
        return pool

    def _connect_all(self, servers: dict[str, Any]) -> None:
        for name, spec in servers.items():
            try:
                parsed = MCPServerSpec.from_dict({"name": name, **spec})
                self.clients[name] = MCPClient.connect(
                    name, parsed.argv, cwd=parsed.cwd, env=parsed.env
                )
            except (MCPRuntimeError, KeyError) as exc:
                log.warning("MCP server %s unavailable: %s", name, exc)
                self.errors[name] = str(exc)

    def connect_all(self) -> MCPClientPool:
        if self.manifest_path.exists():
            import json

            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._connect_all(payload.get("mcpServers", payload))
        return self

    def tools(self) -> list[ToolProxy]:
        return [tool for client in self.clients.values() for tool in client.list_tools()]

    def tool_schemas(self) -> list[Any]:
        return [tool.to_tool_schema() for tool in self.tools()]

    def invoke(self, qualified_name: str, **arguments: Any) -> ToolResult:
        server, _, tool = qualified_name.partition(".")
        client = self.clients.get(server)
        if client is None:
            raise MCPRuntimeError(f"unknown MCP server: {server}")
        return client.call_tool(tool, **arguments)

    def index(self) -> dict[str, Any]:
        return {
            "servers": {name: c.index() for name, c in self.clients.items()},
            "unavailable": dict(self.errors),
            "tool_count": sum(len(c.list_tools()) for c in self.clients.values()),
        }

    def close(self) -> None:
        for client in self.clients.values():
            client.close()

    def __enter__(self) -> MCPClientPool:
        return self.connect_all()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[MCPClient]:
        return iter(self.clients.values())

    def __len__(self) -> int:
        return len(self.clients)

    def names(self) -> list[str]:
        return list(self.clients)

    def __contains__(self, name: object) -> bool:
        return name in self.clients

    def get(self, name: str) -> MCPClient:
        try:
            return self.clients[name]
        except KeyError as exc:
            raise MCPRuntimeError(
                f"unknown MCP server {name!r}; connected: {self.names() or 'none'}"
            ) from exc


__all__ = [
    "InProcessTransport",
    "MCPClient",
    "MCPClientPool",
    "MCPServerSpec",
    "StdioTransport",
    "ToolProxy",
    "Transport",
]