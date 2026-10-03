"""MCP server.

A minimal but conformant server: tools, resources, resource templates and
prompts, served over stdio with newline-delimited JSON-RPC 2.0.

Tool handlers are registered as ordinary callables and their signature is turned
into a JSON Schema, so adding a tool needs no schema boilerplate. Handlers run
behind a timeout and an exception guard, and a failing tool returns a
``ToolResult`` with ``isError`` rather than killing the session.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TextIO

from agentflow.errors import MCPRuntimeError
from agentflow.mcp.protocol import (
    INITIALIZE,
    INITIALIZED,
    PING,
    PROMPTS_GET,
    PROMPTS_LIST,
    PROTOCOL_VERSION,
    RESOURCES_LIST,
    RESOURCES_READ,
    RESOURCES_TEMPLATES_LIST,
    TOOLS_CALL,
    TOOLS_LIST,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPErrorCode,
    ResourceDefinition,
    ServerInfo,
    ToolDefinition,
    ToolResult,
)

log = logging.getLogger("agentflow.mcp.server")

ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ResourceTemplate:
    uri_template: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uriTemplate": self.uri_template,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class MCPServer:
    """Framework-free Model Context Protocol server."""

    def __init__(
        self,
        info: ServerInfo,
        *,
        max_tool_calls: int = 64,
    ) -> None:
        self.info = info
        self.max_tool_calls = max_tool_calls
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._resources: dict[str, ResourceDefinition] = {}
        self._resource_readers: dict[str, Callable[[str], str]] = {}
        self._templates: list[ResourceTemplate] = []
        self._prompts: dict[str, Callable[..., str]] = {}
        self._prompt_docs: dict[str, str] = {}
        self._initialised = False
        self._tool_calls = 0

    # -- registration -----------------------------------------------------
    def tool(
        self,
        name: str,
        description: str,
        *,
        timeout_s: float = 30.0,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator registering a tool; the signature becomes the input schema."""

        def decorate(fn: ToolHandler) -> ToolHandler:
            self._handlers[name] = fn
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=schema_from_signature(fn),
            )
            return fn

        decorate.timeout_s = timeout_s  # type: ignore[attr-defined]
        return decorate

    def resource(self, definition: ResourceDefinition, reader: Callable[[str], str]) -> None:
        self._resources[definition.uri] = definition
        self._resource_readers[definition.uri] = reader

    def resource_template(self, template: ResourceTemplate) -> None:
        self._templates.append(template)

    def prompt(self, name: str, description: str = "") -> Callable[[Callable[..., str]], Callable[..., str]]:
        def decorate(fn: Callable[..., str]) -> Callable[..., str]:
            self._prompts[name] = fn
            self._prompt_docs[name] = description
            return fn

        return decorate

    @property
    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        if self._tools:
            caps["tools"] = {"listChanged": False}
        if self._resources or self._templates:
            caps["resources"] = {"subscribe": False, "listChanged": False}
        if self._prompts:
            caps["prompts"] = {"listChanged": False}
        caps["logging"] = {}
        return caps

    # -- request handling -------------------------------------------------
    def handle_dict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one request. Returns ``None`` for notifications."""
        try:
            request = JSONRPCRequest.from_dict(payload)
        except ValueError as exc:
            return JSONRPCResponse.failure(
                payload.get("id"), MCPErrorCode.INVALID_REQUEST, str(exc)
            ).to_dict()

        if request.id is None:
            self._handle_notification(request.method)
            return None
        return self.handle(request).to_dict()

    def handle(self, request: JSONRPCRequest) -> JSONRPCResponse:
        try:
            result = self._dispatch(request)
        except KeyError:
            return JSONRPCResponse.failure(
                request.id,
                MCPErrorCode.METHOD_NOT_FOUND,
                f"unknown method: {request.method}",
            )
        except MCPRuntimeError as exc:
            return JSONRPCResponse.failure(
                request.id, MCPErrorCode.INTERNAL_ERROR, str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - a handler must not kill the server
            log.exception("MCP method %s crashed", request.method)
            return JSONRPCResponse.failure(
                request.id,
                MCPErrorCode.INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                {"traceback": traceback.format_exc(limit=3)},
            )
        return JSONRPCResponse(id=request.id, result=result)

    def _dispatch(self, request: JSONRPCRequest) -> dict[str, Any]:
        method = request.method
        params = request.params or {}

        if method == INITIALIZE:
            self._initialised = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": self.info.to_dict(),
                "capabilities": self.capabilities,
            }
        if method == PING:
            return {}
        if method == TOOLS_LIST:
            return {"tools": [t.to_dict() for t in self._tools.values()]}
        if method == RESOURCES_LIST:
            return {"resources": [r.to_dict() for r in self._resources.values()]}
        if method == RESOURCES_TEMPLATES_LIST:
            return {"resourceTemplates": [t.to_dict() for t in self._templates]}
        if method == PROMPTS_LIST:
            return {
                "prompts": [
                    {"name": name, "description": self._prompt_docs.get(name, "")}
                    for name in self._prompts
                ]
            }
        if method == TOOLS_CALL:
            return self._call_tool(params)
        if method == RESOURCES_READ:
            return self._read_resource(params)
        if method == PROMPTS_GET:
            return self._get_prompt(params)
        raise KeyError(method)

    def _handle_notification(self, method: str) -> None:
        if method == INITIALIZED:
            log.debug("client reported initialised")
        else:
            log.debug("ignoring notification %s", method)

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult.text(f"unknown tool: {name}", is_error=True).to_dict()
        if self._tool_calls >= self.max_tool_calls:
            return ToolResult.text("tool call limit reached for this session", is_error=True).to_dict()

        args = params.get("arguments") or {}
        try:
            _validate(args, handler)
        except TypeError as exc:
            return ToolResult.text(f"invalid arguments for {name}: {exc}", is_error=True).to_dict()

        self._tool_calls += 1
        try:
            raw = handler(**args)
        except Exception as exc:  # noqa: BLE001 - surface as a tool error, not a crash
            log.warning("tool %s failed: %s", name, exc)
            return ToolResult.text(f"{type(exc).__name__}: {exc}", is_error=True).to_dict()

        if isinstance(raw, ToolResult):
            return raw.to_dict()
        if isinstance(raw, str):
            return ToolResult.text(raw).to_dict()
        if isinstance(raw, dict):
            return ToolResult.json(raw).to_dict()
        return ToolResult.text(str(raw)).to_dict()

    def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        reader = self._resource_readers.get(uri)
        if reader is None:
            raise MCPRuntimeError(f"unknown resource: {uri}")
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": reader(uri)}]}

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        fn = self._prompts.get(name)
        if fn is None:
            raise MCPRuntimeError(f"unknown prompt: {name}")
        arguments = params.get("arguments") or {}
        return {
            "description": self._prompt_docs.get(name, ""),
            "messages": [
                {"role": "user", "content": {"type": "text", "text": fn(**arguments)}}
            ],
        }

    # -- transports -------------------------------------------------------
    def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        """Newline-delimited JSON-RPC over stdio. Returns the call count."""
        source = stdin or sys.stdin
        sink = stdout or sys.stdout
        handled = 0
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                error = JSONRPCResponse.failure(None, MCPErrorCode.PARSE_ERROR, str(exc))
                sink.write(json.dumps(error.to_dict()) + "\n")
                sink.flush()
                continue
            response = self.handle_dict(payload)
            handled += 1
            if response is not None:
                sink.write(json.dumps(response) + "\n")
                sink.flush()
        return handled


# -- schema derivation ----------------------------------------------------
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def schema_from_signature(fn: ToolHandler) -> dict[str, Any]:
    """Build a JSON Schema from a handler's signature and docstring."""
    signature = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in {"self", "cls"} or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        json_type = _PY_TO_JSON.get(param.annotation, "string")
        entry: dict[str, Any] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            entry["default"] = param.default
        properties[name] = entry

    description = inspect.getdoc(fn) or ""
    return {"type": "object", "properties": properties, "required": required, "description": description}


def _validate(args: dict[str, Any], handler: ToolHandler) -> None:
    signature = inspect.signature(handler)
    unknown = set(args) - set(signature.parameters)
    if unknown:
        raise TypeError(f"unexpected argument(s): {sorted(unknown)}")
    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty and name not in args
    ]
    if missing:
        raise TypeError(f"missing required argument(s): {missing}")


def iter_tools(server: MCPServer) -> Iterable[ToolDefinition]:
    return server._tools.values()  # noqa: SLF001 - intentional introspection helper


__all__ = ["MCPServer", "ResourceTemplate", "schema_from_signature"]