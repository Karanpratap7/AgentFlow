"""MCP wire protocol: JSON-RPC 2.0 messages, MCP method names and result shapes.

Implemented directly rather than depending on an SDK, so the transport, framing
and error mapping are all visible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = "2025-06-18"
JSONRPC_VERSION = "2.0"

# Lifecycle
INITIALIZE = "initialize"
INITIALIZED = "notifications/initialized"
PING = "ping"

# Discovery
TOOLS_LIST = "tools/list"
TOOLS_CALL = "tools/call"
RESOURCES_LIST = "resources/list"
RESOURCES_READ = "resources/read"
RESOURCES_TEMPLATES_LIST = "resources/templates/list"
PROMPTS_LIST = "prompts/list"
PROMPTS_GET = "prompts/get"

# Server-initiated
LOGGING_SET_LEVEL = "logging/setLevel"
PROGRESS = "notifications/progress"
TOOL_LIST_CHANGED = "notifications/tools/list_changed"

STANDARD_ERRORS: dict[int, str] = {
    -32700: "Parse error",
    -32600: "Invalid Request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
}

ClientCapability = Literal["tools", "resources", "prompts", "logging", "roots"]


class MCPErrorCode:
    """JSON-RPC reserved codes plus the MCP extensions."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    RESOURCE_NOT_FOUND = -32002
    REQUEST_CANCELLED = -32800
    TOOL_EXECUTION_ERROR = -32003


@dataclass(slots=True)
class JSONRPCRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params:
            payload["params"] = self.params
        if self.id is not None:
            payload["id"] = self.id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JSONRPCRequest:
        if payload.get("jsonrpc") != JSONRPC_VERSION:
            raise ValueError(f"unsupported jsonrpc version: {payload.get('jsonrpc')!r}")
        return cls(
            method=str(payload.get("method", "")),
            params=payload.get("params") or {},
            id=payload.get("id"),
        )


@dataclass(slots=True)
class JSONRPCResponse:
    id: int | str | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            payload["error"] = self.error
        else:
            payload["result"] = self.result or {}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JSONRPCResponse:
        return cls(id=payload.get("id"), result=payload.get("result"), error=payload.get("error"))

    @classmethod
    def failure(cls, id_: int | str | None, code: int, message: str, data: Any = None) -> JSONRPCResponse:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return cls(id=id_, error=error)


@dataclass(slots=True)
class ServerInfo:
    name: str
    version: str
    instructions: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Capabilities are returned as a sibling of serverInfo in the
        # `initialize` result, not nested inside it.
        payload: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.instructions:
            payload["instructions"] = self.instructions
        return payload


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass(slots=True)
class ToolResult:
    """Result of ``tools/call``. Errors are *content*, not protocol errors."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    structured: dict[str, Any] | None = None

    @classmethod
    def text(cls, value: str, *, is_error: bool = False) -> ToolResult:
        return cls(content=[{"type": "text", "text": value}], is_error=is_error)

    @classmethod
    def json(cls, value: Any) -> ToolResult:
        import json as _json

        return cls(
            content=[{"type": "text", "text": _json.dumps(value, indent=2, default=str)}],
            structured=value if isinstance(value, dict) else {"value": value},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content, "isError": self.is_error}
        if self.structured is not None:
            payload["structuredContent"] = self.structured
        return payload

    def as_text(self) -> str:
        return "\n".join(c.get("text", "") for c in self.content if c.get("type") == "text")


@dataclass(slots=True)
class ResourceDefinition:
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


__all__ = [
    "INITIALIZE",
    "INITIALIZED",
    "JSONRPC_VERSION",
    "MCPErrorCode",
    "PROTOCOL_VERSION",
    "PING",
    "PROMPTS_GET",
    "PROMPTS_LIST",
    "PROGRESS",
    "RESOURCES_LIST",
    "RESOURCES_READ",
    "RESOURCES_TEMPLATES_LIST",
    "STANDARD_ERRORS",
    "TOOLS_CALL",
    "TOOLS_LIST",
    "ToolDefinition",
    "ToolResult",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "ResourceDefinition",
    "ServerInfo",
]