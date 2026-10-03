"""Model Context Protocol: protocol types, server and client."""

from __future__ import annotations

from agentflow.mcp.client import (
    InProcessTransport,
    MCPClient,
    MCPClientPool,
    MCPServerSpec,
    StdioTransport,
    ToolProxy,
)
from agentflow.mcp.protocol import (
    INITIALIZE,
    INITIALIZED,
    PROTOCOL_VERSION,
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
from agentflow.mcp.server import MCPServer, ResourceTemplate

__all__ = [
    "INITIALIZE",
    "INITIALIZED",
    "PROTOCOL_VERSION",
    "TOOLS_CALL",
    "TOOLS_LIST",
    "InProcessTransport",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPClient",
    "MCPClientPool",
    "MCPErrorCode",
    "MCPServer",
    "MCPServerSpec",
    "ResourceDefinition",
    "ResourceTemplate",
    "ServerInfo",
    "StdioTransport",
    "ToolDefinition",
    "ToolProxy",
    "ToolResult",
]