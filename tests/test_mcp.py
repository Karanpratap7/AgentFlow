"""MCP: argv normalisation, the JSON-RPC server, clients and the pool."""

import json
import sys
from pathlib import Path

import pytest

from agentflow.errors import MCPRuntimeError
from agentflow.mcp.client import (
    MCPClient,
    MCPClientPool,
    MCPServerSpec,
    StdioTransport,
    _as_argv,
)
from agentflow.mcp.protocol import ToolResult
from agentflow.mcp.server import MCPServer
from agentflow.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "mcp_servers" / "manifest.json"


@pytest.fixture(scope="module")
def kb_server():
    sys.path.insert(0, str(ROOT / "mcp_servers"))
    try:
        import knowledge_base

        return knowledge_base.server
    finally:
        sys.path.remove(str(ROOT / "mcp_servers"))


# -- argv normalisation -----------------------------------------------------
def test_a_bare_command_string_is_split_not_iterated():
    """``list("python3")`` would try to execute ``p``; manifests write strings."""
    assert _as_argv("python3 mcp_servers/knowledge_base.py") == [
        "python3",
        "mcp_servers/knowledge_base.py",
    ]


def test_a_command_list_is_kept_and_stringified():
    assert _as_argv(["python3", 42, "x.py"]) == ["python3", "42", "x.py"]


def test_missing_command_is_empty():
    assert _as_argv(None) == []


def test_spec_argv_joins_command_and_args():
    spec = MCPServerSpec.from_dict(
        {"name": "kb", "command": "python3", "args": ["server.py", "--offline"]}
    )
    assert spec.argv == ["python3", "server.py", "--offline"]


# -- the bundled manifest ---------------------------------------------------
def test_the_bundled_manifest_parses():
    payload = json.loads(MANIFEST.read_text())
    servers = payload.get("mcpServers", payload)
    assert {"knowledge_base", "filesystem"} <= set(servers)
    for name, raw in servers.items():
        spec = MCPServerSpec.from_dict({"name": name, **raw})
        assert spec.argv and spec.description, name


def test_a_manifest_missing_a_name_is_rejected():
    with pytest.raises(KeyError):
        MCPServerSpec.from_dict({"command": "python3"})


# -- in-process client against the bundled server ---------------------------
def test_embedded_client_lists_the_bundled_tools(kb_server):
    client = MCPClient.embed("knowledge_base", kb_server)
    names = {t.name for t in client.list_tools()}
    assert "search_docs" in names
    assert len(names) >= 3


def test_embedded_client_calls_a_tool(kb_server):
    client = MCPClient.embed("knowledge_base", kb_server)
    result = client.call_tool("search_docs", query="retry policy")
    assert not result.is_error
    assert "retry" in result.as_text().lower()


def test_a_tool_error_comes_back_as_content_not_an_exception(kb_server):
    client = MCPClient.embed("knowledge_base", kb_server)
    result = client.call_tool("search_docs")  # missing required argument
    assert result.is_error
    assert result.as_text()


def test_an_unknown_tool_is_reported(kb_server):
    client = MCPClient.embed("knowledge_base", kb_server)
    result = client.call_tool("no_such_tool")
    assert result.is_error


def test_client_index_reports_tools_resources_and_prompts(kb_server):
    index = MCPClient.embed("knowledge_base", kb_server).index()
    assert index["server"] == "knowledge_base"
    assert index["tools"]
    assert index["resources"]
    assert index["prompts"]


def test_capabilities_gate_discovery():
    """A server without resources must not be asked for them."""
    from agentflow.mcp.protocol import ServerInfo

    bare = MCPServer(ServerInfo(name="bare", version="0.0.0"))

    @bare.tool("noop", "does nothing")
    def noop() -> str:
        return "ok"

    index = MCPClient.embed("bare", bare).index()
    assert index["resources"] == []
    assert index["prompts"] == []


def test_tool_schemas_are_qualified_by_server(kb_server):
    client = MCPClient.embed("knowledge_base", kb_server)
    assert all(s.name.startswith("knowledge_base.") for s in client.tool_schemas())


# -- protocol edge cases ----------------------------------------------------
def test_a_notification_returns_no_response(kb_server):
    server = kb_server
    assert server.handle_dict({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_an_unknown_method_is_a_protocol_error(kb_server):
    response = kb_server.handle_dict({"jsonrpc": "2.0", "id": 1, "method": "nope/nope"})
    assert response is not None and response["error"]["code"] == -32601


def test_a_malformed_request_is_rejected(kb_server):
    response = kb_server.handle_dict({"id": 1, "method": 5})
    assert response is not None and "error" in response


def test_stdio_transport_reports_a_missing_binary():
    transport = StdioTransport(command="definitely-not-a-real-binary-xyz")
    with pytest.raises(MCPRuntimeError):
        transport.start()


# -- the pool ---------------------------------------------------------------
@pytest.fixture(scope="module")
def pool():
    with MCPClientPool(MANIFEST) as connected:
        yield connected


def test_pool_connects_to_both_servers(pool):
    assert set(pool.names()) >= {"knowledge_base", "filesystem"}
    assert "knowledge_base" in pool
    assert "not-configured" not in pool


def test_pool_exposes_tools_from_every_server(pool):
    schemas = pool.tool_schemas()
    servers = {s.name.split(".")[0] for s in schemas}
    assert servers == set(pool.names())
    assert len(schemas) >= 10


def test_pool_invokes_a_qualified_tool(pool):
    result = pool.invoke("knowledge_base.search_docs", query="retry")
    assert not result.is_error
    assert result.content


def test_pool_get_returns_a_connected_client(pool):
    assert pool.get("knowledge_base").list_tools()


def test_pool_iterates_its_clients(pool):
    assert len(list(pool)) == len(pool)


def test_a_broken_server_is_recorded_not_raised():
    manifest = json.loads(MANIFEST.read_text())
    servers = manifest.get("mcpServers", manifest)
    servers["broken"] = {"command": "definitely-not-a-real-binary-xyz", "description": "nope"}
    path = ROOT / "mcp_servers" / "manifest_broken.json"
    path.write_text(json.dumps(manifest))
    try:
        with MCPClientPool(path) as broken_pool:
            assert "broken" not in broken_pool
            assert "broken" in broken_pool.errors
            assert "knowledge_base" in broken_pool  # one bad server must not sink the rest
    finally:
        path.unlink()


# -- tool registry integration ---------------------------------------------
def test_attaching_a_pool_adds_qualified_tools(pool):
    registry = ToolRegistry()
    added = registry.attach_mcp(pool)
    assert added > 0
    assert all("." in name or "__" in name for name in registry.names())


def test_the_registry_can_split_tools_by_source(pool):
    registry = ToolRegistry()
    registry.attach_mcp(pool)
    by_source = registry.split_by_source()
    assert "mcp" in by_source or "mcp:knowledge-base" in by_source


def test_tool_token_cost_is_attributed_per_tool(pool):
    registry = ToolRegistry()
    registry.attach_mcp(pool)
    costs = registry.token_cost()
    assert costs and all(v >= 0 for v in costs.values())
    assert max(costs.values()) > 0


def test_tool_result_json_round_trip():
    payload = {"rows": 2}
    assert json.loads(ToolResult.json(payload).as_text()) == payload