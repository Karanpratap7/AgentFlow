"""Wire up an MCP server and let an agent call its tools.

Connects every server in ``mcp_servers/manifest.json``, attaches the discovered
tools to the tool registry, and shows them being invoked.

    python3 examples/02_mcp_integration.py
"""
import sys
from pathlib import Path

# Runnable straight from a fresh clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentflow.mcp.client import MCPClientPool
from agentflow.runtime import build_runtime
from agentflow.tools import ToolRegistry


def main() -> None:
    rt = build_runtime()
    pool = MCPClientPool(rt.settings.mcp_manifest).connect_all()

    try:
        index = pool.index()
        print(f"connected to {len(index['servers'])} server(s), "
              f"{index['tool_count']} tools\n")

        for name, server in index["servers"].items():
            print(f"{name}  ({server['protocol_version']})")
            print(f"  tools     {', '.join(server['tools'])}")
            if server["resources"]:
                print(f"  resources {', '.join(server['resources'])}")
            if server["prompts"]:
                print(f"  prompts   {', '.join(server['prompts'])}")
            print()

        for name, error in index["unavailable"].items():
            print(f"unavailable: {name}: {error}\n")

        # Attach them to a fresh registry so the agent can actually call them.
        registry = ToolRegistry()
        registry.attach_mcp(pool)
        print(f"tool registry now exposes: {', '.join(registry.names())}\n")

        if "knowledge_base" in pool.names():
            client = pool.get("knowledge_base")
            print("calling knowledge_base.search_docs ...")
            result = client.call_tool(
                "search_docs", query="retry policy", limit=2
            )
            print(f"  is_error {result.is_error}")
            print("  " + (result.as_text() or "(empty)")[:500])
    finally:
        pool.close()


if __name__ == "__main__":
    main()