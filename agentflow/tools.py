"""Tool registry: local tools and MCP-proxied tools behind one interface.

An agent sees a flat list of :class:`ToolSchema`. Behind that, a tool may be a
local Python callable or a remote MCP tool, and it does not matter which: the
registry resolves ``server.tool`` names to MCP calls and everything else to
local dispatch. This is what lets a new capability be added by standing up an MCP
server rather than by editing agent code.
"""

from __future__ import annotations

import inspect
import logging
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentflow.errors import MCPRuntimeError
from agentflow.llm.types import ContentBlock, ToolSchema
from agentflow.mcp.client import MCPClientPool

log = logging.getLogger("agentflow.tools")

ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ToolRecord:
    """One registered tool and where it actually runs."""

    schema: ToolSchema
    handler: ToolHandler | None
    source: str = "local"
    server: str = ""
    timeout_s: float = 30.0
    calls: int = 0
    errors: int = 0
    total_ms: float = 0.0

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


@dataclass(slots=True)
class ToolInvocation:
    """Result of one tool call, shaped like an MCP ``tools/call`` result."""

    tool: str
    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    latency_ms: float = 0.0

    def as_text(self) -> str:
        return "\n".join(c.get("text", "") for c in self.content if c.get("type") == "text")

    def to_assistant_block(self, tool_use_id: str) -> ContentBlock:
        return ContentBlock.tool_result(tool_use_id, self.as_text(), is_error=self.is_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "is_error": self.is_error,
            "latency_ms": round(self.latency_ms, 2),
            "preview": self.as_text()[:400],
        }


class ToolRegistry:
    """Local + MCP tools, addressed uniformly."""

    def __init__(self, mcp_pool: MCPClientPool | None = None) -> None:
        self._records: dict[str, ToolRecord] = {}
        self.mcp_pool = mcp_pool

    # -- registration -----------------------------------------------------
    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        *,
        timeout_s: float = 30.0,
        input_schema: dict[str, Any] | None = None,
    ) -> ToolRecord:
        record = ToolRecord(
            schema=ToolSchema(
                name=name,
                description=description,
                input_schema=input_schema or schema_from_signature(handler),
            ),
            handler=handler,
            source="local",
            timeout_s=timeout_s,
        )
        self._records[name] = record
        return record

    def tool(self, name: str, description: str, *, timeout_s: float = 30.0) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator form, mirroring the MCP server's registration API."""

        def decorate(fn: ToolHandler) -> ToolHandler:
            self.register(name, description, fn, timeout_s=timeout_s)
            return fn

        return decorate

    def attach_mcp(self, pool: MCPClientPool | None = None) -> int:
        """Mirror every MCP tool into the registry as ``server.tool``."""
        pool = pool or self.mcp_pool
        if pool is None:
            return 0
        self.mcp_pool = pool
        count = 0
        for proxy in pool.tools():
            server, _, tool = proxy.qualified_name.partition(".")
            client = pool.clients.get(server)
            self._records[proxy.qualified_name] = ToolRecord(
                schema=proxy.to_tool_schema(),
                handler=None,
                source="mcp",
                server=server,
                timeout_s=30.0,
            )
            count += 1
            del client
        return count

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def names(self) -> list[str]:
        return list(self._records)

    def schemas(self, *, include: set[str] | None = None) -> list[ToolSchema]:
        return [
            record.schema
            for name, record in self._records.items()
            if include is None or name in include
        ]

    def split_by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"local": [], "mcp": []}
        for name, record in self._records.items():
            out[record.source].append(name)
        return out

    def token_cost(self) -> dict[str, int]:
        """Per-tool schema footprint, so tool bloat can be attributed."""
        from agentflow.tokens.counter import estimate_text_tokens

        return {
            name: estimate_text_tokens(record.schema.description)
            + estimate_text_tokens(str(record.schema.input_schema))
            for name, record in self._records.items()
        }

    def stats(self) -> dict[str, Any]:
        return {
            "tools": len(self._records),
            "by_source": self.split_by_source(),
            "invocations": sum(r.calls for r in self._records.values()),
            "errors": sum(r.errors for r in self._records.values()),
            "total_ms": round(sum(r.total_ms for r in self._records.values()), 2),
            "schema_tokens": sum(self.token_cost().values()),
        }

    # -- invocation -------------------------------------------------------
    def invoke(self, name: str, **arguments: Any) -> ToolInvocation:
        record = self._records.get(name)
        if record is None:
            return ToolInvocation(
                tool=name,
                content=[
                    {"type": "text", "text": f"unknown tool {name!r}; available: {sorted(self._records)}"}
                ],
                is_error=True,
            )

        started = time.perf_counter()
        record.calls += 1
        try:
            if record.source == "mcp":
                text, error = self._invoke_mcp(record, arguments)
            else:
                assert record.handler is not None
                _validate(arguments, record.handler)
                text, error = _stringify(record.handler(**arguments)), False
        except MCPRuntimeError as exc:
            record.errors += 1
            text, error = f"MCP transport error: {exc}", True
        except Exception as exc:  # noqa: BLE001 - a failing tool must not kill the agent
            record.errors += 1
            log.warning("tool %s raised %s", name, type(exc).__name__, exc_info=exc)
            text, error = f"{type(exc).__name__}: {exc}", True

        elapsed = (time.perf_counter() - started) * 1000
        record.total_ms += elapsed
        return ToolInvocation(
            tool=name,
            content=[{"type": "text", "text": text}],
            is_error=error,
            latency_ms=elapsed,
        )

    def _invoke_mcp(self, record: ToolRecord, arguments: dict[str, Any]) -> tuple[str, bool]:
        assert self.mcp_pool is not None
        result = self.mcp_pool.invoke(record.name, **arguments)
        return result.as_text(), result.is_error

    # -- convenience ------------------------------------------------------
    @classmethod
    def from_mcp_manifest(cls, manifest_path: str) -> ToolRegistry:
        pool = MCPClientPool(manifest_path).connect_all()
        registry = cls(pool)
        registry.attach_mcp(pool)
        return registry


_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def schema_from_signature(fn: ToolHandler) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return {"type": "object", "properties": {}}
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        entry: dict[str, Any] = {"type": _PY_TO_JSON.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            entry["default"] = param.default
        properties[name] = entry
    return {"type": "object", "properties": properties, "required": required}


def _validate(arguments: dict[str, Any], handler: ToolHandler) -> None:
    signature = inspect.signature(handler)
    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty and name not in arguments
    ]
    if missing:
        raise TypeError(f"missing required argument(s): {missing}")


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, indent=2, default=str)
    return str(value)


def build_default_registry() -> ToolRegistry:
    """The built-in local toolset the demo agents use."""
    registry = ToolRegistry()

    @registry.tool("read_file", "Read a UTF-8 text file from the workspace.")
    def read_file(path: str, max_bytes: int = 200_000) -> str:
        from pathlib import Path

        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return f"error: {resolved} does not exist"
        if resolved.stat().st_size > max_bytes:
            return f"error: {resolved} exceeds max_bytes={max_bytes}"
        return resolved.read_text(encoding="utf-8", errors="replace")

    @registry.tool("list_dir", "List the entries of a directory.")
    def list_dir(path: str = ".") -> str:
        from pathlib import Path

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            return f"error: {resolved} is not a directory"
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir()))

    @registry.tool("run_tests", "Run a named test target and report the summary line.")
    def run_tests(target: str = "tests") -> str:
        import subprocess

        proc = subprocess.run(  # noqa: S603,S607 - fixed argv, no shell
            ["python3", "-m", "pytest", target, "-q"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        tail = (proc.stdout or proc.stderr).strip().split("\n")[-1]
        return f"exit={proc.returncode} :: {tail}"

    @registry.tool("http_status", "Report the HTTP status of a URL (no body).")
    def http_status(url: str) -> str:
        import httpx

        try:
            response = httpx.head(url, timeout=10.0, follow_redirects=True)
            return f"{response.status_code} {url}"
        except Exception as exc:  # noqa: BLE001 - network surface is wide
            return f"error: {type(exc).__name__}: {exc}"

    return registry


__all__ = ["ToolInvocation", "ToolRecord", "ToolRegistry", "build_default_registry", "shlex"]