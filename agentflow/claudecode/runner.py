"""Claude Code integration.

Drives the ``claude`` CLI in headless (``-p``) mode so the same runtime can be
driven by coding agents instead of raw API calls. Three capabilities:

* ``ClaudeCodeRunner.run``  - one headless prompt with a permission mode, model
  and tool allow-list; returns text plus structured CLI output.
* ``SubagentSpec``           - declarative parallel subagents (the ``Task`` tool
  made scriptable) executed concurrently.
* ``HookRunner``             - project hooks wired into ``.claude/settings.json``
  that route AgentFlow's guardrails into the CLI's own event pipeline.

The CLI is optional: every method degrades to a clear ``MCPRuntimeError`` when
the binary is missing, so nothing here is a hard dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentflow.errors import MCPRuntimeError
from agentflow.llm.catalog import catalog

PermissionMode = Literal["default", "acceptEdits", "bypassPermissions", "plan"]

HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "SessionStart",
)


def claude_binary() -> str:
    """Locate the CLI, honouring ``CLAUDE_CODE_BIN``."""
    override = os.environ.get("CLAUDE_CODE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found:
        return found
    raise MCPRuntimeError(
        "the `claude` CLI is not on PATH; install Claude Code or set CLAUDE_CODE_BIN"
    )


def claude_available() -> bool:
    try:
        claude_binary()
    except MCPRuntimeError:
        return False
    return True


@dataclass(slots=True)
class ClaudeRunResult:
    """Outcome of one headless Claude Code invocation."""

    prompt: str
    text: str = ""
    session_id: str = ""
    model: str = ""
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    num_turns: int = 0
    is_error: bool = False
    result: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt[:200],
            "text": self.text[:1_000],
            "session_id": self.session_id,
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": round(self.duration_ms, 2),
            "num_turns": self.num_turns,
            "is_error": self.is_error,
        }


class ClaudeCodeRunner:
    """Thin, safe wrapper over the headless Claude Code CLI."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        cwd: str | Path | None = None,
        allowed_tools: Sequence[str] = ("Read", "Grep", "Glob"),
        disallowed_tools: Sequence[str] = ("Edit", "Write", "Bash"),
        permission_mode: PermissionMode = "default",
        timeout_s: float = 600.0,
        append_system_prompt: str = "",
    ) -> None:
        self.binary = binary
        self.cwd = str(cwd) if cwd else None
        self.allowed_tools = list(allowed_tools)
        self.disallowed_tools = list(disallowed_tools)
        self.permission_mode = permission_mode
        self.timeout_s = timeout_s
        self.append_system_prompt = append_system_prompt

    def build_argv(
        self,
        prompt: str,
        *,
        model: str | None = None,
        permission_mode: PermissionMode | None = None,
        output_format: str | None = "json",
        resume: str | None = None,
        max_turns: int | None = None,
        mcp_config: Path | None = None,
    ) -> list[str]:
        """Assemble the CLI argv. Pure function, so it is directly testable."""
        argv = [self.binary or claude_binary(), "-p", prompt]
        mode = permission_mode or self.permission_mode
        if mode != "default":
            argv += ["--permission-mode", mode]
        if model:
            argv += ["--model", model]
        if output_format:
            argv += ["--output-format", output_format]
        if self.allowed_tools:
            argv += ["--allowed-tools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowed-tools", ",".join(self.disallowed_tools)]
        if self.append_system_prompt:
            argv += ["--append-system-prompt", self.append_system_prompt]
        if resume:
            argv += ["--resume", resume]
        if max_turns:
            argv += ["--max-turns", str(max_turns)]
        if mcp_config and mcp_config.exists():
            argv += ["--mcp-config", str(mcp_config)]
        return argv

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        permission_mode: PermissionMode | None = None,
        resume: str | None = None,
        max_turns: int | None = None,
        mcp_config: Path | None = None,
    ) -> ClaudeRunResult:
        """Invoke the CLI once and normalise its JSON output."""
        argv = self.build_argv(
            prompt,
            model=model,
            permission_mode=permission_mode,
            resume=resume,
            max_turns=max_turns,
            mcp_config=mcp_config,
        )
        started = time.perf_counter()
        try:
            proc = subprocess.run(  # noqa: S603 - argv is built by build_argv, no shell
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.cwd,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MCPRuntimeError(f"claude binary not found: {argv[0]!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MCPRuntimeError(f"claude timed out after {self.timeout_s}s") from exc

        duration = (time.perf_counter() - started) * 1000
        payload: dict[str, Any] = {}
        if proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
                payload = parsed if isinstance(parsed, dict) else {"result": str(parsed)}
            except json.JSONDecodeError:
                payload = {"result": proc.stdout.strip()}

        return ClaudeRunResult(
            prompt=prompt,
            text=str(payload.get("result", "")),
            result=str(payload.get("result", "")),
            session_id=str(payload.get("session_id", "")),
            model=str(payload.get("model", model or "")),
            cost_usd=float(payload.get("total_cost_usd", 0.0) or 0.0),
            duration_ms=duration,
            num_turns=int(payload.get("num_turns", 0) or 0),
            is_error=proc.returncode != 0 or bool(payload.get("is_error", False)),
            raw=payload,
        )

    def run_parallel(self, prompts: dict[str, str], **kwargs: Any) -> dict[str, ClaudeRunResult]:
        """Fan out one CLI session per prompt. The ``Task`` tool, scripted."""
        out: dict[str, ClaudeRunResult] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(prompts) or 1)) as pool:
            futures = {pool.submit(self.run, p, **kwargs): name for name, p in prompts.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    out[name] = future.result()
                except MCPRuntimeError as exc:
                    out[name] = ClaudeRunResult(prompt=prompts[name], is_error=True, text=str(exc))
        return out


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """A declarative Claude Code subagent, matching ``.claude/agents/*.md``."""

    name: str
    prompt: str
    description: str = ""
    model: str | None = None
    tools: tuple[str, ...] = ("Read", "Grep", "Glob")
    run_in_background: bool = True

    @classmethod
    def from_frontmatter(cls, text: str) -> SubagentSpec:
        """Parse a Claude Code subagent markdown file."""
        import re

        front = re.match(r"^---\n(.*?)\n---\n?(.*)$", text.strip(), re.DOTALL)
        if not front:
            return cls(name="subagent", prompt=text)
        meta: dict[str, str] = {}
        for line in front.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip("\"'")
        return cls(
            name=meta.get("name", "subagent"),
            prompt=front.group(2).strip(),
            description=meta.get("description", ""),
            model=meta.get("model") or None,
            tools=tuple(t.strip() for t in meta.get("tools", "Read, Grep, Glob").split(",") if t.strip()),
        )


def load_subagents(directory: str | Path) -> dict[str, SubagentSpec]:
    """Load every ``*.md`` subagent definition from ``.claude/agents``."""
    path = Path(directory)
    if not path.is_dir():
        return {}
    return {p.stem: SubagentSpec.from_frontmatter(p.read_text(encoding="utf-8")) for p in path.glob("*.md")}


@dataclass(slots=True)
class HookDecision:
    """What a hook decided to do with an event."""

    decision: Literal["allow", "deny", "ask"]
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_settings(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"decision": self.decision}
        if self.reason:
            payload["reason"] = self.reason
        return payload


HookHandler = Callable[[dict[str, Any]], HookDecision]


class HookRunner:
    """Runs AgentFlow guardrails as Claude Code hooks.

    ``settings.json`` wires these to ``PreToolUse`` / ``UserPromptSubmit`` so the
    same rules that protect the HTTP API also protect the CLI.
    """

    def __init__(self, handlers: dict[str, HookHandler] | None = None) -> None:
        self.handlers: dict[str, HookHandler] = handlers or {}

    def register(self, event: str) -> Callable[[HookHandler], HookHandler]:
        def decorate(fn: HookHandler) -> HookHandler:
            self.handlers[event] = fn
            return fn

        return decorate

    def dispatch(self, event: str, payload: dict[str, Any]) -> HookDecision:
        handler = self.handlers.get(event)
        if handler is None:
            return HookDecision(decision="allow", reason=f"no handler for {event}")
        try:
            return handler(payload)
        except Exception as exc:  # noqa: BLE001 - a crashing hook must fail open
            return HookDecision(decision="allow", reason=f"hook error: {exc}")

    @staticmethod
    def settings(commands: dict[str, list[str]]) -> dict[str, Any]:
        """Render a ``.claude/settings.json`` hooks block."""
        hooks: dict[str, Any] = {}
        for event, cmds in commands.items():
            hooks[event] = [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": cmd} for cmd in cmds],
                }
            ]
        return {"hooks": hooks}


def guardrail_hook(pipeline: Any) -> HookHandler:
    """Bridge a :class:`GuardrailPipeline` into a Claude Code hook."""

    def handler(payload: dict[str, Any]) -> HookDecision:
        text = (
            payload.get("prompt")
            or payload.get("tool_input", {}).get("content")
            or payload.get("tool_input", {}).get("command")
            or ""
        )
        if not text:
            return HookDecision(decision="allow")
        try:
            result = pipeline.check_input(str(text))
        except Exception as exc:  # noqa: BLE001 - GuardrailViolation carries the rule
            return HookDecision(decision="deny", reason=str(exc))
        if result.flagged:
            return HookDecision(
                decision="ask",
                reason=f"flagged by {result.flagged}",
                extra=result.to_dict(),
            )
        return HookDecision(decision="allow", extra=result.to_dict())

    return handler


def recommend_model(capabilities: Sequence[str], *, tokens: int = 0) -> str:
    """Pick a Claude model for a CLI task using the shared catalog."""
    try:
        return catalog.select(
            required_capabilities=capabilities,
            min_quality=catalog.requirements_for(capabilities),
            required_tokens=tokens,
            objective="cost",
        ).id
    except MCPRuntimeError:
        return "claude-sonnet-4-5"


__all__ = [
    "HOOK_EVENTS",
    "ClaudeCodeRunner",
    "ClaudeRunResult",
    "HookDecision",
    "HookHandler",
    "HookRunner",
    "SubagentSpec",
    "claude_available",
    "claude_binary",
    "guardrail_hook",
    "load_subagents",
    "recommend_model",
]