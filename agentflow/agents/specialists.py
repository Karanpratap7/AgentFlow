"""The specialist agent set.

Four agents, each narrow on purpose. Narrow agents get cheaper models, tighter
tool scopes and shorter prompts, which is where most of the cost saving in a
multi-agent system actually comes from.

``TriageAgent``     tool-free first pass that decides who handles the request
``ResearchAgent``   grounded retrieval over the internal knowledge base (RAG)
``CodeAgent``       repository inspection and static analysis (read-only tools)
``SQLAgent``        read-only query construction and validation
``SynthesizerAgent`` fan-in agent that merges parallel findings into one answer
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from agentflow.agents.base import Agent, AgentResult
from agentflow.llm.types import Message, ToolSchema

log = logging.getLogger("agentflow.agents")


class TriageAgent(Agent):
    """First-pass classifier. No tools, shortest prompt, cheapest model."""

    name: ClassVar[str] = "triage"
    description: ClassVar[str] = "Fast, tool-free first pass that decides who handles the request."
    capabilities: ClassVar[set[str]] = set()
    allowed_tools: ClassVar[set[str]] = set()
    max_iterations: ClassVar[int] = 1

    def system_prompt(self) -> str:
        return (
            "You are the triage agent. Reply with exactly one line: "
            "<SEVERITY>|<CATEGORY>|<NEXT_AGENT>\n"
            "SEVERITY is one of low, medium, high, critical.\n"
            "CATEGORY is one of bug, question, request, alert, other.\n"
            "NEXT_AGENT is one of coder, researcher, sql, chat.\n"
        )

    def tools(self) -> list[ToolSchema]:
        return []

    def on_result(self, result: AgentResult) -> AgentResult:
        first = result.output.strip().split("\n")[0] if result.output.strip() else ""
        if first.count("|") != 2:
            result.error = f"malformed triage line: {first!r}"
        return result


class ResearchAgent(Agent):
    """Retrieval-grounded research. Delegates to the RAG pipeline when present."""

    name: ClassVar[str] = "researcher"
    description: ClassVar[str] = "Answers from the internal knowledge base with citations."
    capabilities: ClassVar[set[str]] = {"prompt-caching", "tool-use"}
    allowed_tools: ClassVar[set[str] | None] = None

    def __init__(self, *args: Any, rag: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Optional RAGPipeline. When set, ``run`` short-circuits to it so the
        #: retrieval stages (rewrite / hybrid / rerank / compress) are used.
        self.rag = rag

    def system_prompt(self) -> str:
        return (
            "You are the research agent in a multi-agent engineering team.\n"
            "\n"
            "Rules:\n"
            "1. Ground every claim in the retrieved context. Never invent a fact.\n"
            "2. Cite the bracketed marker of each chunk you used.\n"
            "3. If the context is insufficient, say exactly what is missing.\n"
            "4. Quote configuration values, error codes and numbers verbatim.\n"
            "\n"
            "Answer in at most 150 words, leading with the direct answer.\n"
        )

    def run(  # noqa: D102 - documented in Agent
        self, task: str, *, context: str = "", history: list[Message] | None = None
    ) -> AgentResult:
        if self.rag is None:
            return super().run(task, context=context, history=history)
        try:
            self.rag.load()
        except Exception:  # noqa: BLE001 - a missing index is not fatal
            log.debug("rag index unavailable for %s; using the tool loop", self.name)
        if self.rag.store.is_empty:
            # Nothing indexed yet: degrade to the generic tool loop rather than
            # failing the whole fan-out over a missing corpus.
            return super().run(task, context=context, history=history)

        self.guardrails.check_input(task)
        answer = self.rag.ask(task, history=history)
        sources = "\n".join(f"- {c.citation}" for c in answer.trace.kept)
        result = AgentResult(
            agent=self.name,
            output=f"{answer.text}\n\nSources:\n{sources}" if sources else answer.text,
            model=answer.trace.routing.model if answer.trace.routing else "n/a",
            routing=answer.trace.routing,
        )
        result.tokens_saved = int(answer.trace.optimization.get("saved_tokens", 0))
        result.usage_total = int(answer.trace.prompt_tokens)
        result.guardrail_findings = list(answer.guardrails.get("findings", []))
        result.latency_ms = answer.trace.latency_ms
        if not answer.trace.grounded:
            result.error = f"unsupported citations: {answer.trace.unsupported_citations}"
        return result


class CodeAgent(Agent):
    """Repository inspection and diagnosis. Read-only by construction."""

    name: ClassVar[str] = "coder"
    description: ClassVar[str] = "Inspects the repository and diagnoses defects."
    capabilities: ClassVar[set[str]] = {"tool-use", "extended-thinking"}
    allowed_tools: ClassVar[set[str]] = {"read_file", "list_dir", "run_tests"}

    def system_prompt(self) -> str:
        return (
            "You are the code agent in a multi-agent engineering team.\n"
            "\n"
            "Rules:\n"
            "1. Read before you conclude. Call read_file on the exact path.\n"
            "2. Quote the specific lines responsible for a defect.\n"
            "3. Propose the minimal change; do not redesign working code.\n"
            "4. State the complexity or failure mode you are reasoning about.\n"
            "5. If a tool returns an error, use that signal instead of guessing.\n"
            "\n"
            "Format: DIAGNOSIS / EVIDENCE / FIX / RISK.\n"
        )

    def on_result(self, result: AgentResult) -> AgentResult:
        if result.ok and not result.tool_calls:
            result.error = "code agent produced no tool calls; the analysis was not grounded"
        return result


class SQLAgent(Agent):
    """Read-only query construction and safety validation."""

    name: ClassVar[str] = "sql"
    description: ClassVar[str] = "Builds and validates read-only SQL against the schema."
    capabilities: ClassVar[set[str]] = {"tool-use", "extended-thinking"}

    def system_prompt(self) -> str:
        return (
            "You are the SQL agent in a multi-agent engineering team.\n"
            "\n"
            "Rules:\n"
            "1. Emit SELECT/WITH statements only. Never emit DDL or DML.\n"
            "2. Use the schema from the retrieved context; never invent columns.\n"
            "3. Include an ORDER BY whenever the result is paginated.\n"
            "4. Call run_sql_readonly to validate the statement before answering.\n"
            "5. State the expected row shape and any index the query can use.\n"
        )

    def tools(self) -> list[ToolSchema]:
        """Prefer the MCP-exposed validator; fall back to local tools."""
        discovered = [s for s in self.registry.schemas() if s.name.endswith("run_sql_readonly")]
        return discovered or super().tools()

    def on_result(self, result: AgentResult) -> AgentResult:
        for line in result.output.split("\n"):
            if line.lower().strip().startswith(("drop ", "truncate ", "delete ", "update ", "insert ")):
                result.blocked = True
                result.error = "agent emitted a write statement"
                break
        return result


class SynthesizerAgent(Agent):
    """Fan-in agent. Merges parallel agent outputs into one answer."""

    name: ClassVar[str] = "synthesizer"
    description: ClassVar[str] = "Merges parallel agent findings into a single answer."
    capabilities: ClassVar[set[str]] = {"extended-thinking"}
    allowed_tools: ClassVar[set[str]] = set()

    def system_prompt(self) -> str:
        return (
            "You are the synthesizer in a multi-agent engineering team.\n"
            "\n"
            "Rules:\n"
            "1. Reconcile conflicts explicitly; never silently pick a side.\n"
            "2. Prefer the finding with cited evidence over the one without.\n"
            "3. Note any agent that returned nothing usable.\n"
            "4. Lead with the answer, then the supporting bullets.\n"
            "5. Stay under 200 words.\n"
        )

    def tools(self) -> list[ToolSchema]:
        return []

    @staticmethod
    def build_context(findings: dict[str, AgentResult]) -> str:
        blocks = []
        for agent, result in findings.items():
            body = (result.output or result.error or "(no output)").strip()
            blocks.append(f"### {agent} (model={result.model}, ok={result.ok})\n{body}")
        return "\n\n".join(blocks)


__all__ = [
    "CodeAgent",
    "ResearchAgent",
    "SQLAgent",
    "SynthesizerAgent",
    "TriageAgent",
]