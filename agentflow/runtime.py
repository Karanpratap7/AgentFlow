"""Wiring: one place that builds every collaborator with shared settings.

Both the FastAPI app and the CLI construct the runtime through ``build_runtime``,
so there is a single definition of what "the system" is and no chance of the API
and the CLI drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentflow.agents.pool import AgentRegistry, default_registry
from agentflow.chat.session import ChatAgent, SessionStore
from agentflow.config import Settings, get_settings
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.client import AnthropicClient
from agentflow.llm.fake import ScriptedClient
from agentflow.llm.types import LLMClient
from agentflow.orchestration.workflow import AgentFlowOrchestrator
from agentflow.rag.embeddings import Embedder, default_embedder
from agentflow.rag.pipeline import RAGPipeline
from agentflow.routing.intent import IntentDetector
from agentflow.routing.router import AutoRouter
from agentflow.telemetry import Metrics, metrics
from agentflow.tokens.budget import TokenBudget
from agentflow.tools import ToolRegistry, build_default_registry


@dataclass(slots=True)
class Runtime:
    """The assembled system. Everything is lazily built and cached."""

    settings: Settings
    client: LLMClient
    tools: ToolRegistry
    agents: AgentRegistry
    guardrails: GuardrailPipeline
    router: AutoRouter
    intent: IntentDetector
    metrics: Metrics
    _rag: RAGPipeline | None = None
    _orchestrator: AgentFlowOrchestrator | None = None
    _chat: ChatAgent | None = None
    _sessions: SessionStore | None = None

    # -- lazily built components -----------------------------------------
    @property
    def rag(self) -> RAGPipeline:
        if self._rag is None:
            self._rag = RAGPipeline(
                self.client,
                settings=self.settings,
                embedder=default_embedder(),
                guardrails=self.guardrails,
                router=self.router,
                intent_detector=self.intent,
            )
        return self._rag

    @property
    def orchestrator(self) -> AgentFlowOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = AgentFlowOrchestrator(
                self.client,
                registry=self.agents,
                tools=self.tools,
                settings=self.settings,
                guardrails=self.guardrails,
                router=self.router,
                intent_detector=self.intent,
                # ``self.rag``, not ``self._rag``: the researcher needs the live
                # pipeline, and building it here keeps the wiring order-independent.
                rag=self.rag,
            )
        return self._orchestrator

    @property
    def chat(self) -> ChatAgent:
        if self._chat is None:
            self._chat = ChatAgent(
                self.client,
                settings=self.settings,
                guardrails=self.guardrails,
                router=self.router,
                intent_detector=self.intent,
                sessions=self.sessions,
            )
        return self._chat

    @property
    def sessions(self) -> SessionStore:
        if self._sessions is None:
            self._sessions = SessionStore(token_limit=self.settings.session_token_budget)
        return self._sessions

    def budget(self, session_id: str = "api") -> TokenBudget:
        return TokenBudget(
            session_id=session_id, token_limit=self.settings.session_token_budget
        )

    def health(self) -> dict[str, Any]:
        from agentflow.llm.catalog import catalog

        return {
            "status": "ok",
            "offline_mode": self.settings.offline,
            "provider": "scripted" if self.settings.offline else "anthropic",
            "models": [spec.id for spec in catalog.specs()],
            "tools": self.tools.names(),
            "agents": self.agents.names(),
            "guardrail_rules": len(self.guardrails.rules),
            "index": self.rag.store.stats() if self._rag else {"chunks": 0},
        }


def build_runtime(
    settings: Settings | None = None,
    *,
    with_mcp: bool = False,
    mcp_manifest: str | None = None,
    tools: ToolRegistry | None = None,
    embedder: Embedder | None = None,
    client: LLMClient | None = None,
) -> Runtime:
    """Assemble the runtime. With no API key it runs fully offline."""
    settings = settings or get_settings()
    resolved_client = client or (
        ScriptedClient() if settings.offline else AnthropicClient(settings)
    )

    guardrails = GuardrailPipeline(settings)
    router = AutoRouter(settings)
    intent = IntentDetector(resolved_client)

    tool_registry = tools or build_default_registry()
    if with_mcp:
        from agentflow.mcp.client import MCPClientPool

        pool = MCPClientPool(mcp_manifest or settings.mcp_manifest).connect_all()
        tool_registry.attach_mcp(pool)

    return Runtime(
        settings=settings,
        client=resolved_client,
        tools=tool_registry,
        agents=default_registry(),
        guardrails=guardrails,
        router=router,
        intent=intent,
        metrics=metrics if settings.telemetry_enabled else Metrics(enabled=False),
    )


__all__ = ["Runtime", "build_runtime"]