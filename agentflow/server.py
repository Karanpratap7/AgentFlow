"""FastAPI surface.

Endpoints, grouped by what they expose:

``POST /v1/chat``         one chat turn (auto intent + routing + memory)
``POST /v1/agent``        one agentic workflow run (parallel fan-out/fan-in)
``POST /v1/rag/ingest``   index documents
``POST /v1/rag/ask``      grounded answer with a retrieval trace
``POST /v1/route``        routing decision for a request, with the factor breakdown
``POST /v1/route/compare``per-model cost for one request shape
``POST /v1/guard``        run the guardrail pipeline (also used as a CLI hook)
``GET  /v1/mcp``          connected MCP servers and their discovered tools
``GET  /v1/metrics``      JSON metrics snapshot
``GET  /metrics``         Prometheus exposition
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from agentflow.config import get_settings
from agentflow.errors import (
    AgentFlowError,
    GuardrailViolation,
    NotIndexedError,
    TokenBudgetExceeded,
)
from agentflow.orchestration.workflow import AgentFlowOrchestrator
from agentflow.runtime import Runtime, build_runtime

log = logging.getLogger("agentflow.api")


# -- request models --------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    session_id: str = "default"


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=32_000)
    session_id: str = "api"
    agents: list[str] | None = Field(
        default=None, description="Force a specific agent set, bypassing the planner."
    )


class IngestRequest(BaseModel):
    documents: dict[str, str] = Field(min_length=1)
    persist: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=8, ge=1, le=32)
    keep_k: int = Field(default=4, ge=1, le=32)


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=16_000)
    input_tokens: int | None = Field(default=None, ge=0)
    requires_tools: bool = False
    requires_retrieval: bool = False
    force_model: str | None = None


class CompareRequest(BaseModel):
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    cache_prefix_tokens: int = Field(default=0, ge=0)
    cache_hit_ratio: float = Field(default=0.9, ge=0.0, le=1.0)


class GuardRequest(BaseModel):
    text: str = Field(min_length=1)
    stage: Literal["input", "output"] = "input"
    raise_on_block: bool = False


# -- app -------------------------------------------------------------------
def create_app(runtime: Runtime | None = None) -> FastAPI:
    settings = runtime.settings if runtime else get_settings()
    rt = runtime or build_runtime(settings)

    app = FastAPI(
        title="AgentFlow",
        version="0.4.0",
        description=(
            "Multi-agent orchestration runtime: auto intent detection, automatic model "
            "selection, parallel agents, RAG pipelines, MCP integration and guardrails."
        ),
    )
    app.state.runtime = rt

    def get_rt() -> Runtime:
        return app.state.runtime

    # -- error handling --------------------------------------------------
    @app.exception_handler(GuardrailViolation)
    async def _guardrail_handler(_request: Any, exc: GuardrailViolation) -> Any:
        from fastapi.responses import JSONResponse

        rt.metrics.incr("guardrail_blocks_total", rule=exc.rule, stage=exc.stage)
        return JSONResponse(status_code=400, content={"error": exc.rule, "detail": str(exc)})

    @app.exception_handler(TokenBudgetExceeded)
    async def _budget_handler(_request: Any, exc: TokenBudgetExceeded) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=413, content={"error": "token_budget", "detail": str(exc)})

    @app.exception_handler(NotIndexedError)
    async def _index_handler(_request: Any, exc: NotIndexedError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"error": "not_indexed", "detail": str(exc)})

    @app.exception_handler(AgentFlowError)
    async def _base_handler(_request: Any, exc: AgentFlowError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # -- endpoints -------------------------------------------------------
    @app.get("/health")
    def health(dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        return dep.health()

    @app.post("/v1/chat")
    def chat(body: ChatRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        # ChatAgent already records request/intent/route/context metrics.
        return dep.chat.chat(body.message, session_id=body.session_id).to_dict()

    @app.post("/v1/agent")
    def agent(body: AgentRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        orchestrator: AgentFlowOrchestrator = dep.orchestrator
        if body.agents:
            orchestrator.supervisor.forced_agents = body.agents
        try:
            outcome = orchestrator.run(body.task, session_id=body.session_id)
        finally:
            orchestrator.supervisor.forced_agents = None
        # dep.metrics, not the module-level singleton: every other endpoint
        # records on the runtime it was handed, and this one must not be the
        # odd one out.
        dep.metrics.incr("agent_runs_total")
        return outcome.to_dict()

    @app.post("/v1/rag/ingest")
    def ingest(body: IngestRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        chunks = dep.rag.ingest(body.documents, persist=body.persist)
        dep.metrics.incr("rag_chunks_indexed_total", value=len(chunks))
        return {"indexed": len(chunks), "stats": dep.rag.stats()}

    @app.post("/v1/rag/ask")
    def ask(body: AskRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        dep.rag.top_k = body.top_k
        dep.rag.keep_k = min(body.keep_k, body.top_k)
        answer = dep.rag.ask(body.question)
        return {"answer": answer.text, "trace": answer.trace.to_dict(), "guardrails": answer.guardrails}

    @app.post("/v1/route")
    def route(body: RouteRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        decision = dep.router.route(
            body.text,
            expected_input_tokens=body.input_tokens,
            requires_tools=body.requires_tools,
            requires_retrieval=body.requires_retrieval,
            force_model=body.force_model,
        )
        dep.metrics.incr("route_model_total", model=decision.model)
        dep.metrics.incr("route_task_total", task=decision.task)
        return decision.to_dict()

    @app.post("/v1/route/compare")
    def route_compare(body: CompareRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        return dep.router.compare_models(
            input_tokens=body.input_tokens,
            output_tokens=body.output_tokens,
            cache_prefix_tokens=body.cache_prefix_tokens,
            cache_hit_ratio=body.cache_hit_ratio,
        )

    @app.post("/v1/guard")
    def guard(body: GuardRequest, dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        if body.stage == "input":
            result = dep.guardrails.check_input(body.text, raise_on_block=body.raise_on_block)
        else:
            result = dep.guardrails.check_output(body.text, context=body.text)
        return result.to_dict()

    @app.get("/v1/mcp")
    def mcp_index(dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        pool = dep.tools.mcp_pool
        if pool is None:
            return {"connected": False, "servers": {}, "hint": "start with --with-mcp"}
        return {"connected": True, **pool.index()}

    @app.get("/v1/metrics")
    def metrics_json(dep: Runtime = Depends(get_rt)) -> dict[str, Any]:
        return {"summary": dep.metrics.summary(), "raw": dep.metrics.snapshot()}

    @app.get("/metrics")
    def metrics_prometheus(dep: Runtime = Depends(get_rt)) -> Any:
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(dep.metrics.to_prometheus(), media_type="text/plain; version=0.0.4")

    return app


app = create_app()

__all__ = ["app", "create_app"]