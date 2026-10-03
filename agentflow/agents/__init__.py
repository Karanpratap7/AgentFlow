"""Agent layer: base loop, specialists and the parallel pool."""

from __future__ import annotations

from agentflow.agents.base import Agent, AgentResult, StepTrace
from agentflow.agents.pool import AgentPool, AgentRegistry, ParallelOutcome, default_registry
from agentflow.agents.specialists import (
    CodeAgent,
    ResearchAgent,
    SQLAgent,
    SynthesizerAgent,
    TriageAgent,
)

__all__ = [
    "Agent",
    "AgentPool",
    "AgentRegistry",
    "AgentResult",
    "CodeAgent",
    "ParallelOutcome",
    "ResearchAgent",
    "SQLAgent",
    "StepTrace",
    "SynthesizerAgent",
    "TriageAgent",
    "default_registry",
]