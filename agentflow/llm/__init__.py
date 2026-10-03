"""LLM layer: provider-agnostic client, model catalog and offline stub."""

from __future__ import annotations

from agentflow.llm.catalog import (
    CATALOG,
    ModelCatalog,
    ModelSpec,
    Quality,
    Tier,
    catalog,
)
from agentflow.llm.client import AnthropicClient
from agentflow.llm.fake import ScriptedClient
from agentflow.llm.types import (
    ContentBlock,
    LLMClient,
    LLMResponse,
    Message,
    ToolSchema,
    Usage,
)

__all__ = [
    "CATALOG",
    "AnthropicClient",
    "ContentBlock",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ModelCatalog",
    "ModelSpec",
    "Quality",
    "ScriptedClient",
    "Tier",
    "ToolSchema",
    "Usage",
    "catalog",
]