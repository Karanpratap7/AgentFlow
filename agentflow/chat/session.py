"""Chat agent with sliding-window memory and automatic compaction.

Conversation state is the main driver of context cost: an uncapped session grows
linearly until it hits the context window and then fails. This session manager:

* keeps the last ``window`` turns verbatim,
* folds older turns into a running summary,
* prunes the summary when *it* grows too large (recursively, so memory is
  O(1) in session length),
* enforces the session token budget before every call.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.types import LLMClient, Message
from agentflow.routing.intent import IntentDetector, IntentResult
from agentflow.routing.router import AutoRouter, RoutingDecision
from agentflow.telemetry import metrics
from agentflow.tokens.budget import TokenBudget
from agentflow.tokens.counter import estimate_tokens, squeeze_whitespace
from agentflow.tokens.optimizer import TokenOptimizer

SUMMARY_SYSTEM = (
    "Compress a conversation into durable facts. Preserve: decisions, "
    "constraints, user preferences, names, identifiers, unresolved questions. "
    "Discard: pleasantries, restated context, superseded answers. "
    "Reply with at most 120 words of plain prose."
)


@dataclass
class ChatSession:
    """One conversation's memory and spend."""

    session_id: str
    window: int = 6
    budget: TokenBudget = field(default_factory=TokenBudget)
    turns: list[Message] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    turns_ingested: int = 0
    compactions: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.summary_tokens + sum(
            estimate_tokens([m]) for m in self.turns
        )

    def to_dict(self) -> dict[str, Any]:
        snap = self.budget.snapshot()
        return {
            "session_id": self.session_id,
            "window": self.window,
            "turns_in_memory": len(self.turns),
            "turns_ingested": self.turns_ingested,
            "compactions": self.compactions,
            "summary_tokens": self.summary_tokens,
            "context_tokens": self.prompt_tokens,
            "budget": snap.to_dict(),
        }


class SessionStore:
    """Thread-safe LRU of chat sessions."""

    def __init__(self, *, default_window: int = 6, capacity: int = 256, token_limit: int = 180_000) -> None:
        self.default_window = default_window
        self.capacity = capacity
        self.token_limit = token_limit
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ChatSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(
                    session_id=session_id,
                    window=self.default_window,
                    budget=TokenBudget(session_id=session_id, token_limit=self.token_limit),
                )
                self._sessions[session_id] = session
                while len(self._sessions) > self.capacity:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            return session

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "capacity": self.capacity,
                "context_tokens": sum(s.prompt_tokens for s in self._sessions.values()),
                "compactions": sum(s.compactions for s in self._sessions.values()),
            }

    def __len__(self) -> int:
        return len(self._sessions)


@dataclass(slots=True)
class ChatReply:
    text: str
    session_id: str
    routing: RoutingDecision
    intent: IntentResult
    context_tokens: int
    compactions: int
    turn_index: int
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.text,
            "session_id": self.session_id,
            "routing": self.routing.to_dict(),
            "intent": self.intent.to_dict(),
            "context_tokens": self.context_tokens,
            "compactions": self.compactions,
            "turn_index": self.turn_index,
            "budget": self.budget,
        }


class ChatAgent:
    """Conversational agent with auto intent detection, routing and compaction."""

    system_prompt = (
        "You are AgentFlow's chat assistant for an engineering team.\n"
        "\n"
        "1. Use the conversation summary and recent turns to resolve references.\n"
        "2. Prefer the conversation summary over re-deriving earlier context.\n"
        "3. Keep replies under 150 words unless a longer answer is requested.\n"
        "4. Say so when the available context is insufficient.\n"
        "5. Never restate the user's question back to them.\n"
    )

    def __init__(
        self,
        client: LLMClient,
        *,
        settings: Settings | None = None,
        guardrails: GuardrailPipeline | None = None,
        router: AutoRouter | None = None,
        intent_detector: IntentDetector | None = None,
        optimizer: TokenOptimizer | None = None,
        sessions: SessionStore | None = None,
        window: int = 6,
    ) -> None:
        self.settings = settings or default_settings
        self.client = client
        self.guardrails = guardrails or GuardrailPipeline(self.settings)
        self.router = router or AutoRouter(self.settings)
        self.intent_detector = intent_detector or IntentDetector(client)
        self.optimizer = optimizer or TokenOptimizer(
            cache_min_tokens=self.settings.cache_min_tokens
        )
        # ``is None``, not ``or``: SessionStore defines __len__, so a freshly
        # created empty store is falsy and ``sessions or SessionStore(...)`` would
        # silently hand the caller a second store and lose all conversation state.
        self.sessions = (
            sessions
            if sessions is not None
            else SessionStore(
                default_window=window, token_limit=self.settings.session_token_budget
            )
        )

    # -- public API -------------------------------------------------------
    def chat(self, message: str, *, session_id: str = "default") -> ChatReply:
        session = self.sessions.get(session_id)
        cleaned = self.guardrails.check_input(message).text
        self._record(session, "user", cleaned)

        intent = self.intent_detector.detect(cleaned)
        history = self._build_context(session)
        decision = self.router.route(
            cleaned,
            intent=intent,
            expected_input_tokens=estimate_tokens(history, self.system_prompt),
            cache_prefix_tokens=estimate_tokens(history[:1], self.system_prompt),
        )
        session.budget.check(decision.expected_input_tokens)

        payload = self.optimizer.optimize(
            history,
            system=self.system_prompt,
            summary=session.summary or None,
        )
        response = self.client.complete(
            payload.messages,
            system=payload.system,
            model=decision.model,
            max_output_tokens=decision.max_output_tokens,
            temperature=0.4,
        )
        session.budget.charge(response.model, response.usage, cached=response.cached)

        guard = self.guardrails.check_output(response.text, context=message)
        self._record(session, "assistant", guard.text)
        self._compact_if_needed(session)

        metrics.incr("requests_total")
        metrics.incr("chat_turns_total")
        metrics.incr("intent_total", intent=str(intent.intent))
        metrics.incr("route_model_total", model=decision.model)
        metrics.observe("chat_context_tokens", payload.report.tokens_after)
        if session.compactions:
            metrics.incr("chat_compactions_total")

        return ChatReply(
            text=guard.text,
            session_id=session_id,
            routing=decision,
            intent=intent,
            context_tokens=payload.report.tokens_after,
            compactions=session.compactions,
            turn_index=session.turns_ingested,
            budget=session.budget.snapshot().to_dict(),
        )

    def reset(self, session_id: str) -> bool:
        return self.sessions.drop(session_id)

    def stats(self) -> dict[str, Any]:
        return self.sessions.stats()

    # -- memory -----------------------------------------------------------
    def _build_context(self, session: ChatSession) -> list[Message]:
        messages: list[Message] = []
        if session.summary:
            messages.append(
                Message("user", f"[Summary of earlier conversation]\n{session.summary}")
            )
        messages.extend(session.turns[-session.window :])
        return messages

    def _record(self, session: ChatSession, role: str, text: str) -> None:
        session.turns.append(Message(role, squeeze_whitespace(text)))  # type: ignore[arg-type]
        session.turns_ingested += 1

    def _compact_if_needed(self, session: ChatSession) -> None:
        """Fold turns beyond the window into the running summary."""
        overflow = len(session.turns) - session.window
        if overflow <= 0:
            return
        folded = session.turns[:-session.window]
        session.turns = session.turns[-session.window :]

        transcript = "\n".join(f"{m.role}: {m.plain_text()}" for m in folded)
        prompt = (
            f"Existing summary:\n{session.summary or '(none)'}\n\n"
            f"New turns to fold in:\n{transcript}\n\nUpdated summary:"
        )
        response = self.client.complete(
            [Message("user", prompt)],
            system=SUMMARY_SYSTEM,
            model=self.settings.small_model,
            max_output_tokens=300,
            temperature=0.0,
            cache_system=False,
        )
        session.summary = squeeze_whitespace(response.text)
        session.summary_tokens = estimate_tokens([Message("user", session.summary)])
        session.compactions += 1

        # Recursive bound: if the summary itself grows too large, fold it again.
        if session.summary_tokens > self.settings.cache_min_tokens:
            shrunk = self.client.complete(
                [
                    Message(
                        "user",
                        f"Compress this summary to 60 words, dropping detail:\n\n{session.summary}",
                    )
                ],
                system=SUMMARY_SYSTEM,
                model=self.settings.small_model,
                max_output_tokens=200,
                temperature=0.0,
                cache_system=False,
            )
            session.summary = squeeze_whitespace(shrunk.text)
            session.summary_tokens = estimate_tokens([Message("user", session.summary)])
            session.compactions += 1

    # -- introspection ----------------------------------------------------
    def context_report(self, session_id: str = "default") -> dict[str, Any]:
        session = self.sessions.get(session_id)
        history = self._build_context(session)
        return {
            **session.to_dict(),
            "system_prompt_tokens": estimate_tokens([Message("user", self.system_prompt)]),
            "messages_in_context": len(history),
            "window_tokens": estimate_tokens(history),
        }


__all__ = [
    "SUMMARY_SYSTEM",
    "ChatAgent",
    "ChatReply",
    "ChatSession",
    "SessionStore",
]